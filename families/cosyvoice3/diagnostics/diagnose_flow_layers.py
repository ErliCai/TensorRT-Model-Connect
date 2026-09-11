# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Locate the first native-TRT divergence from pinned official PyTorch.

This diagnostic builds a temporary in-memory TensorRT engine whose additional
outputs are the time embedding, input embedding, every DiT block and final
normalization. It never changes the production component's output contract.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np

from ..checkpoint_mapper import load_flow_weights
from ..config import ShapeProfile, read_config
from ..flow_builder import build_flow_engine
from ..validation.validate_flow_pytorch import ATOL, RTOL, _cases, _ieee_fp32_reference, _official_dit


def _run_debug_plan(plan, values):
    import tensorrt as trt
    import torch

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    if engine is None:
        raise RuntimeError("Could not deserialize debug engine")
    context = engine.create_execution_context()
    tensors = {name: torch.from_numpy(value).cuda().contiguous() for name, value in values.items()}
    tensors["positions"] = torch.arange(values["x"].shape[-1], device="cuda", dtype=torch.int32)
    for name, tensor in tensors.items():
        if not context.set_input_shape(name, tuple(tensor.shape)):
            raise RuntimeError(f"TensorRT rejected {name} shape")
        if not context.set_tensor_address(name, tensor.data_ptr()):
            raise RuntimeError(f"TensorRT rejected {name} address")
    unresolved = context.infer_shapes()
    if unresolved:
        raise RuntimeError(f"Unresolved debug shapes: {unresolved}")
    outputs = {}
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        if engine.get_tensor_mode(name) != trt.TensorIOMode.OUTPUT:
            continue
        shape = tuple(context.get_tensor_shape(name))
        outputs[name] = torch.empty(shape, dtype=torch.float32, device="cuda")
        if not context.set_tensor_address(name, outputs[name].data_ptr()):
            raise RuntimeError(f"TensorRT rejected {name} output address")
    stream = torch.cuda.current_stream()
    try:
        if not context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT debug execution failed")
    finally:
        stream.synchronize()
    result = {name: tensor.cpu().numpy() for name, tensor in outputs.items()}
    del context, engine, runtime, outputs, tensors
    torch.cuda.empty_cache()
    gc.collect()
    return result


def _official_outputs(DiT, model_dir, values):
    import torch

    model = DiT(
        dim=1024, depth=22, heads=16, dim_head=64, ff_mult=2,
        mel_dim=80, mu_dim=80, spk_dim=80, out_channels=80,
        static_chunk_size=50, num_decoding_left_chunks=-1,
    ).eval()
    state = torch.load(model_dir / "flow.pt", map_location="cpu", weights_only=True, mmap=True)
    prefix = "decoder.estimator."
    model.load_state_dict(
        {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)},
        strict=True,
    )
    del state
    gc.collect()
    captured = {}

    def hook(name, transform=lambda output: output):
        def save(_module, _inputs, output):
            captured[name] = transform(output).detach().cpu().numpy()
        return save

    handles = [
        model.time_embed.register_forward_hook(hook("debug_time")),
        model.input_embed.proj.register_forward_hook(hook("debug_input_projection")),
        model.input_embed.conv_pos_embed.conv1.register_forward_hook(
            hook("debug_conv_1", lambda output: output.transpose(1, 2))
        ),
        model.input_embed.conv_pos_embed.conv2.register_forward_hook(
            hook("debug_conv_2", lambda output: output.transpose(1, 2))
        ),
        model.input_embed.register_forward_hook(hook("debug_input")),
    ]
    handles += [block.register_forward_hook(hook(f"debug_block_{index:02d}"))
                for index, block in enumerate(model.transformer_blocks)]
    handles.append(model.norm_out.register_forward_hook(hook("debug_norm_out")))
    model.cuda()
    with torch.inference_mode(), _ieee_fp32_reference(torch):
        captured["velocity"] = model(
            **{name: torch.from_numpy(value).cuda() for name, value in values.items()},
            streaming=False,
        ).cpu().numpy()
    for handle in handles:
        handle.remove()
    del model
    torch.cuda.empty_cache()
    gc.collect()
    return captured


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--cosyvoice-source", type=Path, required=True)
    parser.add_argument("--frames", type=int, choices=(4, 17, 64, 128), default=17)
    parser.add_argument("--masked", action="store_true")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.report.exists():
        raise FileExistsError(f"Choose a new report path: {args.report}")
    selected = next(values for frames, masked, values in _cases()
                    if frames == args.frames and masked == args.masked)
    cfg = read_config(args.model_dir)
    profile = ShapeProfile(4, 32, 128)
    weights = load_flow_weights(args.model_dir, cfg)
    plan = build_flow_engine(weights, cfg, profile, workspace_mib=256, debug_outputs=True)
    del weights
    gc.collect()
    actual = _run_debug_plan(plan, selected)
    del plan
    gc.collect()
    DiT, revision = _official_dit(args.cosyvoice_source)
    expected = _official_outputs(DiT, args.model_dir, selected)
    if actual.keys() != expected.keys():
        raise ValueError(f"Debug output mismatch: TRT={sorted(actual)}, PyTorch={sorted(expected)}")
    order = [
        "debug_time", "debug_input_projection", "debug_conv_1", "debug_conv_2", "debug_input",
    ] + [f"debug_block_{i:02d}" for i in range(cfg.depth)] + ["debug_norm_out", "velocity"]
    rows = []
    for name in order:
        difference = np.abs(actual[name] - expected[name])
        rows.append({
            "stage": name, "passed": bool(np.allclose(actual[name], expected[name], atol=ATOL, rtol=RTOL)),
            "max_abs_error": float(difference.max()), "mean_abs_error": float(difference.mean()),
            "reference_abs_max": float(np.abs(expected[name]).max()),
        })
        print(json.dumps(rows[-1]), flush=True)
    first_failure = next((row["stage"] for row in rows if not row["passed"]), None)
    report = {
        "scope": "layerwise_native_trt_vs_pinned_official_pytorch",
        "frames": args.frames, "masked": args.masked, "atol": ATOL, "rtol": RTOL,
        "cosyvoice_source_revision": revision, "first_failure": first_failure, "stages": rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
