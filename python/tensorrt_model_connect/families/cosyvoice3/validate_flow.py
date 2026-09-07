# SPDX-License-Identifier: Apache-2.0
"""Compare the native component with an externally supplied upstream FP32 ONNX.

ONNX Runtime is an independent validation oracle only, never the build route.
An estimator comparison is not an audio-quality or complete TTS acceptance test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .__main__ import sha256_file
from .flow_runtime import FlowEngine, INPUT_NAMES
from .flow_matching import solve_euler
from .config import MODEL_ID, MODEL_REVISION, ORACLE_SHA256
from .parity_metrics import compare_outputs

# Gates shared with the official-PyTorch validator, which documents their
# FP64-oracle calibration; ONNX Runtime FP32 is another FP32 reference.
from .validate_flow_pytorch import ATOL, INTEGRATED_ATOL, INTEGRATED_RTOL, RTOL


def main(argv=None):
    import onnxruntime as ort
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", type=Path, required=True)
    parser.add_argument("--oracle-onnx", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="+", default=[4, 17, 64])
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.report.exists():
        raise FileExistsError(f"Choose a new report path: {args.report}")
    oracle_hash = sha256_file(args.oracle_onnx)
    if oracle_hash != ORACLE_SHA256:
        raise ValueError(f"Oracle checksum mismatch; use {MODEL_ID} at {MODEL_REVISION}")
    engine = FlowEngine(args.component)
    for frames in args.frames:
        engine.profile.validate_frames(frames)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    # Avoid a second multi-GB optimized copy on memory-constrained hosts.
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    reference = ort.InferenceSession(str(args.oracle_onnx), options, providers=["CPUExecutionProvider"])
    if {x.name for x in reference.get_inputs()} != set(INPUT_NAMES):
        raise ValueError("Oracle must be the upstream offline six-input FP32 estimator")
    if any(x.type != "tensor(float)" for x in reference.get_inputs()) or len(reference.get_outputs()) != 1:
        raise ValueError("Unexpected oracle I/O dtype or output count")
    rng = np.random.default_rng(2512)
    rows = []
    passed = True
    for frames in args.frames:
        for masked in (False, True):
            values = {
                "x": rng.normal(size=(2, 80, frames)).astype(np.float32),
                "mask": np.ones((2, 1, frames), np.float32),
                "mu": rng.normal(size=(2, 80, frames)).astype(np.float32),
                "t": np.array([0.0, 0.7], np.float32),
                "spks": rng.normal(size=(2, 80)).astype(np.float32),
                "cond": rng.normal(size=(2, 80, frames)).astype(np.float32),
            }
            if masked and frames > 1:
                values["mask"][1, :, -min(3, frames - 1):] = 0
            expected = reference.run(None, values)[0]
            tensors = {k: torch.from_numpy(v).to(engine.device) for k, v in values.items()}
            actual = engine(**tensors).cpu().numpy()
            row = {"frames": frames, "masked": masked,
                   **compare_outputs(actual, expected, atol=ATOL, rtol=RTOL)}
            passed &= row["passed"]
            rows.append(row)
            print(json.dumps(row), flush=True)
    # Also exercise repeated integration: per-step errors can accumulate even
    # when every isolated velocity prediction meets the component tolerance.
    frames = args.frames[0]
    mu = torch.from_numpy(rng.normal(size=(1, 80, frames)).astype(np.float32)).to(engine.device)
    mask = torch.ones((1, 1, frames), device=engine.device)
    spks = torch.from_numpy(rng.normal(size=(1, 80)).astype(np.float32)).to(engine.device)
    cond = torch.from_numpy(rng.normal(size=(1, 80, frames)).astype(np.float32)).to(engine.device)
    noise = torch.from_numpy(rng.normal(size=(1, 80, frames)).astype(np.float32)).to(engine.device)

    def oracle(x, mask, mu, t, spks, cond, *, streaming):
        if streaming:
            raise ValueError("Offline oracle only")
        values = dict(zip(INPUT_NAMES, (x, mask, mu, t, spks, cond)))
        output = reference.run(None, {k: v.cpu().numpy() for k, v in values.items()})[0]
        return torch.from_numpy(output).to(engine.device)

    actual = solve_euler(engine, mu, mask, spks, cond, noise).cpu().numpy()
    expected = solve_euler(oracle, mu, mask, spks, cond, noise).cpu().numpy()
    row = {"stage": "10_step_euler", "frames": frames, "atol": INTEGRATED_ATOL, "rtol": INTEGRATED_RTOL,
           **compare_outputs(actual, expected, atol=INTEGRATED_ATOL, rtol=INTEGRATED_RTOL)}
    passed &= row["passed"]
    rows.append(row)
    print(json.dumps(row), flush=True)
    report = {
        "component": "cosyvoice3_flow_estimator", "passed": passed,
        "scope": "estimator_numerical_parity_not_end_to_end_tts",
        "atol": ATOL, "rtol": RTOL, "integrated_atol": INTEGRATED_ATOL, "integrated_rtol": INTEGRATED_RTOL,
        "seed": 2512, "plan_sha256": sha256_file(args.component / "flow.plan"),
        "oracle_sha256": oracle_hash,
        "oracle_model_id": MODEL_ID, "oracle_model_revision": MODEL_REVISION,
        "torch_version": torch.__version__, "onnxruntime_version": ort.__version__,
        "gpu": torch.cuda.get_device_name(), "cases": rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    if not passed:
        raise SystemExit("Flow parity FAILED; inspect the report, do not relax the thresholds")


if __name__ == "__main__":
    main()
