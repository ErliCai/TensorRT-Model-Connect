# SPDX-License-Identifier: Apache-2.0
"""Compare the native component with the pinned official CosyVoice PyTorch DiT.

This validator imports the exact upstream source checkout supplied by the user,
verifies its Git revision and the model checkpoint digest, then compares the
same deterministic inputs used by ``validate_flow.py``. It does not treat a
local reimplementation of the PyTorch equations as the reference.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import importlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from .__main__ import sha256_file
from .config import FLOW_SHA256, MODEL_ID, MODEL_REVISION, SOURCE_REVISION
from .flow_runtime import FlowEngine, INPUT_NAMES
from .parity_metrics import compare_outputs

# Elementwise gates for one estimator call, calibrated on 2026-09-08 against
# a float64 evaluation of the pinned official model (audit_flow_fp64). The
# official FP32 model itself deviates from float64 truth by up to 6.6x the
# former 1e-3 tolerance (synthetic CFG trajectories) and the native engine by
# up to 6.2x, so a native versus official-FP32 comparison must admit their
# sum; the worst observed pair differs by 9.7x. These gates detect wrong
# mathematics, not sub-reference rounding; use the FP64 audit for that.
ATOL = 2e-2
RTOL = 2e-2
# Ten-step Euler integration with classifier-free guidance amplifies rounding:
# on the 256-frame prompt-free acoustic case official FP32 differs from the
# float64 integration by 29x the former tolerance, the native engine by 16x,
# and the two FP32 results from each other by 43.5x.
INTEGRATED_ATOL = 1e-1
INTEGRATED_RTOL = 1e-1


@contextmanager
def _ieee_fp32_reference(torch):
    """Match the native engine's FP32 policy in the official reference.

    PyTorch enables TF32 for cuDNN convolutions by default on Ampere GPUs,
    while ``build_flow_engine`` explicitly clears TensorRT's TF32 flag.  A
    parity test must compare equal precision policies; otherwise the first
    causal convolution differs before any model-specific TensorRT math runs.
    """
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    matmul_precision = torch.get_float32_matmul_precision()
    try:
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        yield
    finally:
        torch.backends.cudnn.allow_tf32 = cudnn_tf32
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        torch.set_float32_matmul_precision(matmul_precision)


def _official_dit(source: Path):
    revision = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if revision != SOURCE_REVISION:
        raise ValueError(f"CosyVoice source must be exactly {SOURCE_REVISION}; got {revision}")
    status = subprocess.run(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=all", "--", "cosyvoice"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if status:
        raise ValueError("Official CosyVoice source has local changes; use a clean pinned checkout")
    sys.path.insert(0, str(source))
    module = importlib.import_module("cosyvoice.flow.DiT.dit")
    resolved = Path(module.__file__).resolve()
    if source.resolve() not in resolved.parents:
        raise ValueError(f"Imported CosyVoice from the wrong checkout: {resolved}")
    return module.DiT, revision


def _cases():
    # Synthetic independent batch rows stress the estimator. These are NOT
    # paired CFG inputs from a speech request, nor a measure of TTS success.
    rng = np.random.default_rng(2512)
    result = []
    for frames in (4, 17, 64, 128):
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
            result.append((frames, masked, values))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--cosyvoice-source", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    # Keep ``--help`` usable in dependency-light environments. Heavy runtime
    # dependencies are only required after argument parsing has completed.
    import torch

    if args.report.exists():
        raise FileExistsError(f"Choose a new report path: {args.report}")
    if sha256_file(args.model_dir / "flow.pt") != FLOW_SHA256:
        raise ValueError(f"flow.pt must be from {MODEL_ID} at {MODEL_REVISION}")
    DiT, source_revision = _official_dit(args.cosyvoice_source)
    cases = _cases()

    # Capture TensorRT results first, then release the engine before allocating
    # the official model. This keeps the proof runnable on an 8 GiB GPU.
    engine = FlowEngine(args.component)
    actual_outputs = []
    for _, _, values in cases:
        tensors = {name: torch.from_numpy(values[name]).to(engine.device) for name in INPUT_NAMES}
        actual_outputs.append(engine(**tensors).cpu().numpy())
    device = engine.device
    plan_hash = sha256_file(args.component / "flow.plan")
    del engine
    torch.cuda.empty_cache()
    gc.collect()

    model = DiT(
        dim=1024, depth=22, heads=16, dim_head=64, ff_mult=2,
        mel_dim=80, mu_dim=80, spk_dim=80, out_channels=80,
        static_chunk_size=50, num_decoding_left_chunks=-1,
    ).eval()
    state = torch.load(
        args.model_dir / "flow.pt", map_location="cpu", weights_only=True, mmap=True,
    )
    prefix = "decoder.estimator."
    model.load_state_dict(
        {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)},
        strict=True,
    )
    del state
    gc.collect()
    model.to(device)

    rows = []
    passed = True
    with torch.inference_mode(), _ieee_fp32_reference(torch):
        for (frames, masked, values), actual in zip(cases, actual_outputs):
            expected = model(
                **{name: torch.from_numpy(values[name]).to(device) for name in INPUT_NAMES},
                streaming=False,
            ).cpu().numpy()
            row = {"frames": frames, "masked": masked,
                   **compare_outputs(actual, expected, atol=ATOL, rtol=RTOL)}
            passed &= row["passed"]
            rows.append(row)
            print(json.dumps(row), flush=True)

    report = {
        "component": "cosyvoice3_flow_estimator", "passed": passed,
        "scope": "estimator_vs_pinned_official_pytorch_not_end_to_end_tts",
        "atol": ATOL, "rtol": RTOL, "seed": 2512,
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "checkpoint_sha256": FLOW_SHA256,
        "cosyvoice_source_revision": source_revision,
        "plan_sha256": plan_hash, "torch_version": torch.__version__,
        "reference_precision": {"cudnn_tf32": False, "matmul_tf32": False,
                                "matmul_precision": "highest", "sdpa_policy": "auto"},
        "x_transformers_version": importlib.metadata.version("x-transformers"),
        "input_distribution": "synthetic_independent_batch_rows_not_cfg_pairs",
        "gpu": torch.cuda.get_device_name(device), "cases": rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    if not passed:
        raise SystemExit("Official PyTorch Flow parity FAILED; inspect the report")


if __name__ == "__main__":
    main()
