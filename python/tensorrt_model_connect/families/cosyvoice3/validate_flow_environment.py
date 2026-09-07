# SPDX-License-Identifier: Apache-2.0
"""Two-process exact-input parity: TensorRT capture, isolated PyTorch reference.

Separating processes allows testing upstream torch 2.3.1 without replacing the
CUDA-13 TensorRT runtime environment. Existing gates/inputs remain unchanged.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

import numpy as np

from .__main__ import sha256_file
from .config import FLOW_SHA256
from .flow_runtime import FlowEngine, INPUT_NAMES
from .parity_metrics import compare_outputs
from .validate_flow_pytorch import ATOL, RTOL, _cases, _ieee_fp32_reference, _official_dit


def validate_case_manifest(cases, *, has_acoustic):
    expected = {(f"stress_{frames}_{masked}", "original_stress") for frames in (4, 17, 64, 128) for masked in (0, 1)}
    if has_acoustic:
        labels = [f"tokens{count}_prompt{prompt}" for count in (8, 32, 64, 128) for prompt in (0, min(25, count // 2))]
        expected |= {(f"{label}_step{step}", "acoustic_trajectory") for label in labels for step in range(10)}
    if (not isinstance(cases, list) or any(not isinstance(case, dict) or set(case) != {"id", "group"} for case in cases)
            or len(cases) != len(expected) or {(case["id"], case["group"]) for case in cases} != expected):
        raise ValueError("Capture must contain every declared case exactly once; no omitted or substituted gates")


def capture(args):
    import torch

    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True, exist_ok=False)
    engine = FlowEngine(args.component)
    cases = [(f"stress_{frames}_{int(masked)}", "original_stress", values) for frames, masked, values in _cases()]
    acoustic_hash = None
    if args.acoustic_evidence:
        report = json.loads((args.acoustic_evidence / "report.json").read_text(encoding="utf-8"))
        artifact = args.acoustic_evidence / "acoustic_evidence.npz"
        acoustic_hash = sha256_file(artifact)
        if acoustic_hash != report["evidence_sha256"] or report["plan_sha256"] != sha256_file(args.component / "flow.plan"):
            raise ValueError("Acoustic evidence hash or plan binding mismatch")
        with np.load(artifact, allow_pickle=False) as saved:
            labels = sorted({row["case_id"] for row in report["cases"]})
            if len(labels) != 8:
                raise ValueError("Expected eight acoustic cases")
            cases.extend((f"{label}_step{step}", "acoustic_trajectory", {k: saved[f"{label}_step{step}_{k}"].copy() for k in INPUT_NAMES})
                         for label in labels for step in range(10))
    saved, metadata = {}, []
    for label, group, values in cases:
        actual = engine(**{k: torch.from_numpy(v).cuda() for k, v in values.items()}).cpu().numpy()
        saved.update({label + "_" + k: v for k, v in values.items()})
        saved[label + "_native"] = actual
        metadata.append({"id": label, "group": group})
    np.savez(args.output / "native_inputs_outputs.npz", **saved)
    manifest = {"scope": "exact_input_native_capture_not_acceptance", "cases": metadata,
                "plan_sha256": sha256_file(args.component / "flow.plan"), "torch_version": torch.__version__,
                "acoustic_evidence_sha256": acoustic_hash,
                "artifact_sha256": sha256_file(args.output / "native_inputs_outputs.npz")}
    validate_case_manifest(metadata, has_acoustic=acoustic_hash is not None)
    with (args.output / "manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(json.dumps({"captured": len(cases), "plan_sha256": manifest["plan_sha256"]}))


def compare(args):
    import torch

    if args.report.exists():
        raise FileExistsError(args.report)
    manifest = json.loads((args.capture / "manifest.json").read_text(encoding="utf-8"))
    validate_case_manifest(manifest["cases"], has_acoustic=manifest["acoustic_evidence_sha256"] is not None)
    artifact = args.capture / "native_inputs_outputs.npz"
    if sha256_file(artifact) != manifest["artifact_sha256"]:
        raise ValueError("Native capture checksum mismatch")
    if sha256_file(args.model_dir / "flow.pt") != FLOW_SHA256:
        raise ValueError("Checkpoint does not match pinned model")
    DiT, revision = _official_dit(args.cosyvoice_source)
    model = DiT(dim=1024, depth=22, heads=16, dim_head=64, ff_mult=2, mel_dim=80, mu_dim=80,
                spk_dim=80, out_channels=80, static_chunk_size=50, num_decoding_left_chunks=-1).eval()
    state = torch.load(args.model_dir / "flow.pt", weights_only=True, mmap=True, map_location="cpu")
    prefix = "decoder.estimator."
    model.load_state_dict({k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}, strict=True)
    del state
    model.cuda()
    rows = []
    with np.load(artifact, allow_pickle=False) as saved, torch.inference_mode(), _ieee_fp32_reference(torch):
        for case in manifest["cases"]:
            label = case["id"]
            values = {k: torch.from_numpy(saved[label + "_" + k].copy()).cuda() for k in INPUT_NAMES}
            expected = model(**values, streaming=False).cpu().numpy()
            rows.append({**case, **compare_outputs(saved[label + "_native"], expected, atol=ATOL, rtol=RTOL)})
    report = {"scope": "native_vs_official_in_separate_environment_not_full_tts", "passed": all(row["passed"] for row in rows),
              "source_revision": revision, "checkpoint_sha256": FLOW_SHA256, "native_capture": manifest,
              "reference_versions": {name: importlib.metadata.version(name) for name in ("torch", "torchaudio", "numpy", "x-transformers")},
              "reference_precision": "FP32_TF32_disabled_auto_SDPA", "atol": ATOL, "rtol": RTOL,
              "gpu": torch.cuda.get_device_name(), "cases": rows,
              "summary": {group: {"passed": sum(row["passed"] for row in rows if row["group"] == group),
                                  "total": sum(row["group"] == group for row in rows)} for group in sorted({r["group"] for r in rows})}}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report["summary"], indent=2))
    if not report["passed"]:
        raise SystemExit("Separate-environment parity FAILED; existing gates remain unchanged")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    native = commands.add_parser("capture")
    native.add_argument("--component", type=Path, required=True)
    native.add_argument("--output", type=Path, required=True)
    native.add_argument("--acoustic-evidence", type=Path)
    reference = commands.add_parser("compare")
    for name in ("capture", "model-dir", "cosyvoice-source", "report"):
        reference.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    (capture if args.command == "capture" else compare)(args)


if __name__ == "__main__":
    main()
