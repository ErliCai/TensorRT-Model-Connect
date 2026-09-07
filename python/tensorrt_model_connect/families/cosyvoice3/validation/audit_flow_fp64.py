# SPDX-License-Identifier: Apache-2.0
"""FP64 oracle audit: native FP32 and official FP32 errors against float64 truth.

The pinned official DiT evaluated in float64 is the oracle. Both the native
engine and the same official model in FP32 (TF32 disabled, default SDPA) are
measured against it with the family tolerance on identical inputs. This
explains FP32-versus-FP32 gate failures; it does not replace those gates and
its exit code reflects execution errors only.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np

from ..artifacts import sha256_file
from ..config import FLOW_SHA256
from ..flow_runtime import FlowEngine, INPUT_NAMES
from .parity_metrics import compare_outputs
from .validate_flow_pytorch import ATOL, RTOL, _cases, _ieee_fp32_reference, _official_dit

COMPARISONS = ("native_vs_fp64", "official_fp32_vs_fp64", "native_vs_official_fp32")


def load_cases(acoustic_evidence):
    cases = [(f"stress_{frames}_{int(masked)}", "original_stress", values) for frames, masked, values in _cases()]
    if acoustic_evidence:
        report = json.loads((acoustic_evidence / "report.json").read_text(encoding="utf-8"))
        with np.load(acoustic_evidence / "acoustic_evidence.npz", allow_pickle=False) as saved:
            for label in sorted({row["case_id"] for row in report["cases"]}):
                cases.extend((f"{label}_step{step}", "acoustic_trajectory",
                              {k: saved[f"{label}_step{step}_{k}"].copy() for k in INPUT_NAMES}) for step in range(10))
    return cases


def summarize(rows):
    summary = {}
    for group in sorted({row["group"] for row in rows}):
        selected = [row for row in rows if row["group"] == group]
        summary[group] = {name: {
            "passed": sum(row[name]["passed"] for row in selected), "total": len(selected),
            "max_abs_error": max(row[name]["max_abs_error"] for row in selected),
            "max_tolerance_ratio": max(row[name]["max_tolerance_ratio"] for row in selected),
            "elements_over_tolerance": sum(row[name]["elements_over_tolerance"] for row in selected),
            "total_elements": sum(row[name]["total_elements"] for row in selected),
        } for name in COMPARISONS}
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("component", "model-dir", "cosyvoice-source", "report"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--acoustic-evidence", type=Path, help="validate_offline_flow output; adds its 80 official trajectory steps")
    args = parser.parse_args(argv)
    import torch

    if args.report.exists():
        raise FileExistsError(f"Choose a new report path: {args.report}")
    if sha256_file(args.model_dir / "flow.pt") != FLOW_SHA256:
        raise ValueError("Checkpoint does not match pinned model")
    DiT, revision = _official_dit(args.cosyvoice_source)
    cases = load_cases(args.acoustic_evidence)

    engine = FlowEngine(args.component)
    native = {}
    for label, _, values in cases:
        native[label] = engine(**{k: torch.from_numpy(v).to(engine.device) for k, v in values.items()}).cpu().numpy()
    plan_hash = sha256_file(args.component / "flow.plan")
    del engine
    gc.collect()
    torch.cuda.empty_cache()

    model = DiT(dim=1024, depth=22, heads=16, dim_head=64, ff_mult=2, mel_dim=80, mu_dim=80,
                spk_dim=80, out_channels=80, static_chunk_size=50, num_decoding_left_chunks=-1).eval()
    state = torch.load(args.model_dir / "flow.pt", map_location="cpu", weights_only=True, mmap=True)
    prefix = "decoder.estimator."
    model.load_state_dict({k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}, strict=True)
    del state
    gc.collect()
    model.cuda()
    official = {}
    with torch.inference_mode(), _ieee_fp32_reference(torch):
        for label, _, values in cases:
            official[label] = model(**{k: torch.from_numpy(v).cuda() for k, v in values.items()}, streaming=False).cpu().numpy()
    # float64 is not a production path; it is the arithmetic truth for this checkpoint.
    model.double()
    rows = []
    with torch.inference_mode():
        for label, group, values in cases:
            oracle = model(**{k: torch.from_numpy(v).cuda().double() for k, v in values.items()},
                           streaming=False).cpu().numpy().astype(np.float32)
            row = {"id": label, "group": group,
                   "native_vs_fp64": compare_outputs(native[label], oracle, atol=ATOL, rtol=RTOL),
                   "official_fp32_vs_fp64": compare_outputs(official[label], oracle, atol=ATOL, rtol=RTOL),
                   "native_vs_official_fp32": compare_outputs(native[label], official[label], atol=ATOL, rtol=RTOL)}
            rows.append(row)
            print(json.dumps({"id": label, **{name: row[name]["max_tolerance_ratio"] for name in COMPARISONS}}), flush=True)
    report = {
        "scope": "fp64_oracle_audit_explains_fp32_gates_not_acceptance", "atol": ATOL, "rtol": RTOL,
        "cosyvoice_source_revision": revision, "checkpoint_sha256": FLOW_SHA256, "plan_sha256": plan_hash,
        "official_fp32_precision": {"cudnn_tf32": False, "matmul_tf32": False, "matmul_precision": "highest", "sdpa_policy": "auto"},
        "torch_version": torch.__version__, "gpu": torch.cuda.get_device_name(),
        "summary": summarize(rows), "cases": rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
