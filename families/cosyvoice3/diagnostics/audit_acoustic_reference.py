# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reference-only backend audit on saved real-audio trajectories; NOT a gate.

No native engine is run and no acceptance threshold is changed. Completion
means evidence was collected, not that model parity has been established.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

import numpy as np

from ..flow_runtime import INPUT_NAMES
from ..validation.parity_metrics import compare_outputs
from ..validation.validate_flow_pytorch import ATOL, RTOL, _ieee_fp32_reference, _official_dit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("evidence", "model-dir", "cosyvoice-source", "report"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel

    if args.report.exists():
        raise FileExistsError(args.report)
    evidence_report = json.loads((args.evidence / "report.json").read_text(encoding="utf-8"))
    artifact = args.evidence / "acoustic_evidence.npz"
    DiT, revision = _official_dit(args.cosyvoice_source)
    if revision != evidence_report["source_revision"]:
        raise ValueError("Evidence has a different upstream reference")
    model = DiT(dim=1024, depth=22, heads=16, dim_head=64, ff_mult=2, mel_dim=80, mu_dim=80,
                spk_dim=80, out_channels=80, static_chunk_size=50, num_decoding_left_chunks=-1).eval()
    state = torch.load(args.model_dir / "flow.pt", weights_only=True, mmap=True, map_location="cpu")
    prefix = "decoder.estimator."
    model.load_state_dict({k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}, strict=True)
    del state
    model.cuda()
    rows = []
    with np.load(artifact, allow_pickle=False) as saved, torch.inference_mode(), _ieee_fp32_reference(torch):
        cases = sorted({row["case_id"] for row in evidence_report["cases"]})
        if len(cases) != 8:
            raise ValueError("Expected all eight acoustic cases")
        for case in cases:
            for step in range(10):
                stem = f"{case}_step{step}_"
                values = {k: torch.from_numpy(saved[stem + k].copy()).cuda() for k in INPUT_NAMES}
                auto = model(**values, streaming=False).cpu().numpy()
                with sdpa_kernel(SDPBackend.MATH):
                    math = model(**values, streaming=False).cpu().numpy()
                for stage, actual, expected in (("auto_replay_vs_recorded_official", auto, saved[stem + "velocity"]),
                                                ("official_math_vs_official_auto", math, auto)):
                    rows.append({"case_id": case, "step": step, "stage": stage,
                                 "bit_exact": bool(np.array_equal(actual, expected)),
                                 **compare_outputs(actual, expected, atol=ATOL, rtol=RTOL)})
            print(json.dumps({"case_id": case, "collected": 20}), flush=True)
    report = {"scope": "diagnostic_reference_only_not_native_acceptance", "collection_complete": True,
              "source_revision": revision, "evidence_path": str(artifact), "torch_version": torch.__version__,
              "x_transformers_version": importlib.metadata.version("x-transformers"),
              "gpu": torch.cuda.get_device_name(), "atol": ATOL, "rtol": RTOL,
              "precision": "FP32_TF32_disabled_auto_vs_MATH_SDPA", "cases": rows,
              "summary": {stage: {"passed": sum(row["passed"] for row in rows if row["stage"] == stage),
                                  "bit_exact": sum(row["bit_exact"] for row in rows if row["stage"] == stage),
                                  "total": 80} for stage in ("auto_replay_vs_recorded_official", "official_math_vs_official_auto")}}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
