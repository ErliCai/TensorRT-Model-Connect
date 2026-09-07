# SPDX-License-Identifier: Apache-2.0
"""Supplemental CFG trajectory checks against the unmodified official solver.

Synthetic conditions obey the CFG pairing contract but are NOT real acoustic
features or an audio-quality test. This does not replace the eight stress gates.
"""

from __future__ import annotations

import argparse
import gc
import importlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

from .__main__ import sha256_file
from .config import FLOW_SHA256
from .flow_matching import solve_euler
from .flow_runtime import FlowEngine, INPUT_NAMES
from .parity_metrics import compare_outputs
from .validate_flow_pytorch import ATOL, RTOL, _official_dit, _ieee_fp32_reference


def _cfg_cases():
    rng = np.random.default_rng(2512)
    for frames in (4, 17, 64, 128):
        for masked in (False, True):
            values = {name: rng.normal(size=(1, 80, frames)).astype(np.float32)
                      for name in ("mu", "cond", "noise")}
            values["spks"] = rng.normal(size=(1, 80)).astype(np.float32)
            values["mask"] = np.ones((1, 1, frames), np.float32)
            if masked:
                values["mask"][:, :, -min(3, frames - 1):] = 0
            # Synthetic prompt conditioning only in the prefix, not all frames.
            values["cond"][:, :, frames // 2:] = 0
            yield frames, masked, values


def _official_solver(source):
    """Verify the required pinned submodule, then import the real upstream class."""
    matcha = source / "third_party/Matcha-TTS"

    def git(root, *args):
        return subprocess.run(["git", "-C", str(root), *args], check=True,
                              capture_output=True, text=True).stdout.strip()

    expected = git(source, "rev-parse", "HEAD:third_party/Matcha-TTS")
    if not (matcha / ".git").exists() or git(matcha, "rev-parse", "HEAD") != expected:
        raise ValueError("Initialize the pinned official Matcha-TTS submodule")
    if git(matcha, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("Official Matcha-TTS submodule must be clean")
    sys.path.insert(0, str(matcha))
    module = importlib.import_module("cosyvoice.flow.flow_matching")
    dependency = importlib.import_module("matcha.models.components.flow_matching")
    for imported, root in ((module, source), (dependency, matcha)):
        if root.resolve() not in Path(imported.__file__).resolve().parents:
            raise ValueError("Official solver imported from the wrong source directory")
    return module.ConditionalCFM, expected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--cosyvoice-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    import torch

    if args.output.exists():
        raise FileExistsError(args.output)
    if sha256_file(args.model_dir / "flow.pt") != FLOW_SHA256:
        raise ValueError("Checkpoint does not match pinned model")
    DiT, revision = _official_dit(args.cosyvoice_source)
    ConditionalCFM, matcha_revision = _official_solver(args.cosyvoice_source)
    args.output.mkdir(parents=True, exist_ok=False)
    solver_source_sha256 = sha256_file(Path(__file__).with_name("flow_matching.py"))
    model = DiT(dim=1024, depth=22, heads=16, dim_head=64, ff_mult=2,
                mel_dim=80, mu_dim=80, spk_dim=80, out_channels=80,
                static_chunk_size=50, num_decoding_left_chunks=-1).eval()
    state = torch.load(args.model_dir / "flow.pt", map_location="cpu", weights_only=True, mmap=True)
    prefix = "decoder.estimator."
    model.load_state_dict({k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}, strict=True)
    del state
    model.cuda()

    class Recorder(torch.nn.Module):
        def __init__(self, estimator):
            super().__init__()
            self.estimator = estimator
            self.calls = []

        def forward(self, x, mask, mu, t, spks, cond, streaming=False):
            output = self.estimator(x, mask, mu, t, spks, cond, streaming=streaming)
            self.calls.append({k: v.detach().cpu().numpy().copy() for k, v in
                               zip((*INPUT_NAMES, "velocity"), (x, mask, mu, t, spks, cond, output))})
            return output

    recorder = Recorder(model)
    params = SimpleNamespace(solver="euler", sigma_min=1e-6, t_scheduler="cosine",
                             training_cfg_rate=.2, inference_cfg_rate=.7)
    official = ConditionalCFM(80, params, n_spks=1, spk_emb_dim=80, estimator=recorder).eval()
    cases, traces, expected_outputs, rows, saved = list(_cfg_cases()), [], [], [], {}
    with torch.inference_mode(), _ieee_fp32_reference(torch):
        for frames, masked, values in cases:
            tensors = {k: torch.from_numpy(v).cuda() for k, v in values.items()}
            times = 1 - torch.cos(torch.linspace(0, 1, 11, device="cuda") * .5 * torch.pi)
            recorder.calls = []
            expected = official.solve_euler(tensors["noise"].clone(), times, tensors["mu"],
                                            tensors["mask"], tensors["spks"], tensors["cond"],
                                            streaming=False).cpu().numpy()
            trace = recorder.calls
            traces.append(trace)
            expected_outputs.append(expected)
            recorder.calls = []
            local = solve_euler(recorder, **tensors).cpu().numpy()
            if len(trace) != 10 or len(recorder.calls) != 10:
                raise ValueError("Both solvers must perform exactly ten estimator calls")
            row = {"stage": "local_solver_vs_official_same_pytorch_estimator",
                   "frames": frames, "masked": masked,
                   **compare_outputs(local, expected, atol=ATOL, rtol=RTOL),
                   "times_exact": all(np.array_equal(a["t"], b["t"])
                                      for a, b in zip(trace, recorder.calls))}
            rows.append(row)
            print(json.dumps(row), flush=True)
            label = f"{frames}_{int(masked)}"
            saved[label + "_mel"] = expected
            for step, call in enumerate(trace):
                saved.update({f"{label}_{step}_{k}": v for k, v in call.items()})
    np.savez(args.output / "official_trajectories.npz", **saved)
    del model, recorder, official, tensors
    gc.collect()
    torch.cuda.empty_cache()

    engine = FlowEngine(args.component)
    for (frames, masked, values), trace, expected in zip(cases, traces, expected_outputs):
        # Identical official x/t isolate estimator error from trajectory drift.
        for step, call in enumerate(trace):
            tensors = {k: torch.from_numpy(call[k]).to(engine.device) for k in INPUT_NAMES}
            actual = engine(**tensors).cpu().numpy()
            row = {"stage": "velocity_on_official_trajectory", "frames": frames,
                   "masked": masked, "step": step,
                   **compare_outputs(actual, call["velocity"], atol=ATOL, rtol=RTOL)}
            rows.append(row)
        tensors = {k: torch.from_numpy(v).to(engine.device) for k, v in values.items()}
        actual = solve_euler(engine, **tensors).cpu().numpy()
        row = {"stage": "native_integration_vs_official", "frames": frames, "masked": masked,
               **compare_outputs(actual, expected, atol=ATOL, rtol=RTOL)}
        rows.append(row)
        print(json.dumps(row), flush=True)
    report = {"scope": "supplemental_synthetic_cfg_not_stress_gate_or_audio_acceptance",
              "source_revision": revision, "matcha_revision": matcha_revision,
              "checkpoint_sha256": FLOW_SHA256,
              "plan_sha256": sha256_file(args.component / "flow.plan"),
              "local_solver_sha256": solver_source_sha256,
              "reference_precision": {"cudnn_tf32": False, "matmul_tf32": False,
                                      "matmul_precision": "highest", "sdpa_policy": "auto"},
              "dependencies": {name: importlib.metadata.version(name) for name in
                               ("x-transformers", "conformer", "lightning", "diffusers")},
              "torch_version": torch.__version__, "gpu": torch.cuda.get_device_name(),
              "atol": ATOL, "rtol": RTOL, "seed": 2512, "steps": 10,
              "passed": all(row["passed"] for row in rows), "cases": rows}
    with (args.output / "report.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    if not report["passed"]:
        raise SystemExit("Supplemental CFG trajectory parity FAILED; original gates remain unchanged")


if __name__ == "__main__":
    main()
