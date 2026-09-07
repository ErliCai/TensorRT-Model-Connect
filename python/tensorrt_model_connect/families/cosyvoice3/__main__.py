# SPDX-License-Identifier: Apache-2.0
"""Explicit component commands; no claim of public trtmc TTS support."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import tempfile

from .config import MODEL_ID, MODEL_REVISION, SOURCE_REVISION, ShapeProfile, read_config


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build(args):
    from tensorrt_model_connect import trt_compat
    from .checkpoint_mapper import load_flow_weights
    from .flow_builder import build_flow_engine

    output = args.output.resolve()
    if output.exists():
        raise ValueError(f"Output already exists; choose a new directory: {output}")
    cfg = read_config(args.model_dir)
    profile = ShapeProfile(args.min_frames, args.opt_frames, args.max_frames)
    implementation_files = [Path(__file__).with_name(name) for name in
                            ("__main__.py", "flow_builder.py", "constants.py", "config.py", "checkpoint_mapper.py")]
    implementation_hashes = {path.name: sha256_file(path) for path in implementation_files}
    weights = load_flow_weights(args.model_dir, cfg)
    plan = build_flow_engine(weights, cfg, profile, workspace_mib=args.workspace_mib)
    if implementation_hashes != {path.name: sha256_file(path) for path in implementation_files}:
        raise RuntimeError("Family implementation changed during build; choose a stable source snapshot")
    metadata = {
        "schema_version": 1, "component": "cosyvoice3_flow_estimator",
        "status": "experimental_component_not_end_to_end_tts",
        "target_model_id": MODEL_ID, "target_model_revision": MODEL_REVISION,
        "equations_source_revision": SOURCE_REVISION,
        "local_checkpoint_revision_verified": False,
        "architecture": asdict(cfg), "profile": asdict(profile),
        "precision": "fp32", "tf32": False, "streaming": False,
        "tensorrt_version": trt_compat.module_version(),
        "plan_sha256": hashlib.sha256(plan).hexdigest(),
        "source_sha256": {name: sha256_file(args.model_dir / name) for name in ("cosyvoice3.yaml", "flow.pt")},
        "implementation_sha256": implementation_hashes,
        "workspace_mib": args.workspace_mib,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".cosyvoice3-", dir=output.parent) as tmp:
        stage = Path(tmp) / "component"
        stage.mkdir()
        (stage / "flow.plan").write_bytes(plan)
        (stage / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        # Output publication occurs only after both complete files are written.
        # Never update the user's checkpoint or an existing component directory.
        if output.exists():
            raise FileExistsError(output)
        os.rename(stage, output)
    print(json.dumps(metadata, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Experimental CosyVoice3 native Flow component (not full TTS)")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="Read the model config safely, without executing YAML constructors")
    inspect.add_argument("--model-dir", type=Path, required=True)
    build_parser = commands.add_parser("build-flow", help="Build a native FP32 offline DiT component engine")
    build_parser.add_argument("--model-dir", type=Path, required=True)
    build_parser.add_argument("--output", type=Path, required=True)
    build_parser.add_argument("--min-frames", type=int, default=4)
    build_parser.add_argument("--opt-frames", type=int, default=64)
    build_parser.add_argument("--max-frames", type=int, default=256)
    build_parser.add_argument("--workspace-mib", type=int, default=512)
    args = parser.parse_args(argv)
    if args.command == "inspect":
        print(json.dumps({"target": MODEL_ID, "flow": asdict(read_config(args.model_dir)), "status": "component_only"}, indent=2))
    else:
        build(args)


if __name__ == "__main__":
    main()
