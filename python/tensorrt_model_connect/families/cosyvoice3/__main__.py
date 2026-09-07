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
    parser = argparse.ArgumentParser(description="Experimental offline CosyVoice3 Python components (not full TTS support in trtmc CLI)")
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
    conditioner = commands.add_parser("build-conditioner", help="Build offline token/speaker preprocessing (not full TTS)")
    conditioner.add_argument("--model-dir", type=Path, required=True)
    conditioner.add_argument("--output", type=Path, required=True)
    conditioner.add_argument("--min-tokens", type=int, default=2)
    conditioner.add_argument("--opt-tokens", type=int, default=32)
    conditioner.add_argument("--max-tokens", type=int, default=128)
    conditioner.add_argument("--workspace-mib", type=int, default=64)
    for component in ("llm", "hift"):
        command = commands.add_parser(f"build-{component}", help=f"Build native offline {component} component")
        command.add_argument("--model-dir", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--workspace-mib", type=int, default=256)
        if component == "llm":
            command.add_argument("--max-context", type=int, default=1024)
            command.add_argument("--opt-tokens", type=int, default=64)
        else:
            command.add_argument("--min-frames", type=int, default=4)
            command.add_argument("--opt-frames", type=int, default=64)
            command.add_argument("--max-frames", type=int, default=256)
    tts = commands.add_parser("synthesize", help="Offline text + prepared voice NPZ -> WAV (experimental, unqualified)")
    for name in ("model-dir", "llm", "conditioner", "flow", "hift", "voice", "output"):
        tts.add_argument(f"--{name}", type=Path, required=True)
    tts.add_argument("--text", required=True)
    tts.add_argument("--instruction", default="You are a helpful assistant.")
    tts.add_argument("--prompt-text", default="", help="Exact reference transcript for zero-shot; omit for instruction mode")
    tts.add_argument("--max-tokens", type=int, default=100)
    tts.add_argument("--seed", type=int, default=2512)
    tts.add_argument("--greedy", action="store_true", help="Deterministic argmax, not the official default RAS sampler")
    args = parser.parse_args(argv)
    if args.command == "inspect":
        print(json.dumps({"target": MODEL_ID, "flow": asdict(read_config(args.model_dir)), "status": "component_only"}, indent=2))
    elif args.command == "build-flow":
        build(args)
    elif args.command == "build-conditioner":
        build_conditioner(args)
    elif args.command == "synthesize":
        from .tts import synthesize

        synthesize(args)
    else:
        build_speech_component(args)


def build_conditioner(args):
    from tensorrt_model_connect import trt_compat
    from .conditioning import TokenProfile, build_engine, load_weights

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    read_config(args.model_dir)
    profile = TokenProfile(args.min_tokens, args.opt_tokens, args.max_tokens)
    files = [Path(__file__).with_name(name) for name in ("__main__.py", "conditioning.py", "config.py")]
    sources = {path.name: sha256_file(path) for path in files}
    plan = build_engine(load_weights(args.model_dir), profile, workspace_mib=args.workspace_mib)
    if sources != {path.name: sha256_file(path) for path in files}:
        raise RuntimeError("Conditioner implementation changed during build")
    manifest = {
        "schema_version": 1, "component": "cosyvoice3_conditioner",
        "status": "experimental_component_not_end_to_end_tts",
        "target_model_id": MODEL_ID, "target_model_revision": MODEL_REVISION,
        "equations_source_revision": SOURCE_REVISION,
        "local_checkpoint_revision_verified": False,
        "precision": "fp32", "tf32": False, "streaming": False,
        "profile": asdict(profile), "plan_sha256": hashlib.sha256(plan).hexdigest(),
        "tensorrt_version": trt_compat.module_version(), "workspace_mib": args.workspace_mib,
        "source_sha256": {name: sha256_file(args.model_dir / name) for name in ("cosyvoice3.yaml", "flow.pt")},
        "implementation_sha256": sources,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".cosyvoice3-conditioning-", dir=output.parent) as tmp:
        stage = Path(tmp) / "component"
        stage.mkdir()
        (stage / "conditioning.plan").write_bytes(plan)
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        if output.exists():
            raise FileExistsError(output)
        os.rename(stage, output)
    print(json.dumps(manifest, indent=2))


def build_speech_component(args):
    from tensorrt_model_connect import trt_compat

    component = args.command.removeprefix("build-")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    files = [Path(__file__).with_name(name) for name in
             ("__main__.py", "components.py", f"{component}.py", "config.py")]
    sources = {path.name: sha256_file(path) for path in files}
    metadata = {}
    if component == "llm":
        from .llm import build_engine, load_weights

        cfg, weights = load_weights(args.model_dir)
        plan = build_engine(weights, cfg, max_context=args.max_context, opt_tokens=args.opt_tokens,
                            workspace_mib=args.workspace_mib)
        metadata.update(architecture=asdict(cfg), max_context=args.max_context, opt_tokens=args.opt_tokens,
                        compact_kv_cache=True, checkpoint="llm.pt")
        checkpoint_files = ("llm.pt", "cosyvoice3.yaml", "CosyVoice-BlankEN/config.json")
    else:
        from .hift import build_engine, load_weights

        profile = ShapeProfile(args.min_frames, args.opt_frames, args.max_frames)
        plan = build_engine(load_weights(args.model_dir), profile, workspace_mib=args.workspace_mib)
        metadata.update(profile=asdict(profile), sample_rate=24000, samples_per_frame=480,
                        f0_precision="fp32_reference_uses_fp64", finalize=True, noise="explicit_uniform_0_1")
        checkpoint_files = ("hift.pt", "cosyvoice3.yaml")
    if sources != {path.name: sha256_file(path) for path in files}:
        raise RuntimeError("Component implementation changed during build; rebuild a stable snapshot")
    metadata.update(schema_version=1, component=f"cosyvoice3_{component}",
                    status="experimental_python_component_not_registered_cpp_tts",
                    target_model_id=MODEL_ID, target_model_revision=MODEL_REVISION,
                    equations_source_revision=SOURCE_REVISION, local_checkpoint_revision_verified=False,
                    precision="fp32", tf32=False, streaming=False,
                    tensorrt_version=trt_compat.module_version(), workspace_mib=args.workspace_mib,
                    plan_sha256=hashlib.sha256(plan).hexdigest(), implementation_sha256=sources,
                    source_sha256={name: sha256_file(args.model_dir / name) for name in checkpoint_files})
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".cosyvoice3-{component}-", dir=output.parent) as tmp:
        stage = Path(tmp) / "component"
        stage.mkdir()
        (stage / f"{component}.plan").write_bytes(plan)
        (stage / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        if output.exists():
            raise FileExistsError(output)
        os.rename(stage, output)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
