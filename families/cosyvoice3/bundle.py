# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Package existing FP32 components and one prepared voice for native C++ TTS.

This does not rebuild or qualify the component plans. Their complete manifests
are retained as provenance; reference-WAV preparation remains a separate step.
"""
from __future__ import annotations

import json
import shutil

from tensorrt_model_connect.bundle_writer import BundleWriter
from .config import MODEL_ID, MODEL_REVISION
from .tts import read_voice, text_tokenizer


def package(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    voice = read_voice(args.voice)
    tokenizer = text_tokenizer(args.model_dir)
    sections, manifests = [], {}
    for name, directory in (("llm", args.llm), ("conditioning", args.conditioner),
                            ("flow", args.flow), ("hift", args.hift)):
        manifest_bytes = (directory / "manifest.json").read_bytes()
        m = json.loads(manifest_bytes)
        expected = {"flow": "cosyvoice3_flow_estimator", "conditioning": "cosyvoice3_conditioner"}.get(
            name, f"cosyvoice3_{name}")
        if (m.get("schema_version") != 1 or m.get("component") != expected
                or m.get("target_model_id") != MODEL_ID or m.get("target_model_revision") != MODEL_REVISION
                or m.get("precision") != "fp32" or m.get("streaming") is not False):
            raise ValueError(f"Unsupported {name} component manifest")
        manifests[name] = m
        plan_path = directory / (name + ".plan")
        if not plan_path.is_file():
            raise FileNotFoundError(plan_path)
        sections.extend([(name + ".plan", plan_path),
                         (name + ".manifest.json", manifest_bytes)])
    versions = {m["tensorrt_version"] for m in manifests.values()}
    if len(versions) != 1:
        raise ValueError("Component TensorRT versions must match")
    prompt_count = voice["prompt_tokens"].shape[1]
    capacity = min(manifests["conditioning"]["profile"]["max_tokens"] - prompt_count,
                   manifests["flow"]["profile"]["max_frames"] // 2 - prompt_count,
                   manifests["hift"]["profile"]["max_frames"] // 2)
    if capacity < 1:
        raise ValueError("Prepared voice leaves no generation capacity")
    max_context = manifests["llm"]["max_context"]
    config = {"cosyvoice3_schema": 1, "precision": "fp32", "engine_backend": "trt",
              "tensorrt_version": versions.pop(),
              "runtime_strategy": "text_to_audio_cosyvoice3", "vocab_size": 6761,
              "hidden_size": 896, "num_layers": 24, "num_heads": 14, "num_kv_heads": 2,
              "max_cache_length": max_context, "model_revision": MODEL_REVISION,
              "qualification": "experimental_unqualified",
              "cosyvoice3": {"instruction": args.instruction, "transcript": args.prompt_text,
                             "greedy": args.greedy, "max_context": max_context,
                             "max_tokens": capacity,
                             **{k: v.reshape(-1).tolist() for k, v in voice.items()}}}
    writer = BundleWriter(args.output)
    try:
        writer.set_header(family="cosyvoice3", task="audio_generation", backend="trt")
        for name, value in sections:
            if isinstance(value, bytes):
                writer.add_bytes(name, value)
            else:
                with value.open("rb") as source, writer.open_section(name) as destination:
                    shutil.copyfileobj(source, destination)
        writer.add_json("config.json", config)
        writer.add_bytes("tokenizer.json", tokenizer.backend_tokenizer.to_str().encode())
        writer.finish()
    except BaseException:
        writer.abort()
        raise
    print(json.dumps({"bundle": str(args.output), "max_tokens": capacity,
                      "status": "packaged_unqualified", "component_plans_rebuilt": False}, indent=2))
