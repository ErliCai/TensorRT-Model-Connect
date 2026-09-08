# SPDX-License-Identifier: Apache-2.0
"""Bundle contract and opt-in native tokenizer comparison (no GPU required)."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from tensorrt_model_connect.bundle_writer import read_bundle_section
from tensorrt_model_connect.families.cosyvoice3 import bundle
from tensorrt_model_connect.families.cosyvoice3.config import MODEL_ID, MODEL_REVISION


@pytest.fixture
def package_args(tmp_path, monkeypatch):
    args = SimpleNamespace(model_dir=tmp_path, voice=tmp_path / "voice.npz", output=tmp_path / "test.bundle",
                           instruction="You are a helpful assistant.", prompt_text="", greedy=False)
    np.savez(args.voice, prompt_tokens=np.array([[5]], dtype=np.int32),
             prompt_features=np.zeros((1, 2, 80), dtype=np.float32), speaker=np.ones((1, 192), dtype=np.float32))
    for option, component in (("llm", "llm"), ("conditioner", "conditioning"), ("flow", "flow"), ("hift", "hift")):
        directory = tmp_path / option
        directory.mkdir()
        setattr(args, option, directory)
        data = b"test-only-not-a-real-engine-" + option.encode()
        (directory / (component + ".plan")).write_bytes(data)
        m = dict(schema_version=1, component={"conditioner": "cosyvoice3_conditioner",
                 "flow": "cosyvoice3_flow_estimator"}.get(option, "cosyvoice3_" + component),
                 target_model_id=MODEL_ID, target_model_revision=MODEL_REVISION,
                 precision="fp32", streaming=False, tensorrt_version="11.1.0.106",
                 plan_sha256=hashlib.sha256(data).hexdigest(), max_context=512,
                 profile=dict(max_tokens=128, max_frames=256))
        (directory / "manifest.json").write_text(json.dumps(m))
    monkeypatch.setattr(bundle, "text_tokenizer", lambda _: SimpleNamespace(
        backend_tokenizer=SimpleNamespace(to_str=lambda: "{}")))
    return args


def test_package_retains_plans_voice_and_provenance(package_args):
    args = package_args
    bundle.package(args)
    config = json.loads(read_bundle_section(args.output, "config.json"))
    assert config["cosyvoice3"]["max_tokens"] == 127
    assert config["cosyvoice3"]["prompt_tokens"] == [5]
    assert config["qualification"] == "experimental_unqualified"
    assert read_bundle_section(args.output, "llm.plan") == (args.llm / "llm.plan").read_bytes()
    assert read_bundle_section(args.output, "llm.manifest.json") == (args.llm / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        bundle.package(args)


@pytest.mark.parametrize("key,value", [("precision", "fp16"), ("streaming", True),
                                      ("target_model_revision", "main"), ("component", "other"),
                                      ("tensorrt_version", "10.9.0")])
def test_package_rejects_incompatible_manifest(package_args, key, value):
    path = package_args.llm / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest[key] = value
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        bundle.package(package_args)
    assert not package_args.output.exists()


def test_package_rejects_modified_plan(package_args):
    (package_args.llm / "llm.plan").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="source changed after validation"):
        bundle.package(package_args)
    assert not package_args.output.exists()


def test_native_tokenizer_matches_python(tmp_path):
    binary, model = os.environ.get("COSYVOICE3_CPP_TEST"), os.environ.get("COSYVOICE3_MODEL_DIR")
    if not binary or not model:
        pytest.skip("Set COSYVOICE3_CPP_TEST and COSYVOICE3_MODEL_DIR for native BPE comparison")
    from tensorrt_model_connect.families.cosyvoice3.tts import encode_request, text_tokenizer

    cases = []
    for text in ("Hello world.", "你好，世界。", "Hello 世界，2026!", "It's a nice day.\nGood morning."):
        for transcript in ("", "This is my voice."):
            packed, _ = encode_request(Path(model), text, prompt_text=transcript, prompt_tokens=[5, 6])
            cases.append(dict(text=text, transcript=transcript, speech=[5, 6], packed=packed))
    path = tmp_path / "tokenizer-cases.json"
    path.write_text(json.dumps(dict(tokenizer=text_tokenizer(model).backend_tokenizer.to_str(), cases=cases)), encoding="utf-8")
    subprocess.run([binary, str(path)], check=True, timeout=60)
