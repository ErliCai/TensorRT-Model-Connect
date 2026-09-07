# SPDX-License-Identifier: Apache-2.0
"""Maintained LLM/HiFT contracts and opt-in, independent GPU parity tests.

CPU: pytest <this-file>
GPU: COSYVOICE3_RUN_GPU_TESTS=1; full checkpoints additionally need
COSYVOICE3_MODEL_DIR, COSYVOICE3_LLM_ENGINE, COSYVOICE3_HIFT_ENGINE and
COSYVOICE3_OFFICIAL_SOURCE. Optional COSYVOICE3_SPEECH_EVIDENCE stores metrics.
The new gates are component-specific, not imported from the Flow comparator.
"""

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tensorrt_model_connect.families.cosyvoice3.hift import fourier_filters, weight_shapes as hift_shapes
from tensorrt_model_connect.families.cosyvoice3.llm import (
    END_OF_PROMPT, EOS, SOS, TASK, LLMConfig, pack_prompt, sample_token,
)

GPU = pytest.mark.skipif(os.environ.get("COSYVOICE3_RUN_GPU_TESTS") != "1", reason="Opt-in TensorRT GPU test")


def test_cosyvoice3_prompt_layout_and_special_ids():
    assert (SOS, EOS, TASK) == (6561, 6562, 6563)
    assert pack_prompt([7, 8], [10, END_OF_PROMPT], [2, 3]) == [158497, 10, 151646, 7, 8, 158499, 151938, 151939]
    for text, prompt, speech in (([], [151646], []), ([7], [], []), ([151646, -1], [], []),
                                 ([151646], [], [6561]), ([True, 151646], [], [])):
        with pytest.raises(ValueError):
            pack_prompt(text, prompt, speech)


def test_sampling_copies_input_and_reports_upstream_stop_semantics():
    scores = np.full(6761, -100., np.float32)
    scores[SOS], scores[EOS] = 100, 90
    before = scores.copy()
    assert sample_token(scores, [], np.random.default_rng(1), min_tokens=2, greedy=True) == EOS
    assert sample_token(scores, [], np.random.default_rng(1), min_tokens=0, greedy=True) == SOS
    np.testing.assert_array_equal(scores, before)
    with pytest.raises(ValueError):
        sample_token(np.full(6761, np.nan), [], np.random.default_rng())


def test_ras_repeat_redraw_and_seed_reproducibility():
    scores = np.full(6761, -100., np.float32)
    scores[7], scores[8] = 10, 0
    assert sample_token(scores, [7], np.random.default_rng(3)) == 8
    scores = np.zeros(6761, np.float32)
    a = [sample_token(scores, [], np.random.default_rng(seed)) for seed in range(20)]
    b = [sample_token(scores, [], np.random.default_rng(seed)) for seed in range(20)]
    assert a == b and len(set(a)) > 1 and all(x < 25 for x in a)


@pytest.mark.parametrize("kwargs", [{"hidden_size": 31}, {"num_key_value_heads": 3},
                                    {"rms_norm_eps": float("nan")}, {"num_hidden_layers": True}])
def test_llm_rejects_invalid_architecture(kwargs):
    with pytest.raises(ValueError):
        LLMConfig(**kwargs)


def test_hift_checkpoint_contract_covers_all_published_parameters():
    assert len(hift_shapes(checkpoint=True)) == 328
    assert hift_shapes()["ups.0.weight"] == (256, 512, 16)  # NOT ConvTranspose layout
    assert hift_shapes()["f0_predictor.condnet.0.weight"] == (512, 80, 4)


@pytest.mark.parametrize("length", [480, 1920, 8160])
def test_fourier_filters_match_torch_stft_and_istft(length):
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F

    stft, inverse, window = map(torch.from_numpy, fourier_filters())
    x = torch.randn(1, length, generator=torch.Generator().manual_seed(12)) * .1
    spectrum = F.conv1d(F.pad(x[:, None], (8, 8), mode="reflect"), stft, stride=4)
    expected = torch.stft(x, 16, 4, 16, window=window, return_complex=True)
    torch.testing.assert_close(spectrum, torch.cat((expected.real, expected.imag), 1), atol=1e-6, rtol=1e-5)
    numerator = F.conv_transpose1d(spectrum, inverse, stride=4)[:, :, 8:-8]
    denominator = F.conv_transpose1d(torch.ones_like(spectrum[:, :1]), window.square()[None, None], stride=4)[:, :, 8:-8]
    restored = (numerator / denominator)[:, 0]
    torch.testing.assert_close(restored, torch.istft(expected, 16, 4, 16, window=window), atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(restored, x, atol=1e-6, rtol=1e-5)


def evidence(name, actual, expected, *, atol, rtol):
    """Persist metrics BEFORE asserting, so a failing case retains its evidence."""
    actual, expected = actual.detach().cpu().float(), expected.detach().cpu().float()
    error = (actual - expected).abs()
    limit = atol + rtol * expected.abs()
    metrics = {"case": name, "shape": list(actual.shape), "atol": atol, "rtol": rtol,
               "max_abs": error.max().item(), "rmse": error.square().mean().sqrt().item(),
               "over_tolerance": (error > limit).sum().item()}
    print(json.dumps(metrics), flush=True)
    path = os.environ.get("COSYVOICE3_SPEECH_EVIDENCE")
    if path:
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / f"{name}.json").open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics, indent=2) + "\n")
    import torch
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


def require_full(component):
    names = ("COSYVOICE3_MODEL_DIR", f"COSYVOICE3_{component.upper()}_ENGINE")
    if any(not os.environ.get(x) for x in names):
        pytest.skip("Full-weight test requires " + ", ".join(names))
    model_dir, engine_dir = (Path(os.environ[x]) for x in names)
    from tensorrt_model_connect.families.cosyvoice3.__main__ import sha256_file
    from tensorrt_model_connect.families.cosyvoice3.config import MODEL_REVISION

    expected = {"llm": "69f43bd545131c30e98947fb360ea8b4dc9916d8e83dded7757c7ea4f5a24970",
                "hift": "b279d7641eb97ae55b3b540cfba4f953c26492a2df758328a89a4d007ab87a65"}[component]
    filename = component + ".pt"
    manifest = json.loads((engine_dir / "manifest.json").read_text(encoding="utf-8"))
    if sha256_file(model_dir / filename) != expected or manifest["source_sha256"][filename] != expected:
        raise ValueError("Full parity requires the pinned checkpoint and an engine built from that checkpoint")
    directory = os.environ.get("COSYVOICE3_SPEECH_EVIDENCE")
    if directory:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        import torch
        import transformers

        metadata = {"model_revision": MODEL_REVISION, "engine_manifest": manifest,
                    "torch": torch.__version__, "transformers": transformers.__version__,
                    "gpu": torch.cuda.get_device_name(), "test_sha256": sha256_file(__file__)}
        with (path / f"{component}_run.json").open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(metadata, indent=2) + "\n")
    return model_dir, engine_dir


@pytest.fixture(scope="module")
def tiny_llm(tmp_path_factory):
    import torch
    from transformers import Qwen2Config, Qwen2Model
    from tensorrt_model_connect.families.cosyvoice3.llm import LLMEngine, build_engine, weight_shapes

    cfg = LLMConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                    num_attention_heads=4, num_key_value_heads=1, vocab_size=256)
    rng = np.random.default_rng(2512)
    weights = {k: rng.normal(0, .02, shape).astype(np.float32) for k, shape in weight_shapes(cfg).items()}
    for k in weights:
        if "layernorm" in k or k == "norm":
            weights[k].fill(1)
    config = Qwen2Config(**asdict(cfg), attention_dropout=0., use_sliding_window=False)
    config._attn_implementation = "eager"
    ref = Qwen2Model(config).float().eval()
    state = {"embed_tokens.weight": torch.from_numpy(weights["text_embedding"]),
             "norm.weight": torch.from_numpy(weights["norm"])}
    for key, value in weights.items():
        if not key.startswith("layers."):
            continue
        layer, index, name, *suffix = key.split(".")
        base = f"{layer}.{index}."
        if name.endswith("layernorm"):
            target = base + name + ".weight"
        else:
            target = base + ("self_attn." if name in ("q_proj", "k_proj", "v_proj", "o_proj") else "mlp.") + name + "." + suffix[0]
        state[target] = torch.from_numpy(value)
    ref.load_state_dict(state, strict=True)
    plan = build_engine(weights, cfg, max_context=64, opt_tokens=8, workspace_mib=32)
    path = tmp_path_factory.mktemp("tiny-cosyvoice3-llm")
    (path / "llm.plan").write_bytes(plan)
    (path / "manifest.json").write_text(json.dumps({"schema_version": 1, "component": "cosyvoice3_llm",
        "precision": "fp32", "streaming": False, "max_context": 64, "architecture": asdict(cfg),
        "plan_sha256": hashlib.sha256(plan).hexdigest()}), encoding="utf-8")
    return LLMEngine(path), ref.cuda(), torch.from_numpy(weights["decoder"]).cuda(), weights


@GPU
@pytest.mark.gpu
@pytest.mark.parametrize("length", [1, 4, 63])
def test_tiny_llm_dynamic_cache_and_profile_boundaries(tiny_llm, length):
    import torch
    import torch.nn.functional as F
    from tensorrt_model_connect.families.cosyvoice3.validate_flow_pytorch import _ieee_fp32_reference

    engine, ref, decoder, _ = tiny_llm
    ids = torch.arange(length, device="cuda", dtype=torch.int32)[None]
    with torch.inference_mode(), _ieee_fp32_reference(torch):
        actual, cache = engine.step(ids)
        expected = F.linear(ref(input_ids=ids.long(), use_cache=False).last_hidden_state[:, -1], decoder)[0]
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
        next_id = torch.tensor([[71]], dtype=torch.int32, device="cuda")
        actual, cache = engine.step(next_id, cache)
        expected = F.linear(ref(input_ids=torch.cat((ids, next_id), 1).long(), use_cache=False).last_hidden_state[:, -1], decoder)[0]
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
        assert cache[0].shape == (2, 1, length + 1, 8)
        if length == 63:
            with pytest.raises(ValueError, match="context"):
                engine.step(next_id, cache)


@GPU
@pytest.mark.gpu
def test_tiny_llm_rejects_invalid_requests(tiny_llm):
    import torch

    engine, _, _, _ = tiny_llm
    for ids in (torch.zeros((1, 0), dtype=torch.int32, device="cuda"),
                torch.tensor([[-1]], dtype=torch.int32, device="cuda"),
                torch.tensor([[256 + 6761]], dtype=torch.int32, device="cuda"),
                torch.ones((1, 2), dtype=torch.int64, device="cuda"),
                torch.ones((1, 2), dtype=torch.int32)):
        with pytest.raises(ValueError):
            engine.step(ids)
    with pytest.raises(ValueError):
        engine.generate([1] * 64, max_tokens=2)


@pytest.fixture(scope="module")
def full_llm():
    model_dir, engine_dir = require_full("llm")
    import torch
    from transformers import Qwen2Config, Qwen2Model
    from tensorrt_model_connect.families.cosyvoice3.llm import LLMEngine

    config = Qwen2Config.from_pretrained(model_dir / "CosyVoice-BlankEN", local_files_only=True)
    config._attn_implementation = "eager"
    state = torch.load(model_dir / "llm.pt", mmap=True, weights_only=True, map_location="cpu")
    # A real CPU construction also initializes nonpersistent RoPE buffers;
    # checkpoint assignment alone cannot materialize those from a meta model.
    reference = Qwen2Model(config)
    prefix = "llm.model.model."
    reference.load_state_dict({k.removeprefix(prefix): v for k, v in state.items() if k.startswith(prefix)}, strict=True, assign=True)
    reference = reference.float().eval().cuda()
    embedding = torch.cat((state[prefix + "embed_tokens.weight"], state["speech_embedding.weight"])).float().cuda()
    decoder = state["llm_decoder.weight"].float().cuda()
    yield LLMEngine(engine_dir), reference, embedding, decoder
    del reference, embedding, decoder
    torch.cuda.empty_cache()


@GPU
@pytest.mark.gpu
@pytest.mark.parametrize("length", [4, 17, 64, 256])
def test_full_llm_prefill_cached_decode_and_reference(full_llm, length):
    import torch
    import torch.nn.functional as F
    from tensorrt_model_connect.families.cosyvoice3.validate_flow_pytorch import _ieee_fp32_reference

    engine, ref, embedding, decoder = full_llm
    ids = torch.tensor([[158497, 100, 151646, 158499] + [151936 + (i * 71) % 6561 for i in range(length - 4)]],
                       dtype=torch.int32, device="cuda")
    with torch.inference_mode(), _ieee_fp32_reference(torch):
        expected = F.linear(ref(inputs_embeds=F.embedding(ids.long(), embedding), use_cache=False).last_hidden_state[:, -1], decoder)[0]
        actual, cache = engine.step(ids)
        evidence(f"llm_prefill_{length}", actual, expected, atol=1e-3, rtol=1e-4)
        assert actual.argmax().item() == expected.argmax().item()
        for j in range(3):
            token = torch.tensor([[151936 + 500 + j]], dtype=torch.int32, device="cuda")
            ids = torch.cat((ids, token), 1)
            actual, cache = engine.step(token, cache)
            recomputed, _ = engine.step(ids)
            expected = F.linear(ref(inputs_embeds=F.embedding(ids.long(), embedding), use_cache=False).last_hidden_state[:, -1], decoder)[0]
            evidence(f"llm_cached_{length}_{j}", actual, expected, atol=1e-3, rtol=1e-4)
            evidence(f"llm_cache_vs_prefill_{length}_{j}", actual, recomputed, atol=1e-3, rtol=1e-4)
            assert actual.argmax().item() == expected.argmax().item()
            assert cache[0].shape == (24, 2, ids.shape[1], 64)


@pytest.fixture(scope="module")
def full_hift():
    model_dir, engine_dir = require_full("hift")
    if not os.environ.get("COSYVOICE3_OFFICIAL_SOURCE"):
        pytest.skip("Requires the pinned official CosyVoice source")
    import torch
    from tensorrt_model_connect.families.cosyvoice3.validate_flow_pytorch import _official_dit
    from tensorrt_model_connect.families.cosyvoice3.hift import HiFTEngine

    _official_dit(Path(os.environ["COSYVOICE3_OFFICIAL_SOURCE"]))
    from cosyvoice.hifigan.generator import CausalHiFTGenerator
    from cosyvoice.hifigan.f0_predictor import CausalConvRNNF0Predictor

    ref = CausalHiFTGenerator(sampling_rate=24000, upsample_rates=[8, 5, 3], upsample_kernel_sizes=[16, 11, 7],
                             source_resblock_kernel_sizes=[7, 7, 11],
                             source_resblock_dilation_sizes=[[1, 3, 5]] * 3,
                             f0_predictor=CausalConvRNNF0Predictor(1, 80, 512)).eval()
    ref.load_state_dict(torch.load(model_dir / "hift.pt", weights_only=True, mmap=True, map_location="cpu"), strict=True)
    yield HiFTEngine(engine_dir), ref.cuda()
    del ref
    torch.cuda.empty_cache()


@GPU
@pytest.mark.gpu
@pytest.mark.parametrize("frames", [4, 17, 64, 256])
def test_full_hift_f0_source_decode_and_waveform(full_hift, frames):
    import torch

    # Not an audio-quality corpus: fixed log-Mel-range numerical stress input.
    rng = np.random.default_rng(2512 + frames)
    mel = torch.from_numpy(rng.normal(-4, 2, (1, 80, frames)).astype(np.float32)).cuda()
    noise = torch.from_numpy(rng.random((1, 9, frames * 480), dtype=np.float32)).cuda()
    check_hift(full_hift, mel, noise, f"stress_{frames}")


def check_hift(full_hift, mel, noise, case):
    import torch
    from tensorrt_model_connect.families.cosyvoice3.validate_flow_pytorch import _ieee_fp32_reference

    engine, ref = full_hift
    frames = mel.shape[2]
    ref.m_source.l_sin_gen.sine_waves = noise.transpose(1, 2).contiguous().cpu()
    with torch.inference_mode(), _ieee_fp32_reference(torch):
        native = engine(mel, noise=noise)
        waveform, source = ref.inference(mel, finalize=True)
        f0 = ref.f0_predictor(mel.double()).float()[:, None]
        # Decode given exactly the SAME source isolates convolution/ISTFT from F0.
        decoded = ref.decode(mel, native["source"], finalize=True)
        source_same_f0 = ref.m_source(native["f0"].repeat_interleave(480, dim=2).transpose(1, 2))[0].transpose(1, 2)
        decoded_same_f0 = ref.decode(mel, source_same_f0, finalize=True)
        checks = [("f0_hz", native["f0"], f0, .01, 1e-5),
                  ("source", native["source"], source, 1e-3, 1e-4),
                  ("source_same_f0", native["source"], source_same_f0, 1e-3, 1e-4),
                  ("decoder_same_source", native["audio"], decoded, 1e-3, 1e-4),
                  ("waveform_same_f0", native["audio"], decoded_same_f0, 1e-3, 1e-4),
                  ("waveform", native["audio"], waveform, 1e-3, 1e-4)]
        failures = []
        for name, actual, expected, atol, rtol in checks:
            try:
                evidence(f"hift_{name}_{case}", actual, expected, atol=atol, rtol=rtol)
            except AssertionError as exc:
                failures.append(str(exc))
        assert native["audio"].shape == (1, frames * 480)
        assert native["audio"].abs().max().item() <= .990001
        assert not failures, "\n".join(failures)


@GPU
@pytest.mark.gpu
@pytest.mark.parametrize("frames", [16, 64, 128, 256])
def test_full_hift_real_acoustic_mel(full_hift, frames):
    import torch

    path = os.environ.get("COSYVOICE3_ACOUSTIC_EVIDENCE")
    if not path:
        pytest.skip("Set COSYVOICE3_ACOUSTIC_EVIDENCE to an existing official acoustic_evidence.npz")
    with np.load(path, allow_pickle=False) as arrays:
        mel = torch.from_numpy(arrays[f"tokens{frames // 2}_prompt0_official_mel"].copy()).cuda()
    noise = torch.from_numpy(np.random.default_rng(2512).random((1, 9, frames * 480), dtype=np.float32)).cuda()
    check_hift(full_hift, mel, noise, f"acoustic_{frames}")


def test_prepared_voice_validation(tmp_path):
    from tensorrt_model_connect.families.cosyvoice3.tts import read_voice

    voice = {"prompt_tokens": np.ones((1, 4), np.int32), "prompt_features": np.zeros((1, 8, 80), np.float32),
             "speaker": np.ones((1, 192), np.float32)}
    path = tmp_path / "voice.npz"
    np.savez(path, **voice)
    assert read_voice(path)["prompt_tokens"].shape == (1, 4)
    for replacement in ({"speaker": np.zeros((1, 192), np.float32)},
                        {"prompt_tokens": np.full((1, 4), 6561, np.int32)},
                        {"prompt_features": np.zeros((1, 7, 80), np.float32)},
                        {"speaker": np.full((1, 192), np.nan, np.float32)}):
        np.savez(path, **(voice | replacement))
        with pytest.raises(ValueError):
            read_voice(path)


@GPU
@pytest.mark.gpu
def test_native_source_phase_boundaries(tmp_path):
    import torch
    from tensorrt_model_connect.families.cosyvoice3.components import Graph, ComponentEngine
    from tensorrt_model_connect.families.cosyvoice3.hift import add_source

    g = Graph()
    f0 = g.net.add_input("f0", g.trt.float32, (1, 1, 256))
    noise = g.net.add_input("noise", g.trt.float32, (1, 9, 256 * 480))
    weights = {"m_source.l_linear.weight": np.ones((1, 9), np.float32) / 9,
               "m_source.l_linear.bias": np.zeros((1,), np.float32)}
    for name, tensor in add_source(g, f0, noise, weights).items():
        g.mark(tensor, name)
    plan = g.build({"f0": [(1, 1, 256)] * 3, "noise": [(1, 9, 256 * 480)] * 3}, 32)
    (tmp_path / "source.plan").write_bytes(plan)
    (tmp_path / "manifest.json").write_text(json.dumps({"schema_version": 1, "component": "cosyvoice3_source",
        "precision": "fp32", "streaming": False, "plan_sha256": hashlib.sha256(plan).hexdigest()}))
    engine = ComponentEngine(tmp_path, "source", {"f0": "float32", "noise": "float32"},
                             {name: "float32" for name in ("source", "rad", "cumulative", "phase")})
    f0 = torch.from_numpy(np.random.default_rng(2512).uniform(0, 400, (1, 1, 256)).astype(np.float32)).cuda()
    noise = torch.zeros((1, 9, 256 * 480), dtype=torch.float32, device="cuda")
    actual = engine.run(f0=f0, noise=noise)
    rad = (f0 * torch.arange(1, 10, dtype=torch.float32, device="cuda")[None, :, None] / 24000) % 1
    cumulative = rad.cumsum(2)
    phase = cumulative * (2 * np.pi) * 480
    sine = phase.repeat_interleave(480, dim=2).sin() * .1 * (f0 > 10).float().repeat_interleave(480, dim=2)
    source = sine.mean(dim=1, keepdim=True).tanh()
    # Intermediate numerical regression checks are separate from final HiFT waveform gates.
    for name, expected, atol in (("rad", rad, 1e-7), ("cumulative", cumulative, 1e-5),
                                 ("phase", phase, .02), ("source", source, 1e-3)):
        evidence("source_boundary_" + name, actual[name], expected, atol=atol, rtol=1e-6)


@GPU
@pytest.mark.gpu
def test_offline_text_to_wav_smoke(tmp_path):
    """Execution smoke only, NOT an ASR/listening/voice-similarity quality gate.

    Reuse the existing maintained acoustic validator's real recording fixture;
    take the first 25 prompt tokens for Flow (instruction mode, no transcript).
    """
    from tensorrt_model_connect.families.cosyvoice3.tts import synthesize
    import soundfile as sf

    needed = ("MODEL_DIR", "LLM_ENGINE", "HIFT_ENGINE", "FLOW_ENGINE", "CONDITIONER_ENGINE", "ACOUSTIC_REQUESTS")
    if any(not os.environ.get("COSYVOICE3_" + key) for key in needed):
        pytest.skip("Full pipeline smoke requires " + ", ".join("COSYVOICE3_" + key for key in needed))
    paths = {key: Path(os.environ["COSYVOICE3_" + key]) for key in needed}
    directory = Path(os.environ["COSYVOICE3_TTS_OUTPUT"]) if os.environ.get("COSYVOICE3_TTS_OUTPUT") else tmp_path
    directory.mkdir(parents=True, exist_ok=True)
    voice_path = directory / "voice.npz"
    if voice_path.exists():
        raise FileExistsError("Use a new COSYVOICE3_TTS_OUTPUT directory to preserve prior evidence")
    with np.load(paths["ACOUSTIC_REQUESTS"], allow_pickle=False) as arrays:
        prefix = "tokens128_prompt25_"
        voice = {key: arrays[prefix + key].copy() for key in ("prompt_tokens", "prompt_features", "speaker")}
    np.savez(voice_path, **voice)
    args = SimpleNamespace(model_dir=paths["MODEL_DIR"], llm=paths["LLM_ENGINE"], hift=paths["HIFT_ENGINE"],
                           flow=paths["FLOW_ENGINE"], conditioner=paths["CONDITIONER_ENGINE"], voice=voice_path,
                           text="你好，欢迎。", instruction="You are a helpful assistant.", prompt_text="",
                           max_tokens=100, seed=2512, greedy=False, output=directory / "synthesis")
    synthesize(args)
    report = json.loads((args.output / "report.json").read_text(encoding="utf-8"))
    waveform, rate = sf.read(args.output / "audio.wav")
    assert report["status"] == "completed_unqualified" and report["finish_reason"] == "stop_token"
    assert rate == 24000 and waveform.shape == (len(report["tokens"]) * 960,)
    assert np.isfinite(waveform).all() and np.max(np.abs(waveform)) > 1e-5
    assert all(0 <= token < 6561 for token in report["tokens"])
