# SPDX-License-Identifier: Apache-2.0
"""Supplemental real-audio token -> mel validation against pinned upstream.

Reconstruction tokens come from a supplied audio clip, NOT a text LLM. This
adds coverage; it does not replace stress gates or constitute TTS acceptance.
The official ONNX speech tokenizer/CAMPPlus are fixture tools only. Native
conditioning and DiT inference use no ONNX parser or PyTorch model fallback.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .__main__ import sha256_file
from .conditioning import ConditioningEngine, weight_shapes
from .config import FLOW_SHA256, ShapeProfile
from .flow_matching import solve_euler
from .flow_runtime import FlowEngine, INPUT_NAMES
from .offline_flow import OfflineFlow
from .parity_metrics import compare_outputs
from .validate_flow_pytorch import ATOL, INTEGRATED_ATOL, INTEGRATED_RTOL, RTOL, _official_dit, _ieee_fp32_reference

# Ten-step integrated mel outputs; every other stage compares one operation.
INTEGRATED_STAGES = ("flow_with_official_conditions", "native_tokens_to_target_mel")
from .validate_flow_trajectory import _official_solver


def acoustic_cases(tokens, features, speaker):
    """Fixed coverage declared before inference: 16/64/128/256 mel frames x prompt/no prompt."""
    if (tokens.ndim != 2 or tokens.shape[0] != 1 or tokens.shape[1] < 128
            or features.ndim != 3 or features.shape[0] != 1 or features.shape[1] < 256
            or features.shape[2] != 80 or speaker.shape != (1, 192)):
        raise ValueError("Audio must provide at least 128 speech tokens and 256 mel frames")
    for count in (8, 32, 64, 128):
        for with_prompt in (False, True):
            prompt = min(25, count // 2) if with_prompt else 0
            yield {"id": f"tokens{count}_prompt{prompt}", "frames": count * 2, "prompt_tokens_count": prompt,
                   "tokens": tokens[:, prompt:count].copy(), "prompt_tokens": tokens[:, :prompt].copy(),
                   "prompt_features": features[:, :prompt * 2].copy(), "speaker": speaker.copy()}


def extract_audio(source, model_dir, audio):
    """Fixture-only audio frontend following upstream extraction equations.

    Decode explicitly with SoundFile, the backend requested by pinned upstream
    but ignored by torchaudio 2.13. This adapter does not alter the independently
    imported official conditioning, CFM or DiT reference below.
    """
    import torch
    import torchaudio
    import soundfile
    import whisper
    import onnxruntime as ort
    import torchaudio.compliance.kaldi as kaldi
    from matcha.utils.audio import mel_spectrogram

    samples, sample_rate = soundfile.read(str(audio), dtype="float32", always_2d=True)
    if sample_rate < 16000 or not 0 < samples.shape[0] / sample_rate <= 30 or not np.isfinite(samples).all():
        raise ValueError("Fixture audio must be finite, 0 < duration <= 30 s, sample rate >= 16 kHz")
    speech = torch.from_numpy(samples.T.copy()).mean(dim=0, keepdim=True)
    speech16 = torchaudio.transforms.Resample(sample_rate, 16000)(speech) if sample_rate != 16000 else speech
    speech24 = torchaudio.transforms.Resample(sample_rate, 24000)(speech) if sample_rate != 24000 else speech
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    camp = ort.InferenceSession(str(model_dir / "campplus.onnx"), options, providers=["CPUExecutionProvider"])
    tokenizer = ort.InferenceSession(str(model_dir / "speech_tokenizer_v3.onnx"), options, providers=["CPUExecutionProvider"])
    token_features = whisper.log_mel_spectrogram(speech16, n_mels=128)
    tokens = tokenizer.run(None, {tokenizer.get_inputs()[0].name: token_features.numpy(),
                                 tokenizer.get_inputs()[1].name: np.array([token_features.shape[2]], np.int32)})[0]
    tokens = np.asarray(tokens, np.int32).reshape(1, -1)
    features = mel_spectrogram(speech24, n_fft=1920, num_mels=80, sampling_rate=24000, hop_size=480,
                              win_size=1920, fmin=0, fmax=None, center=False).transpose(1, 2).numpy()
    fbank = kaldi.fbank(speech16, num_mel_bins=80, dither=0, sample_frequency=16000)
    fbank = fbank - fbank.mean(dim=0, keepdim=True)
    speaker = camp.run(None, {camp.get_inputs()[0].name: fbank[None].numpy()})[0].reshape(1, 192)
    # Same 2:1 alignment as upstream frontend_zero_shot for 24 kHz models.
    count = min(tokens.shape[1], features.shape[1] // 2)
    return tokens[:, :count], features[:, :count * 2], speaker


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("component", "conditioner", "model-dir", "cosyvoice-source", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--audio", type=Path, help="Defaults to pinned upstream asset/cross_lingual_prompt.wav (13.75 s)")
    args = parser.parse_args(argv)
    import torch

    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = json.loads((args.component / "manifest.json").read_text(encoding="utf-8"))
    flow_profile = ShapeProfile(**manifest["profile"])
    for frames in (16, 64, 128, 256):
        flow_profile.validate_frames(frames)
    implementation_files = [Path(__file__).with_name(name) for name in
                            ("validate_offline_flow.py", "conditioning.py", "offline_flow.py", "flow_matching.py")]
    implementation_hashes = {path.name: sha256_file(path) for path in implementation_files}
    if sha256_file(args.model_dir / "flow.pt") != FLOW_SHA256:
        raise ValueError("Checkpoint does not match pinned model")
    DiT, revision = _official_dit(args.cosyvoice_source)
    _, matcha_revision = _official_solver(args.cosyvoice_source)
    from cosyvoice.flow.flow import CausalMaskedDiffWithDiT
    from cosyvoice.flow.flow_matching import CausalConditionalCFM
    from cosyvoice.transformer.upsample_encoder import PreLookaheadLayer

    audio = args.audio or args.cosyvoice_source / "asset/cross_lingual_prompt.wav"
    cases = list(acoustic_cases(*extract_audio(args.cosyvoice_source, args.model_dir, audio)))
    args.output.mkdir(parents=True, exist_ok=False)
    saved, rows, conditions, traces = {}, [], [], []
    np.savez(args.output / "requests.npz", **{case["id"] + "_" + key: case[key] for case in cases
                                           for key in ("tokens", "prompt_tokens", "prompt_features", "speaker")})

    def cpu(value):
        return value.detach().cpu().numpy().copy()

    def cuda(values):
        return {k: torch.from_numpy(v).cuda() for k, v in values.items()}

    def compare(case, stage, actual, expected, **extra):
        atol, rtol = (INTEGRATED_ATOL, INTEGRATED_RTOL) if stage in INTEGRATED_STAGES else (ATOL, RTOL)
        row = {"case_id": case["id"], "frames": case["frames"], "stage": stage, **extra, "atol": atol, "rtol": rtol,
               **compare_outputs(actual, expected, atol=atol, rtol=rtol)}
        rows.append(row)
        with (args.output / "progress.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    class CaptureConditions(torch.nn.Module):
        def forward(self, mu, mask, spks, cond, n_timesteps, streaming):
            assert n_timesteps == 10 and streaming is False
            self.values = {k: cpu(v) for k, v in dict(mu=mu, mask=mask, spks=spks, cond=cond).items()}
            # Placeholder used ONLY to capture upstream preprocessor outputs.
            # Numerical mel reference is computed below with the real CFM/DiT.
            return mu, None

    capture = CaptureConditions()
    prep = CausalMaskedDiffWithDiT(input_size=80, output_size=80, vocab_size=6561, token_mel_ratio=2, input_frame_rate=25,
                                 pre_lookahead_len=3, pre_lookahead_layer=PreLookaheadLayer(80, 1024, 3),
                                 decoder=capture).eval()
    state = torch.load(args.model_dir / "flow.pt", map_location="cpu", weights_only=True, mmap=True)
    prep.load_state_dict({k: state[k] for k in weight_shapes()}, strict=True)
    prep.cuda()
    conditioner = ConditioningEngine(args.conditioner)
    composition = OfflineFlow(conditioner, SimpleNamespace(device=conditioner.device,
                                                          profile=flow_profile))
    with torch.inference_mode(), _ieee_fp32_reference(torch):
        for case in cases:
            request = {k: case[k] for k in ("tokens", "prompt_tokens", "prompt_features", "speaker")}
            values = cuda(request)
            length = lambda n: torch.tensor([n], device="cuda", dtype=torch.int32)
            prep.inference(values["tokens"], length(values["tokens"].shape[1]), values["prompt_tokens"],
                           length(values["prompt_tokens"].shape[1]), values["prompt_features"],
                           length(values["prompt_features"].shape[1]), values["speaker"], False, True)
            expected = capture.values
            actual = {k: cpu(v) for k, v in composition.prepare(**values).items()}
            conditions.append(expected)
            for key in expected:
                compare(case, "conditioning", actual[key], expected[key], tensor=key)
            saved.update({case["id"] + "_request_" + k: v for k, v in request.items()})
    del prep, composition, conditioner, capture, values
    gc.collect()
    torch.cuda.empty_cache()

    model = DiT(dim=1024, depth=22, heads=16, dim_head=64, ff_mult=2, mel_dim=80, mu_dim=80,
                spk_dim=80, out_channels=80, static_chunk_size=50, num_decoding_left_chunks=-1).eval()
    prefix = "decoder.estimator."
    model.load_state_dict({k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}, strict=True)
    del state
    model.cuda()

    class Recorder(torch.nn.Module):
        def __init__(self, estimator):
            super().__init__()
            self.estimator, self.calls = estimator, []

        def forward(self, x, mask, mu, t, spks, cond, streaming=False):
            output = self.estimator(x, mask, mu, t, spks, cond, streaming=streaming)
            self.calls.append({k: cpu(v) for k, v in zip((*INPUT_NAMES, "velocity"), (x, mask, mu, t, spks, cond, output))})
            return output

    recorder = Recorder(model)
    params = SimpleNamespace(solver="euler", sigma_min=1e-6, t_scheduler="cosine", training_cfg_rate=.2, inference_cfg_rate=.7)
    official = CausalConditionalCFM(240, params, n_spks=1, spk_emb_dim=80, estimator=recorder).eval()
    expected_mels, noises = [], []
    with torch.inference_mode(), _ieee_fp32_reference(torch):
        for case, expected_cond in zip(cases, conditions):
            values = cuda(expected_cond)
            recorder.calls = []
            expected, _ = official(**values, n_timesteps=10, streaming=False)
            trace = recorder.calls
            if len(trace) != 10:
                raise ValueError("Official trajectory must contain exactly ten steps")
            traces.append(trace)
            noise = official.rand_noise[:, :, :case["frames"]].cuda().contiguous()
            noises.append(cpu(noise))
            local = solve_euler(model, **values, noise=noise)
            compare(case, "local_solver_vs_official", cpu(local), cpu(expected))
            expected_mels.append(cpu(expected))
            saved[case["id"] + "_noise"] = cpu(noise)
            saved[case["id"] + "_official_mel"] = cpu(expected)
            for step, call in enumerate(trace):
                saved.update({f'{case["id"]}_step{step}_{k}': v for k, v in call.items()})
    del model, recorder, official, values, expected, local, noise
    gc.collect()
    torch.cuda.empty_cache()
    np.savez(args.output / "official_acoustic_traces.npz", **saved)

    estimator = FlowEngine(args.component)
    conditioner = ConditioningEngine(args.conditioner)
    composition = OfflineFlow(conditioner, estimator)
    for case, trace, noise, expected, expected_cond in zip(cases, traces, noises, expected_mels, conditions):
        for step, call in enumerate(trace):
            actual = cpu(estimator(**cuda({k: call[k] for k in INPUT_NAMES})))
            compare(case, "velocity_on_official_trajectory", actual, call["velocity"], step=step)
        # Isolate DiT/solver from conditioning rounding, then test real composition.
        isolated = cpu(solve_euler(estimator, **cuda(expected_cond), noise=torch.from_numpy(noise).cuda()))
        compare(case, "flow_with_official_conditions", isolated, expected)
        request = {k: case[k] for k in ("tokens", "prompt_tokens", "prompt_features", "speaker")}
        actual = cpu(composition(**cuda(request), noise=torch.from_numpy(noise).cuda()))
        compare(case, "native_tokens_to_target_mel", actual, expected[:, :, case["prompt_tokens_count"] * 2:])
        saved[case["id"] + "_native_target_mel"] = actual
    np.savez(args.output / "acoustic_evidence.npz", **saved)
    if implementation_hashes != {path.name: sha256_file(path) for path in implementation_files}:
        raise RuntimeError("Family implementation changed during validation; partial evidence retained")
    stages = sorted({row["stage"] for row in rows})
    report = {
        "scope": "supplemental_audio_reconstruction_tokens_not_text_to_speech_or_replacement_stress_gate",
        "passed": all(row["passed"] for row in rows), "atol": ATOL, "rtol": RTOL,
        "integrated_atol": INTEGRATED_ATOL, "integrated_rtol": INTEGRATED_RTOL, "integrated_stages": list(INTEGRATED_STAGES),
        "source_revision": revision, "matcha_revision": matcha_revision, "checkpoint_sha256": FLOW_SHA256,
        "audio_sha256": sha256_file(audio), "audio_path": str(audio),
        "fixture_frontend": "explicit_soundfile_decode_plus_torchaudio_resample_whisper_kaldi_matcha_features_ort_cpu",
        "fixture_model_sha256": {name: sha256_file(args.model_dir / name) for name in ("campplus.onnx", "speech_tokenizer_v3.onnx")},
        "plan_sha256": sha256_file(args.component / "flow.plan"),
        "conditioner_plan_sha256": sha256_file(args.conditioner / "conditioning.plan"),
        "evidence_sha256": sha256_file(args.output / "acoustic_evidence.npz"),
        "implementation_sha256": implementation_hashes,
        "torch_version": torch.__version__, "gpu": torch.cuda.get_device_name(),
        "dependencies": {name: importlib.metadata.version(name) for name in
                         ("x-transformers", "diffusers", "onnxruntime", "torchaudio", "openai-whisper", "soundfile")},
        "reference_precision": {"cudnn_tf32": False, "matmul_tf32": False, "matmul_precision": "highest", "sdpa_policy": "auto"},
        "summary": {stage: {"passed": sum(row["passed"] for row in rows if row["stage"] == stage),
                            "total": sum(row["stage"] == stage for row in rows)} for stage in stages},
        "cases": rows,
    }
    with (args.output / "report.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report["summary"], indent=2))
    if not report["passed"]:
        raise SystemExit("Supplemental acoustic validation FAILED; original stress gates remain unchanged")


if __name__ == "__main__":
    main()
