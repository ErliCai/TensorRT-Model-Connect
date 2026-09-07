# CosyVoice3: experimental native Flow component

This directory starts the implementation of
`FunAudioLLM/Fun-CosyVoice3-0.5B-2512`. **It does not yet add end-to-end
CosyVoice3 support to `trtmc build` / `trtmc infer`.** There is intentionally
no exported `plugin`, runtime strategy, or E2E manifest claiming otherwise.

Implemented:

- Safe, data-only parsing of `cosyvoice3.yaml`, without executing HyperPyYAML
  constructors or trusting the checkpoint's empty `config.json`.
- Strict shape/key checking of all 322 DiT parameter tensors in `flow.pt`.
- Native FP32 TensorRT DiT: time embedding, causal grouped convolutions,
  adjacent-pair partial RoPE before the head split, 22 attention/feed-forward
  blocks, adaptive normalization, and velocity projection.
- Fixed FP32 timestep coefficients from the published graph, and the original
  rotary-frequency checkpoint buffer, to preserve the model's rounded constants
  independently of the build host's NumPy/exp implementation.
- A shape-checked CUDA component runner and cosine-scheduled Euler solver
  with classifier-free guidance. Noise is explicitly supplied by the caller.
- Native offline token embedding, pre-lookahead convolutions, 2x mel-frame
  expansion and normalized speaker projection, plus `OfflineFlow` composition
  with prompt alignment/cropping and explicit noise (speech tokens -> mel).
- Dependency-light contract tests, solver tests, and an explicit numerical
  comparison command using the upstream FP32 ONNX as an independent oracle.

Not implemented yet:

- Text normalization/tokenization and reference-audio feature extraction.
- Qwen2-based speech-token generation, sampling and termination.
- HiFT/F0 waveform synthesis, streaming and request-time voice cloning.
- Bundle packaging, the family-owned C++ pipeline/DSO, three-root `MODEL.toml`
  registration, and full TTS E2E/reference validation and isolated-family CI.

All model equations/build/runtime code live in this family; no sibling model
implementation is imported. The ONNX graph is used **only as a test oracle**,
not parsed into the native engine. This is a development component, not yet
an upstream-ready model-family contribution.

## Validation status (2026-09-07; baseline before acoustic integration)

- RTX 4070 Laptop, TensorRT 11.1.0.106: real-checkpoint FP32 engine builds.
- Before adding native conditioning: 70 family tests passed with GPU and
  pinned-source tests explicitly enabled. These are not full-model gates.
- **Full-checkpoint stress gate was NOT passing at that time:** the earlier
  `final` plan passed 4/8 against the unmodified pinned official PyTorch model
  under the former `atol=1e-3, rtol=1e-3`. Three subsequent attention experiments were reverted
  because they regressed independent integration. See the sections below.
- Pinning x-transformers to upstream's required 2.11.24 instead of 2.28.3
  reproduced every baseline stress metric exactly; it did not fix the gate.
- Supplemental real-audio-token tests are separate from these stress cases.
  New integration results and limitations are recorded in
  `notes/14-cosyvoice3-next-stage.html`; passing them never erases a stress failure.

### FP64 oracle, GEMM accumulation and calibrated gates (2026-09-07/08)

`audit_flow_fp64` evaluates the pinned official DiT in float64 and measures
both the native engine and the official FP32 model against it. Per-operation
probes with that oracle located the remaining gap: TensorRT FP32 linear layers
carried about twice the rounding error of the cuBLAS kernels behind the PyTorch
reference, depending on which GEMM tactic TensorRT selected in a given build,
and attention inherited it through its four projections. Two graph changes
follow from this evidence, both mathematically identical to the reference:

- Every linear reduction is accumulated in `LINEAR_K_BLOCKS` (4) blocks summed
  pairwise, which bounds the error independent of tactic selection.
- Q is scaled by the exact power of two `64**-0.5` before `QK^T`; the earlier
  `64**-0.25` on both Q and K mirrored the ONNX export's math decomposition, not
  the memory-efficient kernel PyTorch actually selects for FP32.

Plan `e788e44f...` (max 256 frames) against FP64 truth under the former
`1e-3` ruler: acoustic-trajectory velocity 62/80 passed with 80 elements over
tolerance and max error 0.0075, versus official FP32 at 56/80, 367 and 0.0139;
stress 5/8 with 3 elements over versus official FP32 at 6/8 with 9. The native
engine is now at least as accurate as the reference, and the official FP32
model itself passes only 56/80 and 6/8 against float64: no FP32 implementation
of this checkpoint can meet an elementwise `1e-3` gate against another FP32
implementation. On 2026-09-08 the maintainer chose to recalibrate the gates
instead of replacing them with an FP64 criterion:

- One estimator call (stress, trajectory velocities, environment compare, ONNX
  cases): `atol=rtol=2e-2`. Against float64 truth official FP32 deviates by up
  to 6.6x of `1e-3` (synthetic CFG trajectory, 64 frames masked, step 4) and
  the native engine by up to 6.2x; the two FP32 results differ by 9.7x there.
  The gate admits the sum of both deviations with margin.
- Ten-step integrated mel (`flow_with_official_conditions`,
  `native_tokens_to_target_mel`, `native_integration_vs_official`,
  `10_step_euler`): `atol=rtol=1e-1`. Guided Euler integration amplifies
  rounding on the 256-frame prompt-free case: official FP32 differs from the
  float64 integration by 29x of `1e-3` there, the native engine by 16x, and
  the two FP32 results by 43.5x; the other seven cases stay within 1.2x.

These gates catch wrong mathematics, layouts or masks, which produce O(1)
errors; they do not rank sub-reference rounding. `audit_flow_fp64` remains the
instrument for that and is the evidence behind every claim above. With plan
`e788e44f...` all four validators pass: stress 8/8, ONNX 9/9, synthetic CFG
trajectory 96/96 (worst 0.48 of the gate), real-audio suite 136/136 (worst
0.44); the GPU test suite passes with the official source enabled.

```bash
python -m tensorrt_model_connect.families.cosyvoice3.audit_flow_fp64 \
  --component /path/to/new-cosyvoice3-flow-component \
  --model-dir /path/to/Fun-CosyVoice3-0.5B-2512 \
  --cosyvoice-source /path/to/CosyVoice-at-pinned-revision \
  --acoustic-evidence /path/to/new-acoustic-report \
  --report /path/to/new-fp64-audit.json
```

The old v3 ONNX report is historical, not the latest reference acceptance result.
Older development engines/reports remain in the local WSL models directory;
the original checkpoint files have not been modified. No audio has been
generated by this implementation yet.

## Build and compare

Run in a Linux CUDA environment with PyTorch, NumPy, PyYAML and TensorRT
available. TensorRT is accessed through the repository's `trt_compat` boundary.
For comparison, also install ONNX Runtime. The local development run uses
TensorRT 11.1.0.106 with CUDA 13; it does not establish compatibility with all
TensorRT versions or platforms.

From the repository root:

```bash
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"

python -m tensorrt_model_connect.families.cosyvoice3 inspect \
  --model-dir /path/to/Fun-CosyVoice3-0.5B-2512

python -m tensorrt_model_connect.families.cosyvoice3 build-flow \
  --model-dir /path/to/Fun-CosyVoice3-0.5B-2512 \
  --output /path/to/new-cosyvoice3-flow-component \
  --min-frames 4 --opt-frames 32 --max-frames 128 --workspace-mib 256

python -m tensorrt_model_connect.families.cosyvoice3.validate_flow \
  --component /path/to/new-cosyvoice3-flow-component \
  --oracle-onnx /path/to/Fun-CosyVoice3-0.5B-2512/flow.decoder.estimator.fp32.onnx \
  --frames 4 17 64 128 --report /path/to/new-flow-parity.json

python -m tensorrt_model_connect.families.cosyvoice3.validate_flow_pytorch \
  --component /path/to/new-cosyvoice3-flow-component \
  --model-dir /path/to/Fun-CosyVoice3-0.5B-2512 \
  --cosyvoice-source /path/to/CosyVoice-at-pinned-revision \
  --report /path/to/new-official-pytorch-parity.json

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest \
  python/tensorrt_model_connect/families/cosyvoice3/tests -q

# Opt in to the tiny-network TensorRT/PyTorch GPU comparisons as well:
COSYVOICE3_RUN_GPU_TESTS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest \
  python/tensorrt_model_connect/families/cosyvoice3/tests -q
```

Choose **new** output and report paths; commands refuse to overwrite existing
artifacts. The model directory is read-only. The manifest hashes local sources
and the plan; it explicitly does not claim that local files were verified
against the pinned Hugging Face revision. Only load TensorRT plans you trust.
The validation command requires the exact upstream oracle SHA-256 listed in
`config.py`; substituting a different reference is rejected.

Frame limits include both reference and generated frames. 128 mel frames
represent 2.56 seconds at 50 frames/s; this deliberately small component test
profile is not sufficient for many complete voice-cloning requests. Inputs are
FP32; batch 2 means one conditional/unconditional pair, **not two utterances**.
Every mask row must contain at least one valid frame. Streaming is rejected,
not silently approximated with the offline attention mask. TF32 is disabled.
The official-PyTorch validator temporarily disables PyTorch cuDNN TF32 as
well, then restores the caller's settings, so both sides use the same FP32
policy. Otherwise PyTorch's default TF32 convolution creates a false first
divergence in the causal position embedding.

The runner is deliberately synchronous for correctness. It performs input
validation, keeps buffers alive until CUDA completes, and serializes access
to its execution context. No throughput/latency or 8 GB full-model memory
claim is made.

## Validation audit (2026-09-07)

The eight estimator cases are synthetic stress inputs with independent batch
rows, not real CFG request pairs or an audio-quality score. The final production
plan passed 4/8 against pinned official PyTorch. A reference-only audit (no
TensorRT) also found only 5/8 agreement between official PyTorch's automatic
and math FP32 SDPA backends under the then-unchanged `atol=rtol=1e-3` gate. Neither
observation establishes a correct production engine or licenses relaxing CI.

`audit_flow_reference` repeats those eight inputs and saves backend comparisons.
`diagnose_flow_ops` probes operations using captured official inputs;
`diagnose_attention_precision` requires a new, unmasked capture containing mask
metadata. These tools are diagnostics, not alternative acceptance tests: exit
code zero means collection completed, so inspect each JSON `passed` field.

The official validator now rejects locally modified source files, records its
precision policy and x-transformers version, and reports elementwise tolerance
violations. Both parity validators reject mismatched shapes and nonfinite
outputs. Inputs are unchanged; the tolerances were recalibrated on 2026-09-08
as described above. The ONNX Euler
comparison uses the local solver on both sides; it does not independently prove
solver parity with the official implementation. Independent solver traces and
supplemental acoustic reconstruction inputs are now covered separately below;
end-to-end text/audio validation remains work.

Chinese diagrams, evidence and reproduction commands:
`notes/11-cosyvoice3-test-audit.html`.

## Independent solver and trajectory validation

`validate_flow_trajectory` adds three separate checks (single-call and
integrated tolerances as calibrated above): the local solver vs the unmodified pinned
official solver using the same PyTorch estimator, native velocity predictions
on every official trajectory step, and independent ten-step native integration
vs the official result. It stores every official input/output in an NPZ and
fails with a nonzero exit code on any failed comparison. Conditions remain
synthetic, not acoustic features captured from a real request.

Initialize the official checkout's pinned `third_party/Matcha-TTS` submodule
and install its import dependencies before running this supplemental validator.
Set `COSYVOICE3_OFFICIAL_SOURCE` to the clean pinned checkout to opt into the
eight independent CPU solver tests (1/3/10/37 steps, guidance 0/0.7). Without
that setting they are explicitly skipped, not counted as passing.

The local synchronous solver now reuses per-request contiguous x/time buffers;
requests never share these buffers and caller-owned inputs remain unchanged.
This reduces allocation count; no measured latency improvement is claimed.
New component manifests record the five family build-source hashes and workspace
size. A build refuses publication if those source files change during the build.
This records provenance, not deterministic tactic selection.
Full-checkpoint trials of Q-only scaling and two key-centering variants raised
the stress count from 4/8 to 6/8, but regressed integration errors or pass counts.
All three attention variants were reverted; default attention is unchanged.
The final-plan baseline remains 4/8 stress, 45/80 trajectory velocity, and 7/8
integrated results. Local-vs-official solvers using the same real PyTorch model
matched exactly in all eight supplemental cases. No 100% model-parity claim is made.
Detailed Chinese diagrams, dependencies, commands and results:
`notes/12-cosyvoice3-independent-solver.html`.

## Native offline conditioning and real-audio reconstruction evidence

`conditioning.py` owns seven published weight tensors and a separate native
TensorRT graph. `offline_flow.py` composes it with the DiT estimator, preserving
prompt-first token ordering, two mel frames per speech token, conditional-only
prompt features, and target-only output cropping. The caller supplies noise;
this API does not generate text tokens or promise upstream automatic noise
initialization, streaming, or a complete waveform.

```bash
python -m tensorrt_model_connect.families.cosyvoice3 build-conditioner \
  --model-dir /path/to/Fun-CosyVoice3-0.5B-2512 \
  --output /path/to/new-conditioner

# This supplemental suite includes 256-frame inputs: build a matching profile.
python -m tensorrt_model_connect.families.cosyvoice3 build-flow \
  --model-dir /path/to/Fun-CosyVoice3-0.5B-2512 \
  --output /path/to/new-flow-256 --max-frames 256

python -m tensorrt_model_connect.families.cosyvoice3.validate_offline_flow \
  --component /path/to/new-flow-256 --conditioner /path/to/new-conditioner \
  --model-dir /path/to/Fun-CosyVoice3-0.5B-2512 \
  --cosyvoice-source /path/to/CosyVoice-at-pinned-revision \
  --output /path/to/new-acoustic-report
```

This suite uses the upstream 13.75-second `cross_lingual_prompt.wav` asset to
extract reconstruction speech tokens with the official ONNX tokenizer and
speaker features with CAMPPlus. These CPU ONNX sessions are fixture tooling,
never the native conditioning/DiT implementation. Explicit SoundFile decoding
works around torchaudio 2.13 ignoring upstream's requested `soundfile` backend;
feature math still uses torchaudio, Whisper, Kaldi and pinned Matcha functions.
No official model computation is replaced or monkey-patched.

Eight predetermined cases cover 16/64/128/256 mel frames, with/without prompt.
The numerical reference runs unmodified upstream `CausalMaskedDiffWithDiT`
preprocessing and the actual `CausalConditionalCFM.forward`/DiT with its stored
noise. It separately compares conditioning, solver, all 80 official trajectory
steps, integrated Flow with identical conditions, and fully native target mel.
Inputs/traces/outputs and hashes are retained, intermediate comparisons are
appended to `progress.jsonl`, and any numerical failure returns nonzero.
These are eight variants of ONE recording, not eight independent speakers,
not text-generated tokens, and not a speech-quality acceptance suite.

The max=256 stage-two plan (`31f1279c...`) did NOT qualify under `1e-3`: stress
4/8, acoustic conditioning 32/32, official-solver agreement 8/8, acoustic
trajectory velocity 35/80, Flow integration with official conditions 5/8, and
fully native target mel 4/8. The default Flow math has not changed. Do not
promote this larger-profile artifact as an accuracy improvement.

`audit_acoustic_reference` replays saved real-audio inputs without TensorRT.
The recorded PyTorch automatic backend reproduced all 80 velocities bit for
bit; switching only PyTorch SDPA to MATH passed 35/80 under the same gate.
This is backend-sensitivity evidence, not proof that native inference is right.
`validate_flow_environment capture/compare` decouples TensorRT capture from
reference execution in a separate environment, preserving exact inputs,
outputs, plan/checkpoint hashes and all case identities. With torch/torchaudio
2.3.1, numpy 1.26.4 and x-transformers 2.11.24, comparisons still passed 4/8
stress and 35/80 acoustic velocities. Aligning those official core versions
did not resolve the issue. Full TTS acceptance remains blocked; changing any
test acceptance policy requires human review, not an automatic threshold edit;
the 2026-09-08 recalibration above was such a reviewed decision.

## Equation references

- [Published model](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512/tree/29e01c4e8d000f4bcd70751be16fa94bf3d85a18)
- [DiT forward](https://github.com/FunAudioLLM/CosyVoice/blob/074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc/cosyvoice/flow/DiT/dit.py)
- [DiT modules](https://github.com/FunAudioLLM/CosyVoice/blob/074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc/cosyvoice/flow/DiT/modules.py)
- [Flow matching](https://github.com/FunAudioLLM/CosyVoice/blob/074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc/cosyvoice/flow/flow_matching.py)

Chinese implementation walkthrough: `notes/06-cosyvoice3-implementation.html`.
Chronological log, including unsuccessful attempts and all three reports:
`notes/07-cosyvoice3-development-log.html`.
