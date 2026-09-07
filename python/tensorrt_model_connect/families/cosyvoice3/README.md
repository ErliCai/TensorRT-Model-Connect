# CosyVoice3: experimental native offline TTS components

This directory starts the implementation of
`FunAudioLLM/Fun-CosyVoice3-0.5B-2512`. **It does not yet add end-to-end
CosyVoice3 support to `trtmc build` / `trtmc infer`.** There is intentionally
no exported `plugin`, runtime strategy, or E2E manifest claiming otherwise.

## Directory layout

```text
cosyvoice3/
  __main__.py          # inspect / build-* / prepare-voice / synthesize CLI
  tts.py               # reference audio preparation and offline TTS composition
  frontend.py          # native CAMPPlus and speech tokenizer graphs/runtimes
  llm.py, hift.py      # native LLM and vocoder graphs/runtimes
  flow.py              # Euler solver and offline token-to-Mel composition
  conditioning.py      # token and speaker conditioning
  flow_builder.py, flow_runtime.py
  components.py        # family-local TensorRT graph/runtime mechanics
  artifacts.py         # hashing and atomic component publication
  config.py, constants.py, checkpoint_mapper.py
  validation/          # maintained numerical gates and FP64 calibration
  diagnostics/         # development-only failure localization tools
  tests/               # pytest contracts and opt-in GPU comparisons
```

Runtime modules do not import validation or diagnostics. Diagnostic tools are
retained for reproducibility, not required for inference; their inclusion in a
future upstream PR is a separate decision.

Migration: replace `cosyvoice3.validate_*` with
`cosyvoice3.validation.validate_*`, and `cosyvoice3.audit_flow_fp64` with
`cosyvoice3.validation.audit_flow_fp64`. Other `audit_*` and `diagnose_*`
commands now live under `cosyvoice3.diagnostics`. The old module paths have
no forwarding stubs. Import `solve_euler` and `OfflineFlow` from `cosyvoice3.flow`.
Existing `build-*`, `inspect`, and `synthesize` commands and plan formats
are unchanged. New builds record `artifacts.py` in implementation hashes;
old manifests and evidence must not be rewritten.

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
- Native FP32 Qwen2 speech decoder from `llm.pt`: all 24 layers, compact
  two-head GQA KV cache, separate text/speech embeddings, CosyVoice3 prompt
  packing, RAS sampling and all 200 stop classes. This does not select `llm.rl.pt`.
- Native offline HiFT/F0, causal repeat-then-convolve upsampling, Snake blocks,
  explicit uniform excitation noise, Hann STFT and overlap-normalized inverse
  STFT. F0 uses FP32 with short channel reductions; upstream inference uses
  FP64. Long-waveform reference consistency is **not yet qualified**.
- An experimental Python `synthesize` command: plain text plus an explicit
  prepared voice NPZ -> speech tokens -> Mel -> 24 kHz WAV. Each engine is
  unloaded before the next stage to limit memory on an 8 GB GPU.
- Native FP32 CAMPPlus and speech tokenizer v3, including all 617/198 published
  initializer tensors, with exact checkpoint checksums. `prepare-voice` decodes
  reference WAV, computes non-learned features, runs these two TensorRT engines
  sequentially and writes the explicit voice NPZ consumed by `synthesize`.
- Dependency-light contract tests, solver tests, and an explicit numerical
  comparison command using the upstream FP32 ONNX as an independent oracle.

Not implemented yet:

- Text normalization and phoneme/paralinguistic tags. The local text tokenizer handles plain text and the
  instruction boundary, not every upstream tokenizer extension.
- Streaming, a single-call reference-WAV synthesis API and qualified voice cloning.
- Bundle packaging, the family-owned C++ pipeline/DSO, three-root `MODEL.toml`
  registration, and full TTS E2E/reference validation and isolated-family CI.

All model equations/build/runtime code live in this family; no sibling model
implementation is imported. Published ONNX files provide build-time weights
and independent test oracles; no TensorRT ONNX parser is used, and production
inference never calls ONNX Runtime. This is a development component, not yet
an upstream-ready model-family contribution.

## New LLM and HiFT development path (2026-09-08)

```bash
python -m tensorrt_model_connect.families.cosyvoice3 build-llm \
  --model-dir /path/to/Fun-CosyVoice3-0.5B-2512 --output /new/llm \
  --max-context 512 --opt-tokens 64
python -m tensorrt_model_connect.families.cosyvoice3 build-hift \
  --model-dir /path/to/Fun-CosyVoice3-0.5B-2512 --output /new/hift \
  --min-frames 4 --opt-frames 64 --max-frames 256
python -m tensorrt_model_connect.families.cosyvoice3 synthesize \
  --model-dir /path/to/Fun-CosyVoice3-0.5B-2512 \
  --llm /new/llm --conditioner /existing/conditioner --flow /existing/flow \
  --hift /new/hift --voice /path/to/voice.npz \
  --text '你好，欢迎。' --max-tokens 100 --seed 2512 --output /new/synthesis
```

`voice.npz` must contain exactly `prompt_tokens` INT32 `[1,N]`,
`prompt_features` FP32 `[1,2N,80]`, and a nonzero `speaker` FP32 `[1,192]`.
Use the explicit native `prepare-voice` command below to obtain these arrays;
there is no hidden official-model fallback. Without `--prompt-text` the LLM
uses instruction mode (no reference speech tokens in its prompt); Flow still
uses the prepared voice. With `--prompt-text`, supply the exact transcript
corresponding to the complete provided reference speech tokens.

## Native reference-audio frontend

```bash
python -m tensorrt_model_connect.families.cosyvoice3 build-campplus \
  --model-dir /path/to/Fun-CosyVoice3-0.5B-2512 --output /new/campplus
python -m tensorrt_model_connect.families.cosyvoice3 build-speech-tokenizer \
  --model-dir /path/to/Fun-CosyVoice3-0.5B-2512 --output /new/speech-tokenizer
python -m tensorrt_model_connect.families.cosyvoice3 prepare-voice \
  --audio /path/to/reference.wav --campplus /new/campplus \
  --speech-tokenizer /new/speech-tokenizer --output /new/reference-voice
# Pass /new/reference-voice/voice.npz as --voice to synthesize.
```

The input is one unpadded recording, 0.1–30 seconds, at least 16 kHz; stereo is
averaged to mono. Defaults cover 4–3000 feature frames. The tokenizer uses
128-bin Whisper log-mel, two stride-two convolutions, twelve attention/FSMN
blocks, shared published RoPE tables and eight ternary scalar quantizers
(6561 possible tokens). CAMPPlus uses mean-centered 80-bin Kaldi filterbanks,
convolutional/dense context blocks and unbiased temporal statistics to produce
a 192-dimensional speaker vector. Length masks are identically true in this
explicit unpadded batch-one contract; padded batches are not supported.

Build-time dependencies include `onnx` for checkpoint tensor reading, not
conversion or execution. Audio preparation uses `soundfile`, `torchaudio`,
`openai-whisper` (feature function only), `librosa`, NumPy and PyTorch signal
processing. No official CosyVoice checkout or learned PyTorch model is needed
for production. `onnxruntime` is required only by the independent parity tests.

`prepare-voice` records recording/plan/output checksums and crops reference
tokens/Mel to the upstream 2:1 alignment. It refuses existing output paths.
The downstream conditioner/Flow profiles must still accommodate reference
tokens **plus generated tokens**; a 30-second frontend profile does not enlarge
the other engines, and there is no silent reference truncation to fit them.

Run `tests/test_frontend.py` with `COSYVOICE3_RUN_GPU_TESTS=1`,
`COSYVOICE3_MODEL_DIR`, `COSYVOICE3_CAMPPLUS_ENGINE`,
`COSYVOICE3_SPEECH_TOKENIZER_ENGINE` and `COSYVOICE3_OFFICIAL_SOURCE`.
Set `COSYVOICE3_FRONTEND_EVIDENCE` to a fresh directory to retain each comparison
and its exact arrays. Token IDs must match exactly; speaker vectors use
`atol=1e-3, rtol=1e-4`. Tests include segment boundaries, longest shapes and two
recordings. Three sample rates independently compare Mel extraction against
the checksum-pinned official Matcha implementation with zero tolerance.
These checks do not replace the separate, still-unqualified HiFT waveform gates
or establish perceptual voice-cloning quality.

On the recorded WSL RTX 4070 Laptop / TensorRT 11.1.0.106 run, the final frontend
engines passed 11/11 cases each (tokens exact; maximum speaker absolute error
3.505e-5). All 38 frontend tests passed, including the no-ONNX-runtime
preparation test. Default RAS synthesis with the newly prepared one-second
reference produced a finite, nonzero 0.88-second WAV. An additional greedy
scenario hit the LLM length limit and remains a recorded failure; it was not
converted to a success by changing the stop rule. These are distinct workloads.

## Speech generation behavior and existing HiFT qualification

The synthesis entry point inserts `<|endofprompt|>` itself. It deliberately rejects
control/phoneme tags in plain-text arguments. RAS uses the published sampling
rule but a local NumPy RNG; equal seeds do not imply identical sampled tokens
to PyTorch. Greedy argmax is available explicitly with `--greedy`.
The upstream native `sampling_ids` masks only class 6561 before its minimum
length, although CosyVoice3 has 200 stop classes. This behavior is retained,
not silently corrected or interpreted as a guaranteed minimum output length.

All output directories must be new. The pipeline saves tokens, Mel, WAV and
a report binding the plans, voice file, tokenizer and seed. A length limit
without a stop token is an error, not successful truncated speech. A successful
run is marked `completed_unqualified`, not a model-support or quality pass.
The 256-frame Flow limit includes the prompt: 25 prompt tokens leave at most
103 target tokens. Every component profile must fit the requested workload.

### New tests and known failures

LLM/HiFT maintained tests are in `tests/test_speech_components.py`; reference
frontend tests are in `tests/test_frontend.py`, alongside the existing
development-component tests. No new production diagnostic
CLI scripts were added. The family has not yet moved to registered E2E ownership.

```bash
# CPU contracts; GPU tests are opt-in.
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest \
  python/tensorrt_model_connect/families/cosyvoice3/tests/test_speech_components.py -q

# Small native graphs, no full checkpoints required.
COSYVOICE3_RUN_GPU_TESTS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest \
  python/tensorrt_model_connect/families/cosyvoice3/tests/test_speech_components.py \
  -q -s -k 'tiny_llm or source_phase'
```

Full-weight tests require `COSYVOICE3_RUN_GPU_TESTS=1`, `COSYVOICE3_MODEL_DIR`,
`COSYVOICE3_LLM_ENGINE`, `COSYVOICE3_HIFT_ENGINE`, and for HiFT the clean pinned
`COSYVOICE3_OFFICIAL_SOURCE`. Select `-k full_llm` or `-k full_hift` separately
to limit GPU memory. Set `COSYVOICE3_ACOUSTIC_EVIDENCE` to an existing
`validate_offline_flow` `acoustic_evidence.npz` for real-Mel cases. Optional
`COSYVOICE3_SPEECH_EVIDENCE` must name a fresh directory; failures persist
metrics before assertions. The tests verify the fixed LLM/HiFT checkpoint
hashes, engine checkpoint binding and retain environment/plan metadata.

The execution smoke additionally requires `COSYVOICE3_FLOW_ENGINE`,
`COSYVOICE3_CONDITIONER_ENGINE`, `COSYVOICE3_ACOUSTIC_REQUESTS` (the existing
acoustic validator's `requests.npz`) and optionally a new `COSYVOICE3_TTS_OUTPUT`.
Select `-k offline_text`; this also materializes an explicit prepared test voice.
See `notes/08-cosyvoice3-test-guide.html` for this workstation's complete commands.

First complete execution produced 22 speech tokens, EOS 6562 and a 24 kHz,
0.88-second WAV for `你好，欢迎。`. This is not a listening/ASR quality claim.
LLM prefill/cached decoding passed four lengths with 28 numerical comparisons
and matching top-1 against a real-weight Transformers Qwen2 eager reference
(Transformers 5.14.1 in the observed environment).

HiFT gates are separate from Flow: F0 `atol=.01 Hz, rtol=1e-5`; source and
waveform `atol=1e-3, rtol=1e-4`. These are development numerical checks, not
calibrated perceptual-quality criteria. In the first short-reduction build,
same-source decoding passed all tested cases, but full-waveform comparisons
failed on the 256-frame synthetic case and 128/256-frame real-Mel cases.
F0 error improved substantially (64-frame stress: .02224 -> .000755 Hz),
while same-F0 isolation also exposed phase accumulation/scaling rounding.
The failures and original gates are retained. Do not drop cases or widen
thresholds to declare this implementation qualified.

Final clean component rebuild (LLM 02 / HiFT 03) retained the LLM pass (4/4,
28 comparisons; maximum logit error 1.593e-4), but full HiFT waveform parity
was 4/8: synthetic 256 and acoustic 64/128/256 failed. All eight same-source
decoder comparisons and F0 checks passed. The acoustic-64 regression relative
to HiFT 02 is retained, not replaced by the older plan's better pass count.
Reports: `/home/erlic/models/cosyvoice3-speech-final-01/` on the tested WSL host.
Dependency-light regression: WSL 93 passed / 35 skipped; Windows 71 passed /
46 skipped. The small native LLM/cache/source tests passed 5/5. These counts
do not include or erase the failing full-weight checks.

The remaining sections preserve the earlier Flow validation history.

### Python completion follow-up (still incomplete)

The causal source now expresses chronological FP32 phase accumulation with a
TensorRT Loop/Recurrence. On the declared reference environment, this matches
the official interleaved time scan; a parallel cumulative sum rounds
differently. The manifest records `phase_accumulation`. This is reference
consistency work, not a perceptual-quality or higher-precision claim.

HiFT 04 (`trtmc-cosyvoice3-hift-fp32-04`) passed all eight same-F0 waveform
comparisons and five of eight full-waveform cases under the unchanged gates.
The remaining failures are synthetic 256 and acoustic 64/256. An attempted
four-channel F0 reduction (HiFT 05) still failed three cases and regressed a
same-F0 comparison; that experiment was reverted, with its evidence retained.
The retained implementation uses the original 16-channel F0 reductions.

Zero-shot prompt text now requires nonempty reference speech tokens and a
nonblank transcript. Invalid seeds or generation limits are rejected before
creating output directories. Raw-reference-audio frontend integration and
full waveform/quality qualification remain unfinished; prepared NPZ input is
still required. No C++ integration or full Python completion is claimed.

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
python -m tensorrt_model_connect.families.cosyvoice3.validation.audit_flow_fp64 \
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

python -m tensorrt_model_connect.families.cosyvoice3.validation.validate_flow \
  --component /path/to/new-cosyvoice3-flow-component \
  --oracle-onnx /path/to/Fun-CosyVoice3-0.5B-2512/flow.decoder.estimator.fp32.onnx \
  --frames 4 17 64 128 --report /path/to/new-flow-parity.json

python -m tensorrt_model_connect.families.cosyvoice3.validation.validate_flow_pytorch \
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
TensorRT graph. `flow.py` composes it with the DiT estimator, preserving
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

python -m tensorrt_model_connect.families.cosyvoice3.validation.validate_offline_flow \
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
