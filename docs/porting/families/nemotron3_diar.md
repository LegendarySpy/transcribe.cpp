# Nemotron-3 Diarization

Status: validation (Stages 1-7 run locally; not shipped to a public repo yet)

Nemotron-3 Diarization is NVIDIA's Streaming Sortformer v3: a frame-level
end-to-end speaker diarizer (`encoder-diarizer`, like `sortformer`). It
consumes 16 kHz mono audio and emits a `T x 8` matrix of per-speaker
activity probabilities every 10 ms, columns in speaker arrival order, with
the same Arrival-Order Speaker Cache (AOSC) + FIFO streaming scheme as v2.

It is a separate family from `sortformer` because nothing below the
streaming policy is shared: the encoder is a 31-layer pre-LN RoPE
Transformer over 8-frame mel stacking (no FastConformer, no second
Transformer), the head upsamples back to the mel rate with a sub-pixel
convolution, the speaker cache stores embedder outputs and uses a learned
silence embedding, and cache probabilities are pooled from the 10 ms output.
The public surface is shared: the `SFST` run extension, its preset enum,
and the transcript-independent `transcribe_speaker_segment` output, so a
caller that drives Sortformer v2.1 drives this model unchanged. On top of
that, this family implements push-audio live diarization
(`transcribe_stream_*` with the `SFLV` stream extension).

## Identity

- Family key: `nemotron3_diar`
- Upstream architecture string: `Nemotron3DiarizationForAudioFrameClassification` (HF Transformers); NeMo `SortformerEncLabelModel` (`.nemo`)
- Hugging Face repo: `nvidia/Nemotron-3-Diarization`
- Hugging Face revision: `a435e9867d79e789e90053f9b6d6834053af564a`
- License: OpenMDW 1.1 (commercial use allowed)
- Variants: `nemotron-3-diarization` (100M params, fp32 checkpoint)

## References

- Canonical reference: HF Transformers `Nemotron3DiarizationForAudioFrameClassification` + `Nemotron3DiarizationProcessor`, the integration NVIDIA documents on the model card. Pinned at transformers `c8b81b63232be35ab1774dd3cabbf499d8b9808f` in `scripts/envs/nemotron3_diar`.
- Instrumented reference: `scripts/dump_reference_nemotron3_diar_transformers.py` (forward hooks per stage; presets by overriding the offline-mode chunk geometry, which is the same code path the processor's streaming mode drives).
- Cross-check references: NeMo `e2e_diarize_speech.py` (NVIDIA's DER protocol, `diarization_evaluation.md` on the model card); parakeet-rs 0.3.8 `sortformer` ONNX runtime.

## Architecture

```
mel 128 x T (preemph 0.97, log(x + 2^-24), no normalization, constant-pad centered STFT)
 -> stack 8 frames -> Linear 1024 -> 512 (no bias)        enc.embed.proj
 -> LayerNorm                                              enc.norm_in
 -> 31 x [x + attn(LN x)] -> [x + MLP(LN x)]               enc.blocks.{i}
      attn: 8 heads x 64, RoPE (theta 1e4, NeoX halves), q/k/v no bias, out bias
      MLP:  512 -> 2048 GELU(erf) -> 512
 -> LayerNorm                                              enc.norm_out
 -> Linear 512 -> 192                                      diar.proj
 -> Conv1d 192 -> 192*8 (k=3, pad=1), sub-pixel reshape    diar.upsample.tap{0,1,2} + bias
 -> relu -> Linear 192 -> relu -> Linear 8                 diar.head.dense / diar.head.out
 -> sigmoid, one row per 10 ms mel frame
```

Streaming: each step feeds `[speaker cache | FIFO | chunk | lookahead]`
embedder outputs through the encoder with positions restarting at 0. The
chunk's logits are emitted; its embeddings join the FIFO; FIFO overflow
moves at least `spkcache_update_period` frames into the cache, and a cache
over `spkcache_len` (264) frames is compressed to the best-scoring frames
per speaker plus one learned silence slot each (`stream.cpp`).

The processor marks `floor(n / hop)` mel frames valid and zeroes the
trailing center-padding frame; an encoder frame whose first mel frame is
padding is masked as an attention key and gets zero cache probabilities.

## Public API

`include/transcribe/sortformer.h`, shared with `sortformer`. Presets:

| Preset | Geometry (chunk/rc/fifo/update/cache, 80 ms frames) | Lookahead |
| --- | --- | --- |
| `DEFAULT` | GGUF offline config (340/40/40/300/264) | 30.4 s |
| `VERY_HIGH_LATENCY` | 340/40/40/300/264 | 30.4 s |
| `LOW_LATENCY` | 9/4/264/222/264 | 1.04 s |
| `VERY_LOW_LATENCY` | 6/2/264/222/264 | 0.64 s |
| `ULTRA_LOW_LATENCY` | 3/1/264/222/264 | 0.32 s |

`HIGH_LATENCY` has no published v3 operating point and is rejected with
`TRANSCRIBE_ERR_INVALID_ARG` before the previous result is cleared. The two
sub-second values are new enum members; `sortformer` rejects them the same
way. `TRANSCRIBE_NEMOTRON3_DIAR_PRESET` (`offline`, `low_latency`,
`very_low_latency`, `ultra_low_latency`, and the validation-only `small`)
overrides the extension for validation runs, on both the run and the
stream path.

### Push-audio streaming

`transcribe_stream_begin` / `feed` / `finalize` with
`transcribe_sortformer_live_ext` (`SFLV`, STREAM slot) on
`transcribe_stream_params::family`. Accepted presets: `LOW_LATENCY` (the
init default, and the preset of a stream begun without an extension),
`VERY_LOW_LATENCY`, `ULTRA_LOW_LATENCY`. `DEFAULT`, `VERY_HIGH_LATENCY` and
`HIGH_LATENCY` are rejected pre-clear: a 30 s lookahead is a file workload,
and `transcribe_run` already serves it.

Feeds take any piece size. Each stage runs only on input that can no longer
change (`model.cpp`, "Push-audio streaming"):

- mel: a frame is final once its centered 512-sample window has arrived.
  Each feed recomputes from two frames before the first new one, so the left
  half-window and the pre-emphasis carry come from real audio, and keeps
  only that much PCM.
- embedder: an encoder frame is final once its 8 mel frames are. The
  embedder runs in fixed 64-frame blocks on both the run and the stream path,
  because on CPU the quantized matmul result depends on the row count.
- chunks: a chunk runs once the chunk and its lookahead are embedded, with
  the same step input the batch run builds. Finalize flushes the tail with
  the batch run's short lookahead, trailing padding frame and key mask.

The stream therefore reproduces `transcribe_run` at the same preset, bit for
bit. Mid-stream rows follow the rule in `sortformer.h`: the processed
frontier is `transcribe_stream_update::audio_committed_ms`; a row with
`t1_ms < audio_committed_ms` is final, a row with `t1_ms ==
audio_committed_ms` is still open. The stream revision advances whenever the
rows change. Memory is bounded by the speaker cache plus the per-frame
output probabilities (32 bytes per 10 ms, ~11.5 MB per hour).

## Commands

Reference dumps:

```bash
BASE=build/validate/nemotron3_diar/nemotron-3-diarization/sortformer-2spk-mix
uv run --project scripts/envs/nemotron3_diar scripts/dump_reference_nemotron3_diar_transformers.py encoder \
  --model nvidia/Nemotron-3-Diarization --audio samples/sortformer-2spk-mix.wav --out $BASE/ref
uv run --project scripts/envs/nemotron3_diar scripts/dump_reference_nemotron3_diar_transformers.py diarize \
  --preset offline --model nvidia/Nemotron-3-Diarization --audio samples/sortformer-2spk-mix.wav --out $BASE/ref
```

Conversion and quantization:

```bash
uv run --project scripts/envs/nemotron3_diar scripts/convert-nemotron3_diar.py nvidia/Nemotron-3-Diarization
# -> models/nemotron-3-diarization/nemotron-3-diarization-F32.gguf (397 MB, 417 source tensors)
for q in F16 Q8_0 Q4_K_M; do
  build/bin/transcribe-quantize models/nemotron-3-diarization/nemotron-3-diarization-{F32,$q}.gguf --quant $q
done
```

Validation:

```bash
uv run scripts/validate.py all --family nemotron3_diar --backend cpu
VALIDATE_NEMOTRON3_DIAR_PRESET=small uv run scripts/validate.py all --family nemotron3_diar --backend cpu
TRANSCRIBE_NEMOTRON3_DIAR_GGUF=models/nemotron-3-diarization/nemotron-3-diarization-F32.gguf \
  build/tests/transcribe_nemotron3_diar_ext_unit   # needs -DTRANSCRIBE_BUILD_REAL_MODEL_TESTS=ON
```

Acceptance (AMI IHM test, 16 meetings, 9.06 h; forced-alignment `only_words`
RTTMs; collar 0, overlap scored; plain 0.5 threshold at 10 ms, no
post-processing, applied identically to reference and port):

```bash
uv run scripts/diar/ingest_ami.py --config ihm --split test
uv run scripts/diar/fetch_ami_forced_alignment.py --config ihm --split test
uv run --project scripts/envs/nemotron3_diar scripts/diar/run_nemotron3_diar.py ref \
  --manifest samples/diar/ami-ihm-test-fa.manifest.jsonl --preset offline \
  --probs-dir reports/diar/probs/nemotron-3-diarization-REF-offline \
  --pred-dir reports/diar/pred/nemotron-3-diarization-REF-offline
uv run --project scripts/envs/nemotron3_diar scripts/diar/run_nemotron3_diar.py cpp \
  --gguf models/nemotron-3-diarization/nemotron-3-diarization-Q8_0.gguf --backend metal --preset offline \
  --manifest samples/diar/ami-ihm-test-fa.manifest.jsonl \
  --probs-dir reports/diar/probs/nemotron-3-diarization-CPP-Q8_0-metal-offline \
  --pred-dir reports/diar/pred/nemotron-3-diarization-CPP-Q8_0-metal-offline
uv run scripts/diar/score_der.py --manifest samples/diar/ami-ihm-test-fa.manifest.jsonl \
  --pred-dir reports/diar/pred/nemotron-3-diarization-CPP-Q8_0-metal-offline \
  --out reports/diar/nemotron-3-diarization-CPP-Q8_0-metal-offline.ami-ihm-test-fa.score.json
```

## Capability Validation

Diarization substitutes for the forced transcription rows, as for
`sortformer`.

| Capability | Mode | Command / test | Expected observable | Target | Status |
|------------|------|----------------|---------------------|--------|--------|
| Offline diarization (<= 8 spk) | chunked AOSC/FIFO, offline geometry | `validate.py all` + AMI DER | prob parity; DER within 0.01 of reference | MUST PASS | PASS — `diar.probs` 4.2e-6; AMI DER 9.23% vs reference 9.22% |
| Streaming presets | 1.04 / 0.64 / 0.32 s | `VALIDATE_NEMOTRON3_DIAR_PRESET=...` + AMI DER | prob parity; DER reported per preset | MUST PASS | PASS — oracle parity <= 6.3e-6, zero flips; AMI DER below |
| Cache compression | `small` preset (cache 24) | `VALIDATE_NEMOTRON3_DIAR_PRESET=small` | prob parity through compression | MUST PASS | PASS — 3.6e-6, zero flips |
| Speaker-activity tensor | debug dump | `diar.probs` [T, 8] | parity with the reference | MUST PASS | PASS |
| Full-context forward | offline dump | `diar.preds_offline` + per-stage `enc.*` | parity with the reference | MUST PASS | PASS — 9/9 tensors |
| Push-audio live streaming | `transcribe_stream_*`, `SFLV` | `transcribe_nemotron3_diar_stream_unit` + AMI DER (`--stream-chunk-ms`) | stream rows == batch rows at the same preset; final-only mid-stream rows | MUST PASS | PASS — bit-identical to the batch run (CPU/Metal, F32/Q8_0, all stream presets + `small`, 1 sample to 1 s feeds); AMI below |
| Batch (`run_batch`) | n/a | n/a | n/a | ACCEPTED GAP — single-session runs only, as `sortformer` | N/A |
| Transcription / translation / timestamps | n/a | n/a | no text output | OUT OF SCOPE — not a transcription model | N/A |

## Results

Tensor parity (sortformer-2spk-mix, CPU, F32): mel 1.7e-4, embed 1.9e-4,
layer 0 6.5e-5, encoder 2.2e-5, proj 5.4e-5, upsample 7.6e-5, probs 4.2e-6
max_abs. Metal: 5.5e-4 (offline) / 2.3e-3 (`small`), zero flips.

Long audio: over the 16 AMI meetings the Metal port and the reference
disagree on 0.031% of 10 ms speaker decisions (CPU: 1-138 frames per
meeting). As with `sortformer`, float order can flip a near-tie in the
discrete cache compression and the two runs then drift apart locally; DER
is unaffected.

AMI IHM test (DER / JER, missed / false alarm / confusion):

| Run | DER | JER | Miss | FA | Conf |
| --- | --- | --- | --- | --- | --- |
| Reference (Transformers, fp32, offline) | 9.22% | 12.89% | 4.68% | 3.67% | 0.88% |
| C++ F32, Metal, offline | 9.23% | 12.84% | 4.68% | 3.67% | 0.88% |
| C++ F16, Metal, offline | 9.23% | 12.84% | 4.68% | 3.67% | 0.88% |
| C++ Q8_0, Metal, offline | 9.23% | 12.85% | 4.79% | 3.58% | 0.86% |
| C++ Q4_K_M, Metal, offline | 9.41% | 13.31% | 4.72% | 3.77% | 0.92% |

| C++ Q8_0, Metal, low_latency (1.04 s) | 9.48% | 13.02% | 5.02% | 3.51% | 0.95% |
| C++ Q8_0, Metal, low_latency, push-audio stream (100 ms feeds) | 9.48% | 13.02% | 5.02% | 3.51% | 0.95% |
| C++ Q8_0, Metal, very_low_latency (0.64 s) | 9.64% | 13.16% | 5.13% | 3.51% | 1.01% |
| `sortformer` v2.1 Q8_0, CPU, very_high_latency (same pipeline) | 15.99% | 21.20% | 7.42% | 4.97% | 3.60% |

`ultra_low_latency` (0.32 s) passes tensor parity on the oracle clip; its
AMI DER has not been measured yet.

Throughput (Apple M2 Pro, `scripts/bench/run.py`, offline presets; the
`sortformer` rows use `very_high_latency`, the operating point Glimpse
Speech runs):

| Model | Backend | 11 s clip | 10 min meeting |
| --- | --- | --- | --- |
| nemotron3_diar Q8_0 | Metal | 510x | 339x |
| nemotron3_diar F16 | Metal | 549x | 335x |
| sortformer v2.1 Q8_0 | Metal | 170x | 105x |
| nemotron3_diar Q8_0 | CPU | 161x | 59x |
| nemotron3_diar F16 | CPU | 136x | 48x |
| sortformer v2.1 Q8_0 | CPU | 96x | 44x |

The sub-second presets re-encode the whole speaker cache and FIFO (~540
frames) for every 0.24-0.72 s chunk: AMI ran at 16x (low) and 11x (very
low) realtime on Metal, and a 2 minute clip at ~2x / ~0.8x realtime on CPU
for low / ultra-low.

Push-audio streaming reproduces the batch run bit for bit: the 16 AMI
meetings streamed at `low_latency` in 100 ms feeds give the same `[T, 8]`
probabilities as the batch run (hence the same DER), and the
`sortformer-2spk-mix` clip matches for 1 ms to 1 s feeds on CPU and Metal,
F32 and Q8_0, at every stream preset and `small`. Streaming costs what the
batch run costs (Q8_0, M2 Pro, `transcribe-cli --stream-chunk-ms`):

| Run | Backend | Audio | Batch | Stream |
| --- | --- | --- | --- | --- |
| low_latency, 100 ms feeds | Metal | IS1009a, 14 min | 16x | 16x |
| low_latency, 10 ms / 1 s feeds | Metal | IS1009a, 14 min | 16x | 14x / 16x |
| very_low_latency, 100 ms feeds | Metal | IS1009a, 14 min | | 11x |
| ultra_low_latency, 100 ms feeds | Metal | IS1009a, 14 min | | 5x |
| low_latency, 100 ms feeds | CPU | ES2004a, first 5 min | 3x | 3x |
| low_latency, 100 ms feeds | Metal | AMI test, 9.06 h (incl. load) | | 14x |

Quant policy: F32 (reference), F16, Q8_0. Q8_0 matches F32 on AMI at a
quarter of the size (106 MB). Q4_K_M (61 MB) costs 0.2 DER points; like
`sortformer`, low-bit weights can flip discrete cache decisions, so it is
not in the shipped matrix.

## Notes

- The HF feature extractor zeroes and masks the trailing center-padding
  frame; the port reproduces this (see Architecture). Without the mask the
  last 10 ms row and, on multi-chunk runs, the cache probabilities differ.
- Tie-breaking in the cache top-k picks the lower index; torch's `topk` is
  unspecified on exact ties. Exact ties between finite scores did not occur
  on AMI.
