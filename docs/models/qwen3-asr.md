# Qwen3-ASR

Alibaba's [Qwen3-ASR](https://huggingface.co/collections/Qwen/qwen3-asr)
family ported to transcribe.cpp. An audio-LLM design: a bidirectional
audio encoder feeds audio tokens into a Qwen3 causal LM that emits the
transcript autoregressively. Both shipped variants share the contract
— 16 kHz mono PCM in, transcript text out — and auto-detect across
30 languages.

For the architecture deep-dive, validation contract, and porting notes,
see the family doc at
[`docs/porting/families/qwen3_asr.md`](../porting/families/qwen3_asr.md).

## Choosing a variant

- **Smaller, faster, near-realtime CPU.** `qwen3-asr-0.6b` — 600M
  parameters, 811 MB at Q8_0. 18-layer encoder + Qwen3 LM with
  `hidden_size=1024`. ~2.1% WER on LibriSpeech test-clean.
- **Accuracy headroom.** `qwen3-asr-1.7b` widens both halves of the
  model (24-layer encoder, LM `hidden_size=2048`,
  `intermediate_size=6144`) for ~0.5pp WER improvement at ~2.5× the
  storage and decode cost.
- Language coverage is identical across the two variants — pick on
  the accuracy/cost axis, not on languages.

## All variants

WER is on LibriSpeech test-clean for the **Q8_0** preset, measured by
transcribe.cpp's WER pipeline. See each per-variant doc for the full
quant matrix.

| Variant | Params | Q8_0 size | WER (Q8_0) | Languages | Doc |
| --- | ---: | ---: | ---: | --- | --- |
| `qwen3-asr-0.6b` | ~600M | 811 MB  | 2.11% | 30 (auto-detect) | [qwen3-asr-0.6b.md](qwen3-asr-0.6b.md) |
| `qwen3-asr-1.7b` | ~1.7B | 2.08 GB | 1.61% | 30 (auto-detect) | [qwen3-asr-1.7b.md](qwen3-asr-1.7b.md) |

Pre-built GGUFs for every variant and quant are hosted under
[`handy-computer` on Hugging Face](https://huggingface.co/handy-computer);
each per-variant doc has direct download links.

## Input limits

Both variants accept up to about **87 minutes** of 16 kHz mono audio in a single
call — the binding limit is the 65,536-token decoder context, shared across the
family. That ceiling is there to bound memory and sits far beyond any normal
clip; audio past it is rejected up front with `TRANSCRIBE_ERR_INPUT_TOO_LONG`
rather than silently truncated. Lowering `--n-ctx` lowers the limit (and the
KV-cache footprint), and `transcribe_session_get_limits()` reports the exact
per-session value. See the [input-length contract](../input-limits.md).

## Quick start

Pick a variant and run:

```bash
cmake -B build
cmake --build build

build/bin/transcribe-cli \
  -m models/qwen3-asr-0.6b/qwen3-asr-0.6b-Q8_0.gguf \
  samples/jfk.wav
```

The repo doesn't ship the GGUFs — pull them from the corresponding
`handy-computer/<variant>-gguf` repo on Hugging Face, or convert from
the upstream Qwen checkpoint via the per-variant doc's reproduction
section.

## Capabilities

All Qwen3-ASR variants support:

- **Transcription** of 16 kHz mono WAV input.
- **Auto language detection** across 30 languages.

What's not supported (consistent across the family): translation,
real-time streaming, VAD, speaker diarization, timestamps. See the
family doc for the full runtime contract.

## Apple Neural Engine encoder (optional)

Qwen3-ASR uses the [shared Core ML encoder runtime](../coreml.md) for its
audio encoder; the Qwen3 LM decoder stays on ggml. Build with
`-DTRANSCRIBE_COREML=ON`. Core ML uses `CPUAndNeuralEngine`, which excludes
the GPU but permits CPU operations where needed.

Create the companion from the same Handy GGUF used for inference. The
converter dequantizes the encoder tensors and exports the chunked conv
subsampler, the sinusoidal positions, the bidirectional blocks, and the
LN/proj head, so the output is already in the LM width:

```bash
uv run --python 3.11 scripts/convert-qwen3-asr-gguf-to-coreml.py \
  models/qwen3-asr-0.6b/Qwen3-ASR-0.6B-Q8_0.gguf \
  --output models/qwen3-asr-0.6b/Qwen3-ASR-0.6B-Q8_0-encoder.mlpackage --compile

TRANSCRIBE_QWEN3_ASR_COREML_MODEL=models/qwen3-asr-0.6b/Qwen3-ASR-0.6B-Q8_0-encoder.mlmodelc \
  build/bin/transcribe-cli -m models/qwen3-asr-0.6b/Qwen3-ASR-0.6B-Q8_0.gguf \
  --backend cpu --threads 2 --language en samples/jfk.wav
```

The default capacity is 1500 mel frames (15 seconds, a multiple of the
100-frame conv chunk); shorter inputs mask the padded rows out of the
attention keys and longer inputs explicitly fall back to the ggml encoder
without truncation. `--max-frames` exports a different capacity. Batch
requests use the serial per-utterance path when this encoder is enabled. The
companion uses FP16 computation, so transcripts can differ from ggml. The
runtime checks variant and tensor shapes but does not recompute the GGUF
checksum; keep each companion paired with the exact GGUF used to create it.
Invalid paths, mismatched variants, and prediction failures return errors.
Unset the environment variable to use the ggml encoder.

Verified with `qwen3-asr-0.6b` Q8_0 on `samples/jfk.wav`: identical
transcript, `enc.proj.out` within `tests/tolerances/qwen3_asr.json`
(max 2.5e-2, mean 1.1e-3), finite output on the Core ML CPU path, and
matching output on a 3 s clip, a 1500-frame clip, the 1502-frame fallback,
a 22 s fallback, session reuse, and a two-item batch. The compute plan places
454 of 467 assigned operations on the Neural Engine. Encoder time on jfk
(Apple M2 Pro, 4 threads, Q8_0) went from 518 ms on ggml CPU to 34 ms; the LM
decode is unchanged. A 64-clip LibriSpeech screen (32 test-clean, 32
test-other speakers, English hint) measured 2.16% WER on ggml and 2.37% on
Core ML, +0.20 pp with a paired 95% bootstrap interval of [0.0, 0.56] pp;
five hypotheses changed on rare proper nouns and invented words, and seven
more differed only in punctuation. `qwen3-asr-1.7b` converts through the same path but has
not been exercised here.

## Vocabulary context

The RUN-slot `transcribe_qwen3_asr_run_ext` accepts UTF-8 vocabulary/background
context in its `context` field. Initialize it with
`transcribe_qwen3_asr_run_ext_init` and attach `&ext.ext` to `run_params.family`.
The Rust equivalent is `RunExtension::Qwen3Asr(Qwen3AsrRunOptions { context })`.
Context is inserted into the system turn, matching the [upstream prompt builder](https://github.com/QwenLM/Qwen3-ASR/blob/main/qwen_asr/inference/qwen3_asr.py).
It guides recognition; it is not an instruction-following cleanup prompt.

Null or empty context preserves the original prompt exactly. Context applies
only to that request, including every utterance in a native batch. The caller
must reapply it when splitting audio into separate requests. CPU, GPU, and
Core ML encoder paths share the same prompt builder. Context is limited to
4096 UTF-8 bytes and 1024 tokens; larger inputs return `INVALID_ARG`. Ordinary
tokenization treats special-token spellings as text rather than chat delimiters.
