# Parakeet

NVIDIA's [Parakeet](https://huggingface.co/collections/nvidia/parakeet)
family ported to transcribe.cpp. A FastConformer encoder paired with one
of three decoder heads — TDT (transducer with a duration prediction
head), classical RNN-T, or CTC — and a TDT+CTC hybrid that ships both
heads in one checkpoint. All variants take 16 kHz mono PCM through an
80-bin mel frontend; English-only across the family, except
`parakeet-tdt-0.6b-v3` and its German fine-tune `parakeet-primeline`,
which cover 25 European languages.

For the architecture deep-dive, validation contract, and porting notes,
see the family doc at
[`docs/porting/families/parakeet.md`](../porting/families/parakeet.md).

## Choosing a variant

Most users want one of three:

- **English transcription → `parakeet-tdt-0.6b-v2`.** The default pick:
  small, fast, near top-of-family accuracy on English.
- **Multilingual (25 European languages) → `parakeet-tdt-0.6b-v3`.** The
  only multilingual variant; same size as v2, broader coverage at a
  small English-WER cost.
- **German → `parakeet-primeline`.** primeLine's German fine-tune of
  v3. Same size and speed; tuned for German while keeping the other 24
  v3 languages usable.
- **Streaming / real-time → Nemotron streaming.** Most Parakeet variants here
  are offline-only. For low-latency streaming use the FastConformer-lineage
  [`nemotron-3.5-asr-streaming-0.6b`](nemotron-3.5-asr-streaming-0.6b.md)
  (multilingual) or
  [`nemotron-speech-streaming-en-0.6b`](nemotron-speech-streaming-en-0.6b.md)
  (English). Within Parakeet itself, `parakeet-unified-en-0.6b` is the only
  streaming-capable variant.

If you specifically need the lowest WER or a different decoder:

- **Lowest WER, English.** `parakeet-tdt-1.1b` (1.38% Q8_0) is the most
  accurate in the family, narrowly ahead of `parakeet-rnnt-1.1b` (1.46%).
  TDT also decodes faster — the duration head lets the decoder skip frames.
- **Fastest decode at any size.** Use a CTC variant
  (`parakeet-ctc-0.6b` / `parakeet-ctc-1.1b`). Single-pass greedy
  alignment, no transducer loop — at a ~0.2pp WER cost vs the
  same-size RNN-T.
- **Tiny footprint.** `parakeet-tdt_ctc-110m` is the smallest Parakeet. The 1.1B `tdt_ctc` ships both heads but is
  primarily useful when you want TDT speed with CTC as a fallback at
  runtime.

## All variants

WER is on LibriSpeech test-clean for the **Q8_0** preset, measured by
transcribe.cpp's WER pipeline. See each per-variant doc for the full
quant matrix and the comparison to NVIDIA's self-reported numbers.

<!-- catalog:family variants=parakeet-tdt-0.6b-v2,parakeet-tdt-0.6b-v3,parakeet-primeline,parakeet-tdt-1.1b,parakeet-tdt_ctc-110m,parakeet-tdt_ctc-1.1b,parakeet-rnnt-0.6b,parakeet-rnnt-1.1b,parakeet-ctc-0.6b,parakeet-ctc-1.1b,parakeet-unified-en-0.6b -->
| Variant                    | Params | Languages                  | Q8_0 size | Benchmark                    |  Q8_0 | Capabilities                | Doc |
| --- | ---: | --- | ---: | --- | ---: | --- | --- |
| `parakeet-tdt-0.6b-v2`     |   618M | en                         |    730 MB | LibriSpeech test-clean (WER) | 1.69% | token timestamps            | [parakeet-tdt-0.6b-v2.md](parakeet-tdt-0.6b-v2.md) |
| `parakeet-tdt-0.6b-v3`     |   627M | 25 languages + auto-detect |    740 MB | LibriSpeech test-clean (WER) | 1.94% | token timestamps            | [parakeet-tdt-0.6b-v3.md](parakeet-tdt-0.6b-v3.md) |
| `parakeet-primeline`       |   627M | 25 languages + auto-detect |    740 MB | FLEURS de (WER)              | 5.98% | token timestamps            | [parakeet-primeline.md](parakeet-primeline.md) |
| `parakeet-tdt-1.1b`        |   1.1B | en                         |   1.27 GB | LibriSpeech test-clean (WER) | 1.38% | token timestamps            | [parakeet-tdt-1.1b.md](parakeet-tdt-1.1b.md) |
| `parakeet-tdt_ctc-110m`    |   114M | en                         |    135 MB | LibriSpeech test-clean (WER) | 2.43% | token timestamps            | [parakeet-tdt_ctc-110m.md](parakeet-tdt_ctc-110m.md) |
| `parakeet-tdt_ctc-1.1b`    |   1.1B | en                         |   1.27 GB | LibriSpeech test-clean (WER) | 1.87% | token timestamps            | [parakeet-tdt_ctc-1.1b.md](parakeet-tdt_ctc-1.1b.md) |
| `parakeet-rnnt-0.6b`       |   617M | en                         |    730 MB | LibriSpeech test-clean (WER) | 1.62% | token timestamps            | [parakeet-rnnt-0.6b.md](parakeet-rnnt-0.6b.md) |
| `parakeet-rnnt-1.1b`       |   1.1B | en                         |   1.27 GB | LibriSpeech test-clean (WER) | 1.46% | token timestamps            | [parakeet-rnnt-1.1b.md](parakeet-rnnt-1.1b.md) |
| `parakeet-ctc-0.6b`        |   609M | en                         |    722 MB | LibriSpeech test-clean (WER) | 1.87% | token timestamps            | [parakeet-ctc-0.6b.md](parakeet-ctc-0.6b.md) |
| `parakeet-ctc-1.1b`        |   1.1B | en                         |   1.26 GB | LibriSpeech test-clean (WER) | 1.85% | token timestamps            | [parakeet-ctc-1.1b.md](parakeet-ctc-1.1b.md) |
| `parakeet-unified-en-0.6b` |   618M | en                         |    731 MB | LibriSpeech test-clean (WER) | 1.60% | streaming, token timestamps | [parakeet-unified-en-0.6b.md](parakeet-unified-en-0.6b.md) |
<!-- /catalog -->

\* `parakeet-primeline` is scored on FLEURS German (862 utterances),
not LibriSpeech test-clean, so its number is not comparable to the rest
of the column. Its NeMo reference on the same manifest is 5.98%.

Pre-built GGUFs for every variant and quant are hosted under
[`handy-computer` on Hugging Face](https://huggingface.co/handy-computer);
each per-variant doc has direct download links.

## Input limits

No practical per-call length limit (`transcribe_capabilities.max_audio_ms == 0`):
the Conformer encoder's positional encoding is recomputed per call, so audio of
any length is processed in a single pass — pass arbitrarily long recordings. See
the [input-length contract](../input-limits.md).

## Quick start

Pick a variant and run:

```bash
cmake -B build
cmake --build build

build/bin/transcribe-cli \
  -m models/parakeet-tdt-0.6b-v2/parakeet-tdt-0.6b-v2-Q8_0.gguf \
  samples/jfk.wav
```

The repo doesn't ship the GGUFs — pull them from the corresponding
`handy-computer/<variant>-gguf` repo on Hugging Face, or convert from
the upstream NVIDIA `.nemo` checkpoint via the per-variant doc's
reproduction section.

## Capabilities

All Parakeet variants support:

- **Transcription** of 16 kHz mono WAV input.
- **Token-level timestamps** at the encoder frame rate (TDT and RNN-T;
  CTC also exposes frame-level alignment).

**Buffered streaming** is supported on `parakeet-unified-en-0.6b`
across all six published `(L, C, R)` configurations from the model's
training menu (lookahead latency from 160ms at `(70, 1, 1)` through
2.08s at the default `(70, 13, 13)`). See
[parakeet-unified-en-0.6b.md](parakeet-unified-en-0.6b.md#streaming)
for the per-config WER and the `--stream-buf-{left,chunk,right}-ms`
CLI surface. Other Parakeet variants run offline only.

What's not supported (consistent across the family): translation,
VAD, speaker diarization. Language coverage is English-only except
`parakeet-tdt-0.6b-v3` and `parakeet-primeline` (25 European languages,
no auto-detect — language hint required). Note that the v3 lineage,
including `parakeet-primeline`, writes German `ss` where standard
orthography uses `ß`; see
[parakeet-primeline.md](parakeet-primeline.md#orthography-ß-vs-ss) for
why and what to do about it. See the family doc for the full runtime
contract.

## Apple Neural Engine encoder (optional)

Both families use the [shared Core ML encoder runtime](../coreml.md), which
other compatible model families can reuse.

On Apple Silicon with macOS 13 or later, the optional Core ML backend covers
the offline full-context FastConformer encoders: every Parakeet variant except
the streaming and multitalker checkpoints, which the converter and the runtime
reject. Build with `-DTRANSCRIBE_COREML=ON`; use `-DTRANSCRIBE_METAL=OFF` and
`--backend cpu` to keep the decoder on CPU. Core ML uses `CPUAndNeuralEngine`,
which excludes the GPU but permits CPU operations where needed.

Create the companion encoder directly from the same Handy GGUF used for
inference. The converter reads and dequantizes its encoder tensors; it does not
download a separate NVIDIA checkpoint:

```bash
uv run --python 3.11 scripts/convert-parakeet-gguf-to-coreml.py \
  models/parakeet-tdt-0.6b-v3-Q8_0.gguf \
  --output models/parakeet-tdt-0.6b-v3-Q8_0-encoder.mlpackage --compile

TRANSCRIBE_PARAKEET_COREML_MODEL=models/parakeet-tdt-0.6b-v3-Q8_0-encoder.mlmodelc \
  build/bin/transcribe-cli -m models/parakeet-tdt-0.6b-v3-Q8_0.gguf \
  --backend cpu --threads 2 samples/jfk.wav
```

The default encoder capacity is 1501 mel frames (about 15 seconds). Shorter
inputs use a length mask. Longer inputs explicitly fall back to the existing
ggml encoder, without truncation. `--max-frames` can export a different fixed
capacity; attention memory grows quadratically with capacity. Batch requests
use the existing serial per-utterance path when this encoder is enabled.

The companion uses FP16 computation, so transcripts can differ from ggml.
Checkpoints with `xscaling` (the 80-mel `ctc`, `rnnt`, and `tdt_ctc` families)
have block-0 inputs beyond FP16 range; the converter applies that scale to the
block-0 residual contributions instead, which is equivalent up to LayerNorm
epsilon and keeps the graph finite on every compute unit. Its metadata records
the source GGUF SHA256, filename, and variant. The runtime checks variant and
tensor shapes, but does not recompute the GGUF checksum; keep each companion
paired with the exact GGUF used to create it. Invalid paths or prediction
failures return errors. Unset the environment variable to use the normal ggml
encoder. This path does not cover the Nemotron streaming models,
`parakeet-unified-en-0.6b`, or `multitalker-parakeet-streaming-0.6b-v1`.

Exercised with real Q8_0 weights on `samples/jfk.wav`, a 3 s clip, the
1501-frame boundary, the 1502-frame and 22 s ggml fallbacks, session reuse, and
a two-item batch, with `enc.final` inside `tests/tolerances/parakeet.json`:
`tdt-0.6b-v2`, `tdt-0.6b-v3`, `tdt-1.1b`, `tdt_ctc-110m`, `ctc-0.6b`,
`rnnt-0.6b`, and `primeline` (also on `samples/german.wav`). Transcripts
matched ggml on every case except the 1501-frame boundary clip, where
`tdt_ctc-110m` drops a comma and `tdt-1.1b` drops the cut-off final word.
`ctc-1.1b` and `tdt_ctc-1.1b` convert through the same path but are
unverified.


### Decoder-only Parakeet TDT V3 packages

`scripts/extract-parakeet-decoder.py SOURCE.gguf OUTPUT-decoder.gguf` copies the
original GGUF metadata and predictor/joint tensors, omitting encoder tensors.
It sets `stt.parakeet.decoder_only=true` and records
`stt.parakeet.source_sha256`. Use the Core ML encoder exported from SOURCE;
the extraction step does not change the encoder or decoder weights.

`ParakeetModel::decoder_only` skips encoder tensor validation and preparation.
Only offline TDT V3 supports this package. Session creation requires the matching
Core ML companion; builds without Core ML cannot create such a session.
Inputs exceeding the companion capacity return an error requesting chunking.
Full GGUF models retain their existing CPU/GPU path and over-capacity fallback.
