# Whisper

OpenAI's [Whisper](https://github.com/openai/whisper) family ported to
transcribe.cpp. All twelve OpenAI checkpoints — `tiny`, `base`, `small`,
`medium`, `large`, `large-v2`, `large-v3`, `large-v3-turbo`, plus the four
English-only `*.en` siblings — share the same encoder-decoder transformer
architecture and 30-second windowing, so most of the porting and runtime
contract is the same across variants. Pick a size based on the
accuracy/cost tradeoff you want; pick a `.en` variant only if you know the
audio is English.

For the architecture deep-dive, validation contract, and porting notes,
see the family doc at
[`docs/porting/families/whisper.md`](../porting/families/whisper.md).

## Choosing a variant

- **English-only audio?** Prefer the `.en` checkpoint at your chosen size.
  They are typically a touch more accurate than the multilingual sibling
  and slightly cheaper to decode (no language-detection step). They cannot
  transcribe other languages and cannot translate.
- **Other languages, or auto-detect across languages?** Use the
  multilingual checkpoints (no `.en`). They cover 99 languages
  (100 in the v3 family, which adds Cantonese), do automatic language
  identification, and can produce English translations of non-English
  audio when invoked with `task=translate`.
- **Throughput vs accuracy.** WER drops as you go up the size ladder, but
  decode latency scales with parameter count. `tiny` and `base` run in
  near-realtime on CPU; `large` typically wants Metal or a recent CUDA
  GPU. `large-v3-turbo` is large-v3 quality with a much smaller decoder —
  the best accuracy/speed tradeoff for most multilingual workloads.
- **v3 family quirks.** `large-v3` and `large-v3-turbo` use a 128-bin mel
  input (the rest use 80) and add a Cantonese (`yue`) language token. The
  reference dtype shipped is F16, not F32 — they were released in F16
  upstream, so transcribe.cpp follows suit.

## All variants

WER is on LibriSpeech test-clean for the **Q8_0** preset (the default
recommended quant), measured by transcribe.cpp's WER pipeline with
timestamps off (`scripts/wer/run.py --timestamps none`, the WER harness
default). See each per-variant doc for the full quant
matrix (F32/F16/Q8_0/Q6_K/Q5_K_M/Q4_K_M) and a discussion of how our
numbers compare to OpenAI's self-reported figures. Numbers come from single Metal-backed runs; Metal's non-deterministic parallel reductions add ~0.1pp run-to-run variance on the noise floor.

<!-- catalog:family variants=breeze-asr-25,whisper-tiny,whisper-tiny.en,whisper-base,whisper-base.en,whisper-small,whisper-small.en,whisper-medium,whisper-medium.en,whisper-large,whisper-large-v2,whisper-large-v3,whisper-large-v3-turbo -->
| Variant                  | Params | Languages                   | Q8_0 size | Benchmark                    |  Q8_0 | Capabilities               | Doc |
| --- | ---: | --- | ---: | --- | ---: | --- | --- |
| `breeze-asr-25`          |   1.5B | zh, en + auto-detect        |   1.67 GB | LibriSpeech test-clean (WER) | 2.27% | translate, word timestamps | [handy-computer/Breeze-ASR-25-gguf](https://huggingface.co/handy-computer/Breeze-ASR-25-gguf) |
| `whisper-tiny`           |    38M | 99 languages + auto-detect  |     46 MB | LibriSpeech test-clean (WER) | 7.52% | translate, word timestamps | [whisper-tiny.md](whisper-tiny.md) |
| `whisper-tiny.en`        |    38M | en                          |     46 MB | LibriSpeech test-clean (WER) | 5.72% | word timestamps            | [whisper-tiny.en.md](whisper-tiny.en.md) |
| `whisper-base`           |    73M | 99 languages + auto-detect  |     85 MB | LibriSpeech test-clean (WER) | 5.12% | translate, word timestamps | [whisper-base.md](whisper-base.md) |
| `whisper-base.en`        |    73M | en                          |     85 MB | LibriSpeech test-clean (WER) | 4.16% | word timestamps            | [whisper-base.en.md](whisper-base.en.md) |
| `whisper-small`          |   242M | 99 languages + auto-detect  |    270 MB | LibriSpeech test-clean (WER) | 3.33% | translate, word timestamps | [whisper-small.md](whisper-small.md) |
| `whisper-small.en`       |   242M | en                          |    270 MB | LibriSpeech test-clean (WER) | 3.09% | word timestamps            | [whisper-small.en.md](whisper-small.en.md) |
| `whisper-medium`         |   764M | 99 languages + auto-detect  |    832 MB | LibriSpeech test-clean (WER) | 2.64% | translate, word timestamps | [whisper-medium.md](whisper-medium.md) |
| `whisper-medium.en`      |   764M | en                          |    831 MB | LibriSpeech test-clean (WER) | 2.72% | word timestamps            | [whisper-medium.en.md](whisper-medium.en.md) |
| `whisper-large`          |   1.5B | 99 languages + auto-detect  |   1.67 GB | LibriSpeech test-clean (WER) | 2.71% | translate, word timestamps | [whisper-large.md](whisper-large.md) |
| `whisper-large-v2`       |   1.5B | 99 languages + auto-detect  |   1.67 GB | LibriSpeech test-clean (WER) | 2.97% | translate, word timestamps | [whisper-large-v2.md](whisper-large-v2.md) |
| `whisper-large-v3`       |   1.5B | 100 languages + auto-detect |   1.67 GB | LibriSpeech test-clean (WER) | 1.82% | translate, word timestamps | [whisper-large-v3.md](whisper-large-v3.md) |
| `whisper-large-v3-turbo` |   809M | 100 languages + auto-detect |    886 MB | LibriSpeech test-clean (WER) | 2.01% | word timestamps            | [whisper-large-v3-turbo.md](whisper-large-v3-turbo.md) |
<!-- /catalog -->

Pre-built GGUFs for every variant and quant are hosted under
[`handy-computer` on Hugging Face](https://huggingface.co/handy-computer);
each per-variant doc has direct download links.

## Input limits

No practical per-call length limit (`transcribe_capabilities.max_audio_ms == 0`):
Whisper slices long audio into 30-second windows internally and stitches the
results, so you can pass arbitrarily long recordings. See the
[input-length contract](../input-limits.md).

## Quick start

Pick a variant and run:

```bash
cmake -B build
cmake --build build

build/bin/transcribe-cli \
  -m models/whisper-base.en/whisper-base.en-Q8_0.gguf \
  samples/jfk.wav
```

The repo doesn't ship the GGUFs — pull them from the corresponding
`handy-computer/<variant>-gguf` repo on Hugging Face, or convert from
the upstream OpenAI checkpoint via the per-variant doc's reproduction
section.

## Apple Neural Engine encoder (optional)

Whisper uses the [shared Core ML encoder runtime](../coreml.md).

On Apple Silicon with macOS 13 or later, build with Core ML support and
convert the encoder directly from the **same transcribe.cpp GGUF** used by the
runtime. Only Handy's GGUF is downloaded; the converter reads its tensors,
expands quantized weights, and emits an FP16 Core ML encoder. It records the
GGUF's SHA-256 and variant in the model metadata. The decoder and frontend
continue to use the GGUF through the existing runtime.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DTRANSCRIBE_COREML=ON
cmake --build build --target transcribe-cli -j 2

mkdir -p models
curl -fL -o models/whisper-tiny-Q8_0.gguf \
  https://huggingface.co/handy-computer/whisper-tiny-gguf/resolve/main/whisper-tiny-Q8_0.gguf

uv run --python 3.11 scripts/convert-whisper-gguf-to-coreml.py \
  models/whisper-tiny-Q8_0.gguf \
  --output models/whisper-tiny-Q8_0-encoder.mlpackage --compile

TRANSCRIBE_WHISPER_COREML_MODEL=models/whisper-tiny-Q8_0-encoder.mlmodelc \
  build/bin/transcribe-cli -m models/whisper-tiny-Q8_0.gguf --threads 2 samples/jfk.wav
```

Conversion requires Xcode's `coremlcompiler`; `uv` supplies the conversion-only
Python dependencies. No Python dependency is added to the C++ runtime. Use the
same command for other Whisper variants and quantizations. Regenerate the
encoder when changing the GGUF; using a separate unquantized encoder would test
different weights. Conversion to FP16 still introduces numerical differences.

Core ML uses `CPUAndNeuralEngine`, which excludes the GPU but lets macOS place
unsupported operations on the CPU. This setting alone is not proof that every
operation runs on ANE; inspect the Core ML compute plan or use Instruments to
check placement. Leaving `TRANSCRIBE_METAL` enabled permits GPU decoding.

Pass the compiled `.mlmodelc` in `transcribe_session_params::coreml_encoder_path`
(Rust: `SessionOptions::coreml_encoder_path`); an unset path falls back to
`TRANSCRIBE_WHISPER_COREML_MODEL`. The path is read when creating a Whisper
session. Unset or empty leaves the ggml encoder active. An explicitly requested
encoder that cannot load or predict returns an error instead of silently using
ggml. Builds without Core ML support reject a nonempty path. Encoders without
matching variant metadata are rejected. The SHA-256 is provenance metadata, not
a runtime file-hash check.
The session retains the Core ML model for reuse; first load may take longer
while macOS specializes it. GGUF encoder weights remain loaded too.

The CLI's `backend` field describes the ggml decoder backend. The
`Core ML encoder loaded` log identifies the separate encoder path. Tensor
debugging provides `enc.final`; internal Core ML layers are not exposed.
Flash-attention flags apply only to the ggml portions of the run.

Run the optional regression check after building the CLI:

```bash
TRANSCRIBE_WHISPER_COREML_GGUF="$PWD/models/whisper-tiny-Q8_0.gguf" \
TRANSCRIBE_WHISPER_COREML_MODEL="$PWD/models/whisper-tiny-Q8_0-encoder.mlmodelc" \
  ctest --test-dir build -R transcribe_whisper_coreml_smoke --output-on-failure
```

## Capabilities

All Whisper variants support:

- **Transcription** of 16 kHz mono WAV input.
- **Long-form audio** via 30-second chunked decoding with the
  prev-context window assembly described in the family doc.
- **Segment timestamps** from the decoded timestamp tokens (`AUTO` and
  `SEGMENT`).
- **Word timestamps** on an explicit `TRANSCRIBE_TIMESTAMPS_WORD` request
  (`max_timestamp_kind = word`; `AUTO` still means segments). See below.
- **Translation** (any supported language → English) on multilingual
  checkpoints — `.en` variants are transcribe-only.

### Word timestamps

`--timestamps word` (C: `TRANSCRIBE_TIMESTAMPS_WORD`, Rust:
`TimestampKind::Word`) decodes exactly like `SEGMENT`, so the transcript is
byte-identical, and adds one alignment pass per 30-second window. The pass
follows OpenAI whisper's `find_alignment`: a teacher-forced decoder pass over
`<|startoftranscript|> [<|lang|> <|task|>] <|notimestamps|> text <|endoftext|>`
that reads the cross-attention of the model's alignment heads over the real
audio frames, then softmax, per-head normalization, a width-7 median filter,
DTW, and OpenAI's word grouping, punctuation merge and duration rules. Only
the Q·K product of the alignment layers runs without flash attention;
decoding, the encoder and every other attention stay on flash. The pass stops
after the last alignment layer and costs about one decode step per window.

Words are whitespace-trimmed with punctuation attached (`"country."`), carry
absolute times in ms, and are clamped to the window's real audio, so a word
never ends after the input does. Starts are non-decreasing and consecutive
words never overlap. `zh`, `ja`, `th`, `lo`, `my` and `yue` split per
character, as OpenAI does. Segment times may be refined by the OpenAI rules
(a segment starts at its first word unless that word is implausibly long).
Token rows are not exposed (`transcribe_n_tokens` is 0 on word runs).

Alignment heads, first match wins:

1. The GGUF KV `stt.whisper.alignment_heads` (int32 array, flat
   `[layer, head, layer, head, ...]`), written by `convert-whisper.py` from
   the checkpoint's `generation_config.json` when it fits the decoder.
2. OpenAI's per-model table, matched on exact geometry (encoder layers,
   decoder layers, heads, mel bins, multilingual): tiny through large-v3,
   large-v3-turbo, and distil-large-v3 / v3.5. This covers whisper.cpp
   `.bin` files. large-v1 shares large-v2's geometry and needs the
   `whisper-large-v1` variant string or the KV.
3. Every head of the upper half of the decoder layers (OpenAI's default),
   e.g. distil-medium.en and distil-small.en.

A window whose alignment cannot run (under 20 ms of audio, a backend
failure) falls back to timing proportional to word length inside its
segment instead of failing the run; the log says how many windows did.

What's not supported (consistent across the family): real-time
streaming (whisper is not streaming-first; chunked 30-second windows
only), VAD, speaker diarization. See the family doc for the full
runtime contract.
