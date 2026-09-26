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

On Apple Silicon with macOS 14 or later, build with Core ML support and
convert the encoder from the model file the runtime loads: a transcribe.cpp
GGUF or a whisper.cpp `.bin` (F16, F32 or quantized, including the
distil-whisper `.bin` files). The converter reads the encoder tensors, expands
quantized weights and emits an FP16 Core ML encoder with the source's SHA-256
and variant in its metadata. The decoder and frontend continue to use the
model file through the existing runtime.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DTRANSCRIBE_COREML=ON
cmake --build build --target transcribe-cli -j 2

mkdir -p models
curl -fL -o models/ggml-tiny.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin

uv run --python 3.11 scripts/convert-whisper-gguf-to-coreml.py \
  models/ggml-tiny.bin \
  --output models/whisper-tiny-encoder.mlpackage --compile

TRANSCRIBE_WHISPER_COREML_MODEL=models/whisper-tiny-encoder.mlmodelc \
  build/bin/transcribe-cli -m models/ggml-tiny.bin --threads 2 samples/jfk.wav
```

Conversion requires Xcode's `coremlcompiler`; `uv` supplies the conversion-only
Python dependencies. No Python dependency is added to the C++ runtime. One
encoder serves every quantization of a variant, so export it once from the F16
file. The distil-whisper encoders are bit-identical to their teachers
(distil-large-v3.5 to large-v3, distil-medium.en to medium.en, distil-small.en
to small.en; `large-v3-turbo` has its own). Their `.bin` files report the
teacher's variant, their GGUFs their own, so export the teacher's encoder with
`--also-variant distil-large-v3.5` (and so on) to serve both.

The graph is laid out for the Neural Engine rather than translated from the
ggml graph:

- Activations stay channels-first `(1, C, 1, S)`; linears are 1x1 convs and
  LayerNorm normalizes the channel axis.
- Attention is split per head and into four query blocks. Each score tensor
  is `(1, 1500, 1, 375)`, which the ANE runs 1.3 to 1.6x faster than a full
  `1500 x 1500` block, at about 4x the first-load compile time.
  `--query-blocks 1` restores the faster compile.
- The ANE flushes FP16 products below 2^-14 to zero. Most of fc2's terms are
  that small, which cost about 0.2% cosine similarity per layer. The residual
  stream is therefore carried scaled per channel by a power of two, chosen
  from an FP32 NumPy pass over synthetic input so the few outlier channels (up
  to about 2700 in medium) keep a wide margin below the FP16 limit. LayerNorm
  unscales on read.
- Large-v3 and large-v3-turbo are split into two chained programs (a Core ML
  pipeline in one `.mlmodelc`). The ANE compiler rejects the 1.2 GB
  single-program encoder, and Core ML then runs it on the CPU without an error.
  `--int8` stores per-channel int8 weights instead: half the size, one program,
  slightly faster, but it changed punctuation on one test clip.

FP16 still differs from ggml. Against the ggml encoder on the same model
file, transcripts of `jfk`, a 12 s and a 60 s clip and a 298 s file matched
word for word for most sizes; the rest differ in punctuation or a few words,
except distil-medium.en, which fell into a repetition loop in one 30 s window
of the long file.

Check placement with the compute plan (`coremltools.models.compute_plan`); all
operations of every exported size are planned on the Neural Engine. The runtime
requests `CPUAndNeuralEngine`; `ALL` picks the same placement for these graphs.
Leaving `TRANSCRIBE_METAL` enabled permits GPU decoding.

The first load of each encoder on a machine compiles it for the ANE, which
takes seconds for small models and over a minute for large ones (see
[Core ML](../coreml.md)); macOS caches the result per app, and later loads
take under half a second.

Pass the compiled `.mlmodelc` in `transcribe_session_params::coreml_encoder_path`
(Rust: `SessionOptions::coreml_encoder_path`); an unset path falls back to
`TRANSCRIBE_WHISPER_COREML_MODEL`. The path is read when creating a Whisper
session. Unset or empty leaves the ggml encoder active. An explicitly requested
encoder that cannot load or predict returns an error instead of silently using
ggml. Builds without Core ML support reject a nonempty path. Encoders without
matching variant metadata are rejected. The SHA-256 is provenance metadata, not
a runtime file-hash check. The session retains the Core ML model for reuse.
The model file's encoder weights remain loaded too.

The CLI's `backend` field describes the ggml decoder backend. The
`Core ML encoder loaded` log identifies the separate encoder path. Tensor
debugging provides `enc.final`; internal Core ML layers are not exposed.
Flash-attention flags apply only to the ggml portions of the run.

Run the optional regression check after building the CLI:

```bash
TRANSCRIBE_WHISPER_COREML_GGUF="$PWD/models/ggml-tiny.bin" \
TRANSCRIBE_WHISPER_COREML_MODEL="$PWD/models/whisper-tiny-encoder.mlmodelc" \
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

Three refinements go beyond OpenAI: a word that starts without whitespace
joins the previous one (`555-0142`, `2.5` stay whole); word edges drop the
silence (and a neighbor's sliver before a pause) that DTW hands to a word,
using 20 ms frame energy of the input; and speech that DTW gives to trailing
punctuation (`Yes` ending where `.` starts) goes back to the word.

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

### Decoding

The decode follows the HF / OpenAI recipe (greedy, temperature fallback on
compression ratio and average log-probability, no-speech gate) with these
rules on every run:

- Audio past the input is never decoded. A window starts only if more than
  100 ms of real audio remains after the seek point (input of 100 ms or
  less gives an empty transcript), and a window stops generating once a
  timestamp reaches the last 100 ms of the audio. Both are whisper.cpp's;
  without them text is decoded from the zero padding of the last window.
- Long-form input (over 30 s) always decodes timestamp tokens; `NONE`
  only drops the segments. Blind 30-second advances under
  `<|notimestamps|>` cut words at window edges and made windows stop early.
- A fallback tier at temperature T divides the logits by T before the
  suppression and timestamp rules, and its average log-probability comes
  from those scaled logits (whisper.cpp's order).
- The transcript is built from the kept segments: a window's unfinished
  tail after its last closed timestamp pair is decoded again by the next
  window and appears once.
- `initial_prompt` text that looks like a special token (`<|en|>`) is
  encoded as plain text.
- With `condition_on_prev_tokens`, the carried tokens keep each window's
  closing timestamp pair, and no prior context is used once less than 5 s
  of audio remains (both as in whisper.cpp; the short final window tended
  to repeat or continue the carried text).

whisper.cpp's defaults, as used by whisper-rs, map to these run-ext fields
(Rust `WhisperRunOptions`): `suppress_non_speech = false` (silence decodes
as a tag such as `[BLANK_AUDIO]` instead of an invented "Thank you."),
`best_of = 5`, `entropy_thold = 2.4`, `condition_on_prev_tokens = true`,
and `no_speech_thold = TRANSCRIBE_WHISPER_THOLD_DISABLED`. whisper.cpp
reads its no-speech probability from a stale logits row, so its gate
effectively never fires; disabling ours matches that. whisper.cpp has no
compression-ratio check (the entropy check replaces it); keeping ours
changed no transcript in our comparison.

What's not supported (consistent across the family): real-time
streaming (whisper is not streaming-first; chunked 30-second windows
only), VAD, speaker diarization. See the family doc for the full
runtime contract.
