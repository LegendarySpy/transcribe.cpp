# Core ML encoder companion

This branch supports optional Qwen3-ASR, offline Parakeet, and Whisper Core ML encoders. The
audio frontend and decoder remain in the existing transcribe.cpp engine.

## Build and load

Build on Apple Silicon with `TRANSCRIBE_COREML=ON`, or enable the Rust `coreml`
feature. The shared implementation lives in `src/transcribe-coreml.h` and
`src/transcribe-coreml.mm`; Qwen owns its adapter in `src/arch/qwen3_asr/`.

Pass a compiled `.mlmodelc` directory in
`transcribe_session_params::coreml_encoder_path` (Rust:
`SessionOptions::coreml_encoder_path`). An unset path falls back to
`TRANSCRIBE_QWEN3_ASR_COREML_MODEL`. With neither set, inference uses ggml.
The session loads and owns one encoder and reuses it across calls.

See [Qwen3-ASR](models/qwen3-asr.md) for the converter command and model details.
Derive the companion from the exact GGUF used for inference. Runtime validation
checks the variant and tensor shapes, but does not recompute the GGUF checksum.
Invalid paths, mismatched variants, and prediction failures return errors.

## Tensor contract

- Input `logmel_data`: F32 `[1, n_mels, capacity]`.
- Input `mel_length`: I32 `[1]`, the valid prefix length.
- Output `output`: F32 `[1, capacity_out, d_model]`.
- Creator metadata `transcribe.variant`: the loaded model's variant, or a
  comma-separated list when one encoder serves several checkpoints.
- Creator metadata `transcribe.gguf.sha256` and `.filename`: export provenance
  (the source model file, GGUF or whisper.cpp `.bin`).

Qwen's chunked convolution emits 13 rows per 100 mel frames. Its adapter checks
that row count and passes the valid output length to `coreml_encoder_run`.
The shared runtime honors output strides and copies only the valid rows.
The exported graph must mask padding wherever it can affect valid frames.

The default companion capacity is 1,500 mel frames (15 seconds). Longer inputs
fall back to the ggml encoder without truncation. Batch requests use the serial
per-utterance path when the companion is enabled.

## Execution and verification

The compute policy is `CPUAndNeuralEngine`: the GPU is excluded from the encoder,
but Core ML may use CPU operations. The decoder continues on the selected ggml
backend. This is not a guarantee of ANE-only execution or energy savings.

The companion uses FP16 computation, so exact transcript equality with ggml is
not guaranteed. Validate encoder tensors and transcription with the same GGUF,
including short and capacity-boundary inputs, before claiming support for a
new companion. The Qwen model note records its numerical check and limitations.

The shared session struct layout is independent of the private Core ML build
flag. Builds with Core ML disabled retain the existing ggml path.

## Whisper

The Whisper adapter loads the session companion through `coreml_encoder_path`
or `TRANSCRIBE_WHISPER_COREML_MODEL`. Export it with
`scripts/convert-whisper-gguf-to-coreml.py` from a transcribe.cpp GGUF or a
whisper.cpp `.bin`; one encoder serves every quantization of a variant. The
encoder has a fixed 3000-frame (30 second) input, matching the padded window
Whisper always encodes, so there is no capacity fallback.

The exported graph targets macOS 14 and runs entirely on the Neural Engine for
every size (channels-first layout, per-head attention in query blocks,
per-channel residual scaling against FP16 flush-to-zero, two chained programs
for large-v3 and turbo). See `docs/models/whisper.md` for the design and the
numbers behind it. Apple M2 Pro, encoder time per 30 second window inside each
engine (median, shared machine):

| Encoder | ANE ms | whisper.cpp Core ML ms | First load | Cached load |
| --- | ---: | ---: | ---: | ---: |
| tiny | 10 | 20 | 3 s | 0.05 s |
| base | 20 | 39 | 6 s | 0.07 s |
| small, small.en | 66 | 124 | 18 s | 0.1 s |
| medium, medium.en | 206 | 530 | 52 s | 0.3 s |
| large-v3 | 397 | 534 (GPU) | 82 s | 0.4 s |
| large-v3-turbo | 396 | 702 (GPU) | 82 s | 0.4 s |

"whisper.cpp Core ML" is `ggml-<size>-encoder.mlmodelc` loaded the way
whisper.cpp loads it (`MLComputeUnitsAll`); it runs large-v3 and turbo on the
GPU because the ANE compiler rejects them. First load is the one-time ANE
compile; macOS caches it per app across launches.

`transcribe_whisper_coreml_smoke` exercises the supplied real model when
`TRANSCRIBE_WHISPER_COREML_GGUF` and `TRANSCRIBE_WHISPER_COREML_MODEL` are set.

## Parakeet TDT

The offline Parakeet adapter loads the same session companion through
`coreml_encoder_path` or `TRANSCRIBE_PARAKEET_COREML_MODEL`. Export from the exact
GGUF using `scripts/convert-parakeet-gguf-to-coreml.py`. The default capacity is
1501 mel frames (about 15 seconds); shorter inputs are length-masked and longer
inputs use the existing ggml encoder. Streaming variants are rejected.

The encoder uses CPU + Neural Engine, with GPU excluded. Its output feeds the
existing host decoder. A missing or mismatched companion returns an error.
`transcribe_parakeet_coreml_smoke` exercises the supplied real model when
`TRANSCRIBE_PARAKEET_COREML_GGUF` and `TRANSCRIBE_PARAKEET_COREML_MODEL` are set.
See `docs/models/parakeet.md` for the adapter contract and supported variants.


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
