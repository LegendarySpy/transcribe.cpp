# Core ML encoder companion

This branch supports the optional Qwen3-ASR Core ML encoder. The audio frontend
and autoregressive decoder remain in the existing transcribe.cpp engine.

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
- Creator metadata `transcribe.variant`: the loaded GGUF variant.
- Creator metadata `transcribe.gguf.sha256` and `.filename`: export provenance.

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
