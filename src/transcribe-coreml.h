#pragma once

#include <vector>

namespace transcribe {

struct CoreMLEncoder;

// Fixed [1, n_mels, capacity] F32 input and [1, ceil(capacity/subsampling), d_model]
// F32 output. Variable-length encoders also accept mel_length: int32[1].
// capacity_out > 0 replaces ceil(capacity/subsampling) for graphs whose row
// count is not an integer division of the input (Qwen3-ASR's chunked conv);
// < 0 accepts the companion's row count, which the caller then checks.
CoreMLEncoder * coreml_encoder_load(const char * path,
                                    const char * variant,
                                    int          n_mels,
                                    int          capacity,
                                    int          d_model,
                                    int          subsampling,
                                    bool         variable_length,
                                    int          capacity_out = 0);
void            coreml_encoder_free(CoreMLEncoder * encoder) noexcept;
int             coreml_encoder_capacity(const CoreMLEncoder * encoder) noexcept;
int             coreml_encoder_capacity_out(const CoreMLEncoder * encoder) noexcept;
// time_major selects [T, n_mels] vs [n_mels, T] input. Output is [T_enc, d_model]
// with T_enc = ceil(n_frames/subsampling), or frames_out when > 0.
bool            coreml_encoder_run(CoreMLEncoder *      encoder,
                                   const float *        mel,
                                   int                  n_frames,
                                   bool                 time_major,
                                   std::vector<float> & output,
                                   int                  frames_out = 0);

}  // namespace transcribe
