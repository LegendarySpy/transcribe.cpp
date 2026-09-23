// arch/nemotron3_diar/nemotron3_diar.h - Nemotron-3 Diarization (Streaming
// Sortformer v3) internal model and context types. An `encoder-diarizer`:
// 8-frame mel stacking -> 31 pre-LN RoPE Transformer layers -> 512->192
// projection -> sub-pixel upsampling back to the 10 ms mel rate -> 8-way
// speaker head. No tokenizer, no text: the product is a T x 8 per-frame
// speaker-activity probability matrix, streamed chunk by chunk through an
// Arrival-Order Speaker Cache (AOSC) + FIFO of embedder outputs.
//
// Reference: HF Transformers Nemotron3DiarizationForAudioFrameClassification.

#pragma once

#include "transcribe-backend.h"
#include "transcribe-mel.h"
#include "transcribe-model.h"
#include "transcribe-session.h"
#include "transcribe/sortformer.h"  // shared public preset enum + run ext

#include <cstdint>
#include <optional>
#include <vector>

struct gguf_context;
struct ggml_context;
struct ggml_tensor;
struct ggml_backend_buffer;
typedef struct ggml_backend_buffer * ggml_backend_buffer_t;

namespace transcribe::nemotron3_diar {

struct HParams {
    int32_t max_speakers       = 8;
    int32_t frame_hop          = 160;  // samples per output (mel) frame
    int32_t subsampling_factor = 8;    // mel frames per encoder frame

    int32_t n_layers   = 0;
    int32_t d_model    = 0;
    int32_t n_heads    = 0;
    int32_t d_ff       = 0;
    float   rope_theta = 10000.0f;
    int32_t head_d     = 0;  // classifier / upsampler width

    int32_t fe_num_mels     = 0;
    int32_t fe_sample_rate  = 0;
    int32_t fe_n_fft        = 0;
    int32_t fe_win_length   = 0;
    int32_t fe_hop_length   = 0;
    float   fe_pre_emphasis = 0.0f;

    // Offline-mode chunk geometry (checkpoint defaults).
    int32_t offline_chunk_len              = 0;
    int32_t offline_right_context          = 0;
    int32_t offline_fifo_len               = 0;
    int32_t offline_spkcache_update_period = 0;

    // Streaming-mode FIFO geometry and the speaker-cache policy.
    int32_t stream_fifo_len               = 0;
    int32_t stream_spkcache_update_period = 0;
    int32_t spkcache_len                  = 0;
    int32_t sil_frames_per_spk            = 0;
    float   pred_score_threshold          = 0.0f;
    float   latest_frames_score_boost     = 0.0f;
    float   strong_boost_rate             = 0.0f;
    float   weak_boost_rate               = 0.0f;
    float   min_pos_scores_rate           = 0.0f;

    int32_t head_dim() const { return n_heads > 0 ? d_model / n_heads : 0; }
};

struct Block {
    ggml_tensor * norm_attn_w = nullptr;
    ggml_tensor * norm_attn_b = nullptr;
    ggml_tensor * q_w         = nullptr;
    ggml_tensor * k_w         = nullptr;
    ggml_tensor * v_w         = nullptr;
    ggml_tensor * out_w       = nullptr;
    ggml_tensor * out_b       = nullptr;
    ggml_tensor * norm_ff_w   = nullptr;
    ggml_tensor * norm_ff_b   = nullptr;
    ggml_tensor * ff_in_w     = nullptr;
    ggml_tensor * ff_in_b     = nullptr;
    ggml_tensor * ff_out_w    = nullptr;
    ggml_tensor * ff_out_b    = nullptr;
};

struct Weights {
    ggml_tensor *      embed_w   = nullptr;  // [n_mels * sub, d_model], no bias
    ggml_tensor *      norm_in_w = nullptr;
    ggml_tensor *      norm_in_b = nullptr;
    std::vector<Block> blocks;
    ggml_tensor *      norm_out_w = nullptr;
    ggml_tensor *      norm_out_b = nullptr;
    ggml_tensor *      proj_w     = nullptr;                        // [d_model, head_d]
    ggml_tensor *      proj_b     = nullptr;
    ggml_tensor *      up_w[3]    = { nullptr, nullptr, nullptr };  // [head_d, head_d * sub] per conv tap
    ggml_tensor *      up_b       = nullptr;                        // [head_d * sub]
    ggml_tensor *      dense_w    = nullptr;
    ggml_tensor *      dense_b    = nullptr;
    ggml_tensor *      out_w      = nullptr;  // [head_d, max_speakers]
    ggml_tensor *      out_b      = nullptr;
    ggml_tensor *      silence    = nullptr;  // [d_model]
};

transcribe_status read_hparams(const gguf_context * gguf, HParams & hp);
transcribe_status build_weights(ggml_context * ctx_meta, const HParams & hp, Weights & w);

void apply_family_invariants(transcribe_model & model);

struct Model final : public transcribe_model {
    HParams                                hparams;
    Weights                                weights;
    std::vector<float>                     silence_host;  // [d_model], read once at load
    ggml_context *                         ctx_meta = nullptr;
    transcribe::BackendPlan                plan;
    ggml_backend_buffer_t                  backend_buffer = nullptr;
    std::optional<transcribe::MelFrontend> mel;

    Model() = default;
    ~Model() override;

    const transcribe::Tokenizer * tokenizer() const override { return nullptr; }
};

// Chunk geometry of one run, in 80 ms encoder frames.
struct StreamParams {
    int chunk_len              = 0;
    int right_context          = 0;
    int fifo_len               = 0;
    int spkcache_update_period = 0;
    int spkcache_len           = 0;
};

// Resolve the operating point: GGUF offline geometry < public preset <
// TRANSCRIBE_NEMOTRON3_DIAR_PRESET env (validation hook, adds `small`).
// Returns false for a preset this family has no validated geometry for.
bool resolve_stream_params(const HParams & hp, transcribe_sortformer_preset preset, StreamParams & out);

// Host AOSC speaker cache + FIFO over embedder outputs, a port of HF
// Nemotron3DiarizationSpeakerCache (batch 1). Embeddings are row-major
// [n_frames, d_model]; cache probs are [n_frames, n_spk] at the encoder rate.
struct SpeakerCache {
    std::vector<float> embeds;  // [cache_n * D]
    std::vector<float> probs;   // [cache_n * S]
    std::vector<float> fifo;    // [fifo_n * D]
    int                cache_n    = 0;
    int                fifo_n     = 0;
    bool               compressed = false;

    void reset() {
        embeds.clear();
        probs.clear();
        fifo.clear();
        cache_n    = 0;
        fifo_n     = 0;
        compressed = false;
    }
};

// Push one processed chunk. `input` is the step's encoder input [cache |
// fifo | chunk | lookahead] ([n_input, D]); `logits` are the step's
// classifier logits at the mel rate ([n_input * sub, S]). Input frames at or
// past `n_valid` are padding: their speaker probabilities count as zero.
void update_speaker_cache(SpeakerCache &             sc,
                          const HParams &            hp,
                          const StreamParams &       p,
                          const std::vector<float> & silence,
                          const float *              input,
                          int                        n_input,
                          int                        n_valid,
                          const float *              logits,
                          int                        n_chunk);

struct Session final : public transcribe_session {
    std::vector<float> mel_buf;
    std::vector<float> embeds_host;  // [T_enc * D] embedder output for the whole clip
    std::vector<float> input_host;   // [n_input * D] per-step encoder input
    std::vector<float> logits_host;  // [n_input * sub * S] per-step logits
    std::vector<float> probs;        // [T_mel * S] accumulated output
    SpeakerCache       cache;

    Session() = default;
    ~Session() override;
};

}  // namespace transcribe::nemotron3_diar
