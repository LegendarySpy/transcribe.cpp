// arch/nemotron3_diar/model.cpp - Nemotron-3 Diarization load / run / Arch.
//
// Per run: log-mel over the whole clip, one embedder graph (8-frame stacking
// + projection) over every encoder frame, then one step graph per chunk over
// [speaker cache | FIFO | chunk | lookahead] embeddings: 31 pre-LN RoPE
// layers -> projection -> sub-pixel upsampling -> speaker head. The host
// speaker cache (stream.cpp) carries identity across chunks. Mirrors HF
// Nemotron3DiarizationForAudioFrameClassification.forward. Push-audio streams
// run the same chunks as their input arrives ("Push-audio streaming" below).

#include "ggml.h"
#include "gguf.h"
#include "nemotron3_diar.h"
#include "transcribe-arch.h"
#include "transcribe-backend.h"
#include "transcribe-batch-util.h"
#include "transcribe-debug.h"
#include "transcribe-load-common.h"
#include "transcribe-loader.h"
#include "transcribe-log.h"
#include "transcribe-meta.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

namespace transcribe::nemotron3_diar {

extern const Arch arch;

static constexpr char  k_default_variant[] = "nemotron-3-diarization";
static constexpr float k_ln_eps            = 1e-5f;

Model::~Model() {
    if (ctx_meta != nullptr) {
        ggml_free(ctx_meta);
    }
    if (backend_buffer != nullptr) {
        transcribe::safe_buffer_free(backend_buffer);
    }
    for (auto it = plan.scheduler_list.rbegin(); it != plan.scheduler_list.rend(); ++it) {
        transcribe::safe_backend_free(*it);
    }
    plan.scheduler_list.clear();
    plan.primary = nullptr;
}

Session::~Session() = default;

void apply_family_invariants(transcribe_model & model) {
    transcribe_capabilities & caps = model.caps;
    caps.native_sample_rate        = 16000;
    caps.supports_translate        = false;
    caps.max_timestamp_kind        = TRANSCRIBE_TIMESTAMPS_NONE;
    transcribe::set_feature(&model, TRANSCRIBE_FEATURE_DIARIZATION, true);
    transcribe::set_feature(&model, TRANSCRIBE_FEATURE_CANCELLATION, true);
}

namespace {

ggml_tensor * layer_norm(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b) {
    return ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, x, k_ln_eps), w), b);
}

ggml_tensor * linear(ggml_context * ctx, ggml_tensor * w, ggml_tensor * x, ggml_tensor * b) {
    ggml_tensor * y = ggml_mul_mat(ctx, w, x);
    return b != nullptr ? ggml_add(ctx, y, b) : y;
}

// Pre-LN layer: x + attn(LN(x)), then + MLP(LN(x)). x ne = [d, n].
ggml_tensor * encoder_layer(ggml_context *  ctx,
                            const Block &   b,
                            const HParams & hp,
                            ggml_tensor *   x,
                            ggml_tensor *   pos,
                            ggml_tensor *   mask,
                            int             n) {
    const int d  = hp.d_model;
    const int H  = hp.n_heads;
    const int hd = hp.head_dim();

    ggml_tensor * h = layer_norm(ctx, x, b.norm_attn_w, b.norm_attn_b);
    ggml_tensor * q = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, b.q_w, h), hd, H, n);
    ggml_tensor * k = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, b.k_w, h), hd, H, n);
    ggml_tensor * v = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, b.v_w, h), hd, H, n);

    q = ggml_rope_ext(ctx, q, pos, nullptr, hd, GGML_ROPE_TYPE_NEOX, 0, hp.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    k = ggml_rope_ext(ctx, k, pos, nullptr, hd, GGML_ROPE_TYPE_NEOX, 0, hp.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    q                = ggml_permute(ctx, q, 0, 2, 1, 3);                  // [hd, n, H]
    k                = ggml_permute(ctx, k, 0, 2, 1, 3);                  // [hd, n, H]
    ggml_tensor * vt = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3));  // [n, hd, H]

    ggml_tensor * kq = ggml_mul_mat(ctx, k, q);                           // [n_k, n_q, H]
    kq               = ggml_soft_max_ext(ctx, kq, mask, 1.0f / std::sqrt(static_cast<float>(hd)), 0.0f);
    ggml_tensor * o  = ggml_mul_mat(ctx, vt, kq);                         // [hd, n_q, H]
    o                = ggml_cont(ctx, ggml_permute(ctx, o, 0, 2, 1, 3));  // [hd, H, n]
    o                = ggml_reshape_2d(ctx, o, d, n);
    x                = ggml_add(ctx, x, linear(ctx, b.out_w, o, b.out_b));

    h = layer_norm(ctx, x, b.norm_ff_w, b.norm_ff_b);
    h = ggml_gelu_erf(ctx, linear(ctx, b.ff_in_w, h, b.ff_in_b));
    return ggml_add(ctx, x, linear(ctx, b.ff_out_w, h, b.ff_out_b));
}

struct StepGraph {
    ggml_cgraph * graph     = nullptr;
    ggml_tensor * input     = nullptr;  // [d, n]
    ggml_tensor * pos       = nullptr;  // [n] i32
    ggml_tensor * mask      = nullptr;  // [n, n] or null
    ggml_tensor * layer0    = nullptr;  // [d, n] first encoder layer output
    ggml_tensor * audio     = nullptr;  // [d, n] encoder output
    ggml_tensor * proj      = nullptr;  // [head_d, n]
    ggml_tensor * upsampled = nullptr;  // [head_d, n * sub]
    ggml_tensor * logits    = nullptr;  // [S, n * sub]
};

// Encoder + head over one step's n input embeddings. Padding frames (at or
// past n_valid) are excluded as attention keys.
StepGraph build_step_graph(ggml_context * ctx, const Model & m, int n, bool masked) {
    const HParams & hp  = m.hparams;
    const Weights & w   = m.weights;
    const int       hd  = hp.head_d;
    const int       sub = hp.subsampling_factor;

    StepGraph g{};
    g.input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hp.d_model, n);
    ggml_set_name(g.input, "step.input");
    ggml_set_input(g.input);
    g.pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n);
    ggml_set_name(g.pos, "step.pos");
    ggml_set_input(g.pos);
    if (masked) {
        g.mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, n);
        ggml_set_name(g.mask, "step.mask");
        ggml_set_input(g.mask);
    }

    ggml_tensor * x = layer_norm(ctx, g.input, w.norm_in_w, w.norm_in_b);
    for (const Block & b : w.blocks) {
        x = encoder_layer(ctx, b, hp, x, g.pos, g.mask, n);
        if (g.layer0 == nullptr) {
            g.layer0 = x;
        }
    }
    g.audio = layer_norm(ctx, x, w.norm_out_w, w.norm_out_b);
    g.proj  = linear(ctx, w.proj_w, g.audio, w.proj_b);

    // Conv1d(k=3, pad=1) as three taps: out[t] = W0 x[t-1] + W1 x[t] + W2 x[t+1] + b.
    ggml_tensor * a0   = ggml_mul_mat(ctx, w.up_w[0], g.proj);  // [hd*sub, n]
    ggml_tensor * a1   = ggml_mul_mat(ctx, w.up_w[1], g.proj);
    ggml_tensor * a2   = ggml_mul_mat(ctx, w.up_w[2], g.proj);
    ggml_tensor * prev = ggml_pad_ext(ctx, a0, 0, 0, 1, 0, 0, 0, 0, 0);  // [hd*sub, n+1], shifted right
    prev               = ggml_view_2d(ctx, prev, prev->ne[0], n, prev->nb[1], 0);
    ggml_tensor * next = ggml_pad_ext(ctx, a2, 0, 0, 0, 1, 0, 0, 0, 0);  // [hd*sub, n+1]
    next               = ggml_view_2d(ctx, next, next->ne[0], n, next->nb[1], next->nb[1]);
    ggml_tensor * up   = ggml_add(ctx, ggml_add(ctx, ggml_add(ctx, a1, prev), next), w.up_b);
    g.upsampled        = ggml_reshape_2d(ctx, ggml_cont(ctx, up), hd, static_cast<int64_t>(n) * sub);

    ggml_tensor * h = linear(ctx, w.dense_w, ggml_relu(ctx, g.upsampled), w.dense_b);
    g.logits        = linear(ctx, w.out_w, ggml_relu(ctx, h), w.out_b);

    g.graph = ggml_new_graph_custom(ctx, 16384, false);
    ggml_build_forward_expand(g.graph, g.logits);
    return g;
}

transcribe_status ensure_sched(Session * s, Model * m) {
    if (s->sched == nullptr) {
        s->sched = ggml_backend_sched_new(m->plan.scheduler_list.data(), nullptr,
                                          static_cast<int>(m->plan.scheduler_list.size()), 16384, false, true);
        if (s->sched == nullptr) {
            return TRANSCRIBE_ERR_BACKEND;
        }
    }
    return TRANSCRIBE_OK;
}

transcribe_status new_compute_ctx(Session * s, size_t bytes) {
    if (s->compute_ctx != nullptr) {
        ggml_free(s->compute_ctx);
        s->compute_ctx = nullptr;
    }
    ggml_init_params ip{};
    ip.mem_size    = bytes;
    ip.no_alloc    = true;
    s->compute_ctx = ggml_init(ip);
    return s->compute_ctx != nullptr ? TRANSCRIBE_OK : TRANSCRIBE_ERR_BACKEND;
}

// Embedder block, in encoder frames. Fixed so the batch run and the stream
// use one matmul shape: on CPU the quantized matmul result depends on the
// row count.
constexpr int k_embed_rows = 64;

// Embedder (8-frame stacking + projection) over n_enc encoder frames in
// k_embed_rows blocks, the last block zero-padded. fill(first, n, dst)
// writes the stacked time-major input of frames [first, first + n) into dst
// ([n, n_mels * sub]). Output rows go to out ([n_enc, D]).
template <typename Fill> transcribe_status embed_frames(Session * s, Model * m, int n_enc, float * out, Fill && fill) {
    const int rows  = k_embed_rows;
    const int D     = m->hparams.d_model;
    const int width = m->hparams.fe_num_mels * m->hparams.subsampling_factor;

    if (const transcribe_status st = new_compute_ctx(s, 1 * 1024 * 1024); st != TRANSCRIBE_OK) {
        return st;
    }
    ggml_context * ctx = s->compute_ctx;
    ggml_tensor *  in  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, rows);
    ggml_set_input(in);
    ggml_tensor * y     = ggml_mul_mat(ctx, m->weights.embed_w, in);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y);
    ggml_backend_sched_reset(s->sched);
    if (!ggml_backend_sched_alloc_graph(s->sched, graph)) {
        return TRANSCRIBE_ERR_BACKEND;
    }
    transcribe::configure_sched_n_threads(s->sched, s->n_threads);

    std::vector<float> stacked(static_cast<size_t>(rows) * width);
    for (int first = 0; first < n_enc; first += rows) {
        const int n = std::min(rows, n_enc - first);
        std::fill(stacked.begin(), stacked.end(), 0.0f);
        fill(first, n, stacked.data());
        ggml_backend_tensor_set(in, stacked.data(), 0, stacked.size() * sizeof(float));
        if (ggml_backend_sched_graph_compute(s->sched, graph) != GGML_STATUS_SUCCESS) {
            return TRANSCRIBE_ERR_BACKEND;
        }
        ggml_backend_tensor_get(y, out + static_cast<size_t>(first) * D, 0, static_cast<size_t>(n) * D * sizeof(float));
    }
    return TRANSCRIBE_OK;
}

// Embedder over the whole clip from the channel-major mel in s->mel_buf.
transcribe_status run_embedder(Session * s, Model * m, int n_mels, int n_mel_frames, int n_valid_mel, int n_enc) {
    const int sub = m->hparams.subsampling_factor;
    const int D   = m->hparams.d_model;

    s->embeds_host.resize(static_cast<size_t>(n_enc) * D);
    const transcribe_status st = embed_frames(s, m, n_enc, s->embeds_host.data(), [&](int first, int n, float * dst) {
        const int t_end = std::min((first + n) * sub, n_valid_mel);
        for (int t = first * sub; t < t_end; ++t) {
            float * row = dst + static_cast<size_t>(t - first * sub) * n_mels;
            for (int f = 0; f < n_mels; ++f) {
                row[f] = s->mel_buf[static_cast<size_t>(f) * n_mel_frames + t];
            }
        }
    });
    if (st != TRANSCRIBE_OK) {
        return st;
    }
    if (transcribe::debug::enabled()) {
        const long long shape[2] = { n_enc, D };
        transcribe::debug::dump_host_f32("enc.embed.out", s->embeds_host.data(),
                                         static_cast<long long>(s->embeds_host.size()), shape, 2, "encoder");
    }
    return TRANSCRIBE_OK;
}

// One step over s->input_host (n frames, the first n_valid real). Leaves the
// mel-rate logits in s->logits_host ([n * sub, S]).
transcribe_status run_step(Session * s, Model * m, int n, int n_valid, bool dump) {
    const bool masked = n_valid < n;
    const int  mem    = 64 * 1024 * 1024;
    if (const transcribe_status st = new_compute_ctx(s, mem); st != TRANSCRIBE_OK) {
        return st;
    }
    StepGraph g = build_step_graph(s->compute_ctx, *m, n, masked);
    if (dump) {
        transcribe::debug::mark_tensor_for_dump(g.layer0);
        transcribe::debug::mark_tensor_for_dump(g.audio);
        transcribe::debug::mark_tensor_for_dump(g.proj);
        transcribe::debug::mark_tensor_for_dump(g.upsampled);
    }
    ggml_backend_sched_reset(s->sched);
    if (!ggml_backend_sched_alloc_graph(s->sched, g.graph)) {
        return TRANSCRIBE_ERR_BACKEND;
    }
    ggml_backend_tensor_set(g.input, s->input_host.data(), 0,
                            static_cast<size_t>(n) * m->hparams.d_model * sizeof(float));
    std::vector<int32_t> pos(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        pos[static_cast<size_t>(i)] = i;
    }
    ggml_backend_tensor_set(g.pos, pos.data(), 0, pos.size() * sizeof(int32_t));
    if (masked) {
        std::vector<float> mask(static_cast<size_t>(n) * n, 0.0f);
        for (int q = 0; q < n; ++q) {
            for (int k = n_valid; k < n; ++k) {
                mask[static_cast<size_t>(q) * n + k] = -std::numeric_limits<float>::infinity();
            }
        }
        ggml_backend_tensor_set(g.mask, mask.data(), 0, mask.size() * sizeof(float));
    }
    transcribe::configure_sched_n_threads(s->sched, s->n_threads);
    if (ggml_backend_sched_graph_compute(s->sched, g.graph) != GGML_STATUS_SUCCESS) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "nemotron3_diar: step compute failed");
        return TRANSCRIBE_ERR_BACKEND;
    }
    const int sub = m->hparams.subsampling_factor;
    s->logits_host.resize(static_cast<size_t>(n) * sub * m->hparams.max_speakers);
    ggml_backend_tensor_get(g.logits, s->logits_host.data(), 0, s->logits_host.size() * sizeof(float));
    if (dump) {
        transcribe::debug::dump_tensor("enc.layer0.out", g.layer0, "encoder");
        transcribe::debug::dump_tensor("enc.audio.out", g.audio, "encoder");
        transcribe::debug::dump_tensor("enc.proj.out", g.proj, "encoder");
        transcribe::debug::dump_tensor("enc.upsample.out", g.upsampled, "encoder");
    }
    return TRANSCRIBE_OK;
}

void sigmoid_rows(const std::vector<float> & logits, size_t first, size_t count, std::vector<float> & out) {
    for (size_t i = 0; i < count; ++i) {
        out.push_back(1.0f / (1.0f + std::exp(-logits[first + i])));
    }
}

// Full-context forward with no speaker cache, the non-stateful parity
// target (validation only: O(T^2) attention over the whole clip).
transcribe_status run_offline_dump(Session * s, Model * m, int n_enc, int n_valid_enc, int n_mel_frames) {
    const int D = m->hparams.d_model;
    const int S = m->hparams.max_speakers;
    s->input_host.assign(s->embeds_host.begin(), s->embeds_host.begin() + static_cast<size_t>(n_enc) * D);
    if (const transcribe_status st = run_step(s, m, n_enc, n_valid_enc, true); st != TRANSCRIBE_OK) {
        return st;
    }
    std::vector<float> probs;
    sigmoid_rows(s->logits_host, 0, static_cast<size_t>(n_mel_frames) * S, probs);
    const long long shape[2] = { n_mel_frames, S };
    transcribe::debug::dump_host_f32("diar.preds_offline", probs.data(), static_cast<long long>(probs.size()), shape, 2,
                                     "encoder");
    return TRANSCRIBE_OK;
}

// One chunk step: [speaker cache | FIFO | chunk | lookahead]. `emb` holds the
// `take` embeddings from the chunk's first frame on (n_chunk chunk frames,
// the rest lookahead), the first n_valid of them real. Appends the chunk's
// mel-rate probabilities to s->probs.
transcribe_status
run_chunk(Session * s, Model * m, const StreamParams & p, const float * emb, int n_chunk, int take, int n_valid) {
    const HParams & hp     = m->hparams;
    const int       D      = hp.d_model;
    const int       S      = hp.max_speakers;
    const int       sub    = hp.subsampling_factor;
    const int       cached = s->cache.cache_n + s->cache.fifo_n;
    const int       n      = cached + take;

    s->input_host.resize(static_cast<size_t>(n) * D);
    auto dst = s->input_host.begin();
    dst      = std::copy_n(s->cache.embeds.begin(), static_cast<size_t>(s->cache.cache_n) * D, dst);
    dst      = std::copy_n(s->cache.fifo.begin(), static_cast<size_t>(s->cache.fifo_n) * D, dst);
    std::copy_n(emb, static_cast<size_t>(take) * D, dst);

    if (const transcribe_status st = run_step(s, m, n, cached + n_valid, false); st != TRANSCRIBE_OK) {
        return st;
    }
    update_speaker_cache(s->cache, hp, p, m->silence_host, s->input_host.data(), n, cached + n_valid,
                         s->logits_host.data(), n_chunk);
    sigmoid_rows(s->logits_host, static_cast<size_t>(cached) * sub * S, static_cast<size_t>(n_chunk) * sub * S,
                 s->probs);
    return TRANSCRIBE_OK;
}

// Chunked AOSC/FIFO forward over the clip's embeddings; fills s->probs
// ([n_mel_frames, S]).
transcribe_status run_chunks(Session *            s,
                             Model *              m,
                             const StreamParams & p,
                             int                  n_enc,
                             int                  n_valid_enc,
                             int                  n_mel_frames) {
    const HParams & hp = m->hparams;
    const int       S  = hp.max_speakers;

    s->cache.reset();
    s->probs.clear();
    s->probs.reserve(static_cast<size_t>(n_enc) * hp.subsampling_factor * S);

    for (int start = 0; start < n_enc; start += p.chunk_len) {
        if (s->poll_abort()) {
            return TRANSCRIBE_ERR_ABORTED;
        }
        const int end  = std::min(start + p.chunk_len, n_enc);
        const int take = std::min(end + p.right_context, n_enc) - start;
        if (const transcribe_status st =
                run_chunk(s, m, p, s->embeds_host.data() + static_cast<size_t>(start) * hp.d_model, end - start, take,
                          std::clamp(n_valid_enc - start, 0, take));
            st != TRANSCRIBE_OK) {
            return st;
        }
    }
    s->probs.resize(std::min(s->probs.size(), static_cast<size_t>(n_mel_frames) * S));
    return TRANSCRIBE_OK;
}

void emit_segments(Session * s, int n_frames, int n_spk, double ms_per_frame) {
    for (int spk = 0; spk < n_spk; ++spk) {
        int run_start = -1;
        for (int t = 0; t <= n_frames; ++t) {
            const bool active = t < n_frames && s->probs[static_cast<size_t>(t) * n_spk + spk] > 0.5f;
            if (active && run_start < 0) {
                run_start = t;
            } else if (!active && run_start >= 0) {
                transcribe_session::SpeakerSegmentEntry row;
                row.t0_ms      = static_cast<int64_t>(std::llround(run_start * ms_per_frame));
                row.t1_ms      = static_cast<int64_t>(std::llround(t * ms_per_frame));
                row.speaker_id = spk + 1;
                row.p          = std::numeric_limits<float>::quiet_NaN();
                s->speaker_segments.push_back(row);
                run_start = -1;
            }
        }
    }
}

transcribe_status load(Loader & loader, const transcribe_model_load_params * params, transcribe_model ** out_model) {
    const int64_t t_start = ggml_time_us();
    auto          m       = std::make_unique<Model>();
    m->arch               = &arch;
    m->variant            = loader.variant().empty() ? k_default_variant : loader.variant();

    apply_family_invariants(*m);
    m->caps.n_languages = 0;
    m->caps.languages   = nullptr;
    if (const transcribe_status st = read_capability_kv(loader.gguf(), m->caps); st != TRANSCRIBE_OK) {
        return st;
    }
    if (const transcribe_status st = read_languages_kv(loader.gguf(), *m); st != TRANSCRIBE_OK) {
        return st;
    }
    if (const transcribe_status st = read_hparams(loader.gguf(), m->hparams); st != TRANSCRIBE_OK) {
        return st;
    }

    {
        transcribe::MelConfig cfg{};
        cfg.sample_rate  = m->hparams.fe_sample_rate;
        cfg.num_mels     = m->hparams.fe_num_mels;
        cfg.n_fft        = m->hparams.fe_n_fft;
        cfg.win_length   = m->hparams.fe_win_length;
        cfg.hop_length   = m->hparams.fe_hop_length;
        cfg.pre_emphasis = m->hparams.fe_pre_emphasis;
        cfg.f_max        = static_cast<float>(m->hparams.fe_sample_rate) / 2.0f;
        cfg.normalize    = "none";
        cfg.pad_mode     = "constant";
        m->mel.emplace(cfg);
    }

    gguf_init_params init_params{};
    init_params.no_alloc     = true;
    init_params.ctx          = &m->ctx_meta;
    gguf_context * gguf_data = gguf_init_from_file(loader.path().c_str(), init_params);
    if (gguf_data == nullptr) {
        return TRANSCRIBE_ERR_GGUF;
    }
    auto fail = [&](transcribe_status st) {
        gguf_free(gguf_data);
        return st;
    };
    if (const transcribe_status st = build_weights(m->ctx_meta, m->hparams, m->weights); st != TRANSCRIBE_OK) {
        return fail(st);
    }
    const transcribe_backend_request backend_req = params != nullptr ? params->backend : TRANSCRIBE_BACKEND_AUTO;
    if (const transcribe_status st = transcribe::load_common::init_backends(
            backend_req, params != nullptr ? params->device : nullptr, "nemotron3_diar", m->plan);
        st != TRANSCRIBE_OK) {
        return fail(st);
    }
    m->backend         = ggml_backend_name(m->plan.primary);
    m->primary_backend = m->plan.primary;

    m->backend_buffer = ggml_backend_alloc_ctx_tensors(m->ctx_meta, m->plan.primary);
    if (m->backend_buffer == nullptr) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "nemotron3_diar: weight allocation failed");
        return fail(TRANSCRIBE_ERR_GGUF);
    }
    ggml_backend_buffer_set_usage(m->backend_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    if (const transcribe_status st =
            transcribe::load_common::stream_tensor_data(loader.path(), gguf_data, m->ctx_meta, "nemotron3_diar");
        st != TRANSCRIBE_OK) {
        return fail(st);
    }
    gguf_free(gguf_data);

    m->silence_host.resize(static_cast<size_t>(m->hparams.d_model));
    ggml_backend_tensor_get(m->weights.silence, m->silence_host.data(), 0, m->silence_host.size() * sizeof(float));

    m->t_load_us = ggml_time_us() - t_start;
    *out_model   = m.release();
    return TRANSCRIBE_OK;
}

transcribe_status init_context(transcribe_model *                model,
                               const transcribe_session_params * params,
                               transcribe_session **             out_ctx) {
    if (model->arch != &arch) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    auto s       = std::make_unique<Session>();
    s->model     = model;
    s->n_threads = params->n_threads;
    s->kv_type   = params->kv_type;
    *out_ctx     = s.release();
    return TRANSCRIBE_OK;
}

transcribe_sortformer_preset requested_preset(const transcribe_run_params * params) {
    if (params == nullptr || params->family == nullptr) {
        return TRANSCRIBE_SORTFORMER_PRESET_DEFAULT;
    }
    return reinterpret_cast<const transcribe_sortformer_stream_ext *>(params->family)->preset;
}

transcribe_status run(transcribe_session *          session,
                      const float *                 pcm,
                      int                           n_samples,
                      const transcribe_run_params * params) {
    auto * s = static_cast<Session *>(session);
    auto * m = static_cast<Model *>(session->model);
    if (s->poll_abort()) {
        return TRANSCRIBE_ERR_ABORTED;
    }
    if (params != nullptr && params->family != nullptr) {
        if (const transcribe_status st = transcribe_ext_check(params->family, TRANSCRIBE_EXT_KIND_SORTFORMER_STREAM,
                                                              sizeof(struct transcribe_sortformer_stream_ext));
            st != TRANSCRIBE_OK) {
            return st;
        }
    }
    StreamParams sp{};
    if (!resolve_stream_params(m->hparams, requested_preset(params), sp)) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    s->clear_result();
    transcribe::debug::init();
    if (!m->mel.has_value()) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    const int64_t t_start = ggml_time_us();
    int           n_mels = 0, n_mel_frames = 0;
    if (const transcribe_status st =
            m->mel->compute(pcm, static_cast<size_t>(n_samples), s->mel_buf, n_mels, n_mel_frames);
        st != TRANSCRIBE_OK) {
        return st;
    }
    // The processor marks floor(n / hop) frames valid and zeroes the
    // trailing center-padding frame; an encoder frame is valid when its
    // first mel frame is.
    const int sub         = m->hparams.subsampling_factor;
    const int n_valid_mel = std::min(n_samples / m->hparams.fe_hop_length, n_mel_frames);
    for (int f = 0; f < n_mels; ++f) {
        std::fill(s->mel_buf.begin() + static_cast<size_t>(f) * n_mel_frames + n_valid_mel,
                  s->mel_buf.begin() + static_cast<size_t>(f + 1) * n_mel_frames, 0.0f);
    }
    const int n_enc       = (n_mel_frames + sub - 1) / sub;
    const int n_valid_enc = (n_valid_mel + sub - 1) / sub;
    if (transcribe::debug::enabled()) {
        const long long shape[2] = { n_mels, n_mel_frames };
        transcribe::debug::dump_host_f32("enc.mel.in", s->mel_buf.data(), static_cast<long long>(s->mel_buf.size()),
                                         shape, 2, "frontend");
    }

    if (const transcribe_status st = ensure_sched(s, m); st != TRANSCRIBE_OK) {
        return st;
    }
    if (const transcribe_status st = run_embedder(s, m, n_mels, n_mel_frames, n_valid_mel, n_enc);
        st != TRANSCRIBE_OK) {
        return st;
    }
    if (transcribe::debug::enabled() && std::getenv("TRANSCRIBE_NEMOTRON3_DIAR_OFFLINE_DUMP") != nullptr) {
        if (const transcribe_status st = run_offline_dump(s, m, n_enc, n_valid_enc, n_mel_frames);
            st != TRANSCRIBE_OK) {
            return st;
        }
    }
    if (const transcribe_status st = run_chunks(s, m, sp, n_enc, n_valid_enc, n_mel_frames); st != TRANSCRIBE_OK) {
        return st;
    }
    s->t_encode_us = ggml_time_us() - t_start;

    const int S        = m->hparams.max_speakers;
    const int n_frames = static_cast<int>(s->probs.size() / static_cast<size_t>(S));
    if (transcribe::debug::enabled()) {
        const long long shape[2] = { n_frames, S };
        transcribe::debug::dump_host_f32("diar.probs", s->probs.data(), static_cast<long long>(s->probs.size()), shape,
                                         2, "diarize");
    }
    const double ms_per_frame = 1000.0 * m->hparams.frame_hop / m->hparams.fe_sample_rate;
    emit_segments(s, n_frames, S, ms_per_frame);

    s->result_kind = TRANSCRIBE_TIMESTAMPS_NONE;
    s->has_result  = true;
    return TRANSCRIBE_OK;
}

bool accepts_ext_kind(const transcribe_model * model, transcribe_ext_slot slot, uint32_t kind) {
    if (model == nullptr) {
        return false;
    }
    return (slot == TRANSCRIBE_EXT_SLOT_RUN && kind == TRANSCRIBE_EXT_KIND_SORTFORMER_STREAM) ||
           (slot == TRANSCRIBE_EXT_SLOT_STREAM && kind == TRANSCRIBE_EXT_KIND_SORTFORMER_LIVE);
}

transcribe_status run_validate(const transcribe_session * session, const transcribe_run_params * params) {
    if (params == nullptr || params->family == nullptr) {
        return TRANSCRIBE_OK;
    }
    if (const transcribe_status st = transcribe_ext_check(params->family, TRANSCRIBE_EXT_KIND_SORTFORMER_STREAM,
                                                          sizeof(struct transcribe_sortformer_stream_ext));
        st != TRANSCRIBE_OK) {
        return st;
    }
    const auto * m = static_cast<const Model *>(session->model);
    StreamParams sp{};
    return resolve_stream_params(m->hparams, requested_preset(params), sp) ? TRANSCRIBE_OK : TRANSCRIBE_ERR_INVALID_ARG;
}

// ---- Push-audio streaming -------------------------------------------------
// The batch run is already chunked; the stream runs the same chunks as soon
// as their input is final. A mel frame is final once its centered window has
// fully arrived, an encoder frame once its sub mel frames are, and a chunk
// once the chunk and its lookahead are embedded. Stream output therefore
// equals the batch run at the same preset; only the tail (partial lookahead,
// trailing padding frame) waits for finalize.

// Mel frames recomputed ahead of the first new one: the left half-window and
// the pre-emphasis carry of that frame come from already-received audio.
constexpr int k_live_mel_guard = 2;

transcribe_status resolve_live_params(const Model * m, const transcribe_stream_params * sp, StreamParams & out) {
    const transcribe_ext * fam = sp != nullptr ? sp->family : nullptr;
    if (const transcribe_status st = transcribe_ext_check(fam, TRANSCRIBE_EXT_KIND_SORTFORMER_LIVE,
                                                          sizeof(struct transcribe_sortformer_live_ext));
        st != TRANSCRIBE_OK) {
        return st;
    }
    const transcribe_sortformer_preset preset =
        fam != nullptr ? reinterpret_cast<const transcribe_sortformer_live_ext *>(fam)->preset :
                         TRANSCRIBE_SORTFORMER_PRESET_LOW_LATENCY;
    switch (preset) {
        case TRANSCRIBE_SORTFORMER_PRESET_LOW_LATENCY:
        case TRANSCRIBE_SORTFORMER_PRESET_VERY_LOW_LATENCY:
        case TRANSCRIBE_SORTFORMER_PRESET_ULTRA_LOW_LATENCY:
            break;
        default:
            return TRANSCRIBE_ERR_INVALID_ARG;
    }
    return resolve_stream_params(m->hparams, preset, out) ? TRANSCRIBE_OK : TRANSCRIBE_ERR_INVALID_ARG;
}

// Mel frames up to the final count (all frames at finalize, zeroing the
// trailing padding frame as run() does), appended time-major to live.mel.
transcribe_status live_mel(Session * s, Model * m, bool final) {
    LiveState &     lv  = s->live;
    const HParams & hp  = m->hparams;
    const int64_t   N   = lv.n_received;
    const int       hop = hp.fe_hop_length;
    const int       pad = hp.fe_n_fft / 2;
    int             target;
    if (final) {
        target = N >= hop ? static_cast<int>(N / hop) + 1 : lv.n_mel;
    } else {
        target = N >= pad ? static_cast<int>((N - pad) / hop) + 1 : 0;
    }
    if (target <= lv.n_mel) {
        return TRANSCRIBE_OK;
    }
    const int     first  = std::max(0, lv.n_mel - k_live_mel_guard);
    const int64_t a      = static_cast<int64_t>(first) * hop;
    int           n_mels = 0, n_frames = 0;
    if (const transcribe_status st = m->mel->compute(lv.pcm.data() + (a - lv.pcm_base), static_cast<size_t>(N - a),
                                                     s->mel_buf, n_mels, n_frames);
        st != TRANSCRIBE_OK) {
        return st;
    }
    if (n_frames < target - first) {
        return TRANSCRIBE_ERR_BACKEND;
    }
    for (int t = lv.n_mel - first; t < target - first; ++t) {
        for (int f = 0; f < n_mels; ++f) {
            lv.mel.push_back(s->mel_buf[static_cast<size_t>(f) * n_frames + t]);
        }
    }
    lv.n_mel = target;

    const int64_t keep = static_cast<int64_t>(std::max(0, target - k_live_mel_guard)) * hop;
    lv.pcm.erase(lv.pcm.begin(), lv.pcm.begin() + (keep - lv.pcm_base));
    lv.pcm_base = keep;
    return TRANSCRIBE_OK;
}

// Embeds every encoder frame whose mel frames are all computed (at finalize
// the last frame is zero-padded, as in run()).
transcribe_status live_embed(Session * s, Model * m, bool final) {
    LiveState & lv     = s->live;
    const int   sub    = m->hparams.subsampling_factor;
    const int   D      = m->hparams.d_model;
    const int   width  = m->hparams.fe_num_mels * sub;
    const int   target = final ? (lv.n_mel + sub - 1) / sub : lv.n_mel / sub;
    const int   n      = target - lv.n_embedded;
    if (n <= 0) {
        return TRANSCRIBE_OK;
    }
    lv.mel.resize(std::max(lv.mel.size(), static_cast<size_t>(n) * width), 0.0f);
    const size_t old = lv.embeds.size();
    lv.embeds.resize(old + static_cast<size_t>(n) * D);
    const transcribe_status st = embed_frames(s, m, n, lv.embeds.data() + old, [&](int first, int count, float * dst) {
        std::copy_n(lv.mel.begin() + static_cast<size_t>(first) * width, static_cast<size_t>(count) * width, dst);
    });
    if (st != TRANSCRIBE_OK) {
        return st;
    }
    lv.mel.erase(lv.mel.begin(), lv.mel.begin() + static_cast<size_t>(n) * width);
    lv.n_embedded = target;
    return TRANSCRIBE_OK;
}

// Runs every chunk whose chunk + lookahead frames are embedded; at finalize
// also the tail chunks with the batch run's short lookahead and padding mask.
transcribe_status live_chunks(Session * s, Model * m, bool final) {
    LiveState &          lv          = s->live;
    const StreamParams & p           = lv.params;
    const int            D           = m->hparams.d_model;
    const int            n_enc       = lv.n_embedded;
    const int            n_valid_mel = static_cast<int>(lv.n_received / m->hparams.fe_hop_length);
    const int n_valid_enc = (n_valid_mel + m->hparams.subsampling_factor - 1) / m->hparams.subsampling_factor;
    while (lv.next_chunk < n_enc && (final || lv.next_chunk + p.chunk_len + p.right_context <= n_enc)) {
        if (s->poll_abort()) {
            return TRANSCRIBE_ERR_ABORTED;
        }
        const int start = lv.next_chunk;
        const int end   = std::min(start + p.chunk_len, n_enc);
        const int take  = std::min(end + p.right_context, n_enc) - start;
        if (const transcribe_status st =
                run_chunk(s, m, p, lv.embeds.data(), end - start, take, std::clamp(n_valid_enc - start, 0, take));
            st != TRANSCRIBE_OK) {
            return st;
        }
        lv.embeds.erase(lv.embeds.begin(), lv.embeds.begin() + static_cast<size_t>(end - start) * D);
        lv.next_chunk = end;
    }
    if (final) {
        const int S = m->hparams.max_speakers;
        s->probs.resize(std::min(s->probs.size(), static_cast<size_t>(lv.n_mel) * S));
    }
    return TRANSCRIBE_OK;
}

// Turns the new output frames into rows: closed runs are final, a run still
// active at the frontier is published with t1 = frontier (closed at finalize).
// Returns true when the published rows changed.
bool live_rows(Session * s, Model * m, bool final) {
    LiveState &  lv           = s->live;
    const int    S            = m->hparams.max_speakers;
    const double ms_per_frame = 1000.0 * m->hparams.frame_hop / m->hparams.fe_sample_rate;
    const int    n_frames     = static_cast<int>(s->probs.size() / static_cast<size_t>(S));
    const auto   to_ms        = [&](int frame) {
        return static_cast<int64_t>(std::llround(frame * ms_per_frame));
    };
    if (n_frames == lv.n_scanned && !final) {
        return false;
    }
    for (int spk = 0; spk < S; ++spk) {
        int & run_start = lv.open_start[static_cast<size_t>(spk)];
        for (int t = lv.n_scanned; t <= n_frames; ++t) {
            if (t == n_frames && !final) {
                break;
            }
            const bool active = t < n_frames && s->probs[static_cast<size_t>(t) * S + spk] > 0.5f;
            if (active && run_start < 0) {
                run_start = t;
            } else if (!active && run_start >= 0) {
                transcribe_session::SpeakerSegmentEntry row;
                row.t0_ms      = to_ms(run_start);
                row.t1_ms      = to_ms(t);
                row.speaker_id = spk + 1;
                row.p          = std::numeric_limits<float>::quiet_NaN();
                lv.closed[static_cast<size_t>(spk)].push_back(row);
                run_start = -1;
            }
        }
    }
    lv.n_scanned = n_frames;

    std::vector<transcribe_session::SpeakerSegmentEntry> rows;
    for (int spk = 0; spk < S; ++spk) {
        const auto & closed = lv.closed[static_cast<size_t>(spk)];
        rows.insert(rows.end(), closed.begin(), closed.end());
        if (const int run_start = lv.open_start[static_cast<size_t>(spk)]; run_start >= 0) {
            transcribe_session::SpeakerSegmentEntry row;
            row.t0_ms      = to_ms(run_start);
            row.t1_ms      = to_ms(n_frames);
            row.speaker_id = spk + 1;
            row.p          = std::numeric_limits<float>::quiet_NaN();
            rows.push_back(row);
        }
    }
    const bool changed =
        !std::equal(rows.begin(), rows.end(), s->speaker_segments.begin(), s->speaker_segments.end(),
                    [](const auto & a, const auto & b) {
                        return a.t0_ms == b.t0_ms && a.t1_ms == b.t1_ms && a.speaker_id == b.speaker_id;
                    });
    s->speaker_segments.swap(rows);
    return changed;
}

// Runs the pipeline over the received audio and publishes rows + cursors.
transcribe_status live_process(Session * s, Model * m, bool final, transcribe_stream_update * update) {
    const int64_t t_start = ggml_time_us();
    if (const transcribe_status st = live_mel(s, m, final); st != TRANSCRIBE_OK) {
        return st;
    }
    const int64_t t_mel = ggml_time_us();
    s->t_mel_us += t_mel - t_start;
    for (auto stage : { live_embed, live_chunks }) {
        if (const transcribe_status st = stage(s, m, final); st != TRANSCRIBE_OK) {
            return st;
        }
    }
    s->t_encode_us += ggml_time_us() - t_mel;

    const bool changed = live_rows(s, m, final);
    if (changed) {
        s->stream_revision += 1;
    }
    const int     S            = m->hparams.max_speakers;
    const int     sr           = std::max(1, m->hparams.fe_sample_rate);
    const double  ms_per_frame = 1000.0 * m->hparams.frame_hop / sr;
    const int64_t received_ms  = s->live.n_received * 1000 / sr;
    const int64_t committed_ms =
        static_cast<int64_t>(std::llround(static_cast<double>(s->probs.size() / S) * ms_per_frame));
    if (update != nullptr) {
        update->result_changed     = changed;
        update->revision           = s->stream_revision;
        update->input_received_ms  = received_ms;
        update->audio_committed_ms = committed_ms;
        update->buffered_ms        = std::max<int64_t>(0, received_ms - committed_ms);
    }
    if (final && transcribe::debug::enabled()) {
        const long long shape[2] = { static_cast<long long>(s->probs.size() / S), S };
        transcribe::debug::dump_host_f32("diar.probs", s->probs.data(), static_cast<long long>(s->probs.size()), shape,
                                         2, "diarize");
    }
    return TRANSCRIBE_OK;
}

transcribe_status stream_validate(const transcribe_session * session,
                                  const transcribe_run_params * /*run_params*/,
                                  const transcribe_stream_params * stream_params) {
    StreamParams sp{};
    return resolve_live_params(static_cast<const Model *>(session->model), stream_params, sp);
}

transcribe_status stream_begin(transcribe_session * session,
                               const transcribe_run_params * /*run_params*/,
                               const transcribe_stream_params * stream_params) {
    auto * s = static_cast<Session *>(session);
    auto * m = static_cast<Model *>(session->model);
    if (!m->mel.has_value()) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    s->live.reset();
    if (const transcribe_status st = resolve_live_params(m, stream_params, s->live.params); st != TRANSCRIBE_OK) {
        return st;
    }
    if (const transcribe_status st = ensure_sched(s, m); st != TRANSCRIBE_OK) {
        return st;
    }
    transcribe::debug::init();
    const size_t S = static_cast<size_t>(m->hparams.max_speakers);
    s->live.open_start.assign(S, -1);
    s->live.closed.assign(S, {});
    s->cache.reset();
    s->probs.clear();
    s->result_kind = TRANSCRIBE_TIMESTAMPS_NONE;
    s->has_result  = true;
    return TRANSCRIBE_OK;
}

transcribe_status stream_feed(transcribe_session *       session,
                              const float *              pcm,
                              int                        n_samples,
                              transcribe_stream_update * update) {
    auto * s = static_cast<Session *>(session);
    if (s->poll_abort()) {
        return TRANSCRIBE_ERR_ABORTED;
    }
    s->live.pcm.insert(s->live.pcm.end(), pcm, pcm + n_samples);
    s->live.n_received += n_samples;
    return live_process(s, static_cast<Model *>(session->model), false, update);
}

transcribe_status stream_finalize(transcribe_session * session, transcribe_stream_update * update) {
    auto * s = static_cast<Session *>(session);
    if (s->poll_abort()) {
        return TRANSCRIBE_ERR_ABORTED;
    }
    return live_process(s, static_cast<Model *>(session->model), true, update);
}

void stream_reset(transcribe_session * session) {
    auto * s = static_cast<Session *>(session);
    s->live.reset();
    s->cache.reset();
    s->probs.clear();
}

}  // namespace

extern const Arch arch = {
    /* .name             = */ "nemotron3_diar",
    /* .load             = */ load,
    /* .init_context     = */ init_context,
    /* .run              = */ run,
    /* .run_batch        = */ nullptr,
    /* .stream_validate  = */ stream_validate,
    /* .stream_begin     = */ stream_begin,
    /* .stream_feed      = */ stream_feed,
    /* .stream_finalize  = */ stream_finalize,
    /* .stream_reset     = */ stream_reset,
    /* .accepts_ext_kind = */ accepts_ext_kind,
    /* .run_validate     = */ run_validate,
};

}  // namespace transcribe::nemotron3_diar
