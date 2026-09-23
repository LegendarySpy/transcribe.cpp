// arch/nemotron3_diar/model.cpp - Nemotron-3 Diarization load / run / Arch.
//
// Per run: log-mel over the whole clip, one embedder graph (8-frame stacking
// + projection) over every encoder frame, then one step graph per chunk over
// [speaker cache | FIFO | chunk | lookahead] embeddings: 31 pre-LN RoPE
// layers -> projection -> sub-pixel upsampling -> speaker head. The host
// speaker cache (stream.cpp) carries identity across chunks. Mirrors HF
// Nemotron3DiarizationForAudioFrameClassification.forward.

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

// Embedder over the whole clip: time-major mel padded to a multiple of sub,
// viewed as [n_mels * sub, T_enc] and projected to [d, T_enc].
transcribe_status run_embedder(Session * s, Model * m, int n_mels, int n_mel_frames, int n_valid_mel, int n_enc) {
    // Fixed-size blocks keep the stacked input and its projection bounded
    // however long the clip is; the last block is zero-padded.
    constexpr int block = 4096;
    const int     sub   = m->hparams.subsampling_factor;
    const int     D     = m->hparams.d_model;
    const int     width = n_mels * sub;
    const int     rows  = std::min(block, n_enc);

    if (const transcribe_status st = new_compute_ctx(s, 1 * 1024 * 1024); st != TRANSCRIBE_OK) {
        return st;
    }
    ggml_context * ctx = s->compute_ctx;
    ggml_tensor *  in  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, rows);
    ggml_set_input(in);
    ggml_tensor * out   = ggml_mul_mat(ctx, m->weights.embed_w, in);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    ggml_backend_sched_reset(s->sched);
    if (!ggml_backend_sched_alloc_graph(s->sched, graph)) {
        return TRANSCRIBE_ERR_BACKEND;
    }
    transcribe::configure_sched_n_threads(s->sched, s->n_threads);

    s->embeds_host.resize(static_cast<size_t>(n_enc) * D);
    std::vector<float> stacked(static_cast<size_t>(rows) * width);
    for (int first = 0; first < n_enc; first += rows) {
        const int n = std::min(rows, n_enc - first);
        std::fill(stacked.begin(), stacked.end(), 0.0f);
        const int t_end = std::min((first + n) * sub, n_valid_mel);
        for (int t = first * sub; t < t_end; ++t) {
            float * dst = stacked.data() + static_cast<size_t>(t - first * sub) * n_mels;
            for (int f = 0; f < n_mels; ++f) {
                dst[f] = s->mel_buf[static_cast<size_t>(f) * n_mel_frames + t];
            }
        }
        ggml_backend_tensor_set(in, stacked.data(), 0, stacked.size() * sizeof(float));
        if (ggml_backend_sched_graph_compute(s->sched, graph) != GGML_STATUS_SUCCESS) {
            return TRANSCRIBE_ERR_BACKEND;
        }
        ggml_backend_tensor_get(out, s->embeds_host.data() + static_cast<size_t>(first) * D, 0,
                                static_cast<size_t>(n) * D * sizeof(float));
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

// Chunked AOSC/FIFO forward over the clip's embeddings; fills s->probs
// ([n_mel_frames, S]).
transcribe_status run_chunks(Session *            s,
                             Model *              m,
                             const StreamParams & p,
                             int                  n_enc,
                             int                  n_valid_enc,
                             int                  n_mel_frames) {
    const HParams & hp  = m->hparams;
    const int       D   = hp.d_model;
    const int       S   = hp.max_speakers;
    const int       sub = hp.subsampling_factor;

    s->cache.reset();
    s->probs.clear();
    s->probs.reserve(static_cast<size_t>(n_enc) * sub * S);

    for (int start = 0; start < n_enc; start += p.chunk_len) {
        if (s->poll_abort()) {
            return TRANSCRIBE_ERR_ABORTED;
        }
        const int end     = std::min(start + p.chunk_len, n_enc);
        const int n_chunk = end - start;
        const int take    = std::min(end + p.right_context, n_enc) - start;
        const int cached  = s->cache.cache_n + s->cache.fifo_n;
        const int n       = cached + take;
        const int n_valid = cached + std::clamp(n_valid_enc - start, 0, take);

        s->input_host.resize(static_cast<size_t>(n) * D);
        auto dst = s->input_host.begin();
        dst      = std::copy_n(s->cache.embeds.begin(), static_cast<size_t>(s->cache.cache_n) * D, dst);
        dst      = std::copy_n(s->cache.fifo.begin(), static_cast<size_t>(s->cache.fifo_n) * D, dst);
        std::copy_n(s->embeds_host.begin() + static_cast<size_t>(start) * D, static_cast<size_t>(take) * D, dst);

        if (const transcribe_status st = run_step(s, m, n, n_valid, false); st != TRANSCRIBE_OK) {
            return st;
        }
        update_speaker_cache(s->cache, hp, p, m->silence_host, s->input_host.data(), n, n_valid, s->logits_host.data(),
                             n_chunk);
        sigmoid_rows(s->logits_host, static_cast<size_t>(cached) * sub * S, static_cast<size_t>(n_chunk) * sub * S,
                     s->probs);
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
    return model != nullptr && slot == TRANSCRIBE_EXT_SLOT_RUN && kind == TRANSCRIBE_EXT_KIND_SORTFORMER_STREAM;
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

}  // namespace

extern const Arch arch = {
    /* .name             = */ "nemotron3_diar",
    /* .load             = */ load,
    /* .init_context     = */ init_context,
    /* .run              = */ run,
    /* .run_batch        = */ nullptr,
    /* .stream_validate  = */ nullptr,
    /* .stream_begin     = */ nullptr,
    /* .stream_feed      = */ nullptr,
    /* .stream_finalize  = */ nullptr,
    /* .stream_reset     = */ nullptr,
    /* .accepts_ext_kind = */ accepts_ext_kind,
    /* .run_validate     = */ run_validate,
};

}  // namespace transcribe::nemotron3_diar
