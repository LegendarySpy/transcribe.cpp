// arch/nemotron3_diar/weights.cpp - hparam KV reader and weight catalog.
// Tensor names match scripts/convert-nemotron3_diar.py. Linear weights are
// PyTorch [out, in] -> ggml ne [in, out].

#include "ggml.h"
#include "gguf.h"
#include "nemotron3_diar.h"
#include "transcribe-log.h"

#include <cstdio>

namespace transcribe::nemotron3_diar {

namespace {

transcribe_status kv_u32(const gguf_context * g, const char * key, int32_t & out) {
    const int64_t id = gguf_find_key(g, key);
    if (id < 0) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "nemotron3_diar: missing KV %s", key);
        return TRANSCRIBE_ERR_GGUF;
    }
    out = static_cast<int32_t>(gguf_get_val_u32(g, id));
    return TRANSCRIBE_OK;
}

transcribe_status kv_f32(const gguf_context * g, const char * key, float & out) {
    const int64_t id = gguf_find_key(g, key);
    if (id < 0) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "nemotron3_diar: missing KV %s", key);
        return TRANSCRIBE_ERR_GGUF;
    }
    out = gguf_get_val_f32(g, id);
    return TRANSCRIBE_OK;
}

}  // namespace

transcribe_status read_hparams(const gguf_context * g, HParams & hp) {
    if (g == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
#define RD_U32(key, field)                                                          \
    if (const transcribe_status st = kv_u32(g, key, hp.field); st != TRANSCRIBE_OK) \
    return st
#define RD_F32(key, field)                                                          \
    if (const transcribe_status st = kv_f32(g, key, hp.field); st != TRANSCRIBE_OK) \
    return st

    RD_U32("stt.nemotron3_diar.max_speakers", max_speakers);
    RD_U32("stt.nemotron3_diar.frame_hop", frame_hop);
    RD_U32("stt.nemotron3_diar.subsampling_factor", subsampling_factor);
    RD_U32("stt.nemotron3_diar.encoder.n_layers", n_layers);
    RD_U32("stt.nemotron3_diar.encoder.d_model", d_model);
    RD_U32("stt.nemotron3_diar.encoder.n_heads", n_heads);
    RD_U32("stt.nemotron3_diar.encoder.d_ff", d_ff);
    RD_F32("stt.nemotron3_diar.encoder.rope_theta", rope_theta);
    RD_U32("stt.nemotron3_diar.head.d_model", head_d);

    RD_U32("stt.frontend.num_mels", fe_num_mels);
    RD_U32("stt.frontend.sample_rate", fe_sample_rate);
    RD_U32("stt.frontend.n_fft", fe_n_fft);
    RD_U32("stt.frontend.win_length", fe_win_length);
    RD_U32("stt.frontend.hop_length", fe_hop_length);
    RD_F32("stt.frontend.pre_emphasis", fe_pre_emphasis);

    RD_U32("stt.nemotron3_diar.stream.offline.chunk_len", offline_chunk_len);
    RD_U32("stt.nemotron3_diar.stream.offline.right_context", offline_right_context);
    RD_U32("stt.nemotron3_diar.stream.offline.fifo_len", offline_fifo_len);
    RD_U32("stt.nemotron3_diar.stream.offline.spkcache_update_period", offline_spkcache_update_period);
    RD_U32("stt.nemotron3_diar.stream.fifo_len", stream_fifo_len);
    RD_U32("stt.nemotron3_diar.stream.spkcache_update_period", stream_spkcache_update_period);
    RD_U32("stt.nemotron3_diar.stream.spkcache_len", spkcache_len);
    RD_U32("stt.nemotron3_diar.stream.sil_frames_per_spk", sil_frames_per_spk);
    RD_F32("stt.nemotron3_diar.stream.pred_score_threshold", pred_score_threshold);
    RD_F32("stt.nemotron3_diar.stream.latest_frames_score_boost", latest_frames_score_boost);
    RD_F32("stt.nemotron3_diar.stream.strong_boost_rate", strong_boost_rate);
    RD_F32("stt.nemotron3_diar.stream.weak_boost_rate", weak_boost_rate);
    RD_F32("stt.nemotron3_diar.stream.min_pos_scores_rate", min_pos_scores_rate);
#undef RD_U32
#undef RD_F32

    if (hp.n_layers <= 0 || hp.d_model <= 0 || hp.n_heads <= 0 || hp.d_model % hp.n_heads != 0 || hp.head_d <= 0 ||
        hp.max_speakers <= 0 || hp.subsampling_factor <= 0 || hp.spkcache_len < hp.max_speakers ||
        hp.sil_frames_per_spk < 0) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "nemotron3_diar: inconsistent hparams");
        return TRANSCRIBE_ERR_GGUF;
    }
    return TRANSCRIBE_OK;
}

transcribe_status build_weights(ggml_context * ctx, const HParams & hp, Weights & w) {
    auto get = [&](const char * name, int64_t ne0, int64_t ne1) -> ggml_tensor * {
        ggml_tensor * t = ggml_get_tensor(ctx, name);
        if (t == nullptr) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "nemotron3_diar: missing tensor %s", name);
            return nullptr;
        }
        const int64_t want1 = ne1 > 0 ? ne1 : 1;
        if (t->ne[0] != ne0 || t->ne[1] != want1) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                    "nemotron3_diar: tensor %s has shape [%lld, %lld], expected [%lld, %lld]", name,
                    static_cast<long long>(t->ne[0]), static_cast<long long>(t->ne[1]), static_cast<long long>(ne0),
                    static_cast<long long>(want1));
            return nullptr;
        }
        return t;
    };
#define G(dst, name, ne0, ne1)             \
    do {                                   \
        (dst) = get((name), (ne0), (ne1)); \
        if ((dst) == nullptr)              \
            return TRANSCRIBE_ERR_GGUF;    \
    } while (0)

    const int64_t d   = hp.d_model;
    const int64_t hd  = hp.head_d;
    const int64_t sub = hp.subsampling_factor;

    G(w.embed_w, "enc.embed.proj.weight", hp.fe_num_mels * sub, d);
    G(w.norm_in_w, "enc.norm_in.weight", d, 0);
    G(w.norm_in_b, "enc.norm_in.bias", d, 0);
    G(w.norm_out_w, "enc.norm_out.weight", d, 0);
    G(w.norm_out_b, "enc.norm_out.bias", d, 0);

    w.blocks.resize(static_cast<size_t>(hp.n_layers));
    char name[128];
    for (int i = 0; i < hp.n_layers; ++i) {
        Block & b = w.blocks[static_cast<size_t>(i)];
#define GB(dst, suffix, ne0, ne1)                                           \
    do {                                                                    \
        std::snprintf(name, sizeof(name), "enc.blocks.%d.%s", i, (suffix)); \
        G(dst, name, ne0, ne1);                                             \
    } while (0)
        GB(b.norm_attn_w, "norm_attn.weight", d, 0);
        GB(b.norm_attn_b, "norm_attn.bias", d, 0);
        GB(b.q_w, "attn.q.weight", d, d);
        GB(b.k_w, "attn.k.weight", d, d);
        GB(b.v_w, "attn.v.weight", d, d);
        GB(b.out_w, "attn.out.weight", d, d);
        GB(b.out_b, "attn.out.bias", d, 0);
        GB(b.norm_ff_w, "norm_ff.weight", d, 0);
        GB(b.norm_ff_b, "norm_ff.bias", d, 0);
        GB(b.ff_in_w, "ff.in.weight", d, hp.d_ff);
        GB(b.ff_in_b, "ff.in.bias", hp.d_ff, 0);
        GB(b.ff_out_w, "ff.out.weight", hp.d_ff, d);
        GB(b.ff_out_b, "ff.out.bias", d, 0);
#undef GB
    }

    G(w.proj_w, "diar.proj.weight", d, hd);
    G(w.proj_b, "diar.proj.bias", hd, 0);
    for (int k = 0; k < 3; ++k) {
        std::snprintf(name, sizeof(name), "diar.upsample.tap%d.weight", k);
        G(w.up_w[k], name, hd, hd * sub);
    }
    G(w.up_b, "diar.upsample.bias", hd * sub, 0);
    G(w.dense_w, "diar.head.dense.weight", hd, hd);
    G(w.dense_b, "diar.head.dense.bias", hd, 0);
    G(w.out_w, "diar.head.out.weight", hd, hp.max_speakers);
    G(w.out_b, "diar.head.out.bias", hp.max_speakers, 0);
    G(w.silence, "diar.silence_emb", d, 0);
#undef G

    if (w.silence->type != GGML_TYPE_F32) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "nemotron3_diar: diar.silence_emb must be F32");
        return TRANSCRIBE_ERR_GGUF;
    }
    return TRANSCRIBE_OK;
}

}  // namespace transcribe::nemotron3_diar
