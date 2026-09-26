// arch/nemotron3_diar/stream.cpp - latency presets and the host AOSC speaker
// cache, ported from HF Transformers Nemotron3DiarizationSpeakerCache
// (batch 1). Frames are 80 ms encoder frames throughout.

#include "nemotron3_diar.h"
#include "transcribe-env.h"
#include "transcribe-log.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <string>

namespace transcribe::nemotron3_diar {

namespace {

struct Preset {
    const char * name;
    StreamParams p;
};

// NVIDIA's published operating points (model card / processor_config.json
// streaming_modes). `offline` mirrors the checkpoint's offline-mode config;
// `small` is a validation-only geometry that forces several chunks and cache
// compression on a short clip. Keep in sync with PRESETS in
// scripts/dump_reference_nemotron3_diar_transformers.py.
constexpr Preset k_presets[] = {
    { "offline",           { 340, 40, 40, 300, 264 } },
    { "low_latency",       { 9, 4, 264, 222, 264 }   },
    { "very_low_latency",  { 6, 2, 264, 222, 264 }   },
    { "ultra_low_latency", { 3, 1, 264, 222, 264 }   },
    { "small",             { 12, 2, 8, 6, 24 }       },
};

const StreamParams * find_preset(const char * name) {
    for (const Preset & preset : k_presets) {
        if (std::strcmp(preset.name, name) == 0) {
            return &preset.p;
        }
    }
    return nullptr;
}

constexpr float k_neg_inf = -std::numeric_limits<float>::infinity();
constexpr float k_pos_inf = std::numeric_limits<float>::infinity();

// Indices of the `k` largest scores of one speaker column (frames x S,
// column `s`), ties broken by the lower frame index.
void topk_frames(const std::vector<float> & scores, int n_frames, int n_spk, int s, int k, std::vector<int> & out) {
    out.resize(static_cast<size_t>(n_frames));
    std::iota(out.begin(), out.end(), 0);
    k = std::min(k, n_frames);
    std::partial_sort(out.begin(), out.begin() + k, out.end(), [&](int a, int b) {
        const float sa = scores[static_cast<size_t>(a) * n_spk + s];
        const float sb = scores[static_cast<size_t>(b) * n_spk + s];
        return sa > sb || (sa == sb && a < b);
    });
    out.resize(static_cast<size_t>(k));
}

// Keeps spkcache_len frames grouped by speaker, in time order within a
// speaker; sil_frames_per_spk slots per speaker hold the silence embedding.
void compress(std::vector<float> &       embeds,
              std::vector<float> &       probs,
              int &                      n_frames,
              const HParams &            hp,
              const StreamParams &       p,
              const std::vector<float> & silence) {
    const int   S        = hp.max_speakers;
    const int   D        = hp.d_model;
    const int   N        = n_frames;
    const int   cap      = p.spkcache_len;
    const int   n_sil    = hp.sil_frames_per_spk;
    const int   budget   = cap / S - n_sil;
    const int   min_pos  = static_cast<int>(std::floor(budget * hp.min_pos_scores_rate));
    const int   n_strong = static_cast<int>(std::floor(budget * hp.strong_boost_rate));
    const int   n_weak   = static_cast<int>(std::floor(budget * hp.weak_boost_rate));
    const float log_half = std::log(0.5f);
    const float thr      = hp.pred_score_threshold;

    std::vector<float> scores(static_cast<size_t>(N) * S);
    for (int t = 0; t < N; ++t) {
        const float * pr     = probs.data() + static_cast<size_t>(t) * S;
        float         lc_sum = 0.0f;
        for (int s = 0; s < S; ++s) {
            lc_sum += std::log(std::max(1.0f - pr[s], thr));
        }
        for (int s = 0; s < S; ++s) {
            const float lp                         = std::log(std::max(pr[s], thr));
            const float lc                         = std::log(std::max(1.0f - pr[s], thr));
            const float v                          = lp - lc + lc_sum - log_half;
            scores[static_cast<size_t>(t) * S + s] = pr[s] > 0.5f ? v : k_neg_inf;
        }
    }
    for (int s = 0; s < S; ++s) {
        int n_positive = 0;
        for (int t = 0; t < N; ++t) {
            n_positive += scores[static_cast<size_t>(t) * S + s] > 0.0f ? 1 : 0;
        }
        if (n_positive < min_pos) {
            continue;
        }
        for (int t = 0; t < N; ++t) {
            float & v = scores[static_cast<size_t>(t) * S + s];
            if (probs[static_cast<size_t>(t) * S + s] > 0.5f && !(v > 0.0f)) {
                v = k_neg_inf;
            }
        }
    }
    for (int t = cap; t < N; ++t) {
        for (int s = 0; s < S; ++s) {
            scores[static_cast<size_t>(t) * S + s] += hp.latest_frames_score_boost;
        }
    }
    std::vector<int> top;
    for (const auto & [count, boost] : {
             std::pair<int, float>{ n_strong, -2.0f * log_half },
              std::pair<int, float>{ n_weak,   -log_half        }
    }) {
        for (int s = 0; s < S; ++s) {
            topk_frames(scores, N, S, s, count, top);
            for (int t : top) {
                scores[static_cast<size_t>(t) * S + s] += boost;
            }
        }
    }

    // Flatten speaker-major over N frames + n_sil silence slots per speaker,
    // take the cap best, drop -inf picks to the sentinel, sort.
    const int          n_scored = N + n_sil;
    const int          sentinel = n_scored * S;
    std::vector<float> flat(static_cast<size_t>(sentinel));
    for (int s = 0; s < S; ++s) {
        for (int t = 0; t < n_scored; ++t) {
            flat[static_cast<size_t>(s) * n_scored + t] = t < N ? scores[static_cast<size_t>(t) * S + s] : k_pos_inf;
        }
    }
    std::vector<int> order(static_cast<size_t>(sentinel));
    std::iota(order.begin(), order.end(), 0);
    std::partial_sort(order.begin(), order.begin() + cap, order.end(), [&](int a, int b) {
        return flat[static_cast<size_t>(a)] > flat[static_cast<size_t>(b)] ||
               (flat[static_cast<size_t>(a)] == flat[static_cast<size_t>(b)] && a < b);
    });
    order.resize(static_cast<size_t>(cap));
    for (int & idx : order) {
        if (flat[static_cast<size_t>(idx)] == k_neg_inf) {
            idx = sentinel;
        }
    }
    std::sort(order.begin(), order.end());

    std::vector<float> new_embeds(static_cast<size_t>(cap) * D);
    std::vector<float> new_probs(static_cast<size_t>(cap) * S, 0.0f);
    for (int i = 0; i < cap; ++i) {
        const int idx   = order[static_cast<size_t>(i)];
        const int frame = idx == sentinel ? N : std::min(idx % n_scored, N);
        float *   dst   = new_embeds.data() + static_cast<size_t>(i) * D;
        if (frame == N) {
            std::copy(silence.begin(), silence.end(), dst);
        } else {
            std::copy_n(embeds.data() + static_cast<size_t>(frame) * D, D, dst);
            std::copy_n(probs.data() + static_cast<size_t>(frame) * S, S,
                        new_probs.data() + static_cast<size_t>(i) * S);
        }
    }
    embeds.swap(new_embeds);
    probs.swap(new_probs);
    n_frames = cap;
}

}  // namespace

bool resolve_stream_params(const HParams & hp, transcribe_sortformer_preset preset, StreamParams & out) {
    out = { hp.offline_chunk_len, hp.offline_right_context, hp.offline_fifo_len, hp.offline_spkcache_update_period,
            hp.spkcache_len };
    const char * name = nullptr;
    switch (preset) {
        case TRANSCRIBE_SORTFORMER_PRESET_DEFAULT:
            break;
        case TRANSCRIBE_SORTFORMER_PRESET_VERY_HIGH_LATENCY:
            name = "offline";
            break;
        case TRANSCRIBE_SORTFORMER_PRESET_LOW_LATENCY:
            name = "low_latency";
            break;
        case TRANSCRIBE_SORTFORMER_PRESET_VERY_LOW_LATENCY:
            name = "very_low_latency";
            break;
        case TRANSCRIBE_SORTFORMER_PRESET_ULTRA_LOW_LATENCY:
            name = "ultra_low_latency";
            break;
        default:
            return false;
    }
    if (const char * env = transcribe::env::str("TRANSCRIBE_NEMOTRON3_DIAR_PRESET"); env != nullptr && *env != '\0') {
        if (find_preset(env) == nullptr) {
            log_msg(TRANSCRIBE_LOG_LEVEL_WARN, "nemotron3_diar: unknown TRANSCRIBE_NEMOTRON3_DIAR_PRESET=%s", env);
        } else {
            name = env;
        }
    }
    if (name != nullptr) {
        out = *find_preset(name);
    }
    return out.chunk_len > 0 && out.right_context >= 0 && out.spkcache_len >= hp.max_speakers;
}

void update_speaker_cache(SpeakerCache &             sc,
                          const HParams &            hp,
                          const StreamParams &       p,
                          const std::vector<float> & silence,
                          const float *              input,
                          int                        n_input,
                          int                        n_valid,
                          const float *              logits,
                          int                        n_chunk) {
    const int S   = hp.max_speakers;
    const int D   = hp.d_model;
    const int sub = hp.subsampling_factor;

    // Speaker probabilities at the encoder rate: mean of the sigmoids of the
    // sub mel-rate logits of each frame, zero on padding frames.
    std::vector<float> probs(static_cast<size_t>(n_input) * S, 0.0f);
    for (int t = 0; t < std::min(n_input, n_valid); ++t) {
        for (int s = 0; s < S; ++s) {
            float acc = 0.0f;
            for (int j = 0; j < sub; ++j) {
                const float z = logits[(static_cast<size_t>(t) * sub + j) * S + s];
                acc += 1.0f / (1.0f + std::exp(-z));
            }
            probs[static_cast<size_t>(t) * S + s] = acc / static_cast<float>(sub);
        }
    }

    const int cache_n     = sc.cache_n;
    const int chunk_start = cache_n + sc.fifo_n;

    std::vector<float> fifo(sc.fifo.begin(), sc.fifo.begin() + static_cast<size_t>(sc.fifo_n) * D);
    fifo.insert(fifo.end(), input + static_cast<size_t>(chunk_start) * D,
                input + static_cast<size_t>(chunk_start + n_chunk) * D);
    int fifo_n = sc.fifo_n + n_chunk;

    int popped = 0;
    if (fifo_n > p.fifo_len) {
        popped = std::min(std::max(p.spkcache_update_period, fifo_n - p.fifo_len), fifo_n);
    }
    if (popped > 0) {
        std::vector<float> cache_embeds(sc.embeds.begin(), sc.embeds.begin() + static_cast<size_t>(cache_n) * D);
        cache_embeds.insert(cache_embeds.end(), fifo.begin(), fifo.begin() + static_cast<size_t>(popped) * D);

        // An uncompressed cache still holds plain frames whose probabilities
        // this step re-estimated; a compressed one keeps its stored probs.
        std::vector<float> cache_probs;
        if (sc.compressed) {
            cache_probs.assign(sc.probs.begin(), sc.probs.begin() + static_cast<size_t>(cache_n) * S);
        } else {
            cache_probs.assign(probs.begin(), probs.begin() + static_cast<size_t>(cache_n) * S);
        }
        cache_probs.insert(cache_probs.end(), probs.begin() + static_cast<size_t>(cache_n) * S,
                           probs.begin() + static_cast<size_t>(cache_n + popped) * S);

        int n = cache_n + popped;
        if (n > p.spkcache_len) {
            compress(cache_embeds, cache_probs, n, hp, p, silence);
            sc.compressed = true;
        }
        sc.embeds.swap(cache_embeds);
        sc.probs.swap(cache_probs);
        sc.cache_n = n;

        fifo.erase(fifo.begin(), fifo.begin() + static_cast<size_t>(popped) * D);
        fifo_n -= popped;
    }
    sc.fifo.swap(fifo);
    sc.fifo_n = fifo_n;
}

}  // namespace transcribe::nemotron3_diar
