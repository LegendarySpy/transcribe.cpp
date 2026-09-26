// arch/whisper/alignment.cpp - Whisper word timestamps (host math).
//
// Port of openai/whisper timing.py (find_alignment, median_filter, dtw_cpu,
// backtrace, merge_punctuations, add_word_timestamps) and tokenizer.py
// (split_to_word_tokens). Deviations are marked DEVIATION.

#include "alignment.h"

#include "transcribe-tokenizer.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string_view>

namespace transcribe::whisper {

namespace {

// OpenAI whisper/__init__.py _ALIGNMENT_HEADS (== whisper.cpp g_aheads_*), flat
// [layer, head, layer, head, ...].
constexpr int8_t k_aheads_tiny_en[]   = { 1, 0, 2, 0, 2, 5, 3, 0, 3, 1, 3, 2, 3, 3, 3, 4 };
constexpr int8_t k_aheads_tiny[]      = { 2, 2, 3, 0, 3, 2, 3, 3, 3, 4, 3, 5 };
constexpr int8_t k_aheads_base_en[]   = { 3, 3, 4, 7, 5, 1, 5, 5, 5, 7 };
constexpr int8_t k_aheads_base[]      = { 3, 1, 4, 2, 4, 3, 4, 7, 5, 1, 5, 2, 5, 4, 5, 6 };
constexpr int8_t k_aheads_small_en[]  = { 6, 6, 7,  0,  7, 3,  7, 8,  8, 2,  8, 5,  8, 7,  9,  0,  9, 4,  9,
                                          8, 9, 10, 10, 0, 10, 1, 10, 2, 10, 3, 10, 6, 10, 11, 11, 2, 11, 4 };
constexpr int8_t k_aheads_small[]     = { 5, 3, 5, 9, 8, 0, 8, 4, 8, 7, 8, 8, 9, 0, 9, 7, 9, 9, 10, 5 };
constexpr int8_t k_aheads_medium_en[] = { 11, 4,  14, 1, 14, 12, 14, 14, 15, 4, 16, 0, 16, 4, 16, 9,  17, 12,
                                          17, 14, 18, 7, 18, 10, 18, 15, 20, 0, 20, 3, 20, 9, 20, 14, 21, 12 };
constexpr int8_t k_aheads_medium[]    = { 13, 15, 15, 4, 15, 15, 16, 1, 20, 0, 23, 4 };
constexpr int8_t k_aheads_large_v1[]  = { 9, 19, 11, 2, 11, 4, 11, 17, 22, 7, 22, 11, 22, 17, 23, 2, 23, 15 };
constexpr int8_t k_aheads_large_v2[]  = { 10, 12, 13, 17, 16, 11, 16, 12, 16, 13, 17, 15, 17, 16, 18, 4,
                                          18, 11, 18, 19, 19, 11, 21, 2,  21, 3,  22, 3,  22, 9,  22, 12,
                                          23, 5,  23, 7,  23, 13, 25, 5,  26, 1,  26, 12, 27, 15 };
constexpr int8_t k_aheads_large_v3[]  = { 7, 0, 10, 17, 12, 18, 13, 12, 16, 1, 17, 14, 19, 11, 21, 4, 24, 1, 25, 6 };
constexpr int8_t k_aheads_large_v3_turbo[]  = { 2, 4, 2, 11, 3, 3, 3, 6, 3, 11, 3, 14 };
// HF distil-whisper/distil-large-v3 and distil-large-v3.5 generation_config.json.
constexpr int8_t k_aheads_distil_large_v3[] = { 1, 0,  1, 1,  1, 2,  1, 3,  1, 4,  1, 5,  1, 6,  1, 7,  1, 8,  1, 9,
                                                1, 10, 1, 11, 1, 12, 1, 13, 1, 14, 1, 15, 1, 16, 1, 17, 1, 18, 1, 19 };

struct AheadPreset {
    const char *   name;
    int            enc_n_layers;
    int            dec_n_layers;
    int            dec_n_heads;
    int            n_mels;
    bool           multilingual;
    const int8_t * heads;  // flat pairs
    int            n_heads;
};

template <size_t N> constexpr int count_of(const int8_t (&)[N]) {
    return static_cast<int>(N / 2);
}

#define TRANSCRIBE_AHEADS(a) a, count_of(a)
// large-v1 and large-v2 share geometry; v2 is the default (see resolve).
constexpr AheadPreset k_ahead_presets[] = {
    { "tiny.en",         4,  4,  6,  80,  false, TRANSCRIBE_AHEADS(k_aheads_tiny_en)         },
    { "tiny",            4,  4,  6,  80,  true,  TRANSCRIBE_AHEADS(k_aheads_tiny)            },
    { "base.en",         6,  6,  8,  80,  false, TRANSCRIBE_AHEADS(k_aheads_base_en)         },
    { "base",            6,  6,  8,  80,  true,  TRANSCRIBE_AHEADS(k_aheads_base)            },
    { "small.en",        12, 12, 12, 80,  false, TRANSCRIBE_AHEADS(k_aheads_small_en)        },
    { "small",           12, 12, 12, 80,  true,  TRANSCRIBE_AHEADS(k_aheads_small)           },
    { "medium.en",       24, 24, 16, 80,  false, TRANSCRIBE_AHEADS(k_aheads_medium_en)       },
    { "medium",          24, 24, 16, 80,  true,  TRANSCRIBE_AHEADS(k_aheads_medium)          },
    { "large-v2",        32, 32, 20, 80,  true,  TRANSCRIBE_AHEADS(k_aheads_large_v2)        },
    { "large-v3",        32, 32, 20, 128, true,  TRANSCRIBE_AHEADS(k_aheads_large_v3)        },
    { "large-v3-turbo",  32, 4,  20, 128, true,  TRANSCRIBE_AHEADS(k_aheads_large_v3_turbo)  },
    { "distil-large-v3", 32, 2,  20, 128, true,  TRANSCRIBE_AHEADS(k_aheads_distil_large_v3) },
};
#undef TRANSCRIBE_AHEADS

std::vector<AlignHead> to_heads(const int8_t * p, int n) {
    std::vector<AlignHead> out;
    out.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        out.push_back({ p[2 * i], p[2 * i + 1] });
    }
    return out;
}

// Sorts and validates in place: non-empty, in range, no duplicates.
bool validate_heads(std::vector<AlignHead> & heads, const AlignGeometry & g) {
    if (heads.empty()) {
        return false;
    }
    for (const AlignHead & h : heads) {
        if (h.layer < 0 || h.layer >= g.dec_n_layers || h.head < 0 || h.head >= g.dec_n_heads) {
            return false;
        }
    }
    std::sort(heads.begin(), heads.end(), [](const AlignHead & a, const AlignHead & b) {
        return a.layer != b.layer ? a.layer < b.layer : a.head < b.head;
    });
    for (size_t i = 1; i < heads.size(); ++i) {
        if (heads[i].layer == heads[i - 1].layer && heads[i].head == heads[i - 1].head) {
            return false;
        }
    }
    return true;
}

}  // namespace

std::vector<AlignHead> resolve_alignment_heads(const AlignGeometry &        geom,
                                               const std::string &          variant,
                                               const std::vector<int32_t> & kv_heads,
                                               std::string &                source) {
    source.clear();
    if (geom.dec_n_layers <= 0 || geom.dec_n_heads <= 0) {
        source = "none";
        return {};
    }

    if (!kv_heads.empty() && kv_heads.size() % 2 == 0) {
        std::vector<AlignHead> heads;
        for (size_t i = 0; i + 1 < kv_heads.size(); i += 2) {
            heads.push_back({ kv_heads[i], kv_heads[i + 1] });
        }
        if (validate_heads(heads, geom)) {
            source = "gguf";
            return heads;
        }
    }

    if (variant == "whisper-large-v1" && geom.enc_n_layers == 32 && geom.dec_n_layers == 32 && geom.n_mels == 80) {
        std::vector<AlignHead> heads = to_heads(k_aheads_large_v1, count_of(k_aheads_large_v1));
        if (validate_heads(heads, geom)) {
            source = "table:large-v1";
            return heads;
        }
    }

    for (const AheadPreset & p : k_ahead_presets) {
        if (p.enc_n_layers == geom.enc_n_layers && p.dec_n_layers == geom.dec_n_layers &&
            p.dec_n_heads == geom.dec_n_heads && p.n_mels == geom.n_mels && p.multilingual == geom.multilingual) {
            std::vector<AlignHead> heads = to_heads(p.heads, p.n_heads);
            if (validate_heads(heads, geom)) {
                source = std::string("table:") + p.name;
                return heads;
            }
        }
    }

    // OpenAI model.py default: every head of the upper half of the decoder.
    std::vector<AlignHead> heads;
    for (int l = geom.dec_n_layers / 2; l < geom.dec_n_layers; ++l) {
        for (int h = 0; h < geom.dec_n_heads; ++h) {
            heads.push_back({ l, h });
        }
    }
    source = "heuristic";
    return heads;
}

namespace align {

namespace {

constexpr int        k_medfilt_pad       = 3;  // width 7
constexpr double     k_seconds_per_frame = 0.02;
constexpr const char k_prepended[]       = "\"'\xe2\x80\x9c\xc2\xbf([{-";
constexpr const char k_appended[] =
    "\"'.\xe3\x80\x82,\xef\xbc\x8c!\xef\xbc\x81?\xef\xbc\x9f:\xef\xbc\x9a\xe2\x80\x9d)]}\xe3\x80\x81";
constexpr const char k_sentence_end[] = ".\xe3\x80\x82!\xef\xbc\x81?\xef\xbc\x9f";
constexpr const char k_ascii_punct[]  = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";

// Python `s in set` on strings: substring test, "" is in everything.
bool py_in(std::string_view set, std::string_view s) {
    return set.find(s) != std::string_view::npos;
}

bool is_space(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f';
}

std::string_view py_strip(std::string_view s) {
    size_t a = 0;
    size_t b = s.size();
    while (a < b && is_space(s[a])) {
        ++a;
    }
    while (b > a && is_space(s[b - 1])) {
        --b;
    }
    return s.substr(a, b - a);
}

// True unless the bytes end inside a multi-byte UTF-8 sequence. Invalid
// bytes count as complete (OpenAI keeps a U+FFFD that the full decode has).
bool utf8_complete(const std::string & b) {
    const size_t n = b.size();
    if (n == 0) {
        return true;
    }
    size_t pos   = n;
    int    steps = 0;
    while (pos > 0 && steps < 4) {
        --pos;
        ++steps;
        const unsigned char c = static_cast<unsigned char>(b[pos]);
        if ((c & 0xC0) != 0x80) {
            size_t need = 1;
            if (c >= 0xF0 && c <= 0xF7) {
                need = 4;
            } else if (c >= 0xE0) {
                need = 3;
            } else if (c >= 0xC0) {
                need = 2;
            }
            if (c >= 0xF8) {
                return true;
            }
            return n - pos >= need;
        }
    }
    return true;
}

size_t utf8_codepoints(std::string_view s) {
    size_t n = 0;
    for (char ch : s) {
        if ((static_cast<unsigned char>(ch) & 0xC0) != 0x80) {
            ++n;
        }
    }
    return n;
}

int reflect_index(int k, int T) {
    for (int guard = 0; guard < 4; ++guard) {
        if (k < 0) {
            k = -k;
        } else if (k >= T) {
            k = 2 * (T - 1) - k;
        } else {
            break;
        }
    }
    return std::clamp(k, 0, T - 1);
}

double median_of(std::vector<double> v) {
    if (v.empty()) {
        return 0.0;
    }
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return (n % 2 == 1) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

int64_t to_ms(double seconds) {
    if (!std::isfinite(seconds)) {
        return 0;
    }
    const double ms = std::clamp(seconds * 1000.0, -9.0e15, 9.0e15);
    return static_cast<int64_t>(std::llround(ms));
}

}  // namespace

void median_filter_7(const float * in, float * out, int T) {
    if (in == nullptr || out == nullptr || T <= 0) {
        return;
    }
    if (T <= k_medfilt_pad) {
        std::copy(in, in + T, out);
        return;
    }
    float w[2 * k_medfilt_pad + 1];
    for (int t = 0; t < T; ++t) {
        for (int o = -k_medfilt_pad; o <= k_medfilt_pad; ++o) {
            w[o + k_medfilt_pad] = in[reflect_index(t + o, T)];
        }
        // Insertion sort of 7; NaN cannot reach here (inputs are validated).
        for (int i = 1; i < 2 * k_medfilt_pad + 1; ++i) {
            const float v = w[i];
            int         j = i - 1;
            while (j >= 0 && w[j] > v) {
                w[j + 1] = w[j];
                --j;
            }
            w[j + 1] = v;
        }
        out[t] = w[k_medfilt_pad];
    }
}

bool alignment_matrix(const WindowInput & in, Scratch & s, std::vector<float> & matrix) {
    const int H = in.n_heads;
    const int R = in.n_rows;
    const int T = in.n_frames;
    const int N = R - in.sot_len - 1;
    if (in.qk == nullptr || H <= 0 || T <= 0 || in.sot_len < 0 || N < 1 || !std::isfinite(in.qk_scale)) {
        return false;
    }
    const size_t plane = static_cast<size_t>(R) * static_cast<size_t>(T);
    s.weights.assign(in.qk, in.qk + static_cast<size_t>(H) * plane);

    // Softmax over the real frames (timing.py slices before the softmax).
    for (int h = 0; h < H; ++h) {
        for (int r = 0; r < R; ++r) {
            float * w = s.weights.data() + static_cast<size_t>(h) * plane + static_cast<size_t>(r) * T;
            float   m = -std::numeric_limits<float>::infinity();
            for (int t = 0; t < T; ++t) {
                const float v = w[t] * in.qk_scale;
                if (!std::isfinite(v)) {
                    return false;
                }
                w[t] = v;
                m    = std::max(m, v);
            }
            double sum = 0.0;
            for (int t = 0; t < T; ++t) {
                w[t] = std::exp(w[t] - m);
                sum += w[t];
            }
            if (!(sum > 0.0) || !std::isfinite(sum)) {
                return false;
            }
            const float inv = static_cast<float>(1.0 / sum);
            for (int t = 0; t < T; ++t) {
                w[t] *= inv;
            }
        }
    }

    // std/mean over rows (unbiased=False). DEVIATION: a zero-variance column
    // becomes 0 instead of NaN.
    for (int h = 0; h < H; ++h) {
        float * base = s.weights.data() + static_cast<size_t>(h) * plane;
        for (int t = 0; t < T; ++t) {
            double mean = 0.0;
            for (int r = 0; r < R; ++r) {
                mean += base[static_cast<size_t>(r) * T + t];
            }
            mean /= R;
            double var = 0.0;
            for (int r = 0; r < R; ++r) {
                const double d = base[static_cast<size_t>(r) * T + t] - mean;
                var += d * d;
            }
            const double sd = std::sqrt(var / R);
            for (int r = 0; r < R; ++r) {
                float & v = base[static_cast<size_t>(r) * T + t];
                v         = sd < 1e-12 ? 0.0f : static_cast<float>((v - mean) / sd);
            }
        }
    }

    // Median filter the kept rows only (the filter is per row), head mean.
    matrix.assign(static_cast<size_t>(N) * T, 0.0f);
    s.row.resize(static_cast<size_t>(T));
    const float inv_h = 1.0f / static_cast<float>(H);
    for (int h = 0; h < H; ++h) {
        for (int k = 0; k < N; ++k) {
            const int     r   = in.sot_len + k;
            const float * src = s.weights.data() + static_cast<size_t>(h) * plane + static_cast<size_t>(r) * T;
            median_filter_7(src, s.row.data(), T);
            float * dst = matrix.data() + static_cast<size_t>(k) * T;
            for (int t = 0; t < T; ++t) {
                dst[t] += s.row[t] * inv_h;
            }
        }
    }
    return true;
}

bool dtw_path(const float * x, int N, int T, Scratch & s, std::vector<int> & text_idx, std::vector<int> & time_idx) {
    text_idx.clear();
    time_idx.clear();
    if (x == nullptr || N < 1 || T < 1) {
        return false;
    }
    const size_t W = static_cast<size_t>(T) + 1;
    const size_t n = (static_cast<size_t>(N) + 1) * W;
    s.cost.assign(n, std::numeric_limits<float>::infinity());
    s.trace.assign(n, static_cast<int8_t>(-1));
    s.cost[0] = 0.0f;
    for (int j = 1; j <= T; ++j) {
        for (int i = 1; i <= N; ++i) {
            const float c0 = s.cost[(i - 1) * W + (j - 1)];
            const float c1 = s.cost[(i - 1) * W + j];
            const float c2 = s.cost[i * W + (j - 1)];
            float       c;
            int8_t      t;
            if (c0 < c1 && c0 < c2) {
                c = c0;
                t = 0;
            } else if (c1 < c0 && c1 < c2) {
                c = c1;
                t = 1;
            } else {
                c = c2;
                t = 2;
            }
            // OpenAI: float64 x plus float32 cost, stored as float32.
            s.cost[i * W + j]  = static_cast<float>(static_cast<double>(x[(i - 1) * T + (j - 1)]) + c);
            s.trace[i * W + j] = t;
        }
    }
    for (size_t j = 0; j < W; ++j) {
        s.trace[j] = 2;
    }
    for (int i = 0; i <= N; ++i) {
        s.trace[i * W] = 1;
    }

    int          i       = N;
    int          j       = T;
    const size_t max_len = static_cast<size_t>(N) + static_cast<size_t>(T);
    while (i > 0 || j > 0) {
        if (text_idx.size() >= max_len) {
            return false;
        }
        text_idx.push_back(i - 1);
        time_idx.push_back(j - 1);
        switch (s.trace[static_cast<size_t>(i) * W + static_cast<size_t>(j)]) {
            case 0:
                --i;
                --j;
                break;
            case 1:
                --i;
                break;
            case 2:
                --j;
                break;
            default:
                return false;
        }
        if (i < 0 || j < 0) {
            return false;
        }
    }
    std::reverse(text_idx.begin(), text_idx.end());
    std::reverse(time_idx.begin(), time_idx.end());
    return true;
}

bool jump_frames(const std::vector<int> & text_idx,
                 const std::vector<int> & time_idx,
                 int                      N,
                 int                      T,
                 std::vector<int> &       out) {
    out.clear();
    if (text_idx.size() != time_idx.size() || text_idx.empty()) {
        return false;
    }
    for (size_t k = 0; k < text_idx.size(); ++k) {
        if (k == 0 || text_idx[k] != text_idx[k - 1]) {
            if (time_idx[k] < 0 || time_idx[k] >= T || text_idx[k] < 0 || text_idx[k] >= N) {
                return false;
            }
            out.push_back(time_idx[k]);
        }
    }
    return static_cast<int>(out.size()) == N;
}

bool is_no_space_language(const std::string & code) {
    return code == "zh" || code == "ja" || code == "th" || code == "lo" || code == "my" || code == "yue";
}

std::vector<Word> split_words(const Tokenizer & tok, const std::vector<int32_t> & text_ids, bool space_split) {
    // split_tokens_on_unicode. DEVIATION: an 8-token cap and a tail flush keep
    // every token accounted for (OpenAI drops a never-completing tail).
    std::vector<Word> subwords;
    Word              cur;
    for (size_t i = 0; i < text_ids.size(); ++i) {
        const int id = text_ids[i];
        if (cur.n_tokens == 0) {
            cur.first_token = static_cast<int>(i);
        }
        cur.n_tokens += 1;
        cur.text += tok.decode(&id, 1);
        if (utf8_complete(cur.text) || cur.n_tokens >= 8) {
            subwords.push_back(std::move(cur));
            cur = Word{};
        }
    }
    if (cur.n_tokens > 0) {
        subwords.push_back(std::move(cur));
    }
    if (!space_split) {
        return subwords;
    }

    std::vector<Word> words;
    for (Word & sw : subwords) {
        const bool with_space  = !sw.text.empty() && sw.text[0] == ' ';
        const bool punctuation = py_in(k_ascii_punct, py_strip(sw.text));
        if (with_space || punctuation || words.empty()) {
            words.push_back(std::move(sw));
        } else {
            words.back().text += sw.text;
            words.back().n_tokens += sw.n_tokens;
        }
    }
    return words;
}

double apply_duration_heuristics(std::vector<Word> & words, double & max_duration) {
    std::vector<double> durations;
    for (const Word & w : words) {
        const double d = w.end - w.start;
        if (d != 0.0) {
            durations.push_back(d);
        }
    }
    const double median_duration = std::min(0.7, median_of(durations));
    max_duration                 = median_duration * 2.0;
    if (!durations.empty()) {
        for (size_t i = 1; i < words.size(); ++i) {
            if (words[i].end - words[i].start > max_duration) {
                if (py_in(k_sentence_end, words[i].text)) {
                    words[i].end = words[i].start + max_duration;
                } else if (py_in(k_sentence_end, words[i - 1].text)) {
                    words[i].start = words[i].end - max_duration;
                }
            }
        }
    }
    return median_duration;
}

void merge_punctuations(std::vector<Word> & words) {
    const int n = static_cast<int>(words.size());
    if (n < 2) {
        return;
    }
    int i = n - 2;
    int j = n - 1;
    while (i >= 0) {
        Word & prev = words[static_cast<size_t>(i)];
        Word & next = words[static_cast<size_t>(j)];
        if (!prev.text.empty() && prev.text[0] == ' ' && py_in(k_prepended, py_strip(prev.text))) {
            next.text = prev.text + next.text;
            if (prev.n_tokens > 0) {
                next.first_token = prev.first_token;
            }
            next.n_tokens += prev.n_tokens;
            prev.text.clear();
            prev.n_tokens = 0;
        } else {
            j = i;
        }
        --i;
    }
    i = 0;
    j = 1;
    while (j < n) {
        Word & prev = words[static_cast<size_t>(i)];
        Word & next = words[static_cast<size_t>(j)];
        if ((prev.text.empty() || prev.text.back() != ' ') && py_in(k_appended, next.text)) {
            prev.text += next.text;
            if (prev.n_tokens == 0) {
                prev.first_token = next.first_token;
            }
            prev.n_tokens += next.n_tokens;
            next.text.clear();
            next.n_tokens = 0;
        } else {
            i = j;
        }
        ++j;
    }
}

void join_unspaced_words(std::vector<Word> & words) {
    int prev = -1;
    for (size_t i = 0; i < words.size(); ++i) {
        Word & w = words[i];
        if (w.text.empty()) {
            continue;
        }
        if (prev >= 0 && !is_space(w.text[0])) {
            Word & p = words[static_cast<size_t>(prev)];
            p.text += w.text;
            p.n_tokens += w.n_tokens;
            p.end = std::max(p.end, w.end);
            w.text.clear();
            w.n_tokens = 0;
            continue;
        }
        prev = static_cast<int>(i);
    }
}

void trim_silent_edges(std::vector<double> & starts,
                       std::vector<double> & ends,
                       const float *         frame_db,
                       int                   n_frames,
                       double                offset_s) {
    constexpr float k_min_range  = 12.0f;
    // (speech frames at most, then silent frames at least): an edge blip
    // this short before a pause this long belongs to the neighbor word.
    constexpr int   k_blip[2][2] = {
        { 4,  3  },
        { 15, 10 }
    };
    if (frame_db == nullptr || n_frames < 4) {
        return;
    }
    std::vector<float> sorted(frame_db, frame_db + n_frames);
    for (float v : sorted) {
        if (!std::isfinite(v)) {
            return;
        }
    }
    std::sort(sorted.begin(), sorted.end());
    const float floor_db = sorted[static_cast<size_t>(0.10 * (n_frames - 1))];
    const float peak_db  = sorted[static_cast<size_t>(0.99 * (n_frames - 1))];
    if (peak_db - floor_db < k_min_range) {
        return;
    }
    const float thr = floor_db + 0.3f * (peak_db - floor_db);

    struct Run {
        bool speech;
        int  begin;
        int  end;
    };

    auto len = [](const Run & r) {
        return r.end - r.begin;
    };
    auto is_blip = [&](const Run & speech, const Run & silence) {
        for (const auto & t : k_blip) {
            if (len(speech) <= t[0] && len(silence) >= t[1]) {
                return true;
            }
        }
        return false;
    };

    std::vector<Run> runs;
    const size_t     n = std::min(starts.size(), ends.size());
    for (size_t i = 0; i < n; ++i) {
        const double fa = std::round((starts[i] - offset_s) / k_seconds_per_frame);
        const double fb = std::round((ends[i] - offset_s) / k_seconds_per_frame);
        if (!std::isfinite(fa) || !std::isfinite(fb)) {
            continue;
        }
        const int a = static_cast<int>(std::clamp(fa, 0.0, static_cast<double>(n_frames)));
        const int b = static_cast<int>(std::clamp(fb, 0.0, static_cast<double>(n_frames)));
        runs.clear();
        bool any_speech = false;
        for (int f = a; f < b; ++f) {
            const bool sp = frame_db[f] > thr;
            any_speech    = any_speech || sp;
            if (runs.empty() || runs.back().speech != sp) {
                runs.push_back({ sp, f, f + 1 });
            } else {
                runs.back().end = f + 1;
            }
        }
        if (!any_speech) {
            continue;  // no speech inside: leave the DTW times
        }
        // Drop silence and one blip+pause pair from each edge, never the last
        // speech run.
        size_t lo   = 0;
        size_t hi   = runs.size();  // exclusive
        bool   blip = false;
        for (;;) {
            if (!blip && hi - lo >= 3 && runs[lo].speech && !runs[lo + 1].speech && is_blip(runs[lo], runs[lo + 1])) {
                lo += 2;
                blip = true;
            } else if (hi - lo >= 2 && !runs[lo].speech) {
                lo += 1;
            } else {
                break;
            }
        }
        blip = false;
        for (;;) {
            if (!blip && hi - lo >= 3 && runs[hi - 1].speech && !runs[hi - 2].speech &&
                is_blip(runs[hi - 1], runs[hi - 2])) {
                hi -= 2;
                blip = true;
            } else if (hi - lo >= 2 && !runs[hi - 1].speech) {
                hi -= 1;
            } else {
                break;
            }
        }
        if (runs[lo].begin > a) {
            starts[i] = offset_s + runs[lo].begin * k_seconds_per_frame;
        }
        if (runs[hi - 1].end < b) {
            ends[i] = offset_s + runs[hi - 1].end * k_seconds_per_frame;
        }
    }
}

void apply_segment_heuristics(std::vector<double> & starts,
                              std::vector<double> & ends,
                              double &              seg_start,
                              double &              seg_end,
                              double                median_duration,
                              double                max_duration,
                              double &              last_speech) {
    const size_t n = std::min(starts.size(), ends.size());
    if (n == 0) {
        return;
    }
    if (ends[0] - last_speech > median_duration * 4 &&
        (ends[0] - starts[0] > max_duration || (n > 1 && ends[1] - starts[0] > max_duration * 2))) {
        if (n > 1 && ends[1] - starts[1] > max_duration) {
            // OpenAI's literal expression (halves an absolute time).
            const double boundary = std::max(ends[1] / 2, ends[1] - max_duration);
            ends[0]               = boundary;
            starts[1]             = boundary;
        }
        starts[0] = std::max(0.0, ends[0] - max_duration);
    }
    if (seg_start < ends[0] && seg_start - 0.5 > starts[0]) {
        starts[0] = std::max(0.0, std::min(ends[0] - median_duration, seg_start));
    } else {
        seg_start = starts[0];
    }
    if (seg_end > starts[n - 1] && seg_end + 0.5 < ends[n - 1]) {
        ends[n - 1] = std::max(starts[n - 1] + median_duration, seg_end);
    } else {
        seg_end = ends[n - 1];
    }
    last_speech = seg_end;
}

namespace {

// Word -> segment by the global index of its first token. DEVIATION: OpenAI
// counts tokens per segment, which drifts when a word straddles segments.
int segment_of(int first_token, const std::vector<Segment> & segs) {
    int end = 0;
    for (size_t k = 0; k < segs.size(); ++k) {
        end += std::max(0, segs[k].n_text);
        if (first_token < end) {
            return static_cast<int>(k);
        }
    }
    return segs.empty() ? -1 : static_cast<int>(segs.size()) - 1;
}

struct Placed {
    size_t word;
    int    seg;
};

bool dtw_word_times(const WindowInput & in, std::vector<Word> & words, int n_text, Scratch & s) {
    if (in.qk == nullptr || in.n_rows - in.sot_len - 1 != n_text + 1) {
        return false;
    }
    int cursor = 0;
    for (const Word & w : words) {
        if (w.first_token != cursor || w.n_tokens <= 0) {
            return false;
        }
        cursor += w.n_tokens;
    }
    if (cursor != n_text) {
        return false;
    }
    const int N = n_text + 1;
    if (!alignment_matrix(in, s, s.matrix)) {
        return false;
    }
    for (float & v : s.matrix) {
        v = -v;
    }
    if (!dtw_path(s.matrix.data(), N, in.n_frames, s, s.text_idx, s.time_idx)) {
        return false;
    }
    if (!jump_frames(s.text_idx, s.time_idx, N, in.n_frames, s.jumps)) {
        return false;
    }
    for (Word & w : words) {
        w.start = s.jumps[static_cast<size_t>(w.first_token)] * k_seconds_per_frame;
        w.end   = s.jumps[static_cast<size_t>(w.first_token + w.n_tokens)] * k_seconds_per_frame;
    }
    return true;
}

}  // namespace

bool compute_window_words(const WindowInput &    in,
                          std::vector<Word>      words,
                          std::vector<Segment> & segs,
                          RunState &             st,
                          Scratch &              s,
                          int64_t *              prev_word_t1,
                          std::vector<OutWord> & out) {
    out.clear();
    for (Segment & sg : segs) {
        sg.n_words = 0;
    }
    if (segs.empty() || words.empty()) {
        return true;
    }
    const int64_t ws = in.win_start_ms;
    const int64_t we = std::max(in.win_end_ms, ws);

    int n_text = 0;
    for (const Segment & sg : segs) {
        n_text += std::max(0, sg.n_text);
    }

    const bool aligned         = dtw_word_times(in, words, n_text, s);
    double     median_duration = 0.0;
    double     max_duration    = 0.0;
    if (aligned) {
        median_duration = apply_duration_heuristics(words, max_duration);
    } else {
        st.n_fallback += 1;
    }
    merge_punctuations(words);
    if (in.space_split) {
        join_unspaced_words(words);
    }

    std::vector<std::vector<Placed>> by_seg(segs.size());
    for (size_t i = 0; i < words.size(); ++i) {
        if (words[i].text.empty()) {
            continue;
        }
        const int k = segment_of(words[i].first_token, segs);
        if (k >= 0) {
            by_seg[static_cast<size_t>(k)].push_back({ i, k });
        }
    }

    std::vector<double> t0s(words.size(), 0.0);
    std::vector<double> t1s(words.size(), 0.0);
    const double        offset_s = static_cast<double>(ws) / 1000.0;
    for (size_t k = 0; k < segs.size(); ++k) {
        std::vector<Placed> & ws_k = by_seg[k];
        if (ws_k.empty()) {
            continue;
        }
        Segment & sg = segs[k];
        if (aligned) {
            std::vector<double> starts;
            std::vector<double> ends;
            for (const Placed & p : ws_k) {
                starts.push_back(offset_s + words[p.word].start);
                ends.push_back(offset_s + words[p.word].end);
            }
            double seg_start = static_cast<double>(sg.t0_ms) / 1000.0;
            double seg_end   = static_cast<double>(sg.t1_ms) / 1000.0;
            apply_segment_heuristics(starts, ends, seg_start, seg_end, median_duration, max_duration, st.last_speech_s);
            trim_silent_edges(starts, ends, in.frame_db, in.n_frames, offset_s);
            for (size_t i = 0; i < ws_k.size(); ++i) {
                t0s[ws_k[i].word] = starts[i];
                t1s[ws_k[i].word] = ends[i];
            }
            sg.t0_ms = std::clamp(to_ms(seg_start), ws, we);
            sg.t1_ms = std::clamp(to_ms(seg_end), sg.t0_ms, we);
        } else {
            // Proportional to codepoints of the stripped text.
            const int64_t a       = std::clamp(sg.t0_ms, ws, we);
            const int64_t b       = std::clamp(sg.t1_ms, a, we);
            double        c_total = 0.0;
            for (const Placed & p : ws_k) {
                c_total += static_cast<double>(std::max<size_t>(1, utf8_codepoints(py_strip(words[p.word].text))));
            }
            double acc = 0.0;
            for (const Placed & p : ws_k) {
                const double c =
                    static_cast<double>(std::max<size_t>(1, utf8_codepoints(py_strip(words[p.word].text))));
                t0s[p.word] = (static_cast<double>(a) + static_cast<double>(b - a) * acc / c_total) / 1000.0;
                acc += c;
                t1s[p.word] = (static_cast<double>(a) + static_cast<double>(b - a) * acc / c_total) / 1000.0;
            }
            sg.t1_ms = std::max(sg.t1_ms, sg.t0_ms);
        }
    }

    // Emit in segment order with the final clamps.
    for (size_t k = 0; k < segs.size(); ++k) {
        for (const Placed & p : by_seg[k]) {
            std::string_view text = words[p.word].text;
            while (!text.empty() && is_space(text.front())) {
                text.remove_prefix(1);
            }
            if (text.empty()) {
                continue;
            }
            int64_t t0 = std::clamp(to_ms(t0s[p.word]), ws, we);
            int64_t t1 = std::clamp(to_ms(t1s[p.word]), t0, we);
            if (st.has_prev) {
                t0 = std::max(t0, st.prev_t0_ms);
                t1 = std::max(t1, t0);
            }
            if (!out.empty()) {
                if (out.back().t1_ms > t0) {
                    out.back().t1_ms = std::max(out.back().t0_ms, t0);
                }
            } else if (prev_word_t1 != nullptr && *prev_word_t1 > t0) {
                *prev_word_t1 = std::max(st.prev_t0_ms, t0);
            }
            st.prev_t0_ms = t0;
            st.has_prev   = true;
            out.push_back({ std::string(text), t0, t1, p.seg });
            segs[k].n_words += 1;
        }
    }
    for (Segment & sg : segs) {
        sg.t1_ms = std::max(sg.t1_ms, sg.t0_ms);
    }
    return aligned;
}

}  // namespace align
}  // namespace transcribe::whisper
