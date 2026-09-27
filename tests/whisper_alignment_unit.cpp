// Whisper word-timestamp host math (src/arch/whisper/alignment.cpp) against
// numpy ports of openai/whisper timing.py (tests/fixtures/whisper_alignment_ref.inc),
// plus word splitting, head resolution, the proportional fallback, and a fuzz
// loop over the full per-window pipeline.

#include "arch/whisper/alignment.h"
#include "transcribe-tokenizer.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                        \
        }                                                                        \
    } while (0)

namespace wa = transcribe::whisper::align;
using transcribe::whisper::AlignGeometry;
using transcribe::whisper::AlignHead;

struct MedfiltCase {
    int                T;
    std::vector<float> in;
    std::vector<float> out;
};

struct DtwCase {
    int                N;
    int                M;
    std::vector<float> x;
    std::vector<int>   text_idx;
    std::vector<int>   time_idx;
    int                len;
};

struct PipelineCase {
    int                H;
    int                R;
    int                T;
    int                sot_len;
    float              scale;
    std::vector<float> qk;
    std::vector<float> matrix;
    std::vector<int>   jumps;
};

struct WordSpec {
    const char * w;
    double       s;
    double       e;
};

struct WordsCase {
    int      n;
    WordSpec in[8];
    WordSpec out[8];
    double   median;
    double   max_duration;
};

struct SegCase {
    double seg_start;
    double seg_end;
    int    n;
    double in[4][2];
    double median;
    double max_duration;
    double last_speech;
    double out[4][2];
    double exp_seg_start;
    double exp_seg_end;
    double exp_last_speech;
};

#include "fixtures/whisper_alignment_ref.inc"

bool near(double a, double b, double tol) {
    return std::fabs(a - b) <= tol;
}

void test_median_filter() {
    for (const MedfiltCase & c : k_medfilt_cases) {
        std::vector<float> out(static_cast<size_t>(c.T), -99.0f);
        wa::median_filter_7(c.in.data(), out.data(), c.T);
        for (int t = 0; t < c.T; ++t) {
            CHECK(out[t] == c.out[t]);
        }
    }
    wa::median_filter_7(nullptr, nullptr, 5);  // no-op, no crash
}

void test_dtw() {
    wa::Scratch s;
    for (const DtwCase & c : k_dtw_cases) {
        std::vector<int> ti, mi;
        CHECK(wa::dtw_path(c.x.data(), c.N, c.M, s, ti, mi));
        CHECK(static_cast<int>(ti.size()) == c.len);
        CHECK(ti == c.text_idx);
        CHECK(mi == c.time_idx);
        std::vector<int> jf;
        CHECK(wa::jump_frames(ti, mi, c.N, c.M, jf));
        CHECK(static_cast<int>(jf.size()) == c.N);
        for (size_t k = 1; k < jf.size(); ++k) {
            CHECK(jf[k] >= jf[k - 1]);
        }
    }
    std::vector<int> ti, mi;
    const float      x[1] = { 0.0f };
    CHECK(!wa::dtw_path(x, 0, 1, s, ti, mi));
    CHECK(!wa::dtw_path(x, 1, 0, s, ti, mi));
    CHECK(!wa::dtw_path(nullptr, 1, 1, s, ti, mi));
}

void test_pipeline() {
    wa::Scratch s;
    for (const PipelineCase & c : k_pipeline_cases) {
        wa::WindowInput in;
        in.qk       = c.qk.data();
        in.n_heads  = c.H;
        in.n_rows   = c.R;
        in.n_frames = c.T;
        in.sot_len  = c.sot_len;
        in.qk_scale = c.scale;
        std::vector<float> m;
        CHECK(wa::alignment_matrix(in, s, m));
        CHECK(m.size() == c.matrix.size());
        for (size_t i = 0; i < m.size() && i < c.matrix.size(); ++i) {
            CHECK(near(m[i], c.matrix[i], 2e-5));
        }
        const int N = c.R - c.sot_len - 1;
        for (float & v : m) {
            v = -v;
        }
        std::vector<int> ti, mi, jf;
        CHECK(wa::dtw_path(m.data(), N, c.T, s, ti, mi));
        CHECK(wa::jump_frames(ti, mi, N, c.T, jf));
        CHECK(jf == c.jumps);
    }

    // Constant QK: uniform softmax, zero variance -> zeros, never NaN.
    std::vector<float> qk(2 * 5 * 6, 1.5f);
    wa::WindowInput    in;
    in.qk       = qk.data();
    in.n_heads  = 2;
    in.n_rows   = 5;
    in.n_frames = 6;
    in.sot_len  = 1;
    std::vector<float> m;
    CHECK(wa::alignment_matrix(in, s, m));
    for (float v : m) {
        CHECK(v == 0.0f);
    }

    // Non-finite input is rejected.
    qk[7] = std::numeric_limits<float>::quiet_NaN();
    CHECK(!wa::alignment_matrix(in, s, m));
    qk[7] = std::numeric_limits<float>::infinity();
    CHECK(!wa::alignment_matrix(in, s, m));
}

std::vector<wa::Word> make_words(const WordSpec * spec, int n) {
    std::vector<wa::Word> words;
    for (int i = 0; i < n; ++i) {
        wa::Word w;
        w.text        = spec[i].w;
        w.first_token = i;
        w.n_tokens    = 1;
        w.start       = spec[i].s;
        w.end         = spec[i].e;
        words.push_back(w);
    }
    return words;
}

void test_words_heuristics() {
    for (const WordsCase & c : k_words_cases) {
        std::vector<wa::Word> words = make_words(c.in, c.n);
        double                maxd  = -1.0;
        const double          med   = wa::apply_duration_heuristics(words, maxd);
        wa::merge_punctuations(words);
        CHECK(near(med, c.median, 1e-12));
        CHECK(near(maxd, c.max_duration, 1e-12));
        int tokens = 0;
        for (int i = 0; i < c.n; ++i) {
            CHECK(words[i].text == c.out[i].w);
            CHECK(near(words[i].start, c.out[i].s, 1e-12));
            CHECK(near(words[i].end, c.out[i].e, 1e-12));
            tokens += words[i].n_tokens;
            CHECK(words[i].text.empty() == (words[i].n_tokens == 0));
        }
        CHECK(tokens == c.n);
    }
}

void test_join_unspaced() {
    const WordSpec spec[] = {
        { " at",    0.0, 0.2 },
        { " 555",   0.2, 0.6 },
        { "-0142",  0.6, 1.2 },
        { "",       1.2, 1.2 },
        { " v",     1.2, 1.3 },
        { ".5",     1.3, 1.5 },
        { " done.", 1.5, 1.9 }
    };
    std::vector<wa::Word> w = make_words(spec, 7);
    wa::join_unspaced_words(w);
    CHECK(w[1].text == " 555-0142" && w[1].n_tokens == 2 && w[1].start == 0.2 && w[1].end == 1.2);
    CHECK(w[2].text.empty() && w[2].n_tokens == 0);
    CHECK(w[4].text == " v.5" && w[4].end == 1.5);
    CHECK(w[6].text == " done.");
    // A leading unspaced word has nothing to join.
    const WordSpec lead[] = {
        { "-x", 0.0, 0.1 },
        { " y", 0.1, 0.2 }
    };
    w = make_words(lead, 2);
    wa::join_unspaced_words(w);
    CHECK(w[0].text == "-x" && w[1].text == " y");
}

void test_trim_silent_edges() {
    std::vector<float> db(30, -100.0f);
    for (int f = 10; f < 20; ++f) {
        db[f] = -20.0f;
    }
    std::vector<double> st = { 1.0, 1.2 }, en = { 1.6, 1.26 };
    wa::trim_silent_edges(st, en, db.data(), 30, 1.0);
    CHECK(near(st[0], 1.2, 1e-9) && near(en[0], 1.4, 1e-9));
    CHECK(near(st[1], 1.2, 1e-9) && near(en[1], 1.26, 1e-9));  // short silent tail kept

    // Word entirely in silence, and a flat window: untouched.
    st = { 1.0 };
    en = { 1.1 };
    wa::trim_silent_edges(st, en, db.data(), 30, 1.0);
    CHECK(near(st[0], 1.0, 1e-9) && near(en[0], 1.1, 1e-9));
    std::vector<float> flat(30, -30.0f);
    flat[15] = -25.0f;
    st       = { 1.0 };
    en       = { 1.6 };
    wa::trim_silent_edges(st, en, flat.data(), 30, 1.0);
    CHECK(near(st[0], 1.0, 1e-9) && near(en[0], 1.6, 1e-9));

    // One neighbor tail before a pause is skipped at each edge, not a second.
    std::vector<float> blip(40, -100.0f);
    for (int f : { 0, 1, 2, 13, 14, 15, 16, 17, 18, 19, 20, 31, 32, 33, 34, 35 }) {
        blip[f] = -20.0f;
    }
    st = { 0.0 };
    en = { 0.8 };
    wa::trim_silent_edges(st, en, blip.data(), 40, 0.0);
    CHECK(near(st[0], 0.26, 1e-9) && near(en[0], 0.42, 1e-9));

    // extend_to continues an end that sits in speech, never across silence.
    st                      = { 1.2, 1.0 };
    en                      = { 1.3, 1.1 };
    std::vector<double> ext = { 1.9, 1.5 };
    wa::trim_silent_edges(st, en, db.data(), 30, 1.0, &ext);
    CHECK(near(en[0], 1.4, 1e-9));
    CHECK(near(en[1], 1.1, 1e-9));  // ends in silence: no extension

    // Out-of-window times and NaN never index out of range.
    st = { -5.0, std::numeric_limits<double>::quiet_NaN(), 99.0 };
    en = { 50.0, 1.2, 100.0 };
    wa::trim_silent_edges(st, en, db.data(), 30, 1.0);
    CHECK(near(st[0], 1.2, 1e-9) && near(en[0], 1.4, 1e-9));
}

void test_punct_end() {
    const WordSpec spec[] = {
        { " Yes", 0.0, 0.3 },
        { ".",    0.3, 0.5 }
    };
    std::vector<wa::Word> w = make_words(spec, 2);
    wa::merge_punctuations(w);
    CHECK(w[0].text == " Yes." && near(w[0].end, 0.3, 1e-12) && near(w[0].punct_end, 0.5, 1e-12));
    CHECK(w[1].text.empty());
}

void test_segment_heuristics() {
    for (const SegCase & c : k_seg_cases) {
        std::vector<double> st, en;
        for (int i = 0; i < c.n; ++i) {
            st.push_back(c.in[i][0]);
            en.push_back(c.in[i][1]);
        }
        double ss = c.seg_start, se = c.seg_end, last = c.last_speech;
        wa::apply_segment_heuristics(st, en, ss, se, c.median, c.max_duration, last);
        for (int i = 0; i < c.n; ++i) {
            CHECK(near(st[i], c.out[i][0], 1e-12));
            CHECK(near(en[i], c.out[i][1], 1e-12));
        }
        CHECK(near(ss, c.exp_seg_start, 1e-12));
        CHECK(near(se, c.exp_seg_end, 1e-12));
        CHECK(near(last, c.exp_last_speech, 1e-12));
    }
}

void test_split_words() {
    // GPT-2 byte-unicode vocab: "Ġ" is a space, "æĹ" + "¥" are the bytes of 日.
    std::vector<std::string> vocab = {
        "\xc4\xa0Hello", ",", "\xc4\xa0world", ".", "\xc4\xa0trans", "cribe", "\xc3\xa6\xc4\xb9", "\xc2\xa5",
    };
    transcribe::Tokenizer tok;
    CHECK(tok.load_decode_only_gpt2(vocab) == TRANSCRIBE_OK);
    const int cjk[2] = { 6, 7 };
    CHECK(tok.decode(cjk, 2) == "\xe6\x97\xa5");

    std::vector<wa::Word> w = wa::split_words(tok, { 0, 1, 2, 3 }, true);
    CHECK(w.size() == 4);
    if (w.size() == 4) {
        CHECK(w[0].text == " Hello" && w[1].text == "," && w[2].text == " world" && w[3].text == ".");
        CHECK(w[3].first_token == 3 && w[3].n_tokens == 1);
    }

    w = wa::split_words(tok, { 4, 5, 2 }, true);
    CHECK(w.size() == 2);
    if (w.size() == 2) {
        CHECK(w[0].text == " transcribe" && w[0].n_tokens == 2 && w[1].first_token == 2);
    }

    // Character split joins the two byte tokens of one character.
    w = wa::split_words(tok, { 6, 7, 6, 7 }, false);
    CHECK(w.size() == 2);
    if (w.size() == 2) {
        CHECK(w[0].text == "\xe6\x97\xa5" && w[0].n_tokens == 2 && w[1].first_token == 2);
    }

    // A dangling incomplete tail is flushed and keeps its token.
    w     = wa::split_words(tok, { 0, 6 }, true);
    int n = 0;
    for (const wa::Word & x : w) {
        n += x.n_tokens;
    }
    CHECK(n == 2);

    CHECK(wa::is_no_space_language("ja") && wa::is_no_space_language("yue") && !wa::is_no_space_language("en"));
}

void test_heads() {
    struct Case {
        AlignGeometry        g;
        std::vector<int32_t> kv;
        const char *         source;
        size_t               n;
    };

    const std::vector<int32_t> small_en_list = { 6, 6, 7, 0, 7, 3, 7, 8, 8, 2, 8, 5, 8, 7, 9, 0, 9, 4, 9, 8, 9, 10 };
    const Case                 cases[]       = {
        { { 4, 4, 6, 80, true },     {},             "table:tiny",            6  },
        { { 4, 4, 6, 80, false },    {},             "table:tiny.en",         8  },
        { { 6, 6, 8, 80, false },    {},             "table:base.en",         5  },
        { { 6, 6, 8, 80, true },     {},             "table:base",            8  },
        { { 12, 12, 12, 80, false }, {},             "table:small.en",        19 },
        { { 12, 12, 12, 80, true },  {},             "table:small",           10 },
        { { 24, 24, 16, 80, false }, {},             "table:medium.en",       18 },
        { { 24, 24, 16, 80, true },  {},             "table:medium",          6  },
        { { 32, 32, 20, 80, true },  {},             "table:large-v2",        23 },
        { { 32, 32, 20, 128, true }, {},             "table:large-v3",        10 },
        { { 32, 4, 20, 128, true },  {},             "table:large-v3-turbo",  6  },
        { { 32, 2, 20, 128, true },  {},             "table:distil-large-v3", 20 },
        { { 24, 2, 16, 80, false },  {},             "heuristic",             16 },
        { { 12, 4, 12, 80, false },  small_en_list,  "heuristic",             24 },
        { { 5, 3, 7, 80, true },     {},             "heuristic",             14 },
        { { 4, 4, 6, 80, true },     { 3, 1, 2, 0 }, "gguf",                  2  },
        { { 4, 4, 6, 80, true },     { 3, 1, 3, 1 }, "table:tiny",            6  },
        { { 4, 4, 6, 80, true },     { 3 },          "table:tiny",            6  },
    };
    for (const Case & c : cases) {
        std::string                  source;
        const std::vector<AlignHead> h = transcribe::whisper::resolve_alignment_heads(c.g, "whisper", c.kv, source);
        CHECK(source == c.source);
        CHECK(h.size() == c.n);
        for (size_t i = 0; i < h.size(); ++i) {
            CHECK(h[i].layer >= 0 && h[i].layer < c.g.dec_n_layers && h[i].head >= 0 && h[i].head < c.g.dec_n_heads);
            if (i > 0) {
                CHECK(h[i - 1].layer < h[i].layer || (h[i - 1].layer == h[i].layer && h[i - 1].head < h[i].head));
            }
        }
    }
    std::string source;
    CHECK(
        transcribe::whisper::resolve_alignment_heads({ 32, 32, 20, 80, true }, "whisper-large-v1", {}, source).size() ==
        9);
    CHECK(source == "table:large-v1");
    CHECK(transcribe::whisper::resolve_alignment_heads({ 4, 0, 6, 80, true }, "whisper", {}, source).empty());
}

// Invariants every window output must satisfy.
void check_window(const std::vector<wa::OutWord> & out,
                  const std::vector<wa::Segment> & segs,
                  int64_t                          ws,
                  int64_t                          we,
                  int64_t &                        run_prev_t0) {
    for (size_t i = 0; i < out.size(); ++i) {
        CHECK(!out[i].text.empty() && out[i].text[0] != ' ');
        CHECK(out[i].t0_ms <= out[i].t1_ms);
        CHECK(out[i].t0_ms >= ws && out[i].t1_ms <= we);
        CHECK(out[i].t0_ms >= run_prev_t0);
        CHECK(out[i].seg >= 0 && out[i].seg < static_cast<int>(segs.size()));
        if (i > 0) {
            CHECK(out[i].seg >= out[i - 1].seg);
            CHECK(out[i - 1].t1_ms <= out[i].t0_ms);
        }
        run_prev_t0 = out[i].t0_ms;
    }
    for (const wa::Segment & sg : segs) {
        CHECK(sg.t0_ms <= sg.t1_ms);
        CHECK(out.empty() || (sg.t0_ms >= ws && sg.t1_ms <= we));
    }
}

void test_proportional() {
    std::vector<wa::Word> words;
    const char *          texts[] = { " ab", " c", " def" };
    for (int i = 0; i < 3; ++i) {
        wa::Word w;
        w.text        = texts[i];
        w.first_token = i;
        w.n_tokens    = 1;
        words.push_back(w);
    }
    wa::WindowInput in;
    in.win_start_ms = 0;
    in.win_end_ms   = 5000;
    wa::RunState             st;
    wa::Scratch              s;
    std::vector<wa::OutWord> out;
    std::vector<wa::Segment> segs = {
        { 1000, 2000, 3, 0 }
    };
    CHECK(!wa::compute_window_words(in, words, segs, st, s, nullptr, out));
    CHECK(out.size() == 3 && st.n_fallback == 1);
    if (out.size() == 3) {
        CHECK(out[0].text == "ab" && out[0].t0_ms == 1000 && out[0].t1_ms == 1333);
        CHECK(out[1].t0_ms == 1333 && out[1].t1_ms == 1500);
        CHECK(out[2].t0_ms == 1500 && out[2].t1_ms == 2000);
    }
    CHECK(segs[0].n_words == 3 && segs[0].t0_ms == 1000 && segs[0].t1_ms == 2000);

    // Zero-width segment past the window end: everything collapses to the end.
    wa::RunState st2;
    segs = {
        { 7000, 7000, 3, 0 }
    };
    wa::compute_window_words(in, words, segs, st2, s, nullptr, out);
    CHECK(out.size() == 3);
    for (const wa::OutWord & w : out) {
        CHECK(w.t0_ms == 5000 && w.t1_ms == 5000);
    }

    // Two segments, and a trimmed previous word from an earlier window.
    wa::RunState st3;
    st3.has_prev    = true;
    st3.prev_t0_ms  = 900;
    int64_t prev_t1 = 1800;
    segs            = {
        { 1000, 1500, 2, 0 },
        { 1500, 3000, 1, 0 }
    };
    wa::compute_window_words(in, words, segs, st3, s, &prev_t1, out);
    CHECK(out.size() == 3);
    CHECK(prev_t1 == 1000);
    if (out.size() == 3) {
        CHECK(out[0].seg == 0 && out[1].seg == 0 && out[2].seg == 1);
        CHECK(out[2].t0_ms == 1500 && out[2].t1_ms == 3000);
    }
    CHECK(segs[0].n_words == 2 && segs[1].n_words == 1);
}

void test_fuzz() {
    std::mt19937                          rng(12345);
    std::uniform_real_distribution<float> val(-20.0f, 20.0f);
    wa::Scratch                           s;
    int                                   aligned = 0;
    for (int run = 0; run < 400; ++run) {
        wa::RunState st;
        int64_t      run_prev_t0 = 0;
        int64_t      ws          = 0;
        int64_t      prev_we     = 0;
        for (int win = 0; win < 5; ++win) {
            const int H      = 1 + static_cast<int>(rng() % 8);
            const int sot    = (rng() % 2) ? 3 : 1;
            const int n_text = 1 + static_cast<int>(rng() % 36);
            const int R      = sot + 2 + n_text;
            const int T      = static_cast<int>(rng() % 13);
            const int mode   = static_cast<int>(rng() % 10);

            std::vector<float> qk(static_cast<size_t>(H) * R * T);
            for (float & v : qk) {
                v = val(rng);
            }
            if (mode == 0 && !qk.empty()) {
                qk[rng() % qk.size()] = std::numeric_limits<float>::quiet_NaN();
            } else if (mode == 1 && !qk.empty()) {
                qk[rng() % qk.size()] = -std::numeric_limits<float>::infinity();
            }

            std::vector<wa::Word> words;
            for (int i = 0; i < n_text; ++i) {
                wa::Word  w;
                const int kind = static_cast<int>(rng() % 6);
                w.text         = kind == 0 ? "," : kind == 1 ? " \"" : kind == 2 ? " " : " w";
                w.first_token  = i;
                w.n_tokens     = 1;
                words.push_back(w);
            }
            std::vector<wa::Segment> segs;
            int                      left = n_text;
            while (left > 0) {
                const int     take = std::min(left, 1 + static_cast<int>(rng() % 12));
                const int64_t a    = ws + static_cast<int64_t>(rng() % 40000) - 5000;
                const int64_t b    = a + static_cast<int64_t>(rng() % 20000) - 2000;
                segs.push_back({ a, b, take, 0 });
                left -= take;
            }

            std::vector<float> db(static_cast<size_t>(T));
            for (float & v : db) {
                v = (rng() % 3 == 0) ? -100.0f : val(rng) - 30.0f;
            }
            if (mode == 4 && !db.empty()) {
                db[rng() % db.size()] = std::numeric_limits<float>::quiet_NaN();
            }

            wa::WindowInput in;
            in.frame_db     = (mode == 5) ? nullptr : db.data();
            in.qk           = (mode == 2) ? nullptr : qk.data();
            in.n_heads      = H;
            in.n_rows       = (mode == 3) ? R + 1 : R;
            in.n_frames     = T;
            in.sot_len      = sot;
            in.qk_scale     = 0.125f;
            in.win_start_ms = ws;
            in.win_end_ms   = std::max<int64_t>(ws + T * 20, prev_we);  // window ends never move back
            prev_we         = in.win_end_ms;

            std::vector<wa::OutWord> out;
            if (wa::compute_window_words(in, words, segs, st, s, nullptr, out)) {
                ++aligned;
            }
            check_window(out, segs, in.win_start_ms, in.win_end_ms, run_prev_t0);
            ws += static_cast<int64_t>(rng() % 30000);
        }
    }
    CHECK(aligned > 0);
}

}  // namespace

int main() {
    test_median_filter();
    test_dtw();
    test_pipeline();
    test_words_heuristics();
    test_segment_heuristics();
    test_join_unspaced();
    test_punct_end();
    test_trim_silent_edges();
    test_split_words();
    test_heads();
    test_proportional();
    test_fuzz();
    if (g_failures != 0) {
        std::fprintf(stderr, "whisper_alignment_unit: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("whisper_alignment_unit: ok\n");
    return 0;
}
