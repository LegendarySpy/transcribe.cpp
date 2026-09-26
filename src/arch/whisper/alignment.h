// arch/whisper/alignment.h - Whisper word timestamps from cross-attention.
// INTERNAL to src/arch/whisper/.
//
// Host half of the OpenAI whisper timing.py pipeline (find_alignment +
// add_word_timestamps). The decoder runs one teacher-forced pass per window
// and hands back raw Q.K^T for the alignment heads; everything after that
// (softmax, normalization, median filter, DTW, word split, punctuation merge,
// duration heuristics, clamping) lives here. No ggml, no asserts: every
// failure returns false and the caller falls back to proportional timing.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace transcribe {
class Tokenizer;
}

namespace transcribe::whisper {

struct AlignHead {
    int layer = 0;
    int head  = 0;
};

// Model geometry used to pick an alignment-head preset.
struct AlignGeometry {
    int  enc_n_layers = 0;
    int  dec_n_layers = 0;
    int  dec_n_heads  = 0;
    int  n_mels       = 0;
    bool multilingual = true;
};

// Resolution order: GGUF stt.whisper.alignment_heads (flat [l0, h0, l1, h1,
// ...]) > built-in table matched on exact geometry > every head of the upper
// half of the decoder layers. Invalid lists are skipped. The result is sorted
// by (layer, head). `source` names the winner for the load log.
std::vector<AlignHead> resolve_alignment_heads(const AlignGeometry &        geom,
                                               const std::string &          variant,
                                               const std::vector<int32_t> & kv_heads,
                                               std::string &                source);

namespace align {

// One word of a window. Times are seconds, window-relative until
// assign_segments makes them absolute.
struct Word {
    std::string text;
    int         first_token = 0;  // index into the window's text tokens
    int         n_tokens    = 0;
    double      start       = 0.0;
    double      end         = 0.0;
    double      punct_end   = -1.0;  // end of absorbed appended punctuation, or -1
};

// Per-window segment: decoded times plus how many text tokens it owns (in
// text-token order). assign_segments may refine t0_ms / t1_ms.
struct Segment {
    int64_t t0_ms   = 0;
    int64_t t1_ms   = 0;
    int     n_text  = 0;
    int     n_words = 0;  // output: words emitted for this segment
};

struct OutWord {
    std::string text;  // leading whitespace trimmed
    int64_t     t0_ms = 0;
    int64_t     t1_ms = 0;
    int         seg   = 0;  // index into the window's segments
};

// Run-scoped state threaded across windows.
struct RunState {
    double  last_speech_s = 0.0;
    int64_t prev_t0_ms    = 0;
    bool    has_prev      = false;
    int     n_fallback    = 0;
};

// Reusable buffers so windows do not reallocate.
struct Scratch {
    std::vector<float>  weights;
    std::vector<double> col_mean;
    std::vector<double> col_var;
    std::vector<float>  row;
    std::vector<float>  matrix;
    std::vector<float>  cost;
    std::vector<int8_t> trace;
    std::vector<int>    text_idx;
    std::vector<int>    time_idx;
    std::vector<int>    jumps;
};

struct WindowInput {
    // Raw (unscaled) Q.K^T laid out [head][row][frame]; nullptr forces the
    // proportional fallback. Rows are [sot_seq..., notimestamps, text..., eot].
    const float * qk       = nullptr;
    int           n_heads  = 0;
    int           n_rows   = 0;
    int           n_frames = 0;  // real audio frames (20 ms each)
    int           sot_len  = 0;  // rows before <|notimestamps|>
    float         qk_scale = 1.0f;

    bool space_split = true;  // false for zh/ja/th/lo/my/yue (see split_words)

    // Optional frame energy in dB, one value per 20 ms frame (n_frames), used
    // by trim_silent_edges. nullptr skips the trim.
    const float * frame_db = nullptr;

    int64_t win_start_ms = 0;
    int64_t win_end_ms   = 0;
};

// OpenAI median_filter along the last axis, width 7, reflect padding.
// T <= 3 copies the input unchanged.
void median_filter_7(const float * in, float * out, int T);

// Softmax over frames, std/mean normalization over rows, median filter, head
// mean, and the slice that drops the SOT rows and the EOT row. Writes
// matrix [N][T] with N = n_rows - sot_len - 1.
bool alignment_matrix(const WindowInput & in, Scratch & s, std::vector<float> & matrix);

// OpenAI dtw_cpu + backtrace on cost x [N][T].
bool dtw_path(const float * x, int N, int T, Scratch & s, std::vector<int> & text_idx, std::vector<int> & time_idx);

// Frame index of the first path cell of each text row; out.size() == N.
bool jump_frames(const std::vector<int> & text_idx,
                 const std::vector<int> & time_idx,
                 int                      N,
                 int                      T,
                 std::vector<int> &       out);

// OpenAI split_to_word_tokens over the window's text tokens (EOT excluded).
// space_split=false selects the per-character split used for zh/ja/th/lo/my/yue.
std::vector<Word> split_words(const Tokenizer & tok, const std::vector<int32_t> & text_ids, bool space_split);

bool is_no_space_language(const std::string & code);

// OpenAI sentence-boundary truncation. Returns the median word duration and
// sets max_duration (seconds).
double apply_duration_heuristics(std::vector<Word> & words, double & max_duration);

// OpenAI merge_punctuations. A word that absorbs appended punctuation keeps
// its own times (as OpenAI) and records the punctuation's end in punct_end.
void merge_punctuations(std::vector<Word> & words);

// DEVIATION from OpenAI: after merge_punctuations, a word that does not start
// with whitespace joins the previous word, so "555-0142" and "2.5" stay one
// word instead of "555" "-0142" / "2" ".5". The joined word keeps the first
// part's start and takes the last part's end. Space-split languages only.
void join_unspaced_words(std::vector<Word> & words);

// DEVIATION from OpenAI: DTW hands the silence before a word to that word
// (and a pause after it to its end), often with a sliver of the neighbor
// word. From each edge, drop silent frames and one short speech blip
// followed by a pause (<= 4 frames then >= 3 silent, or <= 15 then
// >= 10), never the last speech run. Silent = below floor + 0.3 * (peak -
// floor) of the window (10th / 99th percentile); windows with under 12 dB of
// range are left alone. Times are absolute seconds; frame 0 is at offset_s.
// With extend_to, a word whose end sits in speech continues through speech
// frames up to extend_to[i] (never across a silent frame).
void trim_silent_edges(std::vector<double> &       starts,
                       std::vector<double> &       ends,
                       const float *               frame_db,
                       int                         n_frames,
                       double                      offset_s,
                       const std::vector<double> * extend_to = nullptr);

// OpenAI add_word_timestamps segment rules for one segment's words (absolute
// seconds): pause truncation of the first words, then prefer the segment
// start/end when the edge words are implausibly long. Updates last_speech.
void apply_segment_heuristics(std::vector<double> & starts,
                              std::vector<double> & ends,
                              double &              seg_start,
                              double &              seg_end,
                              double                median_duration,
                              double                max_duration,
                              double &              last_speech);

// Full per-window pipeline. `words` comes from split_words; `segs` lists the
// window's segments in order. Returns true when DTW timing was used, false
// when the window fell back to proportional timing. Output words satisfy
// win_start <= t0 <= t1 <= win_end with non-decreasing starts across the run
// (segment times are clamped to the window too); when the first new word starts before *prev_word_t1 (the last word of an
// earlier window), *prev_word_t1 is trimmed to that start.
bool compute_window_words(const WindowInput &    in,
                          std::vector<Word>      words,
                          std::vector<Segment> & segs,
                          RunState &             st,
                          Scratch &              s,
                          int64_t *              prev_word_t1,
                          std::vector<OutWord> & out);

}  // namespace align
}  // namespace transcribe::whisper
