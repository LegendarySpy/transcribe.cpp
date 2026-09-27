// tokenizer_sentencepiece_unit.cpp - SentencePiece encode() on toy vocabs.
//
// Builds tokenizer.ggml.* KVs in an in-memory gguf_context and checks:
//   - BPE merges the highest-scoring adjacent pair first (leftmost on ties)
//   - unigram takes the Viterbi path over piece scores
//   - a "bpe" label whose scores are not rank-ordered encodes as unigram
//   - whitespace: one U+2581 prefix, each space its own U+2581
//   - unknown characters: one shared unk, or byte-fallback pieces
//   - CONTROL pieces never match text; empty text gives no ids
// Real-vocab parity lives in parakeet_sp_parity.cpp. "_" below stands for
// the U+2581 word marker.

#include "gguf.h"
#include "transcribe-tokenizer.h"

#include <cstdint>
#include <cstdio>
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

struct Piece {
    std::string text;
    float       score;
    int32_t     type;  // 1 normal, 2 unknown, 3 control, 6 byte
};

const std::string k_ws = "\xE2\x96\x81";  // U+2581

bool load(transcribe::Tokenizer & tok, const char * model, const std::vector<Piece> & pieces) {
    gguf_context *            ctx = gguf_init_empty();
    std::vector<const char *> texts;
    std::vector<float>        scores;
    std::vector<int32_t>      types;
    for (const Piece & p : pieces) {
        texts.push_back(p.text.c_str());
        scores.push_back(p.score);
        types.push_back(p.type);
    }
    gguf_set_val_str(ctx, "tokenizer.ggml.model", model);
    gguf_set_arr_str(ctx, "tokenizer.ggml.tokens", texts.data(), texts.size());
    gguf_set_arr_data(ctx, "tokenizer.ggml.scores", GGUF_TYPE_FLOAT32, scores.data(), scores.size());
    gguf_set_arr_data(ctx, "tokenizer.ggml.token_type", GGUF_TYPE_INT32, types.data(), types.size());
    gguf_set_val_u32(ctx, "tokenizer.ggml.unknown_token_id", 0);
    const bool ok = tok.load(ctx) == TRANSCRIBE_OK;
    gguf_free(ctx);
    return ok;
}

std::vector<int32_t> enc(const transcribe::Tokenizer & tok, const std::string & text) {
    std::vector<int32_t> ids;
    if (tok.encode(text, ids) != TRANSCRIBE_OK) {
        ids.assign(1, -99);
    }
    return ids;
}

void test_bpe_merge_order() {
    // Rank-ordered scores (BPE). "the" merges _t first (score 0 beats th
    // at -1), then he, then _t + he.
    transcribe::Tokenizer tok;
    CHECK(load(tok, "bpe",
               {
                   { "<unk>",      0.0f,  2 },
                   { k_ws + "t",   0.0f,  1 }, // 1
                   { "th",         -1.0f, 1 }, // 2
                   { "he",         -2.0f, 1 }, // 3
                   { k_ws + "the", -3.0f, 1 }, // 4
                   { k_ws,         -4.0f, 1 }, // 5
                   { "t",          -5.0f, 1 }, // 6
                   { "h",          -6.0f, 1 }, // 7
                   { "e",          -7.0f, 1 }, // 8
                   { k_ws + "x",   0.0f,  3 }, // 9: CONTROL
                   { "x",          -8.0f, 1 }, // 10
    }));
    CHECK(tok.has_encoder());
    CHECK(enc(tok, "the") == (std::vector<int32_t>{ 4 }));
    // _t (0) beats th (-1), then _t + h has no piece.
    CHECK(enc(tok, "th") == (std::vector<int32_t>{ 1, 7 }));
    // Each space is its own _; runs are kept.
    CHECK(enc(tok, "the the") == (std::vector<int32_t>{ 4, 4 }));
    CHECK(enc(tok, "the  the") == (std::vector<int32_t>{ 4, 5, 4 }));
    // Unknown characters: consecutive ones share one unk.
    CHECK(enc(tok, "the\xE2\x82\xAC\xE2\x82\xAC") == (std::vector<int32_t>{ 4, 0 }));
    // CONTROL pieces never match text, even as a merge result.
    CHECK(enc(tok, "x") == (std::vector<int32_t>{ 5, 10 }));
    CHECK(enc(tok, "").empty());
}

void test_unigram_viterbi() {
    transcribe::Tokenizer tok;
    CHECK(load(tok, "unigram",
               {
                   { "<unk>",      0.0f,  2 },
                   { k_ws + "a",   -1.0f, 1 }, // 1
                   { "b",          -1.0f, 1 }, // 2
                   { k_ws + "ab",  -2.5f, 1 }, // 3
                   { k_ws,         -2.0f, 1 }, // 4
                   { "a",          -3.0f, 1 }, // 5
                   { k_ws + "abc", -2.9f, 1 }, // 6
                   { "c",          -1.0f, 1 }, // 7
    }));
    // _a + b = -2.0 beats _ab = -2.5.
    CHECK(enc(tok, "ab") == (std::vector<int32_t>{ 1, 2 }));
    // _abc = -2.9 beats _a + b + c = -3.0.
    CHECK(enc(tok, "abc") == (std::vector<int32_t>{ 6 }));
    CHECK(enc(tok, "x") == (std::vector<int32_t>{ 4, 0 }));
}

void test_bpe_label_with_unigram_scores() {
    // Same vocab labelled "bpe": ids 2 -> 3 raise the score, so the scores
    // are not merge ranks and the unigram path runs. BPE merging would give
    // _a then _ab.
    transcribe::Tokenizer tok;
    CHECK(load(tok, "bpe",
               {
                   { "<unk>",     0.0f,  2 },
                   { k_ws + "a",  -1.0f, 1 },
                   { "b",         -1.0f, 1 },
                   { k_ws + "ab", -2.5f, 1 },
                   { k_ws,        -2.0f, 1 },
    }));
    CHECK(enc(tok, "ab") == (std::vector<int32_t>{ 1, 2 }));
}

void test_byte_fallback() {
    std::vector<Piece> pieces = {
        { "<unk>",    0.0f,  2 },
        { k_ws + "a", -1.0f, 1 },
    };
    for (int b = 0; b < 256; ++b) {
        char name[8];
        std::snprintf(name, sizeof(name), "<0x%02X>", b);
        pieces.push_back({ name, 0.0f, 6 });
    }
    transcribe::Tokenizer tok;
    CHECK(load(tok, "bpe", pieces));
    // U+20AC = E2 82 AC -> one piece per byte (byte ids start at 2).
    CHECK(enc(tok, "a\xE2\x82\xAC") == (std::vector<int32_t>{ 1, 2 + 0xE2, 2 + 0x82, 2 + 0xAC }));
}

void test_no_scores_no_encoder() {
    gguf_context * ctx     = gguf_init_empty();
    const char *   texts[] = { "<unk>", "a" };
    gguf_set_val_str(ctx, "tokenizer.ggml.model", "bpe");
    gguf_set_arr_str(ctx, "tokenizer.ggml.tokens", texts, 2);
    transcribe::Tokenizer tok;
    CHECK(tok.load(ctx) == TRANSCRIBE_OK);
    gguf_free(ctx);
    CHECK(!tok.has_encoder());
    std::vector<int32_t> ids;
    CHECK(tok.encode("a", ids) == TRANSCRIBE_ERR_NOT_IMPLEMENTED);
}

}  // namespace

int main() {
    test_bpe_merge_order();
    test_unigram_viterbi();
    test_bpe_label_with_unigram_scores();
    test_byte_fallback();
    test_no_scores_no_encoder();
    if (g_failures > 0) {
        std::fprintf(stderr, "tokenizer_sentencepiece_unit: %d failure(s)\n", g_failures);
        return 1;
    }
    std::fprintf(stderr, "tokenizer_sentencepiece_unit: OK\n");
    return 0;
}
