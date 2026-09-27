// arch/parakeet/boost.h - phrase boosting for greedy transducer decoding.
//
// A token trie over the caller's phrases with Aho-Corasick fail links and
// NeMo GPU-PB (TurboBias) arc scores: the first token of a phrase scores
// c0, token d >= 1 scores c0 * beta + ln(d + 1). Greedy decoding keeps the
// model's own blank decision; on a non-blank step the model's token u may
// be replaced by a trie candidate t when
//
//   z[t] + lambda * bonus(t) > z[u] + lambda * bonus(u)
//
// and t's unboosted probability is at least k_boost_floor. bonus(t) is
// NeMo's per-token score minus the score every non-candidate shares, so
// only candidates need to be visited. The decoder verifies each swap by
// forking (decoder.cpp): a swapped branch survives only if it completes a
// phrase and keeps emitting speech afterwards.
//
// Pure host logic, no ggml; built per run / stream from token sequences.

#pragma once

#include <cstdint>
#include <vector>

namespace transcribe::parakeet {

constexpr float k_boost_c0           = 1.0f;         // NeMo context_score
constexpr float k_boost_beta         = 2.0f;         // NeMo depth_scaling for TDT / RNN-T
constexpr float k_boost_floor        = 1e-5f;        // min unboosted probability of a swapped-in token
constexpr int   k_boost_fork_tokens  = 24;           // boosted branch token budget before rejection
constexpr int   k_boost_guard_frames = 6;            // deletion check after a completed phrase (~0.5 s)
constexpr int   k_boost_beam         = 4;            // beam width of the offline boosted decode
constexpr float k_boost_beam_margin  = 8.0f;         // beam keeps hypotheses this close to the best (log prob)
constexpr float k_boost_beam_weight  = 2.5f / 3.0f;  // share of the boost weight the beam uses

// BoostTrie::token_flags bits.
constexpr uint8_t k_boost_token_special = 1;  // never swapped from or into (blank, unk, control, tags)
constexpr uint8_t k_boost_token_word    = 2;  // starts a word (leading U+2581)
constexpr uint8_t k_boost_token_glue    = 4;  // continues a word (letter or digit, no U+2581)

enum class BoostVerdict { Continue, Accept, Reject };

struct BoostTrie {
    struct Node {
        int32_t first_child   = 0;  // span in child_tok / child_node
        int32_t n_children    = 0;
        int32_t fail          = 0;
        float   arc           = 0.0f;  // score of the arc into this node
        float   backoff       = 0.0f;  // 0 on end nodes, else score(fail) - score(node)
        float   chain_backoff = 0.0f;  // sum of backoff from this node down its fail chain
        float   score         = 0.0f;  // sum of arcs from the root
        bool    end           = false;
    };

    std::vector<Node>    nodes;  // nodes[0] is the root
    std::vector<int32_t> child_tok;
    std::vector<int32_t> child_node;
    std::vector<uint8_t> token_flags;  // k_boost_token_* per token class
    float                lambda = 0.0f;

    bool empty() const { return nodes.size() <= 1 || lambda <= 0.0f; }

    // Child of `node` on token `tok`, or -1.
    int child(int node, int tok) const;

    // Aho-Corasick transition: the deepest node reachable by `tok` from
    // `node` or its fail chain, the root when none is.
    int next(int node, int tok) const;

    // Boosted re-ranking of the non-blank model choice `u` at trie state
    // `node`. Returns u or the winning candidate. children_only restricts
    // candidates to direct children of `node` (a branch continuing its
    // own match).
    int pick(int node, const float * logits, int n_classes, int u, bool children_only) const;

    // Beam search step: moves `node` on `tok` and returns the score change,
    // the new arc less the partial match it abandons. A completed phrase
    // that `tok` glues onto loses its score (a phrase is whole words).
    float advance(int & node, int tok) const;

    // A swapped-in token `tok` starts a verified branch at trie state
    // `node`; updates node / completed. Accept when it completes a phrase
    // that nothing extends.
    BoostVerdict fork_begin(int & node, bool & completed, int tok) const;

    // The branch emitted `tok`. Continue while it extends the match; a
    // token that leaves the match accepts a completed branch and rejects
    // an unfinished one or one the token glues onto (a phrase is whole words).
    BoostVerdict fork_advance(int & node, bool & completed, int tok) const;
};

// Deletion check after a completed phrase: over the same frames, the
// boosted branch fails if it started two or more fewer words than the plain
// one (the phrase derailed the decoder). One word of slack covers the plain
// branch still finishing the words the phrase replaced.
inline bool boost_guard_passed(int plain_words, int boosted_words) {
    return boosted_words + 1 >= plain_words;
}

// Build the trie from token sequences (empty sequences are skipped).
// `token_flags` has one entry per token class; lambda <= 0 leaves it empty.
void build_boost_trie(const std::vector<std::vector<int32_t>> & phrases,
                      std::vector<uint8_t>                      token_flags,
                      float                                     lambda,
                      BoostTrie &                               out);

}  // namespace transcribe::parakeet
