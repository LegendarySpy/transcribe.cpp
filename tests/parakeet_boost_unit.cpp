// parakeet_boost_unit.cpp - phrase-boosting trie and decisions (boost.h).
//
// Covers: trie shape (shared prefixes, NeMo arc scores, fail links),
// BoostTrie::next, the pick rule (bonus, probability floor, ties keep
// the model's token, special tokens, children_only, deep-state bonus),
// the fork verdicts (single-token and multi-token phrases, overlaps,
// abandoned matches) and the post-phrase deletion check. No model or backend.

#include "arch/parakeet/boost.h"

#include <cmath>
#include <cstdio>
#include <vector>

using transcribe::parakeet::BoostTrie;
using transcribe::parakeet::BoostVerdict;

namespace {

int g_failures = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                        \
        }                                                                        \
    } while (0)

constexpr int k_n_cls = 32;  // token classes; 31 = blank
constexpr int k_u     = 5;   // the model's own choice in pick tests

// Phrases: [10 11 12], [10 13], [20], [11 12] (a suffix of the first).
BoostTrie make_trie(float lambda = 2.0f) {
    std::vector<uint8_t> flags(k_n_cls, 0);
    flags[31] = transcribe::parakeet::k_boost_token_special;  // blank
    flags[30] = transcribe::parakeet::k_boost_token_special;  // e.g. a language tag
    BoostTrie t;
    build_boost_trie(
        {
            { 10, 11, 12 },
            { 10, 13 },
            { 20 },
            { 11, 12 },
            {}
    },
        flags, lambda, t);
    return t;
}

// Logits with z[u] = 10, every other class 0, plus overrides.
std::vector<float> logits_with(std::initializer_list<std::pair<int, float>> set) {
    std::vector<float> z(k_n_cls, 0.0f);
    z[k_u] = 10.0f;
    for (const auto & kv : set) {
        z[static_cast<size_t>(kv.first)] = kv.second;
    }
    return z;
}

void test_shape() {
    const BoostTrie t = make_trie();
    CHECK(!t.empty());
    // root + {10, 10-11, 10-11-12, 10-13, 20, 11, 11-12}
    CHECK(t.nodes.size() == 8);
    const int n10   = t.child(0, 10);
    const int n1011 = t.child(n10, 11);
    const int n11   = t.child(0, 11);
    CHECK(n10 > 0 && n1011 > 0 && n11 > 0);
    CHECK(t.child(0, 12) == -1);
    CHECK(std::fabs(t.nodes[n10].arc - 1.0f) < 1e-6f);
    CHECK(std::fabs(t.nodes[n1011].arc - (2.0f + std::log(2.0f))) < 1e-5f);
    CHECK(std::fabs(t.nodes[t.child(n1011, 12)].arc - (2.0f + std::log(3.0f))) < 1e-5f);
    // 10-11 falls back to the phrase starting at 11.
    CHECK(t.nodes[n1011].fail == n11);
    CHECK(t.nodes[t.child(n1011, 12)].end);
    // Aho-Corasick transitions.
    CHECK(t.next(0, 10) == n10);
    CHECK(t.next(n10, 11) == n1011);
    CHECK(t.next(n1011, 20) == t.child(0, 20));
    CHECK(t.next(n10, 7) == 0);

    BoostTrie off;
    build_boost_trie(
        {
            { 1, 2 }
    },
        std::vector<uint8_t>(k_n_cls, 0), 0.0f, off);
    CHECK(off.empty());
    BoostTrie none;
    build_boost_trie({}, std::vector<uint8_t>(k_n_cls, 0), 2.0f, none);
    CHECK(none.empty());
}

void test_pick() {
    const BoostTrie t = make_trie();
    // Root bonus = lambda * c0 = 2: 9 + 2 beats 10.
    CHECK(t.pick(0,
                 logits_with({
                                 { 10, 9.0f }
    })
                     .data(),
                 k_n_cls, k_u, false) == 10);
    // A tie keeps the model's token.
    CHECK(t.pick(0,
                 logits_with({
                                 { 10, 8.0f }
    })
                     .data(),
                 k_n_cls, k_u, false) == k_u);
    // Floor: after token 10, z = -3 is below p = 1e-5 (the floor sits near
    // 10 - 11.5) and z = -1 is above it.
    const BoostTrie strong = make_trie(10.0f);
    const int       s10    = strong.child(0, 10);
    CHECK(strong.pick(s10,
                      logits_with({
                                      { 10, -20.0f },
                                      { 11, -3.0f  },
                                      { 13, -20.0f },
                                      { 20, -20.0f }
    })
                          .data(),
                      k_n_cls, k_u, false) == k_u);
    CHECK(strong.pick(s10,
                      logits_with({
                                      { 10, -20.0f },
                                      { 11, -1.0f  },
                                      { 13, -20.0f },
                                      { 20, -20.0f }
    })
                          .data(),
                      k_n_cls, k_u, false) == 11);
    // A special model token is never swapped away from.
    {
        std::vector<float> z = logits_with({
            { 10, 9.9f }
        });
        z[30]                = 10.5f;
        CHECK(t.pick(0, z.data(), k_n_cls, 30, false) == 30);
    }
    // Deeper states pay more: after 10, token 11 gets arc(2 + ln 2) plus the
    // backoff 10 would lose (1), about 3.69 at lambda 1.
    const BoostTrie unit  = make_trie(1.0f);
    const int       n10   = t.child(0, 10);
    const float     bonus = 2.0f + std::log(2.0f) + 1.0f;
    CHECK(unit.pick(n10,
                    logits_with({
                                    { 11, 10.0f - bonus + 0.01f }
    })
                        .data(),
                    k_n_cls, k_u, false) == 11);
    CHECK(unit.pick(n10,
                    logits_with({
                                    { 11, 10.0f - bonus - 0.01f }
    })
                        .data(),
                    k_n_cls, k_u, false) == k_u);
    // Fail-chain candidates compete unless children_only.
    CHECK(t.pick(n10,
                 logits_with({
                                 { 20, 9.0f }
    })
                     .data(),
                 k_n_cls, k_u, false) == 20);
    CHECK(t.pick(n10,
                 logits_with({
                                 { 20, 9.0f }
    })
                     .data(),
                 k_n_cls, k_u, true) == k_u);
    // The model's own token keeps its bonus when it is a candidate.
    CHECK(t.pick(0,
                 logits_with({
                                 { 10, 10.5f },
                                 { 20, 10.2f }
    })
                     .data(),
                 k_n_cls, 10, false) == 10);
}

void test_fork_verdicts() {
    const BoostTrie t = make_trie();
    int             node;
    bool            completed;

    // Single-token phrase: accepted at the fork.
    node = 0;
    CHECK(t.fork_begin(node, completed, 20) == BoostVerdict::Accept);

    // Multi-token phrase: continue, then accept on completion.
    node = 0;
    CHECK(t.fork_begin(node, completed, 10) == BoostVerdict::Continue && !completed);
    CHECK(t.fork_advance(node, completed, 13) == BoostVerdict::Accept);

    // Leaving an unfinished match rejects.
    node = 0;
    CHECK(t.fork_begin(node, completed, 10) == BoostVerdict::Continue);
    CHECK(t.fork_advance(node, completed, 7) == BoostVerdict::Reject);

    // A fork token found on the fail chain continues that match: after 10,
    // swapping in 11 then 12 completes [10 11 12].
    node = t.child(0, 10);
    CHECK(t.fork_begin(node, completed, 11) == BoostVerdict::Continue);
    CHECK(t.fork_advance(node, completed, 12) == BoostVerdict::Accept);

    // Overlap: [5] completes but [5 6] extends it; leaving after the
    // shorter phrase keeps it.
    BoostTrie ov;
    build_boost_trie(
        {
            { 5 },
            { 5, 6 }
    },
        std::vector<uint8_t>(k_n_cls, 0), 2.0f, ov);
    node = 0;
    CHECK(ov.fork_begin(node, completed, 5) == BoostVerdict::Continue && completed);
    CHECK(ov.fork_advance(node, completed, 9) == BoostVerdict::Accept);
    node = 0;
    CHECK(ov.fork_begin(node, completed, 5) == BoostVerdict::Continue);
    CHECK(ov.fork_advance(node, completed, 6) == BoostVerdict::Accept);
}

void test_guard() {
    using transcribe::parakeet::boost_guard_passed;
    CHECK(boost_guard_passed(0, 0));   // silence after the phrase in both
    CHECK(boost_guard_passed(1, 0));   // plain still finishing a replaced word
    CHECK(boost_guard_passed(3, 2));
    CHECK(!boost_guard_passed(4, 1));  // "stand up to ten thirty" -> "Sandy Thirty"
    CHECK(!boost_guard_passed(2, 0));
}

}  // namespace

int main() {
    test_shape();
    test_pick();
    test_fork_verdicts();
    test_guard();
    if (g_failures > 0) {
        std::fprintf(stderr, "parakeet_boost_unit: %d failure(s)\n", g_failures);
        return 1;
    }
    std::fprintf(stderr, "parakeet_boost_unit: OK\n");
    return 0;
}
