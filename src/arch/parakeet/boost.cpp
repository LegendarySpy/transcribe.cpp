// arch/parakeet/boost.cpp - phrase-boosting trie (see boost.h).

#include "boost.h"

#include <algorithm>
#include <cmath>
#include <utility>

namespace transcribe::parakeet {

int BoostTrie::child(int node, int tok) const {
    const Node &    n     = nodes[static_cast<size_t>(node)];
    const int32_t * first = child_tok.data() + n.first_child;
    const int32_t * last  = first + n.n_children;
    const int32_t * it    = std::lower_bound(first, last, tok);
    if (it == last || *it != tok) {
        return -1;
    }
    return child_node[static_cast<size_t>(it - child_tok.data())];
}

int BoostTrie::next(int node, int tok) const {
    for (;;) {
        const int c = child(node, tok);
        if (c >= 0) {
            return c;
        }
        if (node == 0) {
            return 0;
        }
        node = nodes[static_cast<size_t>(node)].fail;
    }
}

int BoostTrie::pick(int node, const float * logits, int n_classes, int u, bool children_only) const {
    if ((token_flags[static_cast<size_t>(u)] & k_boost_token_special) != 0) {
        return u;
    }
    const float dflt = nodes[static_cast<size_t>(node)].chain_backoff;

    // bonus(t) = sum of backoff from `node` down to the first chain state
    // f with a t-arc (f excluded) + arc(f, t) - dflt.
    auto bonus = [&](int tok) -> float {
        float acc = 0.0f;
        for (int f = node;; f = nodes[static_cast<size_t>(f)].fail) {
            const int c = child(f, tok);
            if (c >= 0) {
                return acc + nodes[static_cast<size_t>(c)].arc - dflt;
            }
            if (f == 0 || children_only) {
                return 0.0f;
            }
            acc += nodes[static_cast<size_t>(f)].backoff;
        }
    };

    int   best   = u;
    float best_v = logits[u] + lambda * bonus(u);
    bool  have_f = false;
    float floor  = 0.0f;
    float acc    = 0.0f;
    for (int f = node;; f = nodes[static_cast<size_t>(f)].fail) {
        const Node & fn = nodes[static_cast<size_t>(f)];
        for (int k = 0; k < fn.n_children; ++k) {
            const size_t i   = static_cast<size_t>(fn.first_child + k);
            const int    tok = child_tok[i];
            if (tok == u || (token_flags[static_cast<size_t>(tok)] & k_boost_token_special) != 0) {
                continue;
            }
            // A deeper chain state with the same arc already scored tok.
            bool shadowed = false;
            for (int g = node; g != f; g = nodes[static_cast<size_t>(g)].fail) {
                if (child(g, tok) >= 0) {
                    shadowed = true;
                    break;
                }
            }
            if (shadowed) {
                continue;
            }
            const float v = logits[tok] + lambda * (acc + nodes[static_cast<size_t>(child_node[i])].arc - dflt);
            if (v <= best_v) {
                continue;
            }
            if (!have_f) {
                // log(sum exp z) + log(floor): the floor on the logit scale.
                float mx = logits[0];
                for (int j = 1; j < n_classes; ++j) {
                    mx = std::max(mx, logits[j]);
                }
                double sum = 0.0;
                for (int j = 0; j < n_classes; ++j) {
                    sum += std::exp(static_cast<double>(logits[j] - mx));
                }
                floor  = mx + static_cast<float>(std::log(sum)) + std::log(k_boost_floor);
                have_f = true;
            }
            if (logits[tok] >= floor) {
                best   = tok;
                best_v = v;
            }
        }
        if (f == 0 || children_only) {
            break;
        }
        acc += fn.backoff;
    }
    return best;
}

BoostVerdict BoostTrie::fork_begin(int & node, bool & completed, int tok) const {
    node           = next(node, tok);
    const Node & n = nodes[static_cast<size_t>(node)];
    completed      = n.end;
    return completed && n.n_children == 0 ? BoostVerdict::Accept : BoostVerdict::Continue;
}

BoostVerdict BoostTrie::fork_advance(int & node, bool & completed, int tok) const {
    const int c = child(node, tok);
    if (c < 0) {
        return completed ? BoostVerdict::Accept : BoostVerdict::Reject;
    }
    node           = c;
    const Node & n = nodes[static_cast<size_t>(c)];
    if (n.end) {
        completed = true;
        if (n.n_children == 0) {
            return BoostVerdict::Accept;
        }
    }
    return BoostVerdict::Continue;
}

void build_boost_trie(const std::vector<std::vector<int32_t>> & phrases,
                      std::vector<uint8_t>                      token_flags,
                      float                                     lambda,
                      BoostTrie &                               out) {
    out             = BoostTrie{};
    out.token_flags = std::move(token_flags);
    out.lambda      = lambda;
    if (lambda <= 0.0f) {
        return;
    }

    // Insert into a scratch trie with unsorted (token, node) edges.
    std::vector<std::vector<std::pair<int32_t, int32_t>>> edges(1);
    std::vector<float>                                    arc(1, 0.0f);
    std::vector<float>                                    score(1, 0.0f);
    std::vector<uint8_t>                                  end(1, 0);
    for (const auto & ph : phrases) {
        if (ph.empty()) {
            continue;
        }
        int32_t node = 0;
        for (size_t d = 0; d < ph.size(); ++d) {
            int32_t c = -1;
            for (const auto & e : edges[static_cast<size_t>(node)]) {
                if (e.first == ph[d]) {
                    c = e.second;
                    break;
                }
            }
            if (c < 0) {
                const float s = d == 0 ? k_boost_c0 : k_boost_c0 * k_boost_beta + std::log(static_cast<float>(d + 1));
                c             = static_cast<int32_t>(edges.size());
                edges[static_cast<size_t>(node)].emplace_back(ph[d], c);
                edges.emplace_back();
                arc.push_back(s);
                score.push_back(score[static_cast<size_t>(node)] + s);
                end.push_back(0);
            }
            node = c;
        }
        end[static_cast<size_t>(node)] = 1;
    }

    // Flatten into sorted child spans.
    const size_t n_nodes = edges.size();
    out.nodes.resize(n_nodes);
    for (size_t v = 0; v < n_nodes; ++v) {
        auto & e = edges[v];
        std::sort(e.begin(), e.end());
        BoostTrie::Node & n = out.nodes[v];
        n.first_child       = static_cast<int32_t>(out.child_tok.size());
        n.n_children        = static_cast<int32_t>(e.size());
        n.arc               = arc[v];
        n.end               = end[v] != 0;
        for (const auto & te : e) {
            out.child_tok.push_back(te.first);
            out.child_node.push_back(te.second);
        }
    }

    // Fail links, backoff and chain_backoff in BFS order (a fail target is
    // always shallower, so it is final before its dependents).
    std::vector<int32_t> queue;
    queue.reserve(n_nodes);
    queue.push_back(0);
    for (size_t qi = 0; qi < queue.size(); ++qi) {
        const int32_t     v  = queue[qi];
        BoostTrie::Node & vn = out.nodes[static_cast<size_t>(v)];
        if (v != 0) {
            vn.backoff       = vn.end ? 0.0f : score[static_cast<size_t>(vn.fail)] - score[static_cast<size_t>(v)];
            vn.chain_backoff = vn.backoff + out.nodes[static_cast<size_t>(vn.fail)].chain_backoff;
        }
        for (int k = 0; k < vn.n_children; ++k) {
            const int32_t tok = out.child_tok[static_cast<size_t>(vn.first_child + k)];
            const int32_t c   = out.child_node[static_cast<size_t>(vn.first_child + k)];
            int32_t       f   = 0;
            if (v != 0) {
                f = vn.fail;
                while (out.child(f, tok) < 0 && f != 0) {
                    f = out.nodes[static_cast<size_t>(f)].fail;
                }
                const int32_t g = out.child(f, tok);
                f               = g >= 0 ? g : 0;
            }
            out.nodes[static_cast<size_t>(c)].fail = f;
            queue.push_back(c);
        }
    }
}

}  // namespace transcribe::parakeet
