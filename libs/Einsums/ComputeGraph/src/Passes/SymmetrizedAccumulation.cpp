//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Passes/SymmetrizedAccumulation.hpp>

#include <algorithm>
#include <string>
#include <vector>

namespace einsums::compute_graph::passes {

namespace {

// P is an involution: applying the axis permutation (a_indices -> c_indices)
// twice is the identity. Requires distinct labels (a repeated label is a
// diagonal, not this pattern). "jiba <- ijab" -> perm [1,0,3,2], an involution.
bool is_involution(std::vector<std::string> const &a, std::vector<std::string> const &c) {
    if (a.empty() || a.size() != c.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        for (size_t j = i + 1; j < a.size(); ++j) {
            if (a[i] == a[j]) {
                return false; // repeated label: ill-defined permutation
            }
        }
    }
    std::vector<size_t> perm(c.size());
    for (size_t k = 0; k < c.size(); ++k) {
        auto it = std::find(a.begin(), a.end(), c[k]);
        if (it == a.end()) {
            return false; // c is not a permutation of a
        }
        perm[k] = static_cast<size_t>(it - a.begin());
    }
    for (size_t k = 0; k < perm.size(); ++k) {
        if (perm[perm[k]] != k) {
            return false;
        }
    }
    return true;
}

} // namespace

bool SymmetrizedAccumulation::run(Graph &graph) {
    _num_candidates = 0;
    _num_matched    = 0;

    auto const  &nodes = graph.nodes();
    size_t const n     = nodes.size();

    auto const contains = [](std::vector<TensorId> const &v, TensorId t) { return std::find(v.begin(), v.end(), t) != v.end(); };
    // Node indices writing / reading a tensor. O(n) scans; the matcher slice is
    // correctness-first, the rewrite slice will use the graph UsageAnalysis.
    auto const writers_of = [&](TensorId tid) {
        std::vector<size_t> w;
        for (size_t i = 0; i < n; ++i) {
            if (contains(nodes[i].outputs, tid)) {
                w.push_back(i);
            }
        }
        return w;
    };
    auto const readers_of = [&](TensorId tid) {
        std::vector<size_t> r;
        for (size_t i = 0; i < n; ++i) {
            if (contains(nodes[i].inputs, tid)) {
                r.push_back(i);
            }
        }
        return r;
    };
    // An accumulating axpby reads its destination: `dst` appears in BOTH inputs
    // and outputs (Operations.hpp records inputs = {src, dst} only when beta != 0;
    // an overwrite axpby lists only {src}). This is how the matcher reads the
    // accumulate/overwrite distinction without an axpby descriptor.
    auto const is_accumulating_axpby = [&](Node const &node, TensorId src, TensorId dst) {
        return node.kind == OpKind::Axpby && node.outputs.size() == 1 && node.outputs[0] == dst && contains(node.inputs, src) &&
               contains(node.inputs, dst);
    };

    for (size_t pi = 0; pi < n; ++pi) {
        Node const &permute = nodes[pi];
        if (permute.kind != OpKind::Permute) {
            continue;
        }
        auto const *pd = std::get_if<PermuteDescriptor>(&permute.op_data);
        if (pd == nullptr) {
            continue;
        }
        // Pure overwrite permutation (tmpP freshly written): beta == 0. The
        // source prefactor alpha is left ungated; the rewrite folds it into s2.
        if (pd->beta != 0.0) {
            continue;
        }
        if (!is_involution(pd->a_indices, pd->c_indices)) {
            continue;
        }
        if (permute.inputs.size() != 1 || permute.outputs.size() != 1) {
            continue;
        }
        TensorId const tmp  = permute.inputs[0];
        TensorId const tmpP = permute.outputs[0];

        // tmpP must be sole-produced by this permute and sole-consumed by one axpby.
        if (writers_of(tmpP).size() != 1) {
            continue;
        }
        auto const tmpP_readers = readers_of(tmpP);
        if (tmpP_readers.size() != 1) {
            continue;
        }
        size_t const a2     = tmpP_readers[0];
        Node const  &axpby2 = nodes[a2];
        if (axpby2.outputs.size() != 1) {
            continue;
        }
        TensorId const r2 = axpby2.outputs[0];
        if (!is_accumulating_axpby(axpby2, tmpP, r2)) {
            continue;
        }

        // Sibling accumulating axpby reading the UN-permuted tmp into the same r2.
        // tmp also has the permute as a reader; find the unique sibling axpby.
        long a1 = -1;
        for (size_t const ri : readers_of(tmp)) {
            if (ri == pi) {
                continue; // the permute itself
            }
            if (is_accumulating_axpby(nodes[ri], tmp, r2)) {
                if (a1 != -1) {
                    a1 = -2; // ambiguous: more than one candidate
                    break;
                }
                a1 = static_cast<long>(ri);
            }
        }
        if (a1 < 0) {
            continue;
        }

        ++_num_candidates;

        // Interference guard: over [first, last] of the three matched nodes in
        // node order, no OTHER node may
        //   - write tmp (would change the source mid-fold), or
        //   - touch r2, UNLESS it is an additive accumulation into r2.
        // Merging the two axpby into one node moves r2's un-permuted contribution
        // to the permuted axpby's position; an intervening ADDITIVE r2 += X
        // commutes so the fold stays exact, but a pure r2 reader (would observe
        // the half-symmetrized value), an r2 overwrite (discards the moved
        // contribution), or a non-additive RMW like Scale (r2 *= c does not
        // commute) is unsafe. Sibling symacc sites accumulate into the same r2,
        // so allowing additive accumulations is what makes multi-site folding
        // safe once the topological sort separates a permute from its axpby.
        //
        // Structural additive test: OpKind::Axpby with r2 in BOTH inputs and
        // outputs. This cannot yet distinguish beta == 1 (exact) from beta != 1
        // (non-commuting) - axpby carries no descriptor, so beta is invisible
        // (see docs/symmetrized_accumulation_design.md). The CCSD idiom only ever
        // uses beta == 1, so this is correct for the workload; the precise gate
        // is a follow-up once axpby gains a descriptor.
        size_t const first = std::min({static_cast<size_t>(a1), pi, a2});
        size_t const last  = std::max({static_cast<size_t>(a1), pi, a2});
        bool         clean = true;
        for (size_t i = first + 1; i < last; ++i) {
            if (i == pi || i == static_cast<size_t>(a1) || i == a2) {
                continue;
            }
            bool const writes_tmp     = contains(nodes[i].outputs, tmp);
            bool const touches_r2     = contains(nodes[i].inputs, r2) || contains(nodes[i].outputs, r2);
            bool const additive_accum = nodes[i].kind == OpKind::Axpby && contains(nodes[i].inputs, r2) && contains(nodes[i].outputs, r2);
            if (writes_tmp || (touches_r2 && !additive_accum)) {
                clean = false;
                break;
            }
        }
        if (clean) {
            ++_num_matched;
        }
    }

    return false; // matcher-only slice: never modifies the graph
}

} // namespace einsums::compute_graph::passes
