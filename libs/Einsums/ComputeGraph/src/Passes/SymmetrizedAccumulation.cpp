//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/EinsumSpec.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Passes/SymmetrizedAccumulation.hpp>
#include <Einsums/ComputeGraph/StringDispatch.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <complex>
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
    _num_rewritten  = 0;

    auto        &nodes = graph.nodes();
    size_t const n     = nodes.size();

    auto const contains   = [](std::vector<TensorId> const &v, TensorId t) { return std::find(v.begin(), v.end(), t) != v.end(); };
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
    // An accumulating axpby reads its destination: `dst` in BOTH inputs and
    // outputs (Operations.hpp records inputs = {src, dst} only when beta != 0).
    auto const is_accumulating_axpby = [&](Node const &node, TensorId src, TensorId dst) {
        return node.kind == OpKind::Axpby && node.outputs.size() == 1 && node.outputs[0] == dst && contains(node.inputs, src) &&
               contains(node.inputs, dst);
    };

    // A safely-foldable site (interference-clean). Collected in a first pass so
    // the rewrite does not mutate `nodes` mid-scan.
    struct Site {
        size_t                   permute_idx;
        size_t                   axpby2_idx;
        TensorId                 tmp;
        TensorId                 r2;
        PrefactorScalar          s2; // axpby2's alpha: r2 += s2 * P(tmp)
        std::vector<std::string> a_indices;
        std::vector<std::string> c_indices;
    };
    std::vector<Site> sites;

    for (size_t pi = 0; pi < n; ++pi) {
        Node const &permute = nodes[pi];
        if (permute.kind != OpKind::Permute) {
            continue;
        }
        auto const *pd = std::get_if<PermuteDescriptor>(&permute.op_data);
        if (pd == nullptr) {
            continue;
        }
        // Pure overwrite permutation (tmpP freshly written): beta == 0.
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
        long a1 = -1;
        for (size_t const ri : readers_of(tmp)) {
            if (ri == pi) {
                continue; // the permute itself
            }
            if (is_accumulating_axpby(nodes[ri], tmp, r2)) {
                if (a1 != -1) {
                    a1 = -2; // ambiguous
                    break;
                }
                a1 = static_cast<long>(ri);
            }
        }
        if (a1 < 0) {
            continue;
        }

        ++_num_candidates;

        // Interference guard: over [first, last] of the three matched nodes, no
        // OTHER node may write tmp or touch r2 except an additive accumulation
        // (which commutes). See docs/symmetrized_accumulation_design.md.
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
        if (!clean) {
            continue;
        }
        ++_num_matched;

        // s2 is axpby2's alpha. Requires the axpby descriptor (added alongside
        // this pass); without it the scalar is unreadable and we cannot fold.
        auto const *ad = std::get_if<AxpbyDescriptor>(&axpby2.op_data);
        if (ad == nullptr) {
            continue;
        }
        sites.push_back(Site{pi, a2, tmp, r2, ad->alpha, pd->a_indices, pd->c_indices});
    }

    // ── Rewrite (Level 1): fold each runtime-tensor site by making the permute
    // accumulate directly into r2 (r2 += s2 * P(tmp)) and dropping axpby2 + tmpP.
    std::vector<bool> remove(nodes.size(), false);
    Graph            *graph_ptr = &graph;

    for (auto const &s : sites) {
        // Runtime-tensor / uniform-dtype gate (typed captures fold-out; the
        // executor casts to GeneralRuntimeTensor<T>, which would be UB on a
        // typed Tensor<T, Rank> - the bug-1015 class).
        auto const *th_tmp = graph.find_tensor(s.tmp);
        auto const *th_r2  = graph.find_tensor(s.r2);
        if (th_tmp == nullptr || th_r2 == nullptr || !th_tmp->is_runtime || !th_r2->is_runtime || th_tmp->dtype != th_r2->dtype) {
            continue;
        }
        // Only fold a pure permutation (alpha == 1); a scaled permute would need
        // s2 * alpha folded into the accumulate, deferred until it appears.
        auto *pd = std::get_if<PermuteDescriptor>(&nodes[s.permute_idx].op_data);
        if (pd == nullptr || pd->alpha != 1.0) {
            continue;
        }
        auto const dtype = th_r2->dtype;

        ParsedPermuteSpec pspec;
        pspec.c_indices = s.c_indices;
        pspec.a_indices = s.a_indices;
        pspec.raw       = fmt::format("{} <- {}", fmt::join(s.c_indices, ","), fmt::join(s.a_indices, ","));

        TensorId const        r2  = s.r2;
        TensorId const        tmp = s.tmp;
        PrefactorScalar const s2  = s.s2;

        auto exec = [graph_ptr, r2, tmp, pspec, s2, dtype]() {
            auto build = [&]<typename T>(T /*tag*/) {
                using RT  = GeneralRuntimeTensor<T, std::allocator<T>>;
                auto *dst = static_cast<RT *>(graph_ptr->tensor(r2).tensor_ptr);
                auto *src = static_cast<RT *>(graph_ptr->tensor(tmp).tensor_ptr);
                dispatch::string_permute<RT, RT>(pspec, T{1}, dst, as<T>(s2), *src); // r2 = 1*r2 + s2*P(tmp)
            };
            switch (dtype) {
            case packed_gemm::ScalarType::Float32:
                build(float{});
                break;
            case packed_gemm::ScalarType::Float64:
                build(double{});
                break;
            case packed_gemm::ScalarType::Complex64:
                build(std::complex<float>{});
                break;
            case packed_gemm::ScalarType::Complex128:
                build(std::complex<double>{});
                break;
            default:
                EINSUMS_THROW_EXCEPTION(std::invalid_argument, "SymmetrizedAccumulation: unknown ScalarType");
            }
        };

        Node &perm   = nodes[s.permute_idx];
        perm.execute = std::move(exec);
        perm.inputs  = {tmp, r2}; // accumulate: r2 is read (beta = 1) - RMW convention
        perm.outputs = {r2};
        perm.label   = "symacc: r2 += s2 * P(tmp)";
        pd->beta     = 1.0; // was 0 (overwrite tmpP); now accumulate into r2

        remove[s.axpby2_idx] = true; // axpby2 folded into the permute
        ++_num_rewritten;
    }

    if (_num_rewritten == 0) {
        return false;
    }

    std::vector<Node> filtered;
    filtered.reserve(nodes.size());
    for (size_t i = 0; i < nodes.size(); ++i) {
        if (!remove[i]) {
            filtered.push_back(std::move(nodes[i]));
        }
    }
    nodes = std::move(filtered);
    graph.topological_sort();
    return true;
}

} // namespace einsums::compute_graph::passes
