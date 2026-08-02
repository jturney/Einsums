//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/Detail/ScalarDispatch.hpp>
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

std::vector<std::string> SymmetrizedAccumulation::explain() const {
    if (_num_rewritten == 0) {
        return {};
    }
    return {fmt::format("SymmetrizedAccumulation: folded {} symmetrization site(s) of {} matched ({} candidate(s) examined)",
                        _num_rewritten, _num_matched, _num_candidates)};
}

void SymmetrizedAccumulation::reset_stats() {
    _num_candidates = 0;
    _num_matched    = 0;
    _num_rewritten  = 0;
}

bool SymmetrizedAccumulation::run(Graph &graph) {
    // Per-apply counters: compare against entry values, not zero. The
    // recursive driver calls run() once per subgraph and reset_stats() runs
    // only once per apply, so `_num_x > 0` would report this graph as
    // modified whenever ANY earlier subgraph changed something.
    size_t const num_rewritten_at_entry = _num_rewritten;
    auto        &nodes                  = graph.nodes();
    size_t const n                      = nodes.size();

    auto const contains = [](std::vector<TensorId> const &v, TensorId t) { return std::find(v.begin(), v.end(), t) != v.end(); };
    // Generation bounds for scratch that is REUSED across sites (the CCSD body
    // recycles one tmp/tmpP for every symacc term, so whole-graph
    // writer/reader-uniqueness rejected every site of a shared buffer). A
    // write's value is live exactly until the tensor's next write; matching
    // within that window pairs each producer with its own consumers.
    auto const next_write_after = [&](TensorId tid, size_t pos) {
        for (size_t i = pos + 1; i < n; ++i) {
            if (contains(nodes[i].outputs, tid)) {
                return i;
            }
        }
        return n;
    };
    auto const generation_begin = [&](TensorId tid, size_t pos) -> size_t {
        for (size_t i = pos; i-- > 0;) {
            if (contains(nodes[i].outputs, tid)) {
                return i + 1;
            }
        }
        return 0;
    };
    // An accumulating axpby reads its destination: `dst` in BOTH inputs and
    // outputs (Operations.hpp records inputs = {src, dst} only when beta != 0).
    //
    // cg::axpy records an Axpby with beta == 1, so `r2 += tmp` - the spelling
    // every other library uses - matches here like any other accumulation. A
    // pass-built node with no descriptor is rejected below, where the scalar is
    // read.
    auto const is_accumulating_axpby = [&](Node const &node, TensorId src, TensorId dst) {
        return node.kind == OpKind::Axpby && node.outputs.size() == 1 && node.outputs[0] == dst && contains(node.inputs, src) &&
               contains(node.inputs, dst);
    };
    // The live beta of an axpby node, or null when the node carries no axpby
    // descriptor (a pass-built node, say) and the scalar is therefore unknowable.
    // Prefer the shared params over the descriptor snapshot: an earlier pass that
    // folded a scale into this axpby wrote beta through the params handle, and
    // that value -- not the at-capture snapshot -- is what the executor will read.
    auto const axpby_beta = [](Node const &node) -> PrefactorScalar const * {
        auto const *ad = std::get_if<AxpbyDescriptor>(&node.op_data);
        if (ad == nullptr) {
            return nullptr;
        }
        return ad->params ? &ad->params->beta : &ad->beta;
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
            note_skip("permute accumulates into its output (beta != 0)", fmt::format("permute #{}", pi));
            continue;
        }
        if (!is_involution(pd->a_indices, pd->c_indices)) {
            note_skip("permutation is not an involution", fmt::format("permute #{} '{}'", pi, permute.label));
            continue;
        }
        if (permute.inputs.size() != 1 || permute.outputs.size() != 1) {
            note_skip("permute has unexpected operand arity", fmt::format("permute #{}", pi));
            continue;
        }
        TensorId const tmp  = permute.inputs[0];
        TensorId const tmpP = permute.outputs[0];

        // This permute's tmpP value is live until tmpP's next overwrite; it
        // must be consumed by exactly one reader in that window, an
        // accumulating axpby. Readers outside the window belong to other
        // generations of a reused buffer and are irrelevant here.
        size_t const tmpP_gen_end = next_write_after(tmpP, pi);
        long         a2_found     = -1;
        for (size_t i = pi + 1; i < tmpP_gen_end; ++i) {
            if (contains(nodes[i].inputs, tmpP)) {
                if (a2_found != -1) {
                    a2_found = -2; // second consumer of this generation
                    break;
                }
                a2_found = static_cast<long>(i);
            }
        }
        if (a2_found == -1) {
            note_skip("permuted result has no consumer in its generation", fmt::format("permute #{}", pi));
            continue;
        }
        if (a2_found < 0) {
            note_skip("permuted result has more than one consumer", fmt::format("permute #{}", pi));
            continue;
        }
        size_t const a2     = static_cast<size_t>(a2_found);
        Node const  &axpby2 = nodes[a2];
        if (axpby2.outputs.size() != 1) {
            note_skip("consumer of the permuted result has unexpected output arity", fmt::format("node #{}", a2));
            continue;
        }
        TensorId const r2 = axpby2.outputs[0];
        if (!is_accumulating_axpby(axpby2, tmpP, r2)) {
            note_skip("consumer of the permuted result is not an accumulating axpy/axpby", fmt::format("node #{} '{}'", a2, axpby2.label));
            continue;
        }

        // Sibling accumulating axpby reading the UN-permuted tmp into the same
        // r2, scoped to tmp's generation containing the permute (from tmp's
        // most recent write before pi to its next write after): every read in
        // that window observes the value the permute transposes, and reads of
        // other generations of a reused tmp cannot masquerade as the sibling.
        size_t const tmp_gen_begin = generation_begin(tmp, pi);
        size_t const tmp_gen_end   = next_write_after(tmp, pi);
        long         a1            = -1;
        for (size_t ri = tmp_gen_begin; ri < tmp_gen_end; ++ri) {
            if (ri == pi || !contains(nodes[ri].inputs, tmp)) {
                continue; // the permute itself / not a reader
            }
            if (is_accumulating_axpby(nodes[ri], tmp, r2)) {
                if (a1 != -1) {
                    a1 = -2; // ambiguous
                    break;
                }
                a1 = static_cast<long>(ri);
            }
        }
        if (a1 == -1) {
            note_skip("no sibling accumulate of the un-permuted operand into the same output", fmt::format("permute #{}", pi));
            continue;
        }
        if (a1 < 0) {
            note_skip("more than one candidate sibling accumulate", fmt::format("permute #{}", pi));
            continue;
        }

        ++_num_candidates;

        // Interference guard: over [first, last] of the three matched nodes, no
        // OTHER node may write tmp or touch r2 except an additive accumulation
        // (which commutes). See docs/symmetrized_accumulation_design.md.
        //
        // ALIAS-AWARE: touches are resolved to the owning tensor, so a node
        // reading or writing r2 THROUGH A VIEW (the DF ladder accumulates into
        // r2[i,j] slices, and a topological sort may interleave those nodes
        // with a symacc site) is seen by the guard. The raw-id check missed
        // them entirely and only tripped, incidentally, on the View-creation
        // node's parent input - which is a metadata rebind, not a value read,
        // and is exempted below.
        TensorId const r2_owner      = graph.resolve_alias(r2);
        TensorId const tmp_owner     = graph.resolve_alias(tmp);
        auto const     touches_owner = [&](std::vector<TensorId> const &ids, TensorId owner) {
            return std::any_of(ids.begin(), ids.end(), [&](TensorId raw) { return graph.resolve_alias(raw) == owner; });
        };
        // The live destination prefactor of an einsum, through the shared
        // params when present (a pass that rewrote c_pf wrote it there).
        auto const einsum_c_pf = [](Node const &node) -> PrefactorScalar const * {
            auto const *ed = std::get_if<EinsumDescriptor>(&node.op_data);
            if (ed == nullptr) {
                return nullptr;
            }
            return ed->params ? &ed->params->c_pf : &ed->c_prefactor;
        };

        size_t const first = std::min({static_cast<size_t>(a1), pi, a2});
        size_t const last  = std::max({static_cast<size_t>(a1), pi, a2});
        bool         clean = true;
        for (size_t i = first + 1; i < last; ++i) {
            if (i == pi || i == static_cast<size_t>(a1) || i == a2) {
                continue;
            }
            // A View node only rebinds slice metadata; it observes no values.
            // Its consumers read/write through the view id and are classified
            // below via alias resolution.
            if (nodes[i].kind == OpKind::View) {
                continue;
            }
            bool const writes_tmp = touches_owner(nodes[i].outputs, tmp_owner);
            bool const touches_r2 = touches_owner(nodes[i].inputs, r2_owner) || touches_owner(nodes[i].outputs, r2_owner);
            // Only a PURE accumulation into r2 (or a slice of it) commutes with
            // the fold: an axpby with beta == 1, or an einsum with a live
            // destination prefactor of exactly 1 (the ladder's r2[i,j] += ...).
            // The rewrite moves the permuted contribution from axpby2's
            // position back to the permute's, so anything in between that
            // rescales the running r2 -- a damping/mixing step with beta not
            // in {0, 1}, routine in SCF and DIIS codes -- would apply its
            // scale to a contribution that had not been added yet.
            // beta == 0 needs no check here: a pure overwrite does not list r2
            // as an input, so it fails the inputs test and lands in touches_r2.
            // A node with no readable scalar is treated as non-commuting.
            // Both exemptions require r2 to appear ONLY as the accumulated
            // destination: an op that also reads r2 as a SOURCE operand
            // (axpby's x, an einsum's A or B) consumes the half-symmetrized
            // values and does not commute.
            // A plain `r2 += something_else` between the halves is the
            // commuting accumulation it looks like, so read its beta rather
            // than treating every intervening write as interference.
            PrefactorScalar const *beta           = axpby_beta(nodes[i]);
            bool                   additive_accum = beta != nullptr && is_one(*beta) && nodes[i].inputs.size() == 2 &&
                                  graph.resolve_alias(nodes[i].inputs[0]) != r2_owner && touches_owner(nodes[i].outputs, r2_owner) &&
                                  touches_owner(nodes[i].inputs, r2_owner);
            if (!additive_accum && nodes[i].kind == OpKind::Einsum) {
                PrefactorScalar const *c_pf = einsum_c_pf(nodes[i]);
                additive_accum = c_pf != nullptr && is_one(*c_pf) && nodes[i].inputs.size() >= 2 && nodes[i].outputs.size() == 1 &&
                                 graph.resolve_alias(nodes[i].inputs[0]) != r2_owner &&
                                 graph.resolve_alias(nodes[i].inputs[1]) != r2_owner && touches_owner(nodes[i].outputs, r2_owner);
            }
            if (writes_tmp || (touches_r2 && !additive_accum)) {
                note_skip(writes_tmp ? "an intervening node rewrites the permute source"
                                     : "an intervening node observes the half-symmetrized output",
                          fmt::format("permute #{} blocked by node #{} '{}' inside [{}..{}]", pi, i, nodes[i].label, first, last));
                clean = false;
                break;
            }
        }
        if (!clean) {
            continue;
        }
        ++_num_matched;

        // s2 is axpby2's alpha. Requires the axpby descriptor (which cg::axpy
        // now carries too); without it - a pass-built node - the scalar is
        // unreadable and we cannot fold. Read it through the live params when
        // present: an earlier pass that folded a scale into this accumulate
        // wrote it there, and that is the value the executor will use.
        auto const *ad = std::get_if<AxpbyDescriptor>(&axpby2.op_data);
        if (ad == nullptr) {
            note_skip("accumulate carries no readable scalar (pass-built node without a descriptor)",
                      fmt::format("node #{} '{}'", a2, axpby2.label));
            continue;
        }
        sites.push_back(Site{pi, a2, tmp, r2, ad->params ? ad->params->alpha : ad->alpha, pd->a_indices, pd->c_indices});
    }

    // ── Rewrite (Level 1): fold each runtime-tensor site by making the permute
    // accumulate directly into r2 (r2 += s2 * P(tmp)) and dropping axpby2 + tmpP.
    std::vector<bool> remove(nodes.size(), false);
    Graph            *graph_ptr = &graph;

    for (auto const &s : sites) {
        // Runtime-tensor / uniform-dtype gate (typed captures fold-out; the
        // executor casts to GeneralRuntimeTensor<T>, which would be UB on a
        // typed Tensor<T, Rank>).
        auto const *th_tmp = graph.find_tensor(s.tmp);
        auto const *th_r2  = graph.find_tensor(s.r2);
        if (th_tmp == nullptr || th_r2 == nullptr) {
            note_skip("operand tensor is not registered in the graph", fmt::format("permute #{}", s.permute_idx));
            continue;
        }
        if (!th_tmp->is_runtime || !th_r2->is_runtime) {
            note_skip(
                "operands are statically-typed tensors, not RuntimeTensor - this fold only applies to runtime-ranked operands",
                fmt::format("permute #{}: tmp.is_runtime={}, r2.is_runtime={}", s.permute_idx, th_tmp->is_runtime, th_r2->is_runtime));
            continue;
        }
        if (th_tmp->dtype != th_r2->dtype) {
            note_skip("operands have mixed dtypes", fmt::format("permute #{}", s.permute_idx));
            continue;
        }
        // Only fold a pure permutation (alpha == 1); a scaled permute would need
        // s2 * alpha folded into the accumulate, deferred until it appears.
        auto *pd = std::get_if<PermuteDescriptor>(&nodes[s.permute_idx].op_data);
        if (pd == nullptr || pd->alpha != 1.0) {
            note_skip("permute is scaled (alpha != 1)", fmt::format("permute #{}", s.permute_idx));
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
            detail::dispatch_scalar_type(dtype, build);
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

    if (_num_rewritten == num_rewritten_at_entry) {
        return false;
    }

    graph.erase_nodes(remove);
    graph.topological_sort();
    return true;
}

} // namespace einsums::compute_graph::passes
