//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/EinsumSpec.hpp>
#include <Einsums/ComputeGraph/EscapeAnalysis.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Passes/PassUtil.hpp>
#include <Einsums/ComputeGraph/Passes/SymmetrizedAccumulation.hpp>
#include <Einsums/Config/Namespace.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <string>
#include <unordered_set>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph::passes)

namespace {

// P is an involution: applying the axis permutation (a_indices -> c_indices)
// twice is the identity. Requires distinct labels (a repeated label is a
// diagonal, not this pattern). "jiba <- ijab" -> perm [1,0,3,2], an involution.
bool is_involution(std::vector<std::string> const &a, std::vector<std::string> const &c) {
    auto const perm = permutation_of(a, c);
    if (a.empty() || !perm.has_value()) {
        return false;
    }
    for (std::size_t k = 0; k < perm->size(); ++k) {
        if ((*perm)[(*perm)[k]] != k) {
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

namespace {

/// The storage the ordinary nodes of @p graph reference, as tensor pointers. A control-flow node is
/// left out: its lists name what its body touches, and the body is visited on its own.
void collect_value_ptrs(Graph const &graph, std::unordered_set<void const *> &out) {
    for (auto const &node : graph.nodes()) {
        if (is_control_flow(node.kind) || is_lifecycle(node.kind)) {
            continue;
        }
        for (auto const &ids : {node.inputs, node.outputs}) {
            for (TensorId const id : ids) {
                if (auto const *handle = graph.find_tensor(graph.buffer_of(id)); handle != nullptr && handle->tensor_ptr != nullptr) {
                    out.insert(handle->tensor_ptr);
                }
            }
        }
    }
}

} // namespace

bool SymmetrizedAccumulation::run(Graph &graph) {
    return run_one(graph, {}, {});
}

bool SymmetrizedAccumulation::run_one(Graph &graph, std::unordered_set<void const *> const &external,
                                      std::unordered_set<void const *> const &inherited_scratch) {
    PassCounter const rewritten{_num_rewritten};
    auto             &nodes   = graph.nodes();
    size_t const      n       = nodes.size();
    auto const        escapes = EscapeAnalysis::over(graph);

    auto const contains = [](std::vector<TensorId> const &v, TensorId t) { return std::ranges::find(v, t) != v.end(); };
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
    // every other library uses - matches here like any other accumulation.
    //
    // The LIVE beta must be exactly one, not merely nonzero: the fold rewrites
    // `r2 = beta*r2 + s*P(tmp)` as `r2 += s*P(tmp)`, which drops any other beta.
    // A pass-built node with no descriptor has no readable beta and does not
    // match.
    auto const is_accumulating_axpby = [&](Node const &node, TensorId src, TensorId dst) {
        if (node.kind != OpKind::Axpby || node.outputs.size() != 1 || node.outputs[0] != dst || !contains(node.inputs, src) ||
            !contains(node.inputs, dst)) {
            return false;
        }
        PrefactorScalar const *beta = axpby_beta(node);
        return beta != nullptr && is_one(*beta);
    };
    // A safely-foldable site (interference-clean). Collected in a first pass so
    // the rewrite does not mutate `nodes` mid-scan.
    struct Site {
        size_t          permute_idx;
        size_t          axpby2_idx;
        TensorId        tmp;
        TensorId        r2;
        PrefactorScalar s2; // axpby2's alpha: r2 += s2 * P(tmp)
    };
    std::vector<Site> sites;

    for (size_t pi = 0; pi < n; ++pi) {
        Node const &permute = nodes[pi];
        if (permute.kind != OpKind::Permute) {
            continue;
        }
        if (!understands(graph, permute)) {
            note_skip("the node carries a feature this pass does not understand",
                      fmt::format("node '{}': {}", permute.label, describe_features(features_of(graph, permute))));
            continue;
        }
        auto const *pd = permute.op_data.get_if<PermuteDescriptor>();
        if (pd == nullptr) {
            continue;
        }
        // Pure overwrite permutation (tmpP freshly written): beta == 0. The
        // live beta, which is what the executor applies.
        if (!is_zero(pd->params != nullptr ? pd->params->beta : PrefactorScalar{pd->beta})) {
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

        // The fold stops writing tmpP, so tmpP must be graph-owned scratch nothing outside this
        // graph can observe: a caller's tensor, or one a loop body or branch reads, would be left
        // holding a stale value. It must also be its own storage, not a view or a redirected slot:
        // those land in a buffer other ids read.
        if (auto const *handle = graph.find_tensor(graph.buffer_of(tmpP));
            graph.buffer_of(tmpP) != tmpP || handle == nullptr || handle->tensor_ptr == nullptr ||
            !(handle->is_intermediate || inherited_scratch.contains(handle->tensor_ptr)) || external.contains(handle->tensor_ptr) ||
            escapes.touched_by_subtree(tmpP)) {
            note_skip("the permuted result is not graph-owned scratch this graph alone reads", fmt::format("permute #{}", pi));
            continue;
        }

        // This permute's tmpP value is live until the next node that REPLACES it: a pure
        // overwrite. A read-modify-write of tmpP (tmpP += X, a scale of it) reads this value and
        // so is one of its consumers, not the end of its generation. Compared by buffer, so a read
        // through a view of tmpP counts. It must be consumed by exactly one reader in that
        // window, an accumulating axpby.
        TensorId const tmpP_buffer = graph.buffer_of(tmpP);
        auto const     names_tmpP  = [&](std::vector<TensorId> const &ids) {
            return std::ranges::any_of(ids, [&](TensorId id) { return graph.buffer_of(id) == tmpP_buffer; });
        };
        size_t tmpP_gen_end = n;
        for (size_t i = pi + 1; i < n; ++i) {
            if (names_tmpP(nodes[i].outputs) && !names_tmpP(nodes[i].inputs)) {
                tmpP_gen_end = i;
                break;
            }
        }
        long a2_found = -1;
        for (size_t i = pi + 1; i < tmpP_gen_end; ++i) {
            if (names_tmpP(nodes[i].inputs)) {
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
        // The fold reads tmp in the node that writes r2, where the program read tmp only to write
        // tmpP. A tmp laid over r2's storage (a view of it) would be read while it is overwritten.
        if (graph.buffer_of(tmp) == graph.buffer_of(r2)) {
            note_skip("the permute source shares storage with the output", fmt::format("permute #{}", pi));
            continue;
        }
        if (!understands(graph, axpby2)) {
            note_skip("the node carries a feature this pass does not understand",
                      fmt::format("node '{}': {}", axpby2.label, describe_features(features_of(graph, axpby2))));
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
        TensorId const r2_owner      = graph.buffer_of(r2);
        TensorId const tmp_owner     = graph.buffer_of(tmp);
        auto const     touches_owner = [&](std::vector<TensorId> const &ids, TensorId owner) {
            return std::any_of(ids.begin(), ids.end(), [&](TensorId raw) { return graph.buffer_of(raw) == owner; });
        };
        // The live destination prefactor of an einsum, or null when the node is not one.
        auto const einsum_c_pf = [](Node const &node) -> PrefactorScalar const * {
            auto const *ed = node.op_data.get_if<EinsumDescriptor>();
            return ed != nullptr ? &live_c_prefactor(*ed) : nullptr;
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
            // A loop or branch does not list what its body reads or writes, so it cannot be
            // shown not to observe the half-symmetrized r2.
            if (is_control_flow(nodes[i].kind)) {
                note_skip("a control-flow node sits between the two halves", fmt::format("permute #{} node #{}", pi, i));
                clean = false;
                break;
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
            PrefactorScalar const *beta = axpby_beta(nodes[i]);
            bool additive_accum         = beta != nullptr && is_one(*beta) && nodes[i].inputs.size() == 2 &&
                                          graph.buffer_of(nodes[i].inputs[0]) != r2_owner && touches_owner(nodes[i].outputs, r2_owner) &&
                                          touches_owner(nodes[i].inputs, r2_owner);
            if (!additive_accum && nodes[i].kind == OpKind::Einsum) {
                PrefactorScalar const *c_pf = einsum_c_pf(nodes[i]);
                additive_accum = c_pf != nullptr && is_one(*c_pf) && nodes[i].inputs.size() >= 2 && nodes[i].outputs.size() == 1 &&
                                 graph.buffer_of(nodes[i].inputs[0]) != r2_owner && graph.buffer_of(nodes[i].inputs[1]) != r2_owner &&
                                 touches_owner(nodes[i].outputs, r2_owner);
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
        auto const *ad = axpby2.op_data.get_if<AxpbyDescriptor>();
        if (ad == nullptr) {
            note_skip("accumulate carries no readable scalar (pass-built node without a descriptor)",
                      fmt::format("node #{} '{}'", a2, axpby2.label));
            continue;
        }
        sites.push_back(Site{pi, a2, tmp, r2, live_alpha(*ad)});
    }

    // ── Rewrite (Level 1): fold each site by making the permute accumulate directly into r2
    // (r2 += s2 * alpha * P(tmp)) and dropping axpby2 + tmpP.
    //
    // The folded node is built by the library's own permute factory from one descriptor, so its
    // executor applies exactly what the descriptor says: the permute's operators, both scalars,
    // and operands reached through their impls, whatever tensor type or view they are. The
    // operator is linear and its signs are real, so `tmpP = alpha P(tmp)` then `r2 += s2 tmpP` is
    // `r2 += (s2 alpha) P(tmp)`; the executor applies the destination's one to the first term only.
    std::vector<bool> remove(nodes.size(), false);

    for (auto const &s : sites) {
        auto const *th_tmp = graph.find_tensor(s.tmp);
        auto const *th_r2  = graph.find_tensor(s.r2);
        if (th_tmp == nullptr || th_r2 == nullptr) {
            note_skip("operand tensor is not registered in the graph", fmt::format("permute #{}", s.permute_idx));
            continue;
        }
        if (th_tmp->dtype != th_r2->dtype) {
            note_skip("operands have mixed dtypes", fmt::format("permute #{}", s.permute_idx));
            continue;
        }
        // The factory reaches both operands through their rank-erased impls; only a tile-wise
        // sparse tensor has none.
        if (!th_tmp->impl_fn || !th_r2->impl_fn) {
            note_skip("an operand has no rank-erased impl", fmt::format("permute #{}", s.permute_idx));
            continue;
        }
        auto const *pd = nodes[s.permute_idx].op_data.get_if<PermuteDescriptor>();
        if (pd == nullptr) {
            continue;
        }
        PrefactorScalar const alpha = pd->params != nullptr
                                          ? pd->params->alpha
                                          : (pd->alpha.imag() == 0.0 ? PrefactorScalar{pd->alpha.real()} : PrefactorScalar{pd->alpha});
        // A real destination cannot take an imaginary scale, and the two nodes as captured refuse
        // one at execute. Their product can be real (i times i), so each is asked on its own: a
        // fold must not turn that refusal into an answer.
        bool const complex_destination =
            th_r2->dtype == packed_gemm::ScalarType::Complex64 || th_r2->dtype == packed_gemm::ScalarType::Complex128;
        if (!complex_destination && (!is_real_valued(s.s2) || !is_real_valued(alpha))) {
            note_skip("a real destination would be scaled by a complex factor", fmt::format("permute #{}", s.permute_idx));
            continue;
        }
        PrefactorScalar const factor = multiply_prefactors(s.s2, alpha);

        ParsedPermuteSpec spec;
        spec.c_indices = pd->c_indices;
        spec.a_indices = pd->a_indices;
        spec.operators = pd->operators;
        spec.raw       = spec.render();

        Node  folded = graph.make_permute_node(s.tmp, s.r2, spec, factor, PrefactorScalar{double{1}}, "symacc: r2 += s2 * P(tmp)");
        Node &perm   = nodes[s.permute_idx];
        perm.execute = std::move(folded.execute);
        perm.op_data = std::move(folded.op_data);
        perm.inputs  = std::move(folded.inputs);
        perm.outputs = std::move(folded.outputs);
        perm.label   = std::move(folded.label);

        remove[s.axpby2_idx] = true; // axpby2 folded into the permute
        ++_num_rewritten;
    }

    bool modified = false;
    if (rewritten.moved()) {
        graph.erase_nodes(remove);
        graph.topological_sort();
        modified = true;
    }

    // Descend, the way DeadNodeElimination does. A body sees as external what this graph's own
    // nodes reference and what every sibling sub-tree references, and inherits as scratch every
    // intermediate an enclosing graph owns. A reference that resolves to no pointer means the
    // walk cannot vouch for anything it inherited, so that body folds only its own scratch.
    std::vector<Graph *> children;
    graph.for_each_subgraph([&children](Graph &sub) { children.push_back(&sub); });
    if (children.empty()) {
        return modified;
    }
    std::unordered_set<void const *> own = external;
    collect_value_ptrs(graph, own);
    std::unordered_set<void const *> scratch = inherited_scratch;
    for (auto const &[id, handle] : graph.tensors_map()) {
        if (handle.is_intermediate && handle.tensor_ptr != nullptr) {
            scratch.insert(handle.tensor_ptr);
        }
    }
    std::vector<std::unordered_set<void const *>> subtrees(children.size());
    bool                                          unresolved = false;
    for (std::size_t i = 0; i < children.size(); ++i) {
        collect_value_ptrs(*children[i], subtrees[i]);
        children[i]->collect_subtree_referenced_ptrs(subtrees[i], &unresolved);
    }
    for (std::size_t i = 0; i < children.size(); ++i) {
        std::unordered_set<void const *> child_external = own;
        for (std::size_t j = 0; j < children.size(); ++j) {
            if (j != i) {
                child_external.insert(subtrees[j].begin(), subtrees[j].end());
            }
        }
        if (run_one(*children[i], child_external, unresolved ? std::unordered_set<void const *>{} : scratch)) {
            modified = true;
        }
    }
    return modified;
}

EINSUMS_NAMESPACE_END(compute_graph::passes)
