//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/EscapeAnalysis.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Passes/PassUtil.hpp>
#include <Einsums/ComputeGraph/Passes/PermuteFusion.hpp>
#include <Einsums/ComputeGraph/Prefactor.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Logging.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <unordered_map>

EINSUMS_NAMESPACE_BEGIN(compute_graph::passes)

namespace {

/// Is this permute safe to fuse into a consumer's index pattern?
/// Must be a pure axis reordering: no scaling, no accumulation, no
/// permutation operator (an antisymmetrizer is a sum of terms, not a
/// relabeling), and no duplicate or missing labels (which would be a
/// diagonal/sum, not a permutation). The prefactors are the live ones the
/// executor reads, not the capture snapshot.
bool can_fuse(PermuteDescriptor const &p) {
    bool const pure_scalars = p.params != nullptr ? is_one(p.params->alpha) && is_zero(p.params->beta) : p.alpha == 1.0 && p.beta == 0.0;
    if (!pure_scalars || !p.operators.empty())
        return false;
    // Duplicates would make the inverse map ambiguous.
    return permutation_of(p.a_indices, p.c_indices).has_value();
}

/// Rewrite an einsum slot's subscript to absorb a preceding permute.
///
/// The permute maps input labels @p perm.a_indices to output labels @p perm.c_indices.
/// The einsum's slot currently holds @p old_sub (indexing the permute's output).
/// The returned subscript indexes the permute's *input* such that the einsum
/// computes the same result without the permute in between.
///
/// Math: for each input-axis position p, find the corresponding output-axis
/// position k where perm.c_indices[k] == perm.a_indices[p], then
/// new_sub[p] = old_sub[k].
std::vector<std::string> rewrite_subscript(PermuteDescriptor const &perm, std::vector<std::string> const &old_sub) {
    std::vector<std::string> new_sub(perm.a_indices.size());

    // Lookup from output label → position in c_indices. Safe because
    // can_fuse() ruled out duplicates.
    std::unordered_map<std::string, size_t> out_pos;
    out_pos.reserve(perm.c_indices.size());
    for (size_t k = 0; k < perm.c_indices.size(); k++)
        out_pos.emplace(perm.c_indices[k], k);

    for (size_t p = 0; p < perm.a_indices.size(); p++) {
        auto it = out_pos.find(perm.a_indices[p]);
        // can_fuse() guarantees both index sets are the same multiset,
        // so the lookup always succeeds.
        new_sub[p] = old_sub.at(it->second);
    }
    return new_sub;
}

/// Try to absorb the Permute at @p perm_idx into the einsum at @p einsum_idx,
/// which reads the permute's output at @p slot (0=A, 1=B). Returns true on
/// successful rewrite; the caller is responsible for marking the permute
/// node for removal.
bool try_fuse(Graph &graph, std::vector<Node> &nodes, size_t perm_idx, size_t einsum_idx, size_t slot) {
    auto *perm_desc = nodes[perm_idx].op_data.get_if<PermuteDescriptor>();
    auto *ein_desc  = nodes[einsum_idx].op_data.get_if<EinsumDescriptor>();
    if (!perm_desc || !ein_desc || !ein_desc->indices)
        return false;
    if (!can_fuse(*perm_desc))
        return false;

    // Old subscript: the slot's view of the permute's output.
    auto const  lists        = live_index_lists(*ein_desc);
    auto const &slot_indices = (slot == 0) ? lists.a : lists.b;

    // Sanity: slot rank must match permute output rank. A mismatch means
    // the graph was built inconsistently; skip defensively.
    if (slot_indices.size() != perm_desc->c_indices.size())
        return false;

    // The executor lambda captured `a_slot` / `b_slot` pointers by value,
    // so we can't change WHICH slot it reads. Instead we repoint the
    // PERMUTE-OUTPUT slot's data pointer and dims to match the
    // PRE-PERMUTE tensor, the einsum then reads the original tensor's
    // memory through its existing slot capture. Safe because we already
    // checked the permute's output has exactly one consumer (this
    // einsum); no other node observes the mutated slot.
    TensorId const    perm_input_tid  = nodes[perm_idx].inputs[0];
    TensorId const    perm_output_tid = nodes[perm_idx].outputs[0];
    TensorSlot const *src_slot        = graph.find_slot(perm_input_tid);  // points at the real tensor A
    TensorSlot       *dst_slot        = graph.find_slot(perm_output_tid); // currently points at A_T's temporary
    if (!src_slot || !dst_slot)
        return false;

    auto new_slot_sub = rewrite_subscript(*perm_desc, slot_indices);

    // Commit: live indices, descriptor snapshot, slot redirect, and
    // graph-level edge. Order doesn't matter, none of these are
    // observed until graph.execute() runs next.
    set_operand_indices(*ein_desc, slot == 0 ? EinsumOperand::A : EinsumOperand::B, std::move(new_slot_sub));

    // Durable redirect (not a raw ptr copy) so the einsum keeps following
    // the source tensor if it is rebound after this pass runs.
    graph.redirect_slot(perm_output_tid, perm_input_tid);
    dst_slot->rank = src_slot->rank;
    dst_slot->dims = src_slot->dims;
    dst_slot->name = src_slot->name;

    nodes[einsum_idx].inputs[slot] = perm_input_tid;

    nodes[einsum_idx].label = fmt::format("[fused permute {} -> {}] {}", fmt::join(perm_desc->a_indices, ","),
                                          fmt::join(perm_desc->c_indices, ","), nodes[einsum_idx].label);
    return true;
}

} // namespace

void PermuteFusion::reset_stats() {
    _num_candidates = 0;
    _num_rewrites   = 0;
}

bool PermuteFusion::run(Graph &graph) {
    PassCounter const rewrites{_num_rewrites};
    graph.topological_sort();

    auto &nodes = graph.nodes();

    if (nodes.size() < 2)
        return false;

    // producer[tid] = index of the node that writes tensor tid. Each
    // tensor id has at most one producer in a well-formed graph, so a
    // plain map suffices.
    std::unordered_map<TensorId, size_t> producer;
    for (size_t nd = 0; nd < nodes.size(); nd++)
        for (auto tid : nodes[nd].outputs)
            producer[tid] = nd;

    // The buffer a tensor id lands in: through a view to its parent, and through a slot redirect
    // to the tensor whose storage it now shares (CSE merges two ids that way). Two ids with the
    // same buffer are one tensor to every question below, and asking by id is how an earlier
    // version fused a permute whose output another node still wrote under a different name.
    auto const &redirects = graph.slot_redirects();
    auto const  buffer_of = [&](TensorId id) {
        for (std::size_t hop = 0; hop <= redirects.size(); hop++) {
            TensorId const root = graph.resolve_alias(id);
            auto const     next = redirects.find(root);
            if (next == redirects.end()) {
                return root;
            }
            id = next->second;
        }
        return graph.resolve_alias(id);
    };

    // Readers and value writers of every buffer, by node position. The permute's output has to
    // have exactly one of each, the permute and its consumer: removing the permute leaves the
    // buffer unwritten, and the redirect below hands the source's storage to EVERY id sharing it,
    // so another writer would write into the caller's tensor and another reader would read it.
    std::unordered_map<TensorId, std::vector<size_t>> readers;
    std::unordered_map<TensorId, std::vector<size_t>> writers;
    for (size_t nd = 0; nd < nodes.size(); nd++) {
        for (auto tid : nodes[nd].inputs)
            readers[buffer_of(tid)].push_back(nd);
        if (is_lifecycle(nodes[nd].kind))
            continue;
        for (auto tid : nodes[nd].outputs)
            writers[buffer_of(tid)].push_back(nd);
    }

    auto const        guard = EscapeAnalysis::over(graph);
    std::vector<bool> remove(nodes.size(), false);

    for (size_t nd = 0; nd < nodes.size(); nd++) {
        if (nodes[nd].kind != OpKind::Einsum)
            continue;

        // Check each input slot (0=A, 1=B) of this einsum.
        for (size_t slot = 0; slot < nodes[nd].inputs.size() && slot < 2; slot++) {
            TensorId const input_tid = nodes[nd].inputs[slot];
            auto           prod_it   = producer.find(input_tid);
            if (prod_it == producer.end())
                continue;

            size_t const prod_idx = prod_it->second;
            if (remove[prod_idx])
                continue; // already consumed by an earlier fusion this pass
            if (nodes[prod_idx].kind != OpKind::Permute)
                continue;

            _num_candidates++;

            // Removing the permute leaves its output unwritten, so the output has to be
            // graph-owned scratch nobody can read afterwards, and no sub-graph body may
            // read it: a Loop node does not list its body's reads.
            auto const *handle = graph.find_tensor(input_tid);
            if (handle == nullptr || !handle->is_intermediate) {
                note_skip("the permuted tensor is not a graph-owned intermediate, so it must still be written",
                          fmt::format("permute node {}", nodes[prod_idx].id));
                continue;
            }
            if (guard.touched_by_subtree(input_tid)) {
                note_skip("a child sub-graph references the permuted tensor", fmt::format("permute node {}", nodes[prod_idx].id));
                continue;
            }

            // Safety: exactly one consumer. If the permuted tensor is
            // read by multiple downstream nodes, removing the permute
            // would break them.
            TensorId const permuted     = buffer_of(input_tid);
            auto const     reader_count = readers[permuted].size();
            if (reader_count != 1) {
                EINSUMS_LOG_INFO("PermuteFusion: skip {} (node {}): {} consumers, need exactly 1", nodes[prod_idx].label,
                                 nodes[prod_idx].id, reader_count);
                note_skip("permuted tensor has more than one consumer, so the permute cannot be removed",
                          fmt::format("permute node {} has {} consumers", nodes[prod_idx].id, reader_count));
                continue;
            }
            if (writers[permuted].size() != 1) {
                note_skip("something besides the permute writes the permuted tensor, and the redirect would send that write into the "
                          "permute's source",
                          fmt::format("permute node {}: {} writers", nodes[prod_idx].id, writers[permuted].size()));
                continue;
            }

            // The consumer will read the source where it stands rather than where the permute
            // read it, so the source must hold the same value at both places: nothing between
            // may write it, a control-flow node's hidden writes included, and the consumer must
            // not write it either, or the fused node would read and write one buffer.
            TensorId const source = buffer_of(nodes[prod_idx].inputs[0]);
            bool           moved  = false;
            for (size_t between = prod_idx + 1; between < nd && !moved; between++) {
                moved = is_control_flow(nodes[between].kind) ||
                        std::ranges::any_of(nodes[between].outputs, [&](TensorId out) { return buffer_of(out) == source; });
            }
            if (moved) {
                note_skip("the permute's source is written between the permute and its consumer",
                          fmt::format("permute node {}", nodes[prod_idx].id));
                continue;
            }
            if (std::ranges::any_of(nodes[nd].outputs, [&](TensorId out) { return buffer_of(out) == source; })) {
                note_skip("the consumer writes the permute's source", fmt::format("permute node {}", nodes[prod_idx].id));
                continue;
            }

            if (!try_fuse(graph, nodes, prod_idx, nd, slot)) {
                EINSUMS_LOG_INFO("PermuteFusion: skip {} (node {}): non-pure permute (alpha/beta/operators/dup indices)",
                                 nodes[prod_idx].label, nodes[prod_idx].id);
                note_skip("permute is not pure (scaled, accumulating, antisymmetrized, or repeats an index)",
                          fmt::format("permute node {}", nodes[prod_idx].id));
                continue;
            }

            remove[prod_idx] = true;
            _num_rewrites++;
            EINSUMS_LOG_INFO("PermuteFusion: fused {} (node {}) into {} (node {})", nodes[prod_idx].label, nodes[prod_idx].id,
                             nodes[nd].label, nodes[nd].id);
            report(2, fmt::format("absorb permute node {} ({}) into einsum node {} ({})", nodes[prod_idx].id, nodes[prod_idx].label,
                                  nodes[nd].id, nodes[nd].label));
        }
    }

    if (!rewrites.moved())
        return false;
    report(1, fmt::format("absorbed {} permute(s) into einsum subscripts ({} candidate(s) examined)", _num_rewrites, _num_candidates));

    // Compact: drop marked-for-removal nodes. Same idiom as
    // ScaleAbsorption: cheap single-pass filter, preserves order.
    graph.erase_nodes(remove);

    graph.mark_sorted();
    return true;
}

std::vector<std::string> PermuteFusion::explain() const {
    if (num_rewrites() == 0) {
        return {};
    }
    return {fmt::format("PermuteFusion: folded {} permute(s) into contractions", num_rewrites())};
}

EINSUMS_NAMESPACE_END(compute_graph::passes)
