//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Passes/PermuteFusion.hpp>
#include <Einsums/Logging.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <unordered_map>

namespace einsums::compute_graph::passes {

namespace {

/// Is this permute safe to fuse into a consumer's index pattern?
/// Must be a pure axis reordering — no scaling, no accumulation, no
/// duplicate or missing labels (which would be a diagonal/sum, not a
/// permutation).
bool can_fuse(PermuteDescriptor const &p) {
    if (p.alpha != 1.0 || p.beta != 0.0)
        return false;
    if (p.a_indices.size() != p.c_indices.size())
        return false;
    // c_indices must be a permutation of a_indices: same multiset and no duplicates.
    // Duplicates in c_indices would make the inverse map ambiguous.
    auto sorted_a = p.a_indices;
    auto sorted_c = p.c_indices;
    std::ranges::sort(sorted_a);
    std::ranges::sort(sorted_c);
    if (sorted_a != sorted_c)
        return false;
    if (std::ranges::adjacent_find(sorted_c) != sorted_c.end())
        return false;
    return true;
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
    auto *perm_desc = std::get_if<PermuteDescriptor>(&nodes[perm_idx].op_data);
    auto *ein_desc  = std::get_if<EinsumDescriptor>(&nodes[einsum_idx].op_data);
    if (!perm_desc || !ein_desc || !ein_desc->indices)
        return false;
    if (!can_fuse(*perm_desc))
        return false;

    // Old subscript: the slot's view of the permute's output.
    auto &slot_indices = (slot == 0) ? ein_desc->indices->a_indices : ein_desc->indices->b_indices;
    auto &slot_spec    = (slot == 0) ? ein_desc->spec.a_indices : ein_desc->spec.b_indices;

    // Sanity: slot rank must match permute output rank. A mismatch means
    // the graph was built inconsistently; skip defensively.
    if (slot_indices.size() != perm_desc->c_indices.size())
        return false;

    // The executor lambda captured `a_slot` / `b_slot` pointers by value,
    // so we can't change WHICH slot it reads. Instead we repoint the
    // PERMUTE-OUTPUT slot's data pointer and dims to match the
    // PRE-PERMUTE tensor — the einsum then reads the original tensor's
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
    // graph-level edge. Order doesn't matter — none of these are
    // observed until graph.execute() runs next.
    slot_indices = new_slot_sub;
    slot_spec    = new_slot_sub;

    dst_slot->ptr  = src_slot->ptr;
    dst_slot->rank = src_slot->rank;
    dst_slot->dims = src_slot->dims;
    dst_slot->name = src_slot->name;

    nodes[einsum_idx].inputs[slot] = perm_input_tid;

    nodes[einsum_idx].label = fmt::format("[fused permute {} -> {}] {}", fmt::join(perm_desc->a_indices, ","),
                                          fmt::join(perm_desc->c_indices, ","), nodes[einsum_idx].label);
    return true;
}

} // namespace

bool PermuteFusion::run(Graph &graph) {
    graph.topological_sort();

    auto &nodes     = graph.nodes();
    _num_candidates = 0;
    _num_rewrites   = 0;

    if (nodes.size() < 2)
        return false;

    // producer[tid] = index of the node that writes tensor tid. Each
    // tensor id has at most one producer in a well-formed graph, so a
    // plain map suffices.
    std::unordered_map<TensorId, size_t> producer;
    for (size_t nd = 0; nd < nodes.size(); nd++)
        for (auto tid : nodes[nd].outputs)
            producer[tid] = nd;

    // consumer_count[tid] = how many nodes read tensor tid. Needed so
    // we only fuse when the permute's output has EXACTLY one consumer
    // — otherwise removing the permute would break other readers.
    std::unordered_map<TensorId, size_t> consumer_count;
    for (auto const &n : nodes)
        for (auto tid : n.inputs)
            consumer_count[tid]++;

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
            if (nodes[prod_idx].kind != OpKind::Permute && nodes[prod_idx].kind != OpKind::Transpose)
                continue;

            _num_candidates++;

            // Safety: exactly one consumer. If the permuted tensor is
            // read by multiple downstream nodes, removing the permute
            // would break them.
            if (consumer_count[input_tid] != 1) {
                EINSUMS_LOG_INFO("PermuteFusion: skip {} (node {}) — {} consumers, need exactly 1", nodes[prod_idx].label,
                                 nodes[prod_idx].id, consumer_count[input_tid]);
                report(3,
                       fmt::format("skip permute node {} — {} consumers, need exactly 1", nodes[prod_idx].id, consumer_count[input_tid]));
                continue;
            }

            if (!try_fuse(graph, nodes, prod_idx, nd, slot)) {
                EINSUMS_LOG_INFO("PermuteFusion: skip {} (node {}) — non-pure permute (alpha/beta/dup indices)", nodes[prod_idx].label,
                                 nodes[prod_idx].id);
                report(3, fmt::format("skip permute node {} — non-pure permute (alpha/beta/dup indices)", nodes[prod_idx].id));
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

    if (_num_rewrites == 0)
        return false;
    report(1, fmt::format("absorbed {} permute(s) into einsum subscripts ({} candidate(s) examined)", _num_rewrites, _num_candidates));

    // Compact: drop marked-for-removal nodes. Same idiom as
    // ScaleAbsorption — cheap single-pass filter, preserves order.
    std::vector<Node> filtered;
    filtered.reserve(nodes.size());
    for (size_t i = 0; i < nodes.size(); i++)
        if (!remove[i])
            filtered.push_back(std::move(nodes[i]));
    nodes = std::move(filtered);

    graph.mark_sorted();
    return true;
}

} // namespace einsums::compute_graph::passes
