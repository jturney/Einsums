//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Passes/InplaceOptimization.hpp>
#include <Einsums/ComputeGraph/Prefactor.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Logging.hpp>

#include <algorithm>
#include <cstddef>
#include <functional>
#include <optional>
#include <unordered_map>
#include <unordered_set>

#include "LifecycleNodes.hpp"

EINSUMS_NAMESPACE_BEGIN(compute_graph::passes)

namespace {

/// Consumers whose output may safely alias a (dying) input: element-aligned
/// elementwise ops, where out[i] depends only on the inputs' element i, so
/// writing through the aliased buffer never corrupts a value still to be
/// read. Contractions (Einsum/Gemm/BatchedGemm) and permutes must NOT alias
/// output with input and stay excluded. The pure-overwrite requirement
/// (output not read) is checked separately via the out-tensor-as-input
/// recording convention: an op that reads its destination lists it in
/// Node::inputs.
///
/// Permute keeps coming up as a candidate - a transpose whose source dies is
/// an obvious buffer to reuse - and it is unsound, not merely unimplemented:
/// out[j,i] comes from in[i,j], so writing through a shared buffer destroys
/// elements not yet read. Transposing a 4x4 in place through the out-of-place
/// kernel gives max abs error 1.68 against the correct transpose. An IDENTITY
/// permute (a scaled copy) would be alias-safe, but PermuteFusion already
/// removes those. See the guard test "a permute is never merged onto its dying
/// input". GroupedPermute is excluded for the same reason.
///
/// ElementTransform is element-aligned but records `{c_id}` as both input and
/// output (it is in-place), so the pure-overwrite check excludes it anyway and
/// adding it here would buy nothing.
///
/// The grouped kinds are element-aligned member by member: member i's output
/// depends only on member i's own operands, element by element. Members run
/// at once, so a member may only reuse the storage of one of its OWN inputs;
/// @ref member_sources says which those are.
bool elementwise_alias_safe(OpKind kind) {
    return kind == OpKind::DirectProduct || kind == OpKind::DirectDivision || kind == OpKind::Axpby || kind == OpKind::GroupedAxpby ||
           kind == OpKind::GroupedDirectProduct || kind == OpKind::GroupedDirectDivision;
}

/// The inputs each output of an element-aligned consumer is computed from, one list per output.
///
/// A plain node has one output computed from all of its inputs. A grouped node interleaves its
/// members' inputs (``x_i`` for an axpby, ``a_i, b_i`` for a product or quotient), followed by the
/// member's own destination when its beta is nonzero; the betas are what say where each member's
/// operands sit, exactly as the executor reads them. Empty when the input list does not match the
/// descriptor, which leaves nothing to merge.
std::vector<std::vector<TensorId>> member_sources(Node const &node) {
    if (node.kind == OpKind::DirectProduct || node.kind == OpKind::DirectDivision || node.kind == OpKind::Axpby) {
        if (node.outputs.size() != 1) {
            return {};
        }
        return {node.inputs};
    }
    std::vector<PrefactorScalar> const *betas   = nullptr;
    std::size_t                         sources = 2;
    if (auto const *d = node.op_data.get_if<GroupedAxpbyDescriptor>(); d != nullptr && node.kind == OpKind::GroupedAxpby) {
        betas   = &d->betas;
        sources = 1;
    } else if (auto const *e = node.op_data.get_if<GroupedElementwiseDescriptor>();
               e != nullptr && (node.kind == OpKind::GroupedDirectProduct || node.kind == OpKind::GroupedDirectDivision)) {
        betas = &e->betas;
    }
    if (betas == nullptr || betas->size() != node.outputs.size()) {
        return {};
    }
    std::vector<std::vector<TensorId>> out;
    out.reserve(node.outputs.size());
    std::size_t cursor = 0;
    for (auto const &beta : *betas) {
        std::size_t const width = sources + (is_zero(beta) ? 0 : 1);
        if (cursor + width > node.inputs.size()) {
            return {};
        }
        out.emplace_back(node.inputs.begin() + static_cast<std::ptrdiff_t>(cursor),
                         node.inputs.begin() + static_cast<std::ptrdiff_t>(cursor + width));
        cursor += width;
    }
    if (cursor != node.inputs.size()) {
        return {};
    }
    return out;
}

struct UseInfo {
    size_t writes{0};
    size_t reads{0};
    size_t writer_idx{SIZE_MAX}; ///< only meaningful when writes == 1
    size_t first_read{SIZE_MAX}; ///< position of the earliest reader
};

/// One storage merge: consumer node `node_idx` writes `dst` while reading
/// `src` for the last time; `dst` will reuse `src`'s buffer.
struct MergePlan {
    size_t   node_idx;
    TensorId dst; ///< output tensor merged away
    TensorId src; ///< the tensor owning the dying input's storage
};

/// Whether the pass may rewrite a node; notes the skip when it may not.
using Accepts = std::function<bool(Node const &)>;

/// Reads per BUFFER, not per id. A reader that names a tensor whose slot was redirected onto
/// another's (see Graph::redirect_slot) reads that other buffer, so counting by id saw a tensor
/// die at a node while a later node still read it through the redirect. Writes stay per id: the
/// graph verifier refuses a node that writes a redirected slot, and a write through a view lands
/// in a tensor somebody views, which no merge takes.
std::unordered_map<TensorId, UseInfo> buffer_uses(Graph const &graph) {
    std::unordered_map<TensorId, UseInfo> uses;
    auto const                           &nodes = graph.nodes();
    for (size_t idx = 0; idx < nodes.size(); idx++) {
        if (is_lifecycle(nodes[idx].kind)) {
            continue;
        }
        for (auto tid : nodes[idx].outputs) {
            auto &u = uses[tid];
            u.writes++;
            u.writer_idx = idx;
        }
        for (auto tid : nodes[idx].inputs) {
            auto &u = uses[graph.buffer_of(tid)];
            u.reads++;
            u.first_read = std::min(u.first_read, idx);
        }
    }
    return uses;
}

/// Find the first sound merge in `graph`, or nullopt. Called repeatedly
/// until quiescent because each applied merge deletes lifecycle nodes and
/// rewrites ids (graphs are small and merges are rare, so the restart is
/// cheaper than maintaining incremental state). `accepts` says whether the
/// pass may rewrite a node; a merge rewrites the consumer and every node
/// naming the merged-away tensor, so each of them must be accepted.
std::optional<MergePlan> find_merge(Graph &graph, Accepts const &accepts) {
    graph.topological_sort();

    auto const &nodes   = graph.nodes();
    auto const &tensors = graph.tensors_map();

    // Control-flow bodies reference parent tensors without listing them in
    // the parent nodes' plain input/output lists (only effective_io sees
    // them), so plain use-counts under-count and a merge could corrupt a
    // tensor a body still reads. Bodies are processed on their own recursion
    // level; the parent level is skipped when control flow is present.
    for (auto const &node : nodes) {
        if (is_control_flow(node.kind)) {
            return std::nullopt;
        }
        // GPU placement swaps buffers behind slots (device shadows); storage
        // merging is host-only for now.
        if (node.target == Target::GPU || node.kind == OpKind::HostToDevice || node.kind == OpKind::DeviceToHost) {
            return std::nullopt;
        }
    }

    auto uses = buffer_uses(graph);

    // Tensors somebody views: their storage must not be repurposed.
    std::unordered_set<TensorId> view_targets;
    for (auto const &[tid, handle] : tensors) {
        if (handle.aliases != 0) {
            view_targets.insert(handle.aliases);
        }
    }

    // A tensor redirected onto another's storage needs no test here: its reads and writes are
    // counted under the other tensor, so it never shows the single writer a merge asks for.
    auto mergeable_intermediate = [&](TensorId tid) {
        auto it = tensors.find(tid);
        // Tile-wise sparse tensors are excluded explicitly rather than by accident.
        // A merge redirects one TensorId at another tensor's storage, which assumes
        // a single buffer; a tiled tensor is a map of per-tile buffers, and two
        // tiled tensors with different grids are not interchangeable at all.
        //
        // Today this is unreachable -- tiled ops record as OpKind::Custom, and the
        // whitelist below only admits element-aligned kinds -- but that is
        // incidental. Binding axpby for tiled operands (currently unimplemented,
        // and on the roadmap) would make a tiled node land in the whitelist with
        // nothing else stopping it.
        return it != tensors.end() && it->second.is_intermediate && it->second.aliases == 0 && !it->second.is_tensor_view &&
               !it->second.is_tiled && !view_targets.contains(tid);
    };

    for (size_t idx = 0; idx < nodes.size(); idx++) {
        Node const &node = nodes[idx];
        if (!elementwise_alias_safe(node.kind)) {
            continue;
        }
        auto const members = member_sources(node);
        if (members.empty() || !accepts(node)) {
            continue;
        }

        for (size_t member = 0; member < members.size(); member++) {
            TensorId const dst = node.outputs[member];

            // Pure overwrite: an op that reads its destination lists it as an
            // input (the gemm/direct_product recording convention). Any input
            // of the node, not only the member's own: members run at once, so
            // another member reading dst would read the buffer this one writes.
            if (std::ranges::any_of(node.inputs, [&](TensorId in) { return graph.buffer_of(in) == dst; })) {
                continue;
            }
            if (!mergeable_intermediate(dst)) {
                continue;
            }
            auto const dst_use = uses[dst];
            if (dst_use.writes != 1) {
                continue;
            }
            // Nothing may read dst before this node writes it. Such a reader sees
            // the value dst held before (zero, or the last replay's), and after
            // the merge it would read the dying source instead.
            if (dst_use.first_read < idx) {
                continue;
            }
            // An Initialize on dst would be dead after the merge but deleting it
            // is only sound because of the pure-overwrite check above; keep v1
            // simple and skip such candidates instead.
            bool dst_has_init = false;
            for (auto const &n : nodes) {
                if (n.kind == OpKind::Initialize && std::ranges::find(n.outputs, dst) != n.outputs.end()) {
                    dst_has_init = true;
                    break;
                }
            }
            if (dst_has_init) {
                continue;
            }

            auto const &dst_handle = tensors.at(dst);

            // The merge renames dst in every node naming it and drops dst's
            // lifecycle nodes, so each of those must be one the pass may rewrite.
            // Asked once a src qualifies, so a candidate that would not merge
            // anyway is not reported as skipped.
            auto dst_nodes_accepted = [&] {
                for (auto const &n : nodes) {
                    bool const names_dst = std::ranges::find(n.inputs, dst) != n.inputs.end() ||
                                           std::ranges::find(n.outputs, dst) != n.outputs.end() ||
                                           lifecycle::lifecycle_tensor_name(n) == dst_handle.name;
                    if (names_dst && !accepts(n)) {
                        return false;
                    }
                }
                return true;
            };

            for (auto const read : members[member]) {
                // The storage the member reads. A redirected id reads the tensor it
                // was redirected to, and that tensor is what dst would reuse.
                TensorId const src = graph.buffer_of(read);
                if (src == dst || !mergeable_intermediate(src)) {
                    continue;
                }
                auto const *read_handle = graph.find_tensor(read);
                if (read_handle == nullptr || read_handle->aliases != 0 || read_handle->is_tensor_view) {
                    continue;
                }
                auto const src_use = uses[src];
                // src must die here: one producer that runs before this node, and
                // this node is its only reader. A producer after it would write src
                // again after the merge put dst's value there.
                if (src_use.writes != 1 || src_use.reads != 1 || src_use.writer_idx >= idx) {
                    continue;
                }
                auto const &src_handle = tensors.at(src);
                // Equal dtype as well as equal bytes: a float64 and a complex64
                // buffer of the same size are not interchangeable storage.
                if (src_handle.dims != dst_handle.dims || src_handle.dtype != dst_handle.dtype ||
                    src_handle.total_bytes() != dst_handle.total_bytes()) {
                    continue;
                }
                if (!dst_nodes_accepted()) {
                    break; // the same nodes name dst whatever the src, so no src of this candidate can merge
                }

                return MergePlan{.node_idx = idx, .dst = dst, .src = src};
            }
        }
    }

    return std::nullopt;
}

/// Apply a merge: `dst` disappears from the graph metadata (every reference
/// from the consumer onward becomes `src`), its lifecycle nodes are removed,
/// and its executor slot is durably redirected at `src`'s storage so baked
/// lambdas follow.
void apply_merge(Graph &graph, MergePlan const &plan, size_t &num_merged) {
    auto &nodes = graph.nodes();

    for (auto &node : nodes) {
        std::ranges::replace(node.inputs, plan.dst, plan.src);
        std::ranges::replace(node.outputs, plan.dst, plan.src);
    }

    // dst's Alloc/Materialize (now rewritten to src) duplicate src's own
    // lifecycle; drop the duplicates that originally belonged to dst. After
    // the rewrite they are indistinguishable by id, so match by the tensor
    // their descriptor names.
    auto const &dst_name = graph.tensor(plan.dst).name;
    std::erase_if(nodes, [&](Node const &n) { return lifecycle::lifecycle_tensor_name(n) == dst_name; });

    graph.redirect_slot(plan.dst, plan.src);
    graph.mark_sorted(); // order-preserving rewrite; position-keyed deps are stale
    num_merged++;
}

void process(Graph &graph, Accepts const &accepts, size_t &num_candidates, size_t &num_merged) {
    // Candidate census (kept for introspection parity with the old
    // analysis-only behavior).
    {
        graph.topological_sort();
        auto uses = buffer_uses(graph);
        for (auto const &[tid, handle] : graph.tensors_map()) {
            if (handle.is_intermediate && uses[tid].writes == 1 && uses[tid].reads == 1) {
                num_candidates++;
            }
        }
    }

    while (auto plan = find_merge(graph, accepts)) {
        auto const &src_name = graph.tensor(plan->src).name;
        auto const &dst_name = graph.tensor(plan->dst).name;
        EINSUMS_LOG_INFO("InplaceOptimization: '{}' reuses the storage of dying '{}' ({} bytes saved)", dst_name, src_name,
                         graph.tensor(plan->dst).total_bytes());
        apply_merge(graph, *plan, num_merged);
    }
}

} // namespace

void InplaceOptimization::reset_stats() {
    _num_candidates = 0;
    _num_merged     = 0;
}

bool InplaceOptimization::run(Graph &graph) {
    PassCounter const merged{_num_merged};
    PassCounter const candidates{_num_candidates};
    Accepts const     accepts = [this, &graph](Node const &node) {
        if (understands(graph, node)) {
            return true;
        }
        note_skip("the node carries a feature this pass does not understand",
                  fmt::format("node '{}': {}", node.label, describe_features(features_of(graph, node))));
        return false;
    };
    process(graph, accepts, _num_candidates, _num_merged);

    if (merged.moved()) {
        report(1, fmt::format("merged {} output buffer(s) into dying elementwise inputs ({} candidate(s) found)", _num_merged,
                              _num_candidates));
    } else if (candidates.moved()) {
        report(1, fmt::format("found {} in-place candidate tensor(s), none safely mergeable", _num_candidates));
    }

    return merged.moved();
}

std::vector<std::string> InplaceOptimization::explain() const {
    if (num_merged() == 0 && num_candidates() == 0) {
        return {};
    }
    return {fmt::format("InplaceOptimization: merged {} buffer(s) into dying inputs ({} candidate(s))", num_merged(), num_candidates())};
}

EINSUMS_NAMESPACE_END(compute_graph::passes)
