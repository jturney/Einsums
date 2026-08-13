//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Passes/PassUtil.hpp>
#include <Einsums/ComputeGraph/Passes/ScaleAbsorption.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Logging.hpp>

#include <algorithm>
#include <complex>
#include <ranges>
#include <unordered_map>
#include <variant>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph::passes)

namespace {

/// Where a scale of some tensor can be folded inside a consumer.
enum class FoldSite : std::uint8_t {
    None,        ///< Cannot take the factor.
    Operand,     ///< Reads the tensor as one operand: fold into the source prefactor.
    Accumulator, ///< Accumulates into the tensor: fold into the destination prefactor.
};

/// Where (if anywhere) a scale of @p tensor folds inside @p node.
///
/// **Operand**: the node reads @p tensor as EXACTLY ONE operand. Both einsum
/// and axpby are linear in their source, so scaling the source equals scaling
/// the result: `a` folds into `ab_prefactor` / `alpha`. Reading it as both
/// operands would contribute a², so that is declined.
///
/// **Accumulator**: the node writes @p tensor and reads it only as that
/// destination, with a non-zero destination prefactor. `scale(a, C)` followed
/// by `C = c_pf*C + …` is exactly `C = (c_pf*a)*C + …`, so `a` folds into
/// `c_pf` / `beta`. A node that reads its destination AND uses it as an operand
/// would again need a², so the single-read requirement stands.
///
/// Both require live shared params: the CPU executor reads the scalars from
/// EinsumParams / AxpbyParams, so a descriptor-only edit would desync it. That
/// is why permute and BatchedGemm are not folded into - they bake their
/// prefactors into the executor closure.
///
/// Every other kind - including the grouped ops (GroupedBatchedGemm,
/// GroupedDot, GroupedAxpby), whose per-entry prefactors live in a baked
/// executor closure the same way - falls through to None, which vetoes the
/// whole scale. That default is the pass's safety property: an op kind this
/// function has never heard of is a non-taker, so adding a node kind to the IR
/// cannot silently make an existing fold unsound.
FoldSite fold_site(Node const &node, TensorId tensor) {
    if (node.outputs.empty()) {
        return FoldSite::None;
    }

    // Count reads through the OPERAND slots only, by position: einsum's are
    // inputs[0] and inputs[1], axpby's is inputs[0]. Counting the whole input
    // list would not answer this, because whether an accumulating node also
    // lists its destination there depends on how the node was built -
    // Graph::make_einsum_node appends it, the capture path does not.
    size_t const operand_slots  = (node.kind == OpKind::Einsum) ? 2 : 1;
    size_t const operand_reads  = static_cast<size_t>(std::ranges::count(node.inputs | std::views::take(operand_slots), tensor));
    bool const   is_destination = node.outputs[0] == tensor;

    if (is_destination) {
        // Accumulating into the tensor AND using it as an operand would need
        // the factor twice; only the pure accumulate folds.
        if (operand_reads != 0) {
            return FoldSite::None;
        }
    } else if (operand_reads != 1) {
        return FoldSite::None;
    }

    if (node.kind == OpKind::Einsum) {
        auto const *desc = std::get_if<EinsumDescriptor>(&node.op_data);
        if (desc == nullptr || desc->params == nullptr) {
            return FoldSite::None;
        }
        if (!is_destination) {
            return FoldSite::Operand;
        }
        return is_zero(desc->params->c_pf) ? FoldSite::None : FoldSite::Accumulator;
    }

    if (node.kind == OpKind::Axpby) {
        auto const *desc = std::get_if<AxpbyDescriptor>(&node.op_data);
        if (desc == nullptr || desc->params == nullptr) {
            return FoldSite::None;
        }
        if (!is_destination) {
            return FoldSite::Operand;
        }
        return is_zero(desc->params->beta) ? FoldSite::None : FoldSite::Accumulator;
    }

    return FoldSite::None;
}

/// Multiply @p node's prefactor at @p site by @p factor. Caller guarantees
/// @ref fold_site returned that site. Writes the live params the executor reads
/// AND the snapshot beside them, so later analysis sees the same value.
void apply_fold(Node &node, FoldSite site, double factor) {
    if (auto *desc = std::get_if<EinsumDescriptor>(&node.op_data)) {
        if (site == FoldSite::Operand) {
            desc->params->ab_pf = scale_prefactor(desc->params->ab_pf, factor);
            desc->ab_prefactor  = desc->params->ab_pf;
        } else {
            desc->params->c_pf = scale_prefactor(desc->params->c_pf, factor);
            desc->c_prefactor  = desc->params->c_pf;
        }
        return;
    }
    auto *desc = std::get_if<AxpbyDescriptor>(&node.op_data);
    if (site == FoldSite::Operand) {
        desc->params->alpha = scale_prefactor(desc->params->alpha, factor);
        desc->alpha         = desc->params->alpha;
    } else {
        desc->params->beta = scale_prefactor(desc->params->beta, factor);
        desc->beta         = desc->params->beta;
    }
}

/// A short name for the fold, for logs.
char const *fold_site_name(FoldSite site) {
    return site == FoldSite::Operand ? "operand" : "accumulator";
}

} // namespace

void ScaleAbsorption::reset_stats() {
    _num_absorbed = 0;
    // Per-APPLY state, not per-run: PassManager reads compensated_reads() once,
    // after the recursive descent, and observed_writes() only inspects
    // top-level nodes. Clearing this in run() let any subgraph -- including an
    // empty loop body -- discard a top-level exemption, so the program-order
    // validator threw on a legitimate fold.
    _compensated.clear();
}

bool ScaleAbsorption::run(Graph &graph) {
    // Per-apply counters: compare against entry values, not zero. The
    // recursive driver calls run() once per subgraph and reset_stats() runs
    // only once per apply, so `_num_x > 0` would report this graph as
    // modified whenever ANY earlier subgraph changed something.
    size_t const num_absorbed_at_entry = _num_absorbed;
    graph.topological_sort();

    auto &nodes = graph.nodes();
    if (nodes.size() < 2) {
        return false;
    }

    std::vector<bool> remove(nodes.size(), false);

    auto const touches = [](std::vector<TensorId> const &v, TensorId t) { return std::ranges::find(v, t) != v.end(); };

    // Alias-aware family test, and the reason this pass needs one.
    //
    // Node I/O lists carry TensorIds, and a view of a tensor is a DIFFERENT id
    // whose handle aliases the parent. A raw id comparison therefore does not
    // see an accumulation into a SLICE of the scaled store at all: DLPNO's
    // residual clears a whole pair store with `scale(0, R)`, accumulates into
    // per-pair slice views of it, and finally folds `Rn` into `R` whole. The
    // raw scan saw only that last whole-store axpby, called it the sole
    // observer, and folded the zeroing factor into its beta - deleting every
    // slice accumulation in between. The program-order validator caught it
    // (it resolves aliases), but the fold was wrong, not merely unverifiable.
    //
    // A partial access cannot take the factor: a slice consumer's prefactor
    // scales only its own span, while the scale zeroes (or scales) the WHOLE
    // buffer, padding and untouched slots included. So any access to the
    // scaled tensor's buffer through an id that is not the scaled tensor
    // itself - a view of it, its parent, a sibling view - disqualifies the
    // scale outright, for the dead-scale path as much as for the fold.
    bool has_aliases = false;
    for (auto const &entry : graph.tensors_map()) {
        if (entry.second.aliases != 0) {
            has_aliases = true;
            break;
        }
    }
    // Owner-resolve every registered id once. Per-access resolution would walk
    // the alias chain inside a quadratic window scan; a graph with 13k tensors
    // is not unusual and the chains are path-compressed anyway.
    std::unordered_map<TensorId, TensorId> owner;
    if (has_aliases) {
        owner.reserve(graph.tensors_map().size());
        for (auto const &entry : graph.tensors_map()) {
            owner.emplace(entry.first, graph.resolve_alias(entry.first));
        }
    }
    auto const owner_of = [&owner](TensorId t) {
        auto const it = owner.find(t);
        return it == owner.end() ? t : it->second;
    };

    // What each control-flow node's SUB-GRAPHS touch.
    //
    // A Loop/Conditional node's own inputs/outputs say nothing about what its
    // body reads or writes; Graph::effective_io is what reconstructs that. The
    // window scan below is a liveness question, so a body read is exactly as
    // observable as a top-level one - without this, a scale whose only reader
    // lives inside a loop body looked dead and was dropped, and the body then
    // read the unscaled tensor.
    std::unordered_map<size_t, std::pair<std::vector<TensorId>, std::vector<TensorId>>> subgraph_io;
    for (size_t idx = 0; idx < nodes.size(); idx++) {
        if (is_control_flow(nodes[idx].kind)) {
            subgraph_io.emplace(idx, graph.effective_io(nodes[idx]));
        }
    }
    auto const node_reads = [&](size_t idx, TensorId t) {
        auto const it = subgraph_io.find(idx);
        return touches(it != subgraph_io.end() ? it->second.first : nodes[idx].inputs, t);
    };
    auto const node_writes = [&](size_t idx, TensorId t) {
        auto const it = subgraph_io.find(idx);
        return touches(it != subgraph_io.end() ? it->second.second : nodes[idx].outputs, t);
    };
    // True when node @p idx touches @p scaled's buffer through any id other
    // than @p scaled itself: a view, the parent, or a sibling slice. Control
    // flow is covered because it reads the same effective I/O the scan does.
    auto const touches_alias_of = [&](size_t idx, TensorId scaled, TensorId scaled_owner) {
        if (!has_aliases) {
            return false;
        }
        auto const  it   = subgraph_io.find(idx);
        auto const &ins  = it != subgraph_io.end() ? it->second.first : nodes[idx].inputs;
        auto const &outs = it != subgraph_io.end() ? it->second.second : nodes[idx].outputs;
        for (auto const *list : {&ins, &outs}) {
            for (TensorId const tid : *list) {
                if (tid != scaled && owner_of(tid) == scaled_owner) {
                    return true;
                }
            }
        }
        return false;
    };

    for (size_t sc = 0; sc + 1 < nodes.size(); sc++) {
        if (remove[sc]) {
            continue;
        }
        auto &scale_node = nodes[sc];
        if (scale_node.kind != OpKind::Scale) {
            continue;
        }
        auto *scale_desc = std::get_if<ScaleDescriptor>(&scale_node.op_data);
        if (scale_desc == nullptr || scale_node.outputs.size() != 1) {
            continue;
        }
        TensorId const scaled_tensor = scale_node.outputs[0];
        double const   scale_factor  = scale_desc->factor;

        // Scan the window [sc+1, next-writer-of-scaled_tensor): who observes the
        // scaled value? A `scale` is IN-PLACE, so the tensor's own value is
        // observable until it is overwritten. Folding it away is only safe when
        // the scaled value is consumed by exactly the op we compensate AND is
        // then dead — i.e. overwritten before anything else (including the
        // caller after execute) could read it. Accumulator writes read too, so
        // count them as reads.
        TensorId const      scaled_owner = owner_of(scaled_tensor);
        std::vector<size_t> readers;
        long                writer  = -1;
        long                aliased = -1;
        for (size_t tgt = sc + 1; tgt < nodes.size(); tgt++) {
            if (remove[tgt]) {
                continue;
            }
            // Checked before the exact-id tests: a node may both close the live
            // range with a whole-tensor write AND observe part of the scaled
            // value through a view, and that node must not be mistaken for the
            // sole observer.
            if (touches_alias_of(tgt, scaled_tensor, scaled_owner)) {
                aliased = static_cast<long>(tgt);
                break;
            }
            bool const writes = node_writes(tgt, scaled_tensor);
            // An accumulating node reads its destination whether or not it says
            // so in `inputs` (the capture path does not list it, only
            // Graph::make_einsum_node does), and that read observes the scale.
            bool const reads = node_reads(tgt, scaled_tensor) || (writes && reads_destination(nodes[tgt]));
            if (reads) {
                readers.push_back(tgt);
            }
            if (writes) {
                writer = static_cast<long>(tgt);
                break; // the scaled value's live range ends here
            }
        }

        if (aliased >= 0) {
            // A view of the scaled buffer (or its parent) is touched while the
            // scaled value is live. Neither elimination route is available: the
            // scale's effect on the parts that view does NOT cover is
            // observable, so no prefactor on that consumer can stand in for it,
            // and a later whole-tensor overwrite does not make the scale dead
            // either because the view's own write reads the scaled bytes.
            note_skip(
                "the scaled buffer is also accessed through a view, whose prefactor covers only its own span",
                fmt::format("scale node {} on tensor {}, aliased access by node {}", scale_node.id, scaled_tensor, nodes[aliased].id));
            continue;
        }

        if (writer < 0) {
            // Nothing overwrites the tensor afterwards, so the scaled value is
            // still observable to the caller when execute() returns. Folding
            // the scale away would change what they read back.
            note_skip("scaled tensor is never overwritten, so its scaled value stays observable",
                      fmt::format("scale node {} on tensor {}", scale_node.id, scaled_tensor));
            continue;
        }
        // A control-flow writer is opaque: its body may read the destination
        // before writing it, so it cannot be treated as a pure overwrite.
        bool const writer_overwrites = !is_control_flow(nodes[writer].kind) && pure_overwrite(nodes[writer]);

        if (readers.empty() && writer_overwrites) {
            // Dead scale: the next writer overwrites the tensor without reading
            // it, so the scale's result is discarded. Drop the Scale node.
            remove[sc] = true;
            ++_num_absorbed;
            EINSUMS_LOG_INFO("ScaleAbsorption: removed dead scale({}) of a tensor overwritten by {} node {}", scale_factor,
                             op_kind_name(nodes[writer].kind), nodes[writer].id);
            report(2, fmt::format("remove dead scale({}); {} node {} overwrites the tensor without reading it", scale_factor,
                                  op_kind_name(nodes[writer].kind), nodes[writer].id));
            continue;
        }

        // Fold: every node that observes the scaled value before it dies has to
        // take the factor. A reader that reads the tensor as an operand scales
        // its source prefactor; the accumulating writer scales its destination
        // prefactor. Both are exact, and with all of them compensated the Scale
        // itself is redundant. Bail on the whole scale if any one of them
        // cannot take it - a partial fold would be wrong, not merely missed.
        std::vector<std::pair<size_t, FoldSite>> folds;
        folds.reserve(readers.size());
        bool all_foldable = true;
        for (size_t const r : readers) {
            FoldSite const site = fold_site(nodes[r], scaled_tensor);
            if (site == FoldSite::None) {
                all_foldable = false;
                break;
            }
            folds.emplace_back(r, site);
        }
        // The writer closes the live range. If it does not read the tensor it
        // is not in `readers` and needs nothing; if it does, it was folded
        // above as the accumulator.
        if (!all_foldable) {
            // A partial fold would be wrong rather than merely missed, so one
            // unfoldable observer disqualifies the whole scale.
            note_skip("a node observing the scaled value has no prefactor to fold it into",
                      fmt::format("scale node {} on tensor {}", scale_node.id, scaled_tensor));
            continue;
        }
        if (folds.empty()) {
            note_skip("scaled value is read by nothing foldable before it dies",
                      fmt::format("scale node {} on tensor {}", scale_node.id, scaled_tensor));
            continue;
        }

        for (auto const &[idx, site] : folds) {
            apply_fold(nodes[idx], site, scale_factor);
            // That node's read of scaled_tensor now observes the tensor's
            // initial contents (the scale is gone) but is exact thanks to the
            // fold; waive the program-order guard for exactly that read.
            _compensated.emplace_back(nodes[idx].id, scaled_tensor);
            EINSUMS_LOG_INFO("ScaleAbsorption: folded scale({}) into {} node {} ({} prefactor)", scale_factor,
                             op_kind_name(nodes[idx].kind), nodes[idx].id, fold_site_name(site));
            report(2, fmt::format("fold scale({}) into {} node {} {} prefactor", scale_factor, op_kind_name(nodes[idx].kind), nodes[idx].id,
                                  fold_site_name(site)));
        }
        remove[sc] = true;
        ++_num_absorbed;
    }

    if (_num_absorbed == num_absorbed_at_entry) {
        return false;
    }
    report(1, fmt::format("eliminated {} scale(s) (dead-removed or folded into a consumer)", _num_absorbed));

    graph.erase_nodes(remove);
    graph.mark_sorted();

    return true;
}

std::vector<std::string> ScaleAbsorption::explain() const {
    if (_num_absorbed == 0) {
        return {};
    }
    return {fmt::format("ScaleAbsorption: removed {} dead scale(s)", _num_absorbed)};
}

EINSUMS_NAMESPACE_END(compute_graph::passes)
