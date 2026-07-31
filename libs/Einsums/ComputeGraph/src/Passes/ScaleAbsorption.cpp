//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Passes/PassUtil.hpp>
#include <Einsums/ComputeGraph/Passes/ScaleAbsorption.hpp>
#include <Einsums/Logging.hpp>

#include <algorithm>
#include <complex>
#include <variant>
#include <vector>

namespace einsums::compute_graph::passes {

namespace {

/// Does `target` fully overwrite its output without reading its prior contents
/// (c_prefactor / beta == 0)? If so, a Scale of that tensor immediately
/// preceding it (with no intervening reader) is dead.

/// Can a scale of `tensor` fold into `node`'s operand prefactor? `node` must be
/// an einsum reading `tensor` as EXACTLY ONE operand (not its output, not both
/// operands — that would contribute a², not a). einsum is linear in each
/// operand, so scaling one operand equals scaling the product: fold a into
/// ab_prefactor. Requires live shared params — the CPU executor reads ab_pf from
/// EinsumParams, so a descriptor-only edit would desync it.
bool foldable_einsum_operand(Node const &node, TensorId tensor) {
    if (node.kind != OpKind::Einsum) {
        return false;
    }
    if (node.outputs.empty() || node.outputs[0] == tensor) {
        return false; // tensor is the einsum's output (accumulator), not an operand
    }
    auto const *desc = std::get_if<EinsumDescriptor>(&node.op_data);
    if (desc == nullptr || desc->params == nullptr) {
        return false; // no live params: folding into the descriptor alone would desync
    }
    return std::ranges::count(node.inputs, tensor) == 1;
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
        long reader   = -1;
        bool multiple = false;
        long writer   = -1;
        for (size_t tgt = sc + 1; tgt < nodes.size(); tgt++) {
            if (remove[tgt]) {
                continue;
            }
            auto const &node = nodes[tgt];
            if (touches(node.inputs, scaled_tensor)) {
                if (reader != -1) {
                    multiple = true;
                    break;
                }
                reader = static_cast<long>(tgt);
            }
            if (touches(node.outputs, scaled_tensor)) {
                writer = static_cast<long>(tgt);
                break; // the scaled value's live range ends here
            }
        }

        if (reader == -1 && writer >= 0 && pure_overwrite(nodes[writer])) {
            // Dead scale: the next writer overwrites the tensor without reading
            // it, so the scale's result is discarded. Drop the Scale node.
            remove[sc] = true;
            ++_num_absorbed;
            EINSUMS_LOG_INFO("ScaleAbsorption: removed dead scale({}) of a tensor overwritten by {} node {}", scale_factor,
                             op_kind_name(nodes[writer].kind), nodes[writer].id);
            report(2, fmt::format("remove dead scale({}); {} node {} overwrites the tensor without reading it", scale_factor,
                                  op_kind_name(nodes[writer].kind), nodes[writer].id));
        } else if (reader >= 0 && !multiple && writer >= 0 && foldable_einsum_operand(nodes[static_cast<size_t>(reader)], scaled_tensor)) {
            // Operand fold: the scaled value is read by exactly one einsum operand
            // and then overwritten, so it is dead everywhere else. Fold a into the
            // einsum's ab_prefactor (product scales by a) — both the live params
            // (read by the CPU executor) and the snapshot — and drop the Scale.
            Node &einsum        = nodes[static_cast<size_t>(reader)];
            auto *desc          = std::get_if<EinsumDescriptor>(&einsum.op_data);
            desc->ab_prefactor  = scale_prefactor(desc->ab_prefactor, scale_factor);
            desc->params->ab_pf = scale_prefactor(desc->params->ab_pf, scale_factor);
            remove[sc]          = true;
            ++_num_absorbed;
            // The einsum's read of scaled_tensor now observes the tensor's initial
            // contents (the scale is gone) but is exact thanks to the ab_pf fold;
            // waive the program-order guard for exactly that read.
            _compensated.emplace_back(einsum.id, scaled_tensor);
            EINSUMS_LOG_INFO("ScaleAbsorption: folded scale({}) into einsum node {} ab_prefactor", scale_factor, einsum.id);
            report(2, fmt::format("fold scale({}) into einsum node {} ab_prefactor (sole operand)", scale_factor, einsum.id));
        }
        // else: the scale is live and not (yet) foldable — keep it.
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

} // namespace einsums::compute_graph::passes
