//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Passes/ElementWiseFusion.hpp>
#include <Einsums/ComputeGraph/Passes/PassUtil.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Logging.hpp>

#include <utility>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph::passes)

namespace {

/// Compose two axpby applications on the same (X, Y):
///   `Y = a1*X + b1*Y` then `Y = a2*X + b2*Y`
/// is `Y = (a2 + b2*a1)*X + (b2*b1)*Y`.
///
/// Computed in the prefactors' own element type, which is what the executor
/// will apply. The caller requires all four to share an alternative, so the
/// `as<T>` conversions are exact.
std::pair<PrefactorScalar, PrefactorScalar> compose_axpby(PrefactorScalar const &a1, PrefactorScalar const &b1, PrefactorScalar const &a2,
                                                          PrefactorScalar const &b2) {
    return std::visit(
        [&](auto proto) {
            using T     = decltype(proto);
            T const va1 = as<T>(a1);
            T const vb1 = as<T>(b1);
            T const va2 = as<T>(a2);
            T const vb2 = as<T>(b2);
            return std::pair<PrefactorScalar, PrefactorScalar>{PrefactorScalar{static_cast<T>(va2 + vb2 * va1)},
                                                               PrefactorScalar{static_cast<T>(vb2 * vb1)}};
        },
        a1);
}

/// The axpby source operand, or null when @p nd is not a fusable axpby.
///
/// Requires live shared params: the composed scalars are written there, which
/// is what makes this a real fusion (one sweep with new scalars) rather than
/// two executors called back to back.
AxpbyDescriptor const *fusable_axpby(Node const &nd) {
    if (nd.kind != OpKind::Axpby || nd.outputs.size() != 1 || nd.inputs.empty()) {
        return nullptr;
    }
    auto const *desc = std::get_if<AxpbyDescriptor>(&nd.op_data);
    return (desc != nullptr && desc->params != nullptr) ? desc : nullptr;
}

/// Compose two in-place scales of one tensor: `A *= f1` then `A *= f2` is
/// `A *= f1*f2`.
///
/// Computed in the prefactors' own element type, which is what the executor
/// will apply. The caller requires both to share an alternative, so the
/// `as<T>` conversions are exact.
PrefactorScalar compose_scale(PrefactorScalar const &f1, PrefactorScalar const &f2) {
    return std::visit(
        [&](auto proto) {
            using T = decltype(proto);
            return PrefactorScalar{static_cast<T>(as<T>(f1) * as<T>(f2))};
        },
        f1);
}

/// The scale descriptor, or null when @p nd is not a fusable scale.
///
/// Requires live shared params for the same reason @ref fusable_axpby does:
/// the composed factor is written there, so the merged node applies ONE
/// combined multiply. Until the scale executor read its scalar from the
/// descriptor this pass had to chain the two executors instead, which removed
/// a node but not a sweep.
ScaleDescriptor *fusable_scale(Node &nd) {
    if (nd.kind != OpKind::Scale || nd.outputs.size() != 1) {
        return nullptr;
    }
    auto *desc = std::get_if<ScaleDescriptor>(&nd.op_data);
    return (desc != nullptr && desc->params != nullptr) ? desc : nullptr;
}

} // namespace

void ElementWiseFusion::reset_stats() {
    _num_fused = 0;
}

bool ElementWiseFusion::run(Graph &graph) {
    PassCounter const fused{_num_fused};
    graph.topological_sort();

    auto &nodes = graph.nodes();
    if (nodes.size() < 2) {
        return false;
    }

    std::vector<bool> remove(nodes.size(), false);

    // ── axpby chains ───────────────────────────────────────────────────────
    //
    // `Y = a1*X + b1*Y` immediately followed by `Y = a2*X + b2*Y` on the SAME
    // pair is one axpby with composed scalars. Unlike the scale fusion below
    // this is a real fusion: axpby reads its scalars from live shared params,
    // so rewriting them leaves ONE sweep over Y instead of two.
    for (size_t i = 0; i + 1 < nodes.size(); i++) {
        if (remove[i])
            continue;
        auto const *desc_i = fusable_axpby(nodes[i]);
        if (desc_i == nullptr)
            continue;

        TensorId const y = nodes[i].outputs[0];
        TensorId const x = nodes[i].inputs[0];
        if (x == y) {
            continue; // Y = (a+b)*Y is its own shape; not worth a special case
        }

        for (size_t j = i + 1; j < nodes.size(); j++) {
            if (remove[j])
                continue;

            // Only a directly following axpby on the same (X, Y) may fuse;
            // anything else could observe or disturb Y in between.
            auto const *desc_j = fusable_axpby(nodes[j]);
            if (desc_j == nullptr || nodes[j].outputs[0] != y || nodes[j].inputs[0] != x)
                break;

            // Mixing prefactor alternatives would make the composition lossy;
            // in practice both come from the same tensor's dtype.
            auto const &a1 = desc_i->params->alpha;
            auto const &b1 = desc_i->params->beta;
            auto const &a2 = desc_j->params->alpha;
            auto const &b2 = desc_j->params->beta;
            if (a1.index() != b1.index() || a1.index() != a2.index() || a1.index() != b2.index())
                break;

            auto const [alpha, beta] = compose_axpby(a1, b1, a2, b2);

            auto *live_i          = std::get_if<AxpbyDescriptor>(&nodes[i].op_data);
            live_i->params->alpha = alpha;
            live_i->params->beta  = beta;
            live_i->alpha         = alpha;
            live_i->beta          = beta;

            // Y is read only when beta survives; keep the recorded I/O honest.
            nodes[i].inputs = is_zero(beta) ? std::vector<TensorId>{x} : std::vector<TensorId>{x, y};
            nodes[i].label  = fmt::format("axpby(alpha={}, beta={}) [fused]", to_string(alpha), to_string(beta));

            remove[j] = true;
            _num_fused++;

            EINSUMS_LOG_INFO("ElementWiseFusion: merged axpby node {} into node {} (alpha={}, beta={})", nodes[j].id, nodes[i].id,
                             to_string(alpha), to_string(beta));
            report(2, fmt::format("merge axpby node {} into node {} on the same (X, Y); composed to alpha={}, beta={}", nodes[j].id,
                                  nodes[i].id, to_string(alpha), to_string(beta)));
        }
    }

    for (size_t i = 0; i + 1 < nodes.size(); i++) {
        if (remove[i])
            continue;

        // Look for consecutive Scale ops on the same tensor
        auto *desc_i = fusable_scale(nodes[i]);
        if (desc_i == nullptr)
            continue;

        TensorId const target = nodes[i].outputs[0];

        // Scan forward for more Scale ops on the same tensor
        for (size_t j = i + 1; j < nodes.size(); j++) {
            if (remove[j])
                continue;

            // If this node reads the target but isn't a scale on it, stop
            if (nodes[j].kind != OpKind::Scale)
                break;
            if (nodes[j].outputs.size() != 1 || nodes[j].outputs[0] != target)
                break;

            auto *desc_j = fusable_scale(nodes[j]);
            if (desc_j == nullptr)
                break;

            // Mixing prefactor alternatives would make the composition lossy;
            // in practice both come from the same tensor's dtype.
            if (desc_i->params->alpha.index() != desc_j->params->alpha.index())
                break;

            // Fuse for real: one multiply with the composed factor, written
            // through the live params the executor reads.
            PrefactorScalar const composed = compose_scale(desc_i->params->alpha, desc_j->params->alpha);
            std::string const     was_i    = to_string(desc_i->params->alpha);
            std::string const     was_j    = to_string(desc_j->params->alpha);

            desc_i->params->alpha = composed;
            desc_i->factor        = composed;

            nodes[i].label = fmt::format("scale({}) [fused]", to_string(composed));

            remove[j] = true;
            _num_fused++;

            EINSUMS_LOG_INFO("ElementWiseFusion: merged scale({}) into scale({}) -> scale({})", was_j, was_i, to_string(composed));
            report(2, fmt::format("merge scale({}) into scale({}) on the same tensor; composed to {}", was_j, was_i, to_string(composed)));
        }
    }

    if (!fused.moved())
        return false;
    report(1, fmt::format("fused {} element-wise op chain(s)", _num_fused));

    graph.erase_nodes(remove);
    graph.mark_sorted();

    return true;
}

std::vector<std::string> ElementWiseFusion::explain() const {
    if (num_fused() == 0) {
        return {};
    }
    return {fmt::format("ElementWiseFusion: fused {} elementwise pair(s)", num_fused())};
}

EINSUMS_NAMESPACE_END(compute_graph::passes)
