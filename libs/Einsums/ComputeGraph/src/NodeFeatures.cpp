//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/NodeFeatures.hpp>
#include <Einsums/ComputeGraph/Passes/PassUtil.hpp>
#include <Einsums/ComputeGraph/Prefactor.hpp>
#include <Einsums/Config/Namespace.hpp>

#include <fmt/format.h>

#include <array>
#include <complex>
#include <set>
#include <string>
#include <utility>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

namespace {

bool complex_valued(std::complex<double> const &value) {
    return value.imag() != 0.0;
}

/// Whether the node's live scalars carry an imaginary part. The LIVE ones, because a pass that
/// folded a phase into a node wrote it there and not into the capture snapshot.
bool complex_prefactor(Node const &node) {
    if (auto const *d = node.op_data.get_if<EinsumDescriptor>()) {
        return !is_real_valued(live_c_prefactor(*d)) || !is_real_valued(live_ab_prefactor(*d));
    }
    if (auto const *d = node.op_data.get_if<ScaleDescriptor>()) {
        return !is_real_valued(live_factor(*d));
    }
    if (auto const *d = node.op_data.get_if<AxpbyDescriptor>()) {
        return !is_real_valued(live_alpha(*d)) || !is_real_valued(live_beta(*d));
    }
    if (auto const *d = node.op_data.get_if<ElementwiseBinaryDescriptor>()) {
        return !is_real_valued(live_alpha(*d)) || !is_real_valued(live_beta(*d));
    }
    if (auto const *d = node.op_data.get_if<TiledEinsumDescriptor>(); d != nullptr && d->params != nullptr) {
        return !is_real_valued(d->params->c_pf) || !is_real_valued(d->params->ab_pf);
    }
    if (auto const *d = node.op_data.get_if<PermuteDescriptor>()) {
        return d->params != nullptr ? !is_real_valued(d->params->alpha) || !is_real_valued(d->params->beta)
                                    : complex_valued(d->alpha) || complex_valued(d->beta);
    }
    return false;
}

bool conjugates(Node const &node) {
    if (auto const *d = node.op_data.get_if<EinsumDescriptor>()) {
        return live_conj_a(*d) || live_conj_b(*d);
    }
    if (auto const *d = node.op_data.get_if<DotDescriptor>()) {
        return d->conjugated;
    }
    if (auto const *d = node.op_data.get_if<TiledDotDescriptor>()) {
        return d->conjugated;
    }
    return false;
}

bool grouped(OpKind kind) {
    switch (kind) {
    case OpKind::GroupedPermute:
    case OpKind::GroupedBatchedGemm:
    case OpKind::GroupedDot:
    case OpKind::GroupedAxpby:
    case OpKind::GroupedSandwich:
    case OpKind::GroupedGatherRotate:
    case OpKind::GroupedDirectProduct:
    case OpKind::GroupedDirectDivision:
        return true;
    default:
        return false;
    }
}

constexpr std::array<std::pair<NodeFeature, std::string_view>, 10> kNames{{
    {NodeFeature::PermutationOperators, "permutation operators"},
    {NodeFeature::Views, "views"},
    {NodeFeature::Conjugation, "conjugation"},
    {NodeFeature::ComplexPrefactor, "complex prefactor"},
    {NodeFeature::MixedPrecision, "mixed precision"},
    {NodeFeature::Grouped, "grouped family"},
    {NodeFeature::ControlFlow, "control flow"},
    {NodeFeature::Tiled, "tiled operand"},
    {NodeFeature::RawScalar, "bare scalar"},
    {NodeFeature::RedirectedSlot, "redirected slot"},
}};

} // namespace

NodeFeatures features_of(Graph const &graph, Node const &node) {
    NodeFeatures out;
    if (is_control_flow(node.kind)) {
        out = out | NodeFeature::ControlFlow;
    }
    if (grouped(node.kind)) {
        out = out | NodeFeature::Grouped;
    }
    if (passes::carries_permutation_operators(node)) {
        out = out | NodeFeature::PermutationOperators;
    }
    if (conjugates(node)) {
        out = out | NodeFeature::Conjugation;
    }
    if (complex_prefactor(node)) {
        out = out | NodeFeature::ComplexPrefactor;
    }

    std::set<packed_gemm::ScalarType> dtypes;
    auto const                        operand = [&](TensorId id) {
        if (graph.slot_redirects().contains(id)) {
            out = out | NodeFeature::RedirectedSlot;
        }
        TensorHandle const *handle = graph.find_tensor(id);
        if (handle == nullptr) {
            return;
        }
        // A view over a parent the graph never registered links to nothing, so the alias field
        // alone misses it; the handle knows it is a view regardless.
        if (handle->aliases != 0 || handle->is_tensor_view) {
            out = out | NodeFeature::Views;
        }
        if (handle->is_tiled) {
            out = out | NodeFeature::Tiled;
        }
        if (handle->raw_scalar) {
            out = out | NodeFeature::RawScalar;
        }
        if (handle->dtype != packed_gemm::ScalarType::Unknown) {
            dtypes.insert(handle->dtype);
        }
    };
    for (TensorId const id : node.inputs) {
        operand(id);
    }
    for (TensorId const id : node.outputs) {
        operand(id);
    }
    if (dtypes.size() > 1) {
        out = out | NodeFeature::MixedPrecision;
    }
    return out;
}

std::string describe_features(NodeFeatures features) {
    std::string out;
    for (auto const &[feature, name] : kNames) {
        if (features.covers(feature)) {
            out += out.empty() ? "" : ", ";
            out += name;
        }
    }
    return out.empty() ? std::string{"none"} : out;
}

EINSUMS_NAMESPACE_END(compute_graph)
