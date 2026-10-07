//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include "NodeCost.hpp"

#include <Einsums/ComputeGraph/CostModel.hpp>
#include <Einsums/ComputeGraph/DescriptorHooks.hpp>
#include <Einsums/ComputeGraph/DestinationRead.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/Config/Namespace.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <optional>
#include <unordered_set>

EINSUMS_NAMESPACE_BEGIN(compute_graph::detail)

namespace {

/// How a node's kind behaves, for the two questions this pass asks of it: which
/// speedup curve it scales along, and how its serial time is modeled.
///
/// One partition answering both, so the two cannot drift (a kind missing from
/// one would be priced as a memcpy).
enum class KindClass {
    Gemm,        ///< A single dense contraction, priced from its GEMM shape.
    BatchedGemm, ///< Many small contractions behind one launch, priced from their flops.
    Permute,     ///< A physical reordering, priced from bytes and rank.
    TileMove,    ///< A tile gather/scatter: scales like a permute, priced as plain traffic.
    ControlFlow, ///< Loop / Conditional, priced by the body it runs.
    Other        ///< Elementwise and everything else, priced as the traffic it moves.
};

KindClass kind_class(OpKind kind) {
    switch (kind) {
    case OpKind::Einsum:
    case OpKind::Gemm:
    case OpKind::SymmGemm:
    case OpKind::Gemv:
    case OpKind::Ger:
    case OpKind::Dot:
        return KindClass::Gemm;

    case OpKind::BatchedGemm:
    case OpKind::GroupedBatchedGemm:
    case OpKind::GroupedSandwich:
    case OpKind::GroupedGatherRotate:
        return KindClass::BatchedGemm;

    case OpKind::Permute:
    case OpKind::GroupedPermute:
    case OpKind::Transpose:
    case OpKind::HPTTPermute:
        return KindClass::Permute;

    case OpKind::TileGather:
    case OpKind::TileScatter:
        return KindClass::TileMove;

    case OpKind::Loop:
    case OpKind::Conditional:
        return KindClass::ControlFlow;

    default:
        return KindClass::Other;
    }
}

/// Which speedup curve a node's kernel scales along.
KernelFamily family_for(Node const &node, std::size_t bytes, DeviceProfile const &profile, double flops) {
    switch (kind_class(node.kind)) {
    case KindClass::Gemm:
        // Small and large GEMM are different code, not the same code on less
        // data, so the flop count picks between two curves rather than a size
        // class within one.
        return flops > 0.0 ? (flops < DeviceProfile::kGemmSmallFlops ? KernelFamily::GemmSmall : KernelFamily::GemmLarge)
               : profile.size_class_for_bytes(bytes) == SizeClass::L1Resident ? KernelFamily::GemmSmall
                                                                              : KernelFamily::GemmLarge;
    case KindClass::BatchedGemm:
        return KernelFamily::BatchedGemm;

    case KindClass::Permute:
    case KindClass::TileMove:
        // A tile gather/scatter scales the way a permute does, so it shares the
        // curve. It is still PRICED as plain traffic below: the two questions
        // have different answers for this one class, deliberately.
        return KernelFamily::Permute;

    case KindClass::ControlFlow:
    case KindClass::Other:
        break;
    }
    return KernelFamily::Elementwise;
}

/// The bytes a node touches: every distinct buffer it reads or writes, counted
/// once. Views resolve to the buffer they alias, so a node working through two
/// slices of one tensor is not charged for it twice.
std::size_t working_set_bytes(Graph const &graph, Node const &node) {
    std::unordered_set<TensorId> seen;
    std::size_t                  bytes = 0;
    auto const                  &map   = graph.tensors_map();

    auto add = [&](TensorId raw) {
        TensorId const tid = graph.buffer_of(raw);
        if (!seen.insert(tid).second) {
            return;
        }
        if (auto it = map.find(tid); it != map.end()) {
            bytes += it->second.total_bytes();
        }
    };
    for (TensorId const tid : node.inputs) {
        add(tid);
    }
    for (TensorId const tid : node.outputs) {
        add(tid);
    }
    return bytes;
}

/// Elements in a tensor, or 0 when the graph does not know the tensor.
std::size_t elems_of(Graph const &graph, TensorId raw) {
    auto const &map = graph.tensors_map();
    auto        it  = map.find(graph.buffer_of(raw));
    return it == map.end() ? 0 : it->second.total_elems();
}

/// A GEMM shape for a contraction node, or all zeros when one cannot be derived.
///
/// A capture that matched the 2D x 2D -> 2D pattern carries the exact shape in
/// its GemmHint and that is used verbatim. Anything else is reduced to the
/// square GEMM with the same flop count: with `C = A x B` over a link of extent
/// `K`, `|A| * |B| / |C| = K^2` whatever the index topology, and `|C| = M * N`.
/// The result prices the contraction's ARITHMETIC exactly and its aspect ratio
/// only approximately, which is the right trade for a cost model whose GEMM
/// table is interpolated anyway.
struct GemmShape {
    std::size_t m{0}, n{0}, k{0};

    [[nodiscard]] bool   valid() const { return m > 0 && n > 0 && k > 0; }
    [[nodiscard]] double flops() const { return 2.0 * static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k); }
};

/// Serial time for a batched or grouped-batched GEMM node.
///
/// Sums the arithmetic of every member and charges one launch for the call.
/// Returns 0 when the node carries no descriptor the model can read, which
/// leaves it below the fork floor and so at width 1 - the same "do not guess"
/// behaviour the default branch has.
double batched_gemm_us(Node const &node, DeviceProfile const &profile) {
    if (auto const *gr = node.op_data.get_if<GroupedGatherRotateDescriptor>(); gr != nullptr) {
        // The only member of this family that is not arithmetic alone. Per
        // member the rotation is 2 nq nu^2 nt for the first contraction and
        // 2 nq nu nt^2 for the second, and the gather ahead of them streams
        // nq nu^2 elements out of the shared parent. Those two happen in
        // series inside the kernel - stage a tile, then rotate it - so they
        // add rather than max, and leaving the streaming term out would price
        // the node the way its predecessor emission was priced: as arithmetic
        // that happened to touch memory, when the traffic is the reason the
        // node exists.
        double work = 0.0;
        for (int i = 0; i < gr->total; i++) {
            auto const idx = static_cast<std::size_t>(i);
            auto const nq  = static_cast<double>(gr->nq[idx]);
            auto const nu  = static_cast<double>(gr->nu[idx]);
            auto const nt  = static_cast<double>(gr->nt[idx]);
            if (nq <= 0.0 || nu <= 0.0 || nt <= 0.0) {
                continue;
            }
            double const flops = 2.0 * nq * nu * nt * (nu + nt);
            double const gflops =
                profile.estimate_gemm_gflops(static_cast<std::size_t>(nt), static_cast<std::size_t>(nu), static_cast<std::size_t>(nu));
            work += flops / (gflops * 1e3);
            work += profile.estimate_memory_time_us(static_cast<std::size_t>(nq * nu * nu * static_cast<double>(gr->elem_bytes)), 1);
        }
        return work > 0.0 ? work + profile.kernel_launch_overhead_us : 0.0;
    }
    if (auto const *sw = node.op_data.get_if<GroupedSandwichDescriptor>(); sw != nullptr) {
        // Per member: the dress (2 nq nk na^2) plus the two sandwich GEMMs
        // (4 nq na^3), priced as one batched call the way the descriptor's
        // sibling kinds are.
        double work = 0.0;
        for (int i = 0; i < sw->total; i++) {
            auto const   nq    = static_cast<double>(sw->nq[static_cast<std::size_t>(i)]);
            auto const   nk    = static_cast<double>(sw->nk[static_cast<std::size_t>(i)]);
            auto const   na    = static_cast<double>(sw->na[static_cast<std::size_t>(i)]);
            double const flops = nq * na * na * (2.0 * nk + 4.0 * na);
            double const gflops =
                profile.estimate_gemm_gflops(static_cast<std::size_t>(na), static_cast<std::size_t>(na), static_cast<std::size_t>(na));
            work += flops / (gflops * 1e3);
        }
        return work > 0.0 ? work + profile.kernel_launch_overhead_us : 0.0;
    }

    auto member_us = [&profile](std::size_t m, std::size_t n, std::size_t k, double count) {
        if (m == 0 || n == 0 || k == 0 || count <= 0.0) {
            return 0.0;
        }
        double const each = profile.estimate_gemm_time_us(m, n, k, 1);
        return count * std::max(0.0, each - profile.kernel_launch_overhead_us);
    };

    double work = 0.0;
    if (auto const *b = node.op_data.get_if<BatchedGemmDescriptor>(); b != nullptr) {
        work = member_us(static_cast<std::size_t>(b->m), static_cast<std::size_t>(b->n), static_cast<std::size_t>(b->k),
                         static_cast<double>(b->batch_count));
    } else if (auto const *g = node.op_data.get_if<GroupedBatchedGemmDescriptor>(); g != nullptr) {
        for (auto const &grp : g->groups) {
            work += member_us(static_cast<std::size_t>(grp.m), static_cast<std::size_t>(grp.n), static_cast<std::size_t>(grp.k),
                              static_cast<double>(grp.count));
        }
    } else {
        return 0.0;
    }

    return work > 0.0 ? work + profile.kernel_launch_overhead_us : 0.0;
}

GemmShape gemm_shape_of(Graph const &graph, Node const &node) {
    if (auto const *ein = node.op_data.get_if<EinsumDescriptor>(); ein != nullptr && ein->gemm_hint) {
        auto const &hint = *ein->gemm_hint;
        if (hint.m > 0 && hint.n > 0 && hint.k > 0) {
            return {.m = static_cast<std::size_t>(hint.m), .n = static_cast<std::size_t>(hint.n), .k = static_cast<std::size_t>(hint.k)};
        }
    }
    if (operand_inputs(node).size() != 2 || node.outputs.size() != 1) {
        return {};
    }
    double const a = static_cast<double>(elems_of(graph, node.inputs[0]));
    double const b = static_cast<double>(elems_of(graph, node.inputs[1]));
    double const c = static_cast<double>(elems_of(graph, node.outputs[0]));
    if (a <= 0.0 || b <= 0.0 || c <= 0.0) {
        return {};
    }
    auto const k    = static_cast<std::size_t>(std::llround(std::sqrt(a * b / c)));
    auto const side = static_cast<std::size_t>(std::llround(std::sqrt(c)));
    if (k == 0 || side == 0) {
        return {};
    }
    return {.m = side, .n = side, .k = k};
}

/// The GEMM curve for a contraction of @p flops, as family_for picks it.
KernelFamily gemm_family(double flops) {
    return flops < DeviceProfile::kGemmSmallFlops ? KernelFamily::GemmSmall : KernelFamily::GemmLarge;
}

/// A registered descriptor's own estimate, priced on @p profile, or empty when it gives none.
std::optional<ModelledCost> hooked_cost(Graph const &graph, Node const &node, DeviceProfile const &profile, std::size_t bytes) {
    DescriptorHooks const *hooks = descriptor_hooks(node);
    if (hooks == nullptr || !hooks->cost) {
        return std::nullopt;
    }
    auto const estimate = hooks->cost(node.op_data, CostQuery{.graph = graph, .node = node, .profile = profile, .target = Target::CPU});
    if (!estimate) {
        return std::nullopt;
    }

    ModelledCost cost;
    cost.bytes   = estimate->bytes > 0 ? estimate->bytes : bytes;
    cost.family  = estimate->family.value_or(estimate->flops > 0.0 ? gemm_family(estimate->flops) : KernelFamily::Elementwise);
    cost.speedup = estimate->speedup;
    if (estimate->serial_us) {
        cost.t1_us = *estimate->serial_us;
        return cost;
    }
    // A roofline: the flops priced as the square GEMM of that many flops, the bytes as traffic,
    // and the node as whichever of the two takes longer.
    double compute_us = 0.0;
    if (estimate->flops > 0.0) {
        auto const side = static_cast<std::size_t>(std::max(1.0, std::round(std::cbrt(estimate->flops / 2.0))));
        compute_us      = profile.estimate_gemm_time_us(side, side, side, 1);
    }
    double const memory_us = cost.bytes > 0 ? profile.estimate_memory_time_us(cost.bytes, 1) : 0.0;
    cost.t1_us             = std::max(compute_us, memory_us);
    return cost;
}

} // namespace

ModelledCost model_cost(Graph const &graph, Node const &node, DeviceProfile const &profile) {
    std::size_t const bytes = working_set_bytes(graph, node);
    if (auto hooked = hooked_cost(graph, node, profile, bytes)) {
        return std::move(*hooked);
    }

    GemmShape const shape = gemm_shape_of(graph, node);
    ModelledCost    cost;
    cost.bytes  = bytes;
    cost.family = family_for(node, bytes, profile, shape.valid() ? shape.flops() : 0.0);

    switch (kind_class(node.kind)) {
    case KindClass::ControlFlow:
        // Priced by the body it runs, which only the planner walking it can do.
        break;
    case KindClass::Gemm:
        cost.t1_us = shape.valid() ? profile.estimate_gemm_time_us(shape.m, shape.n, shape.k, 1)
                                   : (bytes > 0 ? profile.estimate_memory_time_us(bytes, 1) : 0.0);
        break;
    case KindClass::BatchedGemm:
        // A batch is arithmetic, not traffic. Pricing it as the bytes it
        // moves is the wrong dimension entirely: the flops scale with the
        // members' m*n*k while the bytes only scale with the operands.
        //
        // One batched call pays ONE launch and then does every member's
        // arithmetic, so the per-member estimate has its launch term
        // removed and a single one is added back at the end. That is
        // the whole reason the batch exists.
        cost.t1_us = batched_gemm_us(node, profile);
        break;
    case KindClass::Permute: {
        auto const &map  = graph.tensors_map();
        std::size_t rank = 2;
        if (!node.inputs.empty()) {
            if (auto it = map.find(graph.buffer_of(node.inputs[0])); it != map.end() && it->second.rank > 0) {
                rank = it->second.rank;
            }
        }
        cost.t1_us = bytes > 0 ? profile.estimate_permute_time_us(bytes, rank, 1) : 0.0;
        break;
    }
    case KindClass::TileMove:
    case KindClass::Other:
        // Everything else - elementwise, axpby, scale, the tiled
        // lowerings, Custom - is priced as the traffic it moves. A node
        // whose buffers the graph does not know moves no bytes it can
        // see and gets a serial time of 0, which is below the floor and
        // so stays at width 1 instead of being guessed at.
        cost.t1_us = bytes > 0 ? profile.estimate_memory_time_us(bytes, 1) : 0.0;
        break;
    }
    return cost;
}

EINSUMS_NAMESPACE_END(compute_graph::detail)
