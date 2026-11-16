//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Optimizer.hpp>
#include <Einsums/Config/Namespace.hpp>

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph::passes)

/**
 * @brief Whether the executor has a device kernel for @p node's operation and shape.
 *
 * The descriptor-shape half of @ref compute_device_eligibility: an einsum that lowers to a GEMM,
 * a GEMV or a strided-batched GEMM, a scale, or an axpby. It says nothing about element types or
 * operand storage, so a caller that wants the whole answer reads @ref Node::eligible instead.
 *
 * Keep it in step with try_gpu_blas_dispatch (src/Graph/GpuDispatch.cpp).
 * @param[in] node  The node.
 * @param[in] graph The graph that owns @p node.
 * @return True when a device dispatcher recognizes the node's shape.
 * @versionadded{2.0.0}
 */
[[nodiscard]] EINSUMS_EXPORT bool node_has_device_kernel(Node const &node, Graph const &graph);

/**
 * @brief Set @ref Node::eligible on every node of @p graph and of the graphs beneath it.
 *
 * A node is eligible for device 0 when the executor can dispatch it there: its shape has a
 * device kernel (@ref node_has_device_kernel), every operand's element type is one the compiled
 * backend handles (MPS takes only float32; CUDA and the mock take real types, and complex only
 * through the strided-batched GEMM), and no operand is tile-wise sparse (a tiled tensor has no
 * single buffer to transfer). Every other node is host only.
 *
 * Whether placing the node on the device would pay, and whether its operands fit in device
 * memory, are not asked here; that is @ref GPUPlacement's decision. This does not ask whether a
 * device is present either: it is a statement about the node, and the @ref DeviceEligibility pass
 * and GPUPlacement check availability themselves.
 *
 * Idempotent: it overwrites every set, so a node whose operands changed loses a stale device.
 * Keep it in step with try_gpu_blas_dispatch (src/Graph/GpuDispatch.cpp).
 * @param[in,out] graph The graph to annotate.
 * @return True when any node's set changed.
 * @versionadded{2.0.0}
 */
EINSUMS_EXPORT bool compute_device_eligibility(Graph &graph);

/**
 * @brief Records which nodes a device can run, so placement can ask "can" apart from "should".
 *
 * Calls @ref compute_device_eligibility over the whole graph tree. It changes no node's target,
 * operands or executor; @ref GPUPlacement reads the result. On a machine with no usable device,
 * or with `--einsums:gpu:disable` set, it leaves every node host only.
 *
 * Structural-resource, because the answer depends on the backend compiled in and the device
 * present. It writes only @ref Node::eligible, which a save leaves out, so it does not make a
 * graph refuse to be saved.
 *
 * @par Limitations
 * - One device: the set names device 0 only.
 * - Compiled into the default pipeline on every build, but without a device it records nothing.
 */
class EINSUMS_EXPORT DeviceEligibility : public OptimizerPass {
  public:
    [[nodiscard]] std::string name() const override { return "DeviceEligibility"; }

    /// @copydoc OptimizerPass::phase
    [[nodiscard]] PassPhase phase() const override { return PassPhase::StructuralResource; }
    /// Only @ref Node::eligible, which a save does not write.
    [[nodiscard]] bool changes_saved_form() const override { return false; }
    /// Every feature: it reads each node's shape and operand types and rewrites nothing.
    [[nodiscard]] std::optional<NodeFeatures> understood_features() const override { return NodeFeatures::all(); }
    void                                      reset_stats() override { _stats = {}; }

    [[nodiscard]] std::vector<std::string> explain() const override;

    /// Nodes that held a device in their eligible set after the last run.
    [[nodiscard]] std::size_t num_eligible() const { return _stats.num_eligible; }

  protected:
    bool transform(Graph &graph) override;

  private:
    /// What one run found, which reset_stats starts afresh.
    struct Stats {
        std::size_t num_eligible{0};
        std::size_t num_changed{0};
    };
    Stats _stats{};
};

EINSUMS_NAMESPACE_END(compute_graph::passes)
