//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/ComputeGraph/Detail/RunContext.hpp>
#include <Einsums/ComputeGraph/TensorHandle.hpp>
#include <Einsums/ComputeGraphTypes/Descriptors.hpp>
#include <Einsums/ComputeGraphTypes/Enums.hpp>
#include <Einsums/Config/Namespace.hpp>

#include <cstddef>
#include <functional>
#include <memory>

// The executors of the lifecycle kinds, which build_executor dispatches to. Each reads only the
// tensor's handle and the graph's run context, so a node of one of these kinds rebuilds from its
// operand lists like any other.

EINSUMS_NAMESPACE_BEGIN(compute_graph::detail)

/**
 * @brief The tensor a lifecycle node acts on, named by its declaring graph and its id there.
 *
 * Looked up each time the node runs rather than when it is built, because a rebind can hand the
 * slot to the caller's storage after the node exists. The node then does nothing: allocating,
 * filling and freeing belong to storage the graph owns, and a Free acting on the bound tensor
 * would release the caller's.
 */
struct LifecycleTarget {
    std::shared_ptr<GraphAnchor const> graph; ///< The declaring graph.
    TensorId                           id{0}; ///< The tensor's id in it.

    /// @brief The tensor's handle as it is now.
    /// @throws std::logic_error If the declaring graph has been destroyed.
    [[nodiscard]] TensorHandle const &handle() const;

    /// @brief The handle when the graph still owns the tensor's storage, null once a rebind has
    ///        given the slot to the caller's (@ref TensorHandle::caller_bound).
    [[nodiscard]] TensorHandle const *graph_owned() const;
};

/**
 * @brief Allocate @p target's tensor; a distributed one is first resized to its local partition.
 *
 * Idempotent: a no-op when the tensor is already allocated, a reallocation on a replay after a Free
 * reclaimed it. Only the reallocation is counted on @p context as memory the graph took.
 * @param[in] target  The tensor.
 * @param[in] context The run context that counts the bytes allocated, or null.
 * @return The executor.
 */
[[nodiscard]] std::function<void()> build_materialize(LifecycleTarget target, std::shared_ptr<RunContext> context);

/**
 * @brief Release @p target's tensor.
 * @param[in] target  The tensor.
 * @param[in] context The run context that counts the bytes released, or null.
 * @return The executor.
 */
[[nodiscard]] std::function<void()> build_free(LifecycleTarget target, std::shared_ptr<RunContext> context);

struct ArenaBlock;

/**
 * @brief Attach @p target's tensor at @p offset of @p arena instead of allocating it.
 *
 * Idempotent across replays: attaching the same pointer again is a no-op. Counts nothing, since
 * the arena is counted whole.
 * @param[in] target The tensor.
 * @param[in] arena  The graph's arena, held for as long as the executor lives.
 * @param[in] offset Where in the arena the tensor's bytes start.
 * @return The executor.
 */
[[nodiscard]] std::function<void()> build_attach(LifecycleTarget target, std::shared_ptr<ArenaBlock> arena, std::size_t offset);

/**
 * @brief Detach @p target's tensor from its arena slice.
 *
 * Releasing a tensor on external storage parks it without freeing anything, and the arena persists
 * for the next replay. Counts nothing, as @ref build_attach does not.
 * @param[in] target The tensor.
 * @return The executor.
 */
[[nodiscard]] std::function<void()> build_detach(LifecycleTarget target);

/**
 * @brief Fill @p target's freshly allocated tensor as @p init says: zeros, random values, the
 *        caller's fill function, or the whole of its backing.
 * @param[in] init   How to fill it.
 * @param[in] target The tensor.
 * @return The executor.
 * @throws std::invalid_argument If @p init is @ref InitKind::None, which no Initialize node carries.
 * @throws std::logic_error If @p init reads a backing the handle does not carry, the handle has
 *         no storage accessor to fill, or @p init is @ref InitKind::Custom and it has no function.
 */
[[nodiscard]] std::function<void()> build_initialize(InitKind init, LifecycleTarget target);

EINSUMS_NAMESPACE_END(compute_graph::detail)
