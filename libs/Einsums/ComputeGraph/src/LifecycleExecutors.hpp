//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/ComputeGraph/Detail/RunContext.hpp>
#include <Einsums/ComputeGraph/TensorHandle.hpp>
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
 * @brief Allocate @p handle's tensor through its handle; a distributed one is first resized to its
 *        local partition.
 *
 * Idempotent: a no-op when the tensor is already allocated, a reallocation on a replay after a Free
 * reclaimed it. Only the reallocation is counted on @p context as memory the graph took.
 * @param[in] handle  The tensor's handle.
 * @param[in] context The run context that counts the bytes allocated, or null.
 * @return The executor.
 */
[[nodiscard]] std::function<void()> build_materialize(TensorHandle const &handle, std::shared_ptr<RunContext> context);

/**
 * @brief Release @p handle's tensor through its handle.
 * @param[in] handle  The tensor's handle.
 * @param[in] context The run context that counts the bytes released, or null.
 * @return The executor.
 */
[[nodiscard]] std::function<void()> build_free(TensorHandle const &handle, std::shared_ptr<RunContext> context);

struct ArenaBlock;

/**
 * @brief Attach @p handle's tensor at @p offset of @p arena instead of allocating it.
 *
 * Idempotent across replays: attaching the same pointer again is a no-op. Counts nothing, since
 * the arena is counted whole.
 * @param[in] handle The tensor's handle.
 * @param[in] arena  The graph's arena, held for as long as the executor lives.
 * @param[in] offset Where in the arena the tensor's bytes start.
 * @return The executor.
 */
[[nodiscard]] std::function<void()> build_attach(TensorHandle const &handle, std::shared_ptr<ArenaBlock> arena, std::size_t offset);

/**
 * @brief Detach @p handle's tensor from its arena slice.
 *
 * Releasing a tensor on external storage parks it without freeing anything, and the arena persists
 * for the next replay. Counts nothing, as @ref build_attach does not.
 * @param[in] handle The tensor's handle.
 * @return The executor.
 */
[[nodiscard]] std::function<void()> build_detach(TensorHandle const &handle);

/**
 * @brief Fill @p handle's freshly allocated tensor as @p init says: zeros, random values, or the
 *        whole of its backing.
 * @param[in] init   How to fill it.
 * @param[in] handle The tensor's handle.
 * @return The executor.
 * @throws std::invalid_argument If @p init is @ref InitKind::None, which no Initialize node carries.
 * @throws std::logic_error If @p init reads a backing the handle does not carry, or the handle has
 *         no storage accessor to fill.
 */
[[nodiscard]] std::function<void()> build_initialize(InitKind init, TensorHandle const &handle);

EINSUMS_NAMESPACE_END(compute_graph::detail)
