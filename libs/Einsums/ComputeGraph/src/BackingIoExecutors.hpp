//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/TensorBacking.hpp>
#include <Einsums/ComputeGraph/TensorHandle.hpp>
#include <Einsums/Config/Namespace.hpp>

#include <functional>
#include <memory>
#include <vector>

// The executors of the storage transfers the passes place, which build_executor dispatches to by
// the transfer's DiskIOMode. Each reaches the tensor's storage through its handle's accessor when it
// runs, since the memory planner may place it anywhere.

EINSUMS_NAMESPACE_BEGIN(compute_graph::detail)

/**
 * @brief Store the whole of @p handle's tensor, produced in memory, into its write backing.
 * @param[in] handle The tensor's handle.
 * @return The executor.
 * @throws std::logic_error If the handle carries no backing.
 */
[[nodiscard]] std::function<void()> build_store_whole(TensorHandle const &handle);

/**
 * @brief Publish @p backings as one group, and count the publish on each.
 * @param[in] backings The group, the first of which publishes it.
 * @return The executor.
 * @throws std::logic_error If @p backings is empty.
 */
[[nodiscard]] std::function<void()> build_publish(std::vector<std::shared_ptr<TensorBacking const>> backings);

/**
 * @brief Move @p window of its backing into @p handle's tensor, or out of it when @p store.
 * @param[in] store  True to write the tensor into the window, false to read the window into it.
 * @param[in] handle The tensor the window is held in.
 * @param[in] window Which part of which backing.
 * @param[in] params The graph's parameter table, which a box's bounds are resolved against.
 * @return The executor.
 */
[[nodiscard]] std::function<void()> build_window(bool store, TensorHandle const &handle, std::shared_ptr<BackingWindow const> window,
                                                 std::shared_ptr<ParamTable> params);

EINSUMS_NAMESPACE_END(compute_graph::detail)
