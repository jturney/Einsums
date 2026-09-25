//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/TensorHandle.hpp>
#include <Einsums/Config/Namespace.hpp>

#include <span>
#include <string_view>

EINSUMS_NAMESPACE_BEGIN(compute_graph::passes::lifecycle)

/**
 * @brief A Materialize node for @p handle, written as @p emit_tid.
 *
 * Allocates the tensor through its handle; a distributed tensor is first resized to its local
 * partition. Carries an @ref AllocDescriptor naming the tensor, which is how a later pass asks
 * whether one already exists. The id is left for the caller, which knows which graph the node
 * lands in.
 * @param[in] handle   The tensor's handle.
 * @param[in] emit_tid The id the node writes: the handle's own, or a parent id for a hoisted body tensor.
 */
[[nodiscard]] Node make_materialize_node(TensorHandle const &handle, TensorId emit_tid);

/**
 * @brief A Free node for @p handle, releasing it through its handle.
 *
 * The tensor is both an input and an output: the input orders the Free after the last writer, and
 * the output makes it a writer itself, so the dependency builder orders it after every prior reader
 * too. With only the input a concurrent executor could release the buffer under a reader.
 */
[[nodiscard]] Node make_free_node(TensorHandle const &handle, TensorId emit_tid);

/// The tensor a lifecycle node (Alloc, Materialize, Free) allocates or releases, or empty for any other node.
[[nodiscard]] std::string_view lifecycle_tensor_name(Node const &node) noexcept;

/// Whether @p nodes hold a lifecycle node of @p kind for the tensor named @p name.
[[nodiscard]] bool has_lifecycle_node(std::span<Node const> nodes, OpKind kind, std::string_view name) noexcept;

EINSUMS_NAMESPACE_END(compute_graph::passes::lifecycle)
