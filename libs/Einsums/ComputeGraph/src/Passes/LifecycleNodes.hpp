//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/ComputeGraph/Graph.hpp>
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
 * whether one already exists. The id is left for the caller. The executor is the one
 * @ref build_executor gives the node, so it rebuilds after an edit like any other, and it counts
 * the bytes it allocates on @p graph's run context for @ref Graph::last_run: only when it
 * allocates, since a replay that finds the tensor still held does nothing.
 * @param[in] graph    The graph the node lands in, which resolves @p emit_tid.
 * @param[in] owner    The graph that declares the tensor, whose registry holds @p handle. When it is
 *                     not @p graph, the node's descriptor names it, since the copy of the handle in
 *                     @p graph need not carry the hooks the executor reads.
 * @param[in] handle   The tensor's handle, as @p owner declares it.
 * @param[in] emit_tid The id the node writes: the handle's own, or a parent id for a hoisted body tensor.
 */
[[nodiscard]] Node make_materialize_node(Graph &graph, Graph &owner, TensorHandle const &handle, TensorId emit_tid);

/**
 * @brief A Free node for @p handle, releasing it through its handle.
 *
 * The tensor is both an input and an output: the input orders the Free after the last writer, and
 * the output makes it a writer itself, so the dependency builder orders it after every prior reader
 * too. With only the input a concurrent executor could release the buffer under a reader.
 * @param[in] graph    The graph the node lands in, which resolves @p emit_tid; its run context
 *                     counts the bytes the node releases, as for make_materialize_node.
 * @param[in] owner    The graph that declares the tensor; see make_materialize_node.
 * @param[in] handle   The tensor's handle, as @p owner declares it.
 * @param[in] emit_tid The id the node writes.
 */
[[nodiscard]] Node make_free_node(Graph &graph, Graph &owner, TensorHandle const &handle, TensorId emit_tid);

/**
 * @brief Where a lifecycle node placed in @p graph finds its tensor's declaration.
 * @param[in] graph    The graph the node lands in.
 * @param[in] owner    The graph that declares the tensor.
 * @param[in] handle   The tensor's handle, as @p owner declares it.
 * @param[in] emit_tid The id the node names the tensor by in @p graph.
 * @return Empty when the node's own operand names the declaration, else @p owner and its id.
 */
[[nodiscard]] DeclaredTensor declared_tensor(Graph &graph, Graph &owner, TensorHandle const &handle, TensorId emit_tid);

/// The tensor a lifecycle node (Alloc, Materialize, Free) allocates or releases, or empty for any other node.
[[nodiscard]] std::string_view lifecycle_tensor_name(Node const &node) noexcept;

/// Whether @p nodes hold a lifecycle node of @p kind for the tensor named @p name.
[[nodiscard]] bool has_lifecycle_node(std::span<Node const> nodes, OpKind kind, std::string_view name) noexcept;

EINSUMS_NAMESPACE_END(compute_graph::passes::lifecycle)
