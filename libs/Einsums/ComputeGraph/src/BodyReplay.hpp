//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/// @file BodyReplay.hpp
/// @brief Which executor runs a loop body, a branch or a setup body.

#include <Einsums/Config/Namespace.hpp>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

class Executor;
class Graph;
class PlacementPolicy;

namespace detail {

/**
 * @brief Names the executor running the node in progress on this thread, for as long as it lives.
 *
 * The node runner opens one around every node it runs, so a control-flow node's callable, which
 * takes no arguments, can find the executor running the graph that holds it. A thread-local
 * rather than anything the replay shares: a node runs on whichever thread its executor chose, and
 * the executor differs from one sub-graph to the next when a body has one of its own. Scopes nest,
 * and each puts back what it found, so a worker that runs an unrelated node while it waits for a
 * nested replay restores the waiting node's executor when that node is done.
 */
class EnclosingExecutor {
  public:
    explicit EnclosingExecutor(Executor &executor) noexcept;
    ~EnclosingExecutor();
    EnclosingExecutor(EnclosingExecutor const &)            = delete;
    EnclosingExecutor &operator=(EnclosingExecutor const &) = delete;
    EnclosingExecutor(EnclosingExecutor &&)                 = delete;
    EnclosingExecutor &operator=(EnclosingExecutor &&)      = delete;

  private:
    Executor *_previous;
};

/**
 * @brief Names the placement policy of the node in progress on this thread, for as long as it lives.
 *
 * The same device as @ref EnclosingExecutor, for the same reason: a body with no policy of its own
 * is replayed under the policy of the replay that holds it, and the callable that replays it takes
 * no arguments. The node runner reads it when it is made for the body.
 */
class EnclosingPlacement {
  public:
    explicit EnclosingPlacement(PlacementPolicy const &policy) noexcept;
    ~EnclosingPlacement();
    EnclosingPlacement(EnclosingPlacement const &)            = delete;
    EnclosingPlacement &operator=(EnclosingPlacement const &) = delete;
    EnclosingPlacement(EnclosingPlacement &&)                 = delete;
    EnclosingPlacement &operator=(EnclosingPlacement &&)      = delete;

  private:
    PlacementPolicy const *_previous;
};

/**
 * @brief Replay @p body, a child graph of the node running on this thread.
 *
 * Under the body's own executor when it has one installed, otherwise under the executor running
 * the graph that holds it, otherwise with the built-in walk.
 *
 * @param[in,out] body The loop body, branch or setup body to replay.
 */
void replay_body(Graph &body);

} // namespace detail

EINSUMS_NAMESPACE_END(compute_graph)
