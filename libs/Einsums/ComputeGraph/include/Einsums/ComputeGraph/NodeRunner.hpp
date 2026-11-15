//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/// @file NodeRunner.hpp
/// @brief What running one node of a replay means, defined once for every executor.

#include <Einsums/Config.hpp>

#include <Einsums/ComputeGraph/Detail/DeviceCoherence.hpp>
#include <Einsums/ComputeGraphTypes/Ids.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Profile.hpp>

#include <cstddef>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

class Executor;
class Graph;

/**
 * @brief Runs the nodes of one replay of one graph, with everything running a node means.
 *
 * An @ref Executor decides when, and on which thread, each node runs. What running it means is
 * decided here, so that a graph computes the same thing, on the same devices, with the same
 * profile, under every executor. Per node that is: the profiler zone and its annotations, the
 * observer's node events, the host and device transfers, the offer of the node to the GPU and the
 * CPU fallback when it declines, keeping the host and device copies of each tensor coherent, and
 * the node's timing sample, and which executor replays a body the node holds: its own, if it has
 * one installed, and otherwise the executor running this replay.
 *
 * The graph creates one runner per replay and hands it to the executor; an executor never makes
 * one, and never calls a node's callable itself.
 *
 * @par Concurrency
 * @ref run, @ref start_async and @ref finish_async may be called from several threads at once for
 * different nodes, provided the executor honors @ref Graph::dependencies.
 *
 * @versionadded{2.0.0}
 */
class EINSUMS_EXPORT NodeRunner {
  public:
    NodeRunner(NodeRunner const &)            = delete;
    NodeRunner &operator=(NodeRunner const &) = delete;
    NodeRunner(NodeRunner &&)                 = delete;
    NodeRunner &operator=(NodeRunner &&)      = delete;
    ~NodeRunner()                             = default;

    /// @brief The graph being replayed.
    /// @return The graph, whose @ref Graph::nodes and @ref Graph::dependencies the executor orders.
    [[nodiscard]] Graph &graph() const noexcept { return _graph; }

    /**
     * @brief Run the node at @p index of the graph's node list.
     *
     * @param[in] index Position in the node list.
     * @param[in] width The thread width the executor runs the node at, recorded with its timing
     *                  sample; 0 when it placed the node at no chosen width. The executor sets
     *                  the width up itself (see @ref Node::thread_width).
     */
    void run(std::size_t index, unsigned width = 0);

    /// @brief Whether the node at @p index has asynchronous halves an executor may run apart.
    /// @param[in] index Position in the node list.
    /// @return True when @ref start_async and @ref finish_async apply to it. @ref run runs both
    ///         halves in turn, so an executor that does not overlap them need not ask.
    [[nodiscard]] bool is_async(std::size_t index) const;

    /// @brief Begin the asynchronous node at @p index; its timing covers @ref finish_async alone.
    /// @param[in] index Position in the node list.
    void start_async(std::size_t index);

    /// @brief Wait for the asynchronous node at @p index started by @ref start_async.
    /// @param[in] index Position in the node list.
    void finish_async(std::size_t index);

  private:
    friend class Graph;

    /// @param[in,out] graph     The graph being replayed; it must outlive the runner.
    /// @param[in,out] executor  The executor running this replay, which also runs every body a
    ///                          node replays that has no executor of its own.
    /// @param[in]     recording Whether the profiler records this replay, read once by the
    ///                          caller so every zone this replay opens is also closed.
    NodeRunner(Graph &graph, Executor &executor, bool recording);

    /// Copy back every tensor whose device copy is the only current one, and hand the graph the
    /// replay's timing samples in node order. Called once by the graph, after the executor returns.
    void finish();

    /// Track every tensor a device node or transfer touches, allocate its device shadow, and
    /// record each control-flow node's subtree I/O. Called by the constructor on a build whose
    /// device memory is separate, and only when the graph places a node there or holds a transfer.
    void prepare_device_work();

    /// Make @p id current on the host, copying it down when only its device copy is.
    void host_current(TensorId id);

    /// The part of @ref finish that copies device-only results back to the host.
    void flush_device();

    /// Open node @p index's profiler zone into @p zone, with its annotations, when recording.
    void open_zone(std::optional<waggle::ScopedZone> &zone, std::size_t index) const;

    Graph    &_graph;
    Executor &_executor;
    bool      _recording;

    /// Which of each device-touched tensor's two copies is current, on a build whose device memory
    /// is separate; untouched (and empty) on a unified-memory build.
    ///
    /// Tracked at run time rather than read off TensorHandle::residency, which
    /// is a static annotation left by TransferInsertion and says nothing about a
    /// tensor produced by one GPU node and consumed by the next: no transfer is
    /// inserted for those, so residency stays Host while the live value is on the
    /// device. Copying the host side up on that basis would clobber the producer's
    /// result.
    detail::DeviceCoherence _coherence;

    /// Whether this replay places work on a separate device at all: the coherence record and the
    /// two members below are filled only when it does.
    bool _device_work{false};

    /// The buffers each control-flow node's subtree reads and writes, by node position, worked
    /// out before any node runs: Graph::effective_io may register a handle, which must not
    /// happen while nodes run on other threads.
    std::unordered_map<std::size_t, std::pair<std::vector<TensorId>, std::vector<TensorId>>> _control_io;
};

EINSUMS_NAMESPACE_END(compute_graph)
