//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/// @file NodeRunner.hpp
/// @brief What running one node of a replay means, defined once.
///
/// Private to the `Graph` sources. A replay creates one runner, asks it to run each node in the
/// order its schedule chose, and calls @ref detail::NodeRunner::finish when the last has run.

#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraphTypes/Ids.hpp>
#include <Einsums/Config/Namespace.hpp>

#include <cstddef>
#include <unordered_set>

EINSUMS_NAMESPACE_BEGIN(compute_graph::detail)

/**
 * @brief Runs the nodes of one replay of one graph, with everything running a node means.
 *
 * Per node: the profiler zone and its annotations, the observer's node events, the host and
 * device transfers, the offer of the node to the GPU and the CPU fallback when it declines,
 * keeping the host and device copies of each tensor coherent, and the node's timing sample.
 * At the end of the replay, @ref finish copies back every tensor whose device copy is the only
 * current one.
 *
 * One runner per replay: it holds which tensors' device copies are current, which is state of
 * that replay alone.
 */
class NodeRunner {
  public:
    /**
     * @param[in,out] graph     The graph being replayed; it must outlive the runner.
     * @param[in]     recording Whether the profiler records this replay, read once by the caller
     *                          so every zone this replay opens is also closed.
     */
    NodeRunner(Graph &graph, bool recording) noexcept : _graph(graph), _recording(recording) {}

    NodeRunner(NodeRunner const &)            = delete;
    NodeRunner &operator=(NodeRunner const &) = delete;
    NodeRunner(NodeRunner &&)                 = delete;
    NodeRunner &operator=(NodeRunner &&)      = delete;
    ~NodeRunner()                             = default;

    /// @brief Run the node at @p index of the graph's node list.
    /// @param[in] index Position in the node list.
    void run(std::size_t index);

    /// @brief Copy back every tensor whose device copy is the only current one. Call once,
    ///        after the last node of the replay has run.
    void finish();

  private:
    Graph &_graph;
    bool   _recording;

    /// Tensors whose DEVICE shadow currently holds the authoritative data.
    ///
    /// Tracked at run time rather than read off TensorHandle::residency, which
    /// is a static annotation left by TransferInsertion and says nothing about a
    /// tensor produced by one GPU node and consumed by the next: no transfer is
    /// inserted for those, so residency stays Host while the live value is on the
    /// device. Copying the host side up on that basis would clobber the producer's
    /// result.
    std::unordered_set<TensorId> _device_valid;
};

EINSUMS_NAMESPACE_END(compute_graph::detail)
