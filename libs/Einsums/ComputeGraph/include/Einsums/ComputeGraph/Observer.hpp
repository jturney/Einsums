//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/// @file Observer.hpp
/// @brief Hearing a replay as it runs: node starts and ends, storage reads and writes, publishes.

#include <Einsums/ComputeGraph/RunMetrics.hpp>
#include <Einsums/ComputeGraphTypes/Enums.hpp>
#include <Einsums/ComputeGraphTypes/Ids.hpp>
#include <Einsums/Config/ExportDefinitions.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Python/Annotations.hpp>

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

/**
 * @brief What an @ref ExecutionEvent reports.
 * @versionadded{2.0.0}
 */
enum class APIARY_EXPOSE APIARY_MODULE("graph") EventKind : std::uint8_t{
    ReplayStart, ///< The outermost replay began.
    NodeStart,   ///< A node began, on the thread that runs it.
    NodeEnd,     ///< That node finished, or threw.
    Read,        ///< A slab, box or whole tensor was read from a backing's storage.
    Write,       ///< One was written to it.
    Publish,     ///< An update backing's new value was made current.
    ReplayEnd,   ///< The outermost replay finished.
};

/**
 * @brief One thing that happened while a replay ran.
 *
 * Every event carries the fields its kind has: a node event its node, a storage event its tensor,
 * storage and bytes. @ref at_ms is measured from the start of the outermost replay on one steady
 * clock, so events from different threads order by it.
 * @versionadded{2.0.0}
 */
struct APIARY_EXPOSE APIARY_MODULE("graph") ExecutionEvent {
    APIARY_EXPOSE APIARY_READONLY EventKind kind{EventKind::NodeStart};
    /// The graph or body the node or storage belongs to.
    APIARY_EXPOSE APIARY_READONLY std::string graph;
    /// The node, for a node event.
    APIARY_EXPOSE APIARY_READONLY NodeId node{0};
    /// The node's kind, for a node event.
    APIARY_EXPOSE APIARY_READONLY OpKind op{OpKind::Custom};
    /// The node's label, for a node event.
    APIARY_EXPOSE APIARY_READONLY std::string label;
    /// The backed tensor, for a storage event.
    APIARY_EXPOSE APIARY_READONLY std::string tensor;
    /// Its storage, as @ref TensorBacking::describe names it, for a storage event.
    APIARY_EXPOSE APIARY_READONLY std::string storage;
    /// Milliseconds from the start of the outermost replay.
    APIARY_EXPOSE APIARY_READONLY double at_ms{0.0};
    /// How long the node ran, for a node end.
    APIARY_EXPOSE APIARY_READONLY double ms{0.0};
    /// Bytes moved, for a storage event.
    APIARY_EXPOSE APIARY_READONLY std::uint64_t bytes{0};
    /// Which thread it happened on: equal values are the same thread within one process.
    APIARY_EXPOSE APIARY_READONLY std::uint64_t thread{0};
};

/**
 * @brief What hears a graph's replays as they run, once attached with @ref Graph::set_observer.
 *
 * Events arrive on the thread they happen on: under the OpenMP and dataflow executors two calls can
 * overlap, so an observer must be thread-safe. One that throws is detached and the failure logged
 * once; the replay goes on. With no observer attached an executor tests one pointer per node.
 *
 * Every member is inline, so a program can derive its own observer: an out-of-line one would keep
 * the class's vtable inside the library.
 * @versionadded{2.0.0}
 */
class APIARY_EXPOSE APIARY_MODULE("graph") APIARY_HOLDER(std::shared_ptr) ExecutionObserver {
  public:
    ExecutionObserver()                                     = default;
    ExecutionObserver(ExecutionObserver const &)            = delete;
    ExecutionObserver &operator=(ExecutionObserver const &) = delete;
    ExecutionObserver(ExecutionObserver &&)                 = delete;
    ExecutionObserver &operator=(ExecutionObserver &&)      = delete;
    virtual ~ExecutionObserver()                            = default;

    /// @brief The outermost replay began.
    /// @param[in] event The event.
    virtual void on_replay_start(ExecutionEvent const &event) { static_cast<void>(event); }
    /// @brief A node began.
    /// @param[in] event The event.
    virtual void on_node_start(ExecutionEvent const &event) { static_cast<void>(event); }
    /// @brief A node finished, or threw; @ref ExecutionEvent::ms is how long it ran.
    /// @param[in] event The event.
    virtual void on_node_end(ExecutionEvent const &event) { static_cast<void>(event); }
    /// @brief A backing's storage was read, written or published.
    /// @param[in] event The event.
    virtual void on_io(ExecutionEvent const &event) { static_cast<void>(event); }
    /// @brief The outermost replay finished.
    /// @param[in] event The event.
    /// @param[in] run What it measured, as @ref Graph::last_run will report it.
    virtual void on_replay_end(ExecutionEvent const &event, RunMetrics const &run) {
        static_cast<void>(event);
        static_cast<void>(run);
    }
};

/**
 * @brief An observer that keeps every event, for reading after the replay: the way to observe a
 *        replay from Python, whose callbacks would serialize the executors on the GIL.
 *
 * @code
 * log = cg.EventLog()
 * g.set_observer(log)
 * g.execute()
 * slowest = sorted((e for e in log.events if e.kind == cg.EventKind.NodeEnd), key=lambda e: e.ms)[-5:]
 * log.write_chrome_trace("replay.json")      # open in ui.perfetto.dev
 * @endcode
 * @versionadded{2.0.0}
 */
class APIARY_EXPOSE APIARY_MODULE("graph") APIARY_HOLDER(std::shared_ptr) EINSUMS_EXPORT EventLog : public ExecutionObserver {
  public:
    /// @brief An empty log.
    APIARY_EXPOSE EventLog() = default;

    void on_replay_start(ExecutionEvent const &event) override;
    void on_node_start(ExecutionEvent const &event) override;
    void on_node_end(ExecutionEvent const &event) override;
    void on_io(ExecutionEvent const &event) override;
    void on_replay_end(ExecutionEvent const &event, RunMetrics const &run) override;

    /// @brief Every event so far, in the order they arrived.
    /// @return A copy of the events.
    APIARY_EXPOSE APIARY_GETTER("events") [[nodiscard]] std::vector<ExecutionEvent> events() const;

    /// @brief Forget every event.
    APIARY_EXPOSE void clear();

    /**
     * @brief Write the events as a Chrome trace: one track per thread, a span per node, an instant
     *        per storage event, readable in ``ui.perfetto.dev`` or ``chrome://tracing``.
     * @param[in] path The file to write.
     * @throws std::runtime_error when the file cannot be written.
     */
    APIARY_EXPOSE void write_chrome_trace(std::string const &path) const;

  private:
    void keep(ExecutionEvent const &event);

    mutable std::mutex          _mutex;
    std::vector<ExecutionEvent> _events;
};

/**
 * @brief An observer that labels the profiler's zones with what the graph was doing in them.
 *
 * Every executor opens a zone per node while the profiler records. This adds to it the graph and
 * the node it belongs to, and each read, write and publish of a backing the node made, with its
 * bytes (``read t2``, ``write t2_next``), so the live viewer and the saved profile name the graph
 * node behind a zone and the storage traffic inside it. Without the profiler recording, it does
 * nothing.
 *
 * @code
 * g.set_observer(cg.WaggleBridge())
 * @endcode
 * @versionadded{2.0.0}
 */
class APIARY_EXPOSE APIARY_MODULE("graph") APIARY_HOLDER(std::shared_ptr) EINSUMS_EXPORT WaggleBridge : public ExecutionObserver {
  public:
    /// @brief A bridge to the profiler this process records with.
    APIARY_EXPOSE WaggleBridge() = default;

    void on_node_start(ExecutionEvent const &event) override;
    void on_io(ExecutionEvent const &event) override;
};

EINSUMS_NAMESPACE_END(compute_graph)
