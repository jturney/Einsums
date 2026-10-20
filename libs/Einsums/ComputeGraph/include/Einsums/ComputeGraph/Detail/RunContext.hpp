//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/// @file RunContext.hpp
/// @brief What a graph and every body under it share while a replay runs: the memory they hold,
///        and the node times their replays recorded.

#include <Einsums/ComputeGraphTypes/Enums.hpp>
#include <Einsums/ComputeGraphTypes/Ids.hpp>
#include <Einsums/Config/ExportDefinitions.hpp>
#include <Einsums/Config/Namespace.hpp>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string_view>
#include <utility>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

class Graph;
class ExecutionObserver;
struct Node;
struct RunMetrics;
struct TensorBacking;
enum class EventKind : std::uint8_t;

namespace detail {

/// One node's run, as a replay of some graph in the tree recorded it.
struct RunSample {
    Graph const *graph{nullptr}; ///< The graph or body that ran it, alive as long as the root is.
    NodeId       id{0};
    OpKind       kind{OpKind::Custom};
    double       ms{0.0};
    unsigned     width{0};
};

/**
 * @brief The state one replay of a graph tree shares across its bodies and its worker threads.
 *
 * One per tree: a body joins its parent's when it is attached (@ref Graph::share_run_context),
 * so the lifecycle nodes and replays of every body count into the same place. Nodes run on the
 * executors' worker threads, which is why the counts are atomics and the samples sit behind a
 * mutex, and why this is reached through what each node already holds rather than through the
 * thread that started the replay.
 */
class RunContext {
  public:
    /// Enter a replay of some graph in the tree. True for the outermost one, the root's, which
    /// is the replay the metrics describe.
    bool enter() noexcept { return _depth.fetch_add(1, std::memory_order_acq_rel) == 0; }

    /// Leave the replay @ref enter entered.
    void leave() noexcept { _depth.fetch_sub(1, std::memory_order_acq_rel); }

    /// Whether a replay of the tree is running, on any thread.
    [[nodiscard]] bool replaying() const noexcept { return _depth.load(std::memory_order_acquire) > 0; }

    /// Start the outermost replay holding @p held bytes: what the tensors the lifecycle nodes
    /// manage hold right now, whoever allocated them. Counting from that, rather than from the
    /// previous replay's count, keeps a tensor allocated outside a lifecycle node (eagerly, or
    /// before the first replay) from being freed without ever having been counted. The samples of
    /// the previous replay are dropped.
    void begin_replay(std::uint64_t held) {
        _start = std::chrono::steady_clock::now();
        _live.store(static_cast<std::int64_t>(held), std::memory_order_relaxed);
        _peak.store(static_cast<std::int64_t>(held), std::memory_order_relaxed);
        std::scoped_lock const lock(_mutex);
        _samples.clear();
    }

    /// A lifecycle node allocated @p bytes.
    void allocated(std::uint64_t bytes) noexcept {
        auto const now  = _live.fetch_add(static_cast<std::int64_t>(bytes), std::memory_order_relaxed) + static_cast<std::int64_t>(bytes);
        auto       peak = _peak.load(std::memory_order_relaxed);
        while (now > peak && !_peak.compare_exchange_weak(peak, now, std::memory_order_relaxed)) {
        }
    }

    /// A lifecycle node released @p bytes.
    void released(std::uint64_t bytes) noexcept { _live.fetch_sub(static_cast<std::int64_t>(bytes), std::memory_order_relaxed); }

    /// MemoryPlanning placed intermediates in an arena of @p bytes, held for every replay.
    void set_arena(std::uint64_t bytes) noexcept { _arena.store(bytes, std::memory_order_relaxed); }

    /// The most bytes held at once since @ref begin_replay, the arena included.
    [[nodiscard]] std::uint64_t peak_bytes() const noexcept {
        auto const peak = _peak.load(std::memory_order_relaxed);
        return (peak > 0 ? static_cast<std::uint64_t>(peak) : 0) + _arena.load(std::memory_order_relaxed);
    }

    /// The arena's size, 0 without one.
    [[nodiscard]] std::uint64_t arena_bytes() const noexcept { return _arena.load(std::memory_order_relaxed); }

    /// Add a replay's node times, from the root or any body.
    void add_samples(std::vector<RunSample> &&samples) {
        std::scoped_lock const lock(_mutex);
        _samples.insert(_samples.end(), std::make_move_iterator(samples.begin()), std::make_move_iterator(samples.end()));
    }

    /// Take the samples the replay recorded, leaving none.
    [[nodiscard]] std::vector<RunSample> take_samples() {
        std::scoped_lock const lock(_mutex);
        return std::exchange(_samples, {});
    }

    /// Attach @p observer to every replay of the tree; null detaches. Between replays only.
    void set_observer(std::shared_ptr<ExecutionObserver> observer) {
        std::scoped_lock const lock(_mutex);
        _observer = std::move(observer);
        _active.store(_observer.get(), std::memory_order_release);
    }

    /// The observer to tell, or null: one atomic load, which is all an unobserved replay pays.
    [[nodiscard]] ExecutionObserver *observer() const noexcept { return _active.load(std::memory_order_acquire); }

    /// Stop telling the observer anything, after it threw; it stays held until replaced.
    void detach_observer() noexcept { _active.store(nullptr, std::memory_order_release); }

    /// Milliseconds since the outermost replay began.
    [[nodiscard]] double elapsed_ms() const noexcept {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - _start).count();
    }

  private:
    std::atomic<int>           _depth{0};
    std::atomic<std::int64_t>  _live{0};
    std::atomic<std::int64_t>  _peak{0};
    std::atomic<std::uint64_t> _arena{0};
    std::mutex                 _mutex;
    std::vector<RunSample>     _samples;

    std::shared_ptr<ExecutionObserver>    _observer;
    std::atomic<ExecutionObserver *>      _active{nullptr};
    std::chrono::steady_clock::time_point _start{std::chrono::steady_clock::now()};
};

/// @name Telling the observer
/// Each does nothing when no observer is attached, and detaches one that throws.
/// @{
/// The outermost replay of @p graph began, or ended with @p run measured.
EINSUMS_EXPORT void notify_replay(RunContext &context, EventKind kind, Graph const &graph, RunMetrics const *run);
/// @p node of @p graph began, or ended after @p ms.
EINSUMS_EXPORT void notify_node(RunContext &context, EventKind kind, Graph const &graph, Node const &node, double ms);
/// @p bytes of @p tensor's storage were read, written or published.
EINSUMS_EXPORT void notify_io(RunContext &context, EventKind kind, std::string_view tensor, TensorBacking const &backing,
                              std::uint64_t bytes);
/// @}

/// Tells the observer a node began when made, and that it ended, however it ends, when destroyed.
class EINSUMS_EXPORT NodeObservation {
  public:
    NodeObservation(RunContext &context, Graph const &graph, Node const &node);
    ~NodeObservation();
    NodeObservation(NodeObservation const &)            = delete;
    NodeObservation &operator=(NodeObservation const &) = delete;
    NodeObservation(NodeObservation &&)                 = delete;
    NodeObservation &operator=(NodeObservation &&)      = delete;

  private:
    RunContext                           *_context{nullptr}; ///< Null when nothing observes.
    Graph const                          &_graph;
    Node const                           &_node;
    std::chrono::steady_clock::time_point _start;
};

} // namespace detail

EINSUMS_NAMESPACE_END(compute_graph)
