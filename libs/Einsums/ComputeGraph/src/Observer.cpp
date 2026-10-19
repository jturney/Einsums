//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/Detail/RunContext.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Observer.hpp>
#include <Einsums/ComputeGraph/TensorBacking.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/Logging.hpp>
#include <Einsums/TypeSupport/JsonEscape.hpp>

#include <fmt/format.h>

#include <atomic>
#include <cstdint>
#include <exception>
#include <fstream>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

void EventLog::keep(ExecutionEvent const &event) {
    std::scoped_lock const lock(_mutex);
    _events.push_back(event);
}

void EventLog::on_replay_start(ExecutionEvent const &event) {
    keep(event);
}
void EventLog::on_node_start(ExecutionEvent const &event) {
    keep(event);
}
void EventLog::on_node_end(ExecutionEvent const &event) {
    keep(event);
}
void EventLog::on_io(ExecutionEvent const &event) {
    keep(event);
}
void EventLog::on_replay_end(ExecutionEvent const &event, RunMetrics const & /*run*/) {
    keep(event);
}

std::vector<ExecutionEvent> EventLog::events() const {
    std::scoped_lock const lock(_mutex);
    return _events;
}

void EventLog::clear() {
    std::scoped_lock const lock(_mutex);
    _events.clear();
}

void EventLog::write_chrome_trace(std::string const &path) const {
    auto const events = this->events();
    // Chrome's trace event format: microseconds, a complete span ("X") per node, an instant ("i")
    // per storage event, all in one process with a track per thread.
    std::string out   = "{\"traceEvents\":[";
    bool        first = true;
    auto const  add   = [&](std::string const &entry) {
        out += first ? "\n" : ",\n";
        out += entry;
        first = false;
    };
    for (auto const &event : events) {
        switch (event.kind) {
        case EventKind::NodeEnd:
            add(fmt::format(
                R"({{"name":"{}","cat":"node","ph":"X","pid":1,"tid":{},"ts":{:.3f},"dur":{:.3f},"args":{{"graph":"{}","node":{}}}}})",
                json_escape(event.label), event.thread, (event.at_ms - event.ms) * 1000.0, event.ms * 1000.0, json_escape(event.graph),
                event.node));
            break;
        case EventKind::Read:
        case EventKind::Write:
        case EventKind::Publish: {
            char const *what = event.kind == EventKind::Read ? "read" : event.kind == EventKind::Write ? "write" : "publish";
            add(fmt::format(
                R"({{"name":"{} {}","cat":"io","ph":"i","s":"t","pid":1,"tid":{},"ts":{:.3f},"args":{{"storage":"{}","bytes":{}}}}})", what,
                json_escape(event.tensor), event.thread, event.at_ms * 1000.0, json_escape(event.storage), event.bytes));
            break;
        }
        default:
            break;
        }
    }
    out += "\n]}\n";
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file) {
        EINSUMS_THROW_EXCEPTION(std::runtime_error, "EventLog::write_chrome_trace: cannot open '{}' for writing", path);
    }
    file << out;
    if (!file) {
        EINSUMS_THROW_EXCEPTION(std::runtime_error, "EventLog::write_chrome_trace: writing '{}' failed", path);
    }
}

namespace detail {

namespace {

std::uint64_t this_thread() {
    return static_cast<std::uint64_t>(std::hash<std::thread::id>{}(std::this_thread::get_id()));
}

/// Call @p tell on the observer, detaching it if it throws: observing a calculation must never be
/// the reason it stops. Logged once per process, which is enough to say it happened.
template <typename Tell>
void deliver(RunContext &context, ExecutionObserver &observer, Tell &&tell) {
    try {
        tell(observer);
    } catch (std::exception const &error) {
        context.detach_observer();
        static std::once_flag once;
        std::call_once(once, [&] { EINSUMS_LOG_WARN("ComputeGraph: an execution observer threw and was detached: {}", error.what()); });
    } catch (...) {
        context.detach_observer();
        static std::once_flag once;
        std::call_once(once, [] { EINSUMS_LOG_WARN("ComputeGraph: an execution observer threw and was detached"); });
    }
}

} // namespace

void notify_replay(RunContext &context, EventKind kind, Graph const &graph, RunMetrics const *run) {
    ExecutionObserver *const observer = context.observer();
    if (observer == nullptr) {
        return;
    }
    ExecutionEvent const event{.kind = kind, .graph = graph.name(), .at_ms = context.elapsed_ms(), .thread = this_thread()};
    deliver(context, *observer, [&](ExecutionObserver &to) {
        if (kind == EventKind::ReplayEnd && run != nullptr) {
            to.on_replay_end(event, *run);
        } else {
            to.on_replay_start(event);
        }
    });
}

void notify_node(RunContext &context, EventKind kind, Graph const &graph, Node const &node, double ms) {
    ExecutionObserver *const observer = context.observer();
    if (observer == nullptr) {
        return;
    }
    ExecutionEvent const event{.kind   = kind,
                               .graph  = graph.name(),
                               .node   = node.id,
                               .op     = node.kind,
                               .label  = node.label,
                               .at_ms  = context.elapsed_ms(),
                               .ms     = ms,
                               .thread = this_thread()};
    deliver(context, *observer, [&](ExecutionObserver &to) {
        if (kind == EventKind::NodeEnd) {
            to.on_node_end(event);
        } else {
            to.on_node_start(event);
        }
    });
}

void notify_io(RunContext &context, EventKind kind, std::string_view tensor, TensorBacking const &backing, std::uint64_t bytes) {
    ExecutionObserver *const observer = context.observer();
    if (observer == nullptr) {
        return;
    }
    ExecutionEvent const event{.kind    = kind,
                               .tensor  = std::string{tensor},
                               .storage = backing.describe(),
                               .at_ms   = context.elapsed_ms(),
                               .bytes   = bytes,
                               .thread  = this_thread()};
    deliver(context, *observer, [&](ExecutionObserver &to) { to.on_io(event); });
}

NodeObservation::NodeObservation(RunContext &context, Graph const &graph, Node const &node) : _graph(graph), _node(node) {
    if (context.observer() != nullptr) {
        _context = &context;
        _start   = std::chrono::steady_clock::now();
        notify_node(context, EventKind::NodeStart, graph, node, 0.0);
    }
}

NodeObservation::~NodeObservation() {
    if (_context != nullptr) {
        double const ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - _start).count();
        notify_node(*_context, EventKind::NodeEnd, _graph, _node, ms);
    }
}

} // namespace detail

EINSUMS_NAMESPACE_END(compute_graph)
