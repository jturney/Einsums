//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/DestinationRead.hpp>
#include <Einsums/ComputeGraph/ExecutorBuilder.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Errors/ThrowException.hpp>

#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

namespace {

/// A call through @p shared, so every copy of the result runs the one callable.
std::function<void()> calling(std::shared_ptr<std::function<void()>> shared) {
    return [shared = std::move(shared)]() { (*shared)(); };
}

/// @p core with @p hooks run ahead of it, or @p core alone when there are none.
std::function<void()> with_pre_run(std::vector<std::function<void()>> const &hooks, std::function<void()> core) {
    if (hooks.empty()) {
        return core;
    }
    return [hooks, core = std::move(core)]() {
        for (auto const &hook : hooks) {
            hook();
        }
        if (core) {
            core();
        }
    };
}

} // namespace

void NodeEdit::set_executor(std::function<void()> executor) {
    if (!executor) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "NodeEdit::set_executor: node '{}' was given an empty executor", _node.label);
    }
    _node.custom_execute = std::make_shared<std::function<void()>>(std::move(executor));
    _executor_given      = true;
}

void NodeEdit::add_pre_run(std::function<void()> hook) {
    if (!hook) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "NodeEdit::add_pre_run: node '{}' was given an empty hook", _node.label);
    }
    _node.pre_run.push_back(std::move(hook));
}

void Graph::edit(std::size_t index, std::function<void(NodeEdit &)> const &change) {
    std::scoped_lock const lock(_content_mutex);
    if (index >= _nodes.size()) {
        EINSUMS_THROW_EXCEPTION(std::out_of_range, "Graph '{}': edit of node {}, but the graph holds {}", _name, index, _nodes.size());
    }
    Node      &node   = _nodes[index];
    Node const before = node;

    try {
        // The executor is the commit's to give; a pass that assigns it directly is the omission
        // this API exists to prevent, so it is cleared for the edit and checked after.
        node.execute = nullptr;
        NodeEdit edit(node);
        change(edit);
        if (node.execute) {
            EINSUMS_THROW_EXCEPTION(std::logic_error,
                                    "Graph '{}': an edit of node '{}' assigned its executor directly; give one the builder cannot "
                                    "make through NodeEdit::set_executor",
                                    _name, node.label);
        }

        sync_destination_input(node);
        bool const operands_changed = node.kind != before.kind || node.inputs != before.inputs || node.outputs != before.outputs;

        std::function<void()> core;
        if (node.folded) {
            core = [] {};
        } else if (node.custom_execute) {
            if (operands_changed && !edit._executor_given) {
                EINSUMS_THROW_EXCEPTION(std::logic_error,
                                        "Graph '{}': node '{}' runs an executor its pass wrote, and this edit changed what it reads or "
                                        "writes without giving it a new one through NodeEdit::set_executor",
                                        _name, node.label);
            }
            core = calling(node.custom_execute);
        } else {
            try {
                core = build_executor(node, *this);
            } catch (std::exception const &e) {
                EINSUMS_THROW_EXCEPTION(std::logic_error,
                                        "Graph '{}': node '{}' cannot be given an executor from its fields ({}); a pass that writes "
                                        "its executor gives it through NodeEdit::set_executor",
                                        _name, node.label, e.what());
            }
        }
        node.execute = with_pre_run(node.pre_run, std::move(core));

        if (node.folded) {
            node.async_start  = nullptr;
            node.async_finish = nullptr;
        } else if (node.async_start || node.custom_async_start) {
            // A pass always writes the asynchronous halves itself; the first commit keeps its
            // start phase apart so the hooks wrap it once however often the node is edited.
            if (!node.custom_async_start) {
                node.custom_async_start = std::make_shared<std::function<void()>>(std::move(node.async_start));
            }
            node.async_start = with_pre_run(node.pre_run, calling(node.custom_async_start));
        }
        if (before.cpu_fallback) {
            node.cpu_fallback = node.execute;
        }
    } catch (...) {
        node = before;
        throw;
    }
    note_node_edit();
}

EINSUMS_NAMESPACE_END(compute_graph)
