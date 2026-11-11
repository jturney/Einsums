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

#include <algorithm>
#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <unordered_map>
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

void NodeEdit::replace(Node replacement) {
    NodeId const id     = _node.id;
    Target const target = _node.target;
    _node               = std::move(replacement);
    _node.id            = id;
    _node.target        = target;
    _node.execute       = nullptr; // the commit's to give, from the custom executor or the builder
    _executor_given     = _node.custom_execute != nullptr;
}

void NodeEdit::add_pre_run(std::function<void()> hook) {
    if (!hook) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "NodeEdit::add_pre_run: node '{}' was given an empty hook", _node.label);
    }
    _node.pre_run.push_back(std::move(hook));
}

void Graph::give_executor(Node &node, bool placed) {
    std::function<void()> core;
    if (node.folded) {
        core = [] {};
    } else if (node.custom_execute) {
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
    if (placed) {
        node.cpu_fallback = node.execute;
    }
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

        if (node.target != before.target && !edit._placed) {
            EINSUMS_THROW_EXCEPTION(std::logic_error,
                                    "Graph '{}': an edit of node '{}' changed its target directly; place it through "
                                    "NodeEdit::place_on, which keeps its CPU fallback in step",
                                    _name, node.label);
        }

        sync_destination_input(node);
        bool const operands_changed = node.kind != before.kind || node.inputs != before.inputs || node.outputs != before.outputs;

        if (node.custom_execute && operands_changed && !edit._executor_given && !node.folded) {
            EINSUMS_THROW_EXCEPTION(std::logic_error,
                                    "Graph '{}': node '{}' runs an executor its pass wrote, and this edit changed what it reads or "
                                    "writes without giving it a new one through NodeEdit::set_executor",
                                    _name, node.label);
        }
        // The same for what it computes: that executor holds its own copy of the descriptor, so a
        // change to the descriptor would never reach it. A descriptor type without operator== cannot
        // say, and is let through.
        if (node.custom_execute && !edit._executor_given && !node.folded &&
            node.op_data.equals(before.op_data) == std::optional<bool>{false}) {
            EINSUMS_THROW_EXCEPTION(std::logic_error,
                                    "Graph '{}': node '{}' runs an executor its pass wrote, and this edit changed its {} without "
                                    "giving it a new one through NodeEdit::set_executor",
                                    _name, node.label, node.op_data.name());
        }
        bool const placed = edit._placed ? node.target != Target::CPU : static_cast<bool>(before.cpu_fallback);
        if (!placed) {
            node.cpu_fallback = nullptr;
        }
        give_executor(node, placed);
    } catch (...) {
        node = before;
        throw;
    }
    note_node_edit();
}

void Graph::restore_nodes(std::vector<Node> saved) {
    std::scoped_lock const lock(_content_mutex);
    _nodes = std::move(saved);
    note_node_edit();
    note_structural_change();
}

void Graph::set_schedule(std::size_t index, NodeSchedule const &schedule) {
    std::scoped_lock const lock(_content_mutex);
    if (index >= _nodes.size()) {
        EINSUMS_THROW_EXCEPTION(std::out_of_range, "Graph '{}': schedule of node {}, but the graph holds {}", _name, index, _nodes.size());
    }
    Node &node              = _nodes[index];
    node.thread_width       = schedule.thread_width;
    node.admission_priority = schedule.admission_priority;
    node.stream_id          = schedule.stream_id;
}

void Graph::reorder(std::span<std::size_t const> order) {
    std::scoped_lock const lock(_content_mutex);
    std::size_t const      count = _nodes.size();
    if (order.size() != count) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "Graph '{}': reorder names {} positions, but the graph holds {} nodes", _name,
                                order.size(), count);
    }
    std::vector<bool> taken(count, false);
    for (std::size_t const from : order) {
        if (from >= count || taken[from]) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument, "Graph '{}': reorder is not a permutation; position {} is {}", _name, from,
                                    from >= count ? "past the last node" : "named twice");
        }
        taken[from] = true;
    }
    std::vector<Node> ordered;
    ordered.reserve(count);
    for (std::size_t const from : order) {
        ordered.push_back(std::move(_nodes[from]));
    }
    _nodes = std::move(ordered);
    mark_sorted();
}

void Graph::move_node(std::size_t from, std::size_t to) {
    std::scoped_lock const lock(_content_mutex);
    if (from >= _nodes.size() || to >= _nodes.size()) {
        EINSUMS_THROW_EXCEPTION(std::out_of_range, "Graph '{}': move of node {} to {}, but the graph holds {}", _name, from, to,
                                _nodes.size());
    }
    if (from == to) {
        return;
    }
    auto const at = [this](std::size_t i) { return _nodes.begin() + static_cast<std::ptrdiff_t>(i); };
    if (from < to) {
        std::rotate(at(from), at(from + 1), at(to + 1));
    } else {
        std::rotate(at(to), at(from), at(from + 1));
    }
    mark_sorted();
}

std::vector<std::size_t> Graph::hoist(Graph &child, std::span<std::size_t const> picks, std::size_t position) {
    std::scoped_lock const lock(_content_mutex, child._content_mutex);
    bool                   is_body = false;
    for (auto const &node : _nodes) {
        for_each_child_graph(node, [&](Graph const &sub) { is_body = is_body || &sub == &child; });
    }
    if (!is_body) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "Graph '{}': hoist from '{}', which is not a body of this graph's nodes", _name,
                                child._name);
    }
    if (position > _nodes.size()) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "Graph '{}': hoist to position {}, past the end of {} nodes", _name, position,
                                _nodes.size());
    }
    for (std::size_t k = 0; k < picks.size(); ++k) {
        if (picks[k] >= child._nodes.size() || (k > 0 && picks[k] <= picks[k - 1])) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument, "Graph '{}': hoist picks must be ascending positions in '{}'", _name,
                                    child._name);
        }
    }
    if (picks.empty()) {
        return {};
    }
    // Refused before anything moves: a node with no executor its pass gave must be one the
    // builder can make. It is asked in the body, where the node's ids still resolve.
    for (std::size_t const pick : picks) {
        Node const &node = child._nodes[pick];
        if (node.custom_execute || node.folded) {
            continue;
        }
        try {
            (void)build_executor(node, child);
        } catch (std::exception const &e) {
            EINSUMS_THROW_EXCEPTION(std::logic_error,
                                    "Graph '{}': hoisted node '{}' cannot be given an executor from its fields ({}); a pass that writes "
                                    "its executor gives it through NodeEdit::set_executor",
                                    _name, node.label, e.what());
        }
    }

    std::unordered_map<TensorId, TensorId>     parent_id;
    std::vector<std::pair<TensorId, TensorId>> handed_over; // (body id, this graph's id)
    auto const                                 lift = [&](TensorId body_tid) -> TensorId {
        if (auto const it = parent_id.find(body_tid); it != parent_id.end()) {
            return it->second;
        }
        TensorHandle const &handle = child.tensor(body_tid);
        TensorId const      mine   = find_or_register_tensor_ptr(handle);
        parent_id[body_tid]        = mine;
        if (handle.alloc_state == AllocState::Deferred && handle.is_intermediate) {
            handed_over.emplace_back(body_tid, mine);
        }
        return mine;
    };

    std::vector<Node> moved;
    moved.reserve(picks.size());
    auto const body_anchor = child.anchor();
    // Copies, so that nothing in the body has moved until every one of them has its executor.
    for (std::size_t const pick : picks) {
        Node node = child._nodes[pick];
        // The body numbered this node; this graph issues its own id.
        node.id = unassigned_node_id;
        if (node.custom_execute) {
            // Its executor reaches its operands through the body's slots, which the body re-seats
            // only when the body itself runs. Moved out, it runs first, after this graph may have
            // placed a buffer anew, so the slots it names are re-seated ahead of it: those only,
            // since moved nodes run concurrently here and another may be reading the rest.
            std::vector<TensorId> touched(node.inputs);
            touched.insert(touched.end(), node.outputs.begin(), node.outputs.end());
            node.pre_run.emplace_back([body_anchor, touched = std::move(touched)]() { body_anchor->graph().resync_slot_storage(touched); });
        }
        for (auto &tid : node.inputs) {
            tid = lift(tid);
        }
        for (auto &tid : node.outputs) {
            tid = lift(tid);
        }
        give_executor(node, static_cast<bool>(node.cpu_fallback));
        moved.push_back(std::move(node));
    }

    std::vector<Node> kept;
    kept.reserve(child._nodes.size() - picks.size());
    for (std::size_t i = 0, k = 0; i < child._nodes.size(); ++i) {
        if (k < picks.size() && picks[k] == i) {
            ++k;
            continue;
        }
        kept.push_back(std::move(child._nodes[i]));
    }
    child._nodes = std::move(kept);

    // The writer of a deferred intermediate the body declared moves out with it, so the declaration
    // does too. Left the body's, a pass run on the body alone would see only the body's readers and
    // treat the tensor as its own to change: LayoutAssignment once re-ordered a hoisted operand under
    // the builder that writes it in the parent.
    for (auto const &[body_tid, mine] : handed_over) {
        TensorHandle &stand_in          = child.tensor(body_tid);
        TensorHandle &declared          = tensor(mine);
        declared.is_intermediate        = true;
        declared.materialize_on_execute = stand_in.materialize_on_execute;
        stand_in.is_intermediate        = false;
        stand_in.materialize_on_execute = false;
    }
    child.note_structural_change();
    child.mark_sorted();

    _nodes.insert(_nodes.begin() + static_cast<std::ptrdiff_t>(position), std::make_move_iterator(moved.begin()),
                  std::make_move_iterator(moved.end()));
    note_structural_change();
    mark_sorted();

    std::vector<std::size_t> positions(picks.size());
    for (std::size_t k = 0; k < positions.size(); ++k) {
        positions[k] = position + k;
    }
    return positions;
}

EINSUMS_NAMESPACE_END(compute_graph)
