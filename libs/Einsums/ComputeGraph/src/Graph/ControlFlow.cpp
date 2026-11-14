//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file ControlFlow.cpp
/// @brief Conditionals, loops, setup regions, and the accuracy contract.
///
/// The construction half of control flow: the calls that add a node holding a
/// subgraph and hand back a reference the caller captures into. How such a node
/// is later EXECUTED is in `Execute.cpp`; how its subtree contributes inputs and
/// outputs to the dependence graph is in `Schedule.cpp`.
///
/// The accuracy contract shares this file because it shares the subject. A setup
/// region is the thing an approximation gets installed into, `_setup_key` is
/// what decides whether an installed one is still current, and a budget is what
/// decides whether the next approximation may be installed at all.

#include <Einsums/CXX23/Expected.hpp>
#include <Einsums/ComputeGraph/CaptureContext.hpp>
#include <Einsums/ComputeGraph/Detail/ScalarDispatch.hpp>
#include <Einsums/ComputeGraph/EinsumSpec.hpp>
#include <Einsums/ComputeGraph/Error.hpp>
#include <Einsums/ComputeGraph/ExecutorBuilder.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Optimizer.hpp> // For OptimizerPass and PassManager
#include <Einsums/ComputeGraph/Options.hpp>
#include <Einsums/ComputeGraph/SpaceRegistryAccess.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/Profile/Profile.hpp>
#include <Einsums/Tensor/Tensor.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <optional>
#include <ostream>
#include <queue>
#include <ranges>
#include <set>
#include <span>
#include <unordered_set>
#include <utility>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

std::tuple<Graph &, Graph &> Graph::add_conditional(std::string label, std::function<bool()> predicate) {
    // An EMPTY function is wrapped rather than turned into a literal, so a
    // caller who passed one still gets the std::bad_function_call it has always
    // got instead of silently taking a branch.
    return add_conditional(std::move(label), PredExpr::callback(std::move(predicate)));
}

std::tuple<Graph &, Graph &> Graph::add_conditional(std::string label, PredExpr predicate) {
    auto then_graph = std::make_shared<Graph>(label + "/then");
    auto else_graph = std::make_shared<Graph>(label + "/else");
    // Branches are scopes of this graph: a parameter written before the conditional is the one
    // they resolve against.
    then_graph->set_params_ptr(_param_store.table());
    else_graph->set_params_ptr(_param_store.table());
    // And one run context, so a branch's replay counts into this graph's (last_run).
    share_run_context(*then_graph);
    share_run_context(*else_graph);

    ConditionalDescriptor desc;
    desc.predicate   = std::move(predicate);
    desc.then_branch = then_graph;
    desc.else_branch = else_graph;

    // Dtype and rank are meaningless for a control-flow node, which has no
    // tensor destination; the builder neither dispatches on nor validates them.
    OpData op_data(std::move(desc));
    auto   executor = build_executor(OpKind::Conditional, packed_gemm::ScalarType::Unknown, 0, op_data, *this, {}, {});

    Node node;
    node.kind    = OpKind::Conditional;
    node.label   = std::move(label);
    node.execute = std::move(executor);
    node.op_data = std::move(op_data);

    add_node(std::move(node));

    return {*then_graph, *else_graph};
}

std::tuple<Graph &, Graph &> Graph::add_conditional_flag(std::string label, GateFlags const &flags, size_t index) {
    // The buffer, not the handle: the node has to keep reading the same array after the caller's
    // GateFlags goes out of scope, and a shared_ptr copy is what makes that true. PredExpr::FlagTest
    // expresses the load as data, so the node is saveable as well as GIL-free.
    return add_conditional(std::move(label), PredExpr::flag(flags, index));
}

// NOLINTNEXTLINE(misc-no-recursion): sub-graphs nest.
void Graph::set_params_ptr(std::shared_ptr<ParamTable> params) {
    _param_store.set_table(std::move(params));
    for (auto &node : _nodes) {
        for_each_child_graph(node, [this](Graph &child) { child.set_params_ptr(_param_store.table()); });
    }
}

Graph &Graph::add_loop(std::string label, size_t max_iterations, std::function<bool(size_t)> condition) {
    // An absent condition has always meant "run to max_iterations", and a
    // default PredExpr is an unconditional true, which says exactly that.
    return add_loop(std::move(label), max_iterations, condition ? PredExpr::callback(std::move(condition)) : PredExpr{});
}

Graph &Graph::add_loop(std::string label, size_t max_iterations, PredExpr condition) {
    return add_loop_at(std::move(label), max_iterations, std::move(condition), _nodes.size());
}

Graph &Graph::add_loop_at(std::string label, size_t max_iterations, PredExpr condition, std::size_t position) {
    auto body_graph = std::make_shared<Graph>(label + "/body");
    // One table with the parent, as for a setup body below. The loop's condition is evaluated
    // against this graph's table, so a body holding its own would write parameters the condition
    // never sees, and could not read one written before the loop.
    body_graph->set_params_ptr(_param_store.table());
    share_run_context(*body_graph);

    LoopDescriptor desc;
    desc.body           = body_graph;
    desc.max_iterations = max_iterations;
    desc.condition      = std::move(condition);
    // Shared with the executor, so the iteration count the replay writes is
    // observable on the node afterwards. See LoopDescriptor::last_iteration_count.
    desc.state = std::make_shared<LoopState>();

    OpData op_data(std::move(desc));
    auto   executor = build_executor(OpKind::Loop, packed_gemm::ScalarType::Unknown, 0, op_data, *this, {}, {});

    Node node;
    node.kind    = OpKind::Loop;
    node.label   = std::move(label);
    node.execute = std::move(executor);
    node.op_data = std::move(op_data);
    // Through insert_node_groups rather than add_node, which is what assigns an id when a
    // caller appends: the splice does not, so the id is reserved here.
    node.id = reserve_node_id();

    std::vector<std::pair<std::size_t, std::vector<Node>>> group;
    group.emplace_back(std::min(position, _nodes.size()), std::vector<Node>{std::move(node)});
    insert_node_groups(std::move(group));

    return *body_graph;
}

void Graph::add_loop(std::string label, size_t max_iterations, std::function<bool(size_t)> condition, std::function<void()> body_fn) {
    auto              &body = add_loop(std::move(label), max_iterations, std::move(condition));
    CaptureGuard const g(body);
    body_fn();
}

// ── Setup subgraphs ─────────────────────────────────────────────────────────

Graph &Graph::add_setup(std::string label) {
    return add_setup_at(std::move(label), _nodes.size());
}

Graph &Graph::add_setup_at(std::string label, std::size_t position) {
    auto body_graph = std::make_shared<Graph>(label + "/setup");

    // One parameter table with the parent, not a fresh one. A setup body is a PHASE of this
    // graph rather than a separate scope: a value it writes out with cg::write_param exists to
    // be read by the caller or by a later node here, and a body holding its own table would
    // write it somewhere nobody looks. That is how the fitting diagnostic went missing the
    // first time it was asked for.
    body_graph->set_params_ptr(_param_store.table());
    share_run_context(*body_graph);

    SetupDescriptor desc;
    desc.body = body_graph;
    // Shared with the executor, so the "already computed" answer a replay writes is the
    // one a later bind clears. See LoopDescriptor::state for why this is not a plain field.
    desc.state = std::make_shared<SetupState>();
    // A setup node added after a key was declared still belongs to the problem the caller
    // named; the key is graph state and the node is where it has to be readable from.
    desc.state->pending_key = _binding.setup_key();

    OpData op_data(std::move(desc));
    auto   executor = build_executor(OpKind::Setup, packed_gemm::ScalarType::Unknown, 0, op_data, *this, {}, {});

    Node node;
    node.kind    = OpKind::Setup;
    node.label   = std::move(label);
    node.execute = std::move(executor);
    node.op_data = std::move(op_data);
    node.id      = reserve_node_id();

    // Through insert_node_groups rather than add_node, because that is the splice that
    // already knows how to keep positions valid; appending and then moving would be a second
    // way of doing it and one more thing to keep in step.
    std::vector<std::pair<std::size_t, std::vector<Node>>> group;
    group.emplace_back(std::min(position, _nodes.size()), std::vector<Node>{std::move(node)});
    insert_node_groups(std::move(group));

    return *body_graph;
}

void Graph::add_setup(std::string label, std::function<void()> body_fn) {
    {
        auto              &body = add_setup(std::move(label));
        CaptureGuard const g(body);
        body_fn();
    }
    // The one add_setup spelling that HAS a moment when the body is complete, so the node's
    // lists are right from here on without waiting for a sort or a pass.
    refresh_setup_io();
}

bool Graph::has_setup() const noexcept {
    return std::any_of(_nodes.begin(), _nodes.end(), [](Node const &node) { return node.kind == OpKind::Setup; });
}

void Graph::run_setup(bool force) {
    for (auto &node : _nodes) {
        auto *desc = node.op_data.get_if<SetupDescriptor>();
        if (desc == nullptr || !desc->body) {
            continue;
        }
        if (force && desc->state != nullptr) {
            // Clear both, not just the flag: a stale key would otherwise let the very next
            // guard skip the body that this call exists to force.
            desc->state->computed = false;
            desc->state->computed_key.clear();
        }
        // Through the node's own executor rather than by calling body->execute() here, so
        // there is one place that decides whether a setup body runs. A second copy of the
        // two guards is a second thing to keep in step with the first.
        node.execute();
    }
}

void Graph::invalidate_setup() {
    for (auto &node : _nodes) {
        auto *desc = node.op_data.get_if<SetupDescriptor>();
        if (desc == nullptr || desc->state == nullptr) {
            continue;
        }
        desc->state->computed = false;
    }
}

// ── The accuracy contract ───────────────────────────────────────────────────

std::string Graph::can_approximate(ApproximationRecord const &candidate) const {
    return _optimization.refusal(candidate);
}

void Graph::note_approximation(ApproximationRecord record) {
    if (std::string reason = _optimization.refusal(record); !reason.empty()) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "Graph '{}': {}", _name, reason);
    }
    _optimization.add_approximation(std::move(record));
    // A record is saved structure, so adding one changes what a save writes and what a
    // content hash covers.
    note(Change::NodeSetChanged);
}

void Graph::restore_approximations(std::vector<ApproximationRecord> records) {
    _optimization.restore_approximations(std::move(records));
    note(Change::NodeSetChanged);
}

double Graph::accuracy_spent(ApproximationEffect effect, std::string const &output) const {
    return _optimization.accuracy_spent(effect, output);
}

ApproximationTolerance Graph::approximation_tolerance(std::string const &output) const {
    return _optimization.approximation_tolerance(output);
}

void Graph::set_accuracy_budget(ApproximationEffect effect, double value) {
    if (!std::isfinite(value) || value < 0) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                "Graph '{}': an accuracy budget of {} is not a bound anything can be checked against", _name, value);
    }
    _optimization.set_budget(effect, value);
}

void Graph::clear_accuracy_budget() {
    _optimization.clear_budget();
}

void Graph::set_setup_key(std::string key) {
    _binding.set_setup_key(std::move(key));
    for (auto &node : _nodes) {
        auto *desc = node.op_data.get_if<SetupDescriptor>();
        if (desc == nullptr || desc->state == nullptr) {
            continue;
        }
        desc->state->pending_key = _binding.setup_key();
    }
}

EINSUMS_NAMESPACE_END(compute_graph)
