//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/Optimizer.hpp>
#include <Einsums/ComputeGraph/Pipeline.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Profile/Profile.hpp>

#include <fmt/format.h>

#include <stdexcept>
#include <unordered_set>
#include <utility>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

Pipeline::Pipeline(std::string name) : _name(std::move(name)) {
}

void Pipeline::attach_scope_maps(Graph &stage) const {
    // Workspace first, pipeline second: the narrower scope wins a tie, and a
    // tensor cannot honestly be declared by both.
    if (_workspace != nullptr) {
        stage.add_scope_map(_workspace->scope_map());
    }
    stage.add_scope_map(_scopes);
}

Graph &Pipeline::add_stage(std::string const &name) {
    _stages.push_back(Stage{.name = name, .content = Graph{name}});
    auto &g = std::get<Graph>(_stages.back().content);
    g.set_params_ptr(_params);
    attach_scope_maps(g);
    return g;
}

Graph &Pipeline::add_loop(std::string name, size_t max_iterations, LoopCondition condition) {
    LoopNode loop;
    loop.body           = Graph{name};
    loop.max_iterations = max_iterations;
    loop.condition      = std::move(condition);
    _stages.push_back(Stage{.name = std::move(name), .content = std::move(loop)});
    auto &g = std::get<LoopNode>(_stages.back().content).body;
    g.set_params_ptr(_params);
    attach_scope_maps(g);
    return g;
}

void Pipeline::run(PassManager &pm) {
    if (_workspace == nullptr) {
        throw std::runtime_error(fmt::format("Pipeline::run() called on '{}' with no associated workspace — "
                                             "call set_workspace(...) or use cg::run(name, workspace, build)",
                                             _name));
    }
    apply(pm);
    _workspace->materialize_all();
    execute();
}

void Pipeline::run() {
    auto pm = PassManager::create_default();
    run(pm);
}

void Pipeline::propagate_stage_metadata() {
    std::string const ws_name = _workspace ? _workspace->name() : "";

    int si = 0;
    for_each_stage([&](Stage &stage, Graph &g, LoopNode *loop) {
        g.set_pipeline_name(_name);
        g.set_workspace_name(ws_name);
        g.set_stage_name(stage.name);
        g.set_stage_type(loop != nullptr ? "loop" : "graph");
        g.set_stage_index(si++);
    });
}

void Pipeline::run_stages(std::function<void(Graph &)> const &run) {
    for_each_stage([&](Stage &stage, Graph &graph, LoopNode *loop) {
        if (loop == nullptr) {
            LabeledSection("stage:{}", stage.name);
            run(graph);
            return;
        }

        LabeledSection("loop:{}", stage.name);
        loop->last_iteration_count = 0;
        for (size_t iter = 0; iter < loop->max_iterations; iter++) {
            // The iteration zone closes before the bookkeeping below, which
            // is what the manual push/pop pair it replaces did.
            {
                LabeledSection("iteration:{}", iter);
                run(graph);
            }
            loop->last_iteration_count = iter + 1;
            if (loop->condition && !loop->condition(iter)) {
                break;
            }
        }
    });
}

void Pipeline::execute() {
    LabeledSection("Pipeline::execute({})", _name);

    propagate_stage_metadata();
    run_stages([](Graph &graph) { graph.execute(); });
}

void Pipeline::execute(Executor &executor) {
    LabeledSection("Pipeline::execute({}, executor={})", _name, executor.name());

    propagate_stage_metadata();
    run_stages([&executor](Graph &graph) { graph.execute(executor); });
}

bool Pipeline::apply(PassManager &pm) {
    // Pre-register pipeline-declared tensors with each stage graph.
    // This ensures that MaterializationPass can find them and their
    // materialize_fn/zero_fn/random_fn lambdas.
    // Track which tensors have been assigned a Materialize node (first use only).
    std::unordered_set<void *> materialized_in_earlier_stage;

    auto register_in_graph = [&](Graph &graph) {
        for (auto const &handle : _handles) {
            // The stage graph's own id for this buffer, through the address index rather than a
            // scan of its tensor table: a pipeline with many declared tensors and stages holding
            // many tensors made the pair of loops quadratic for a lookup the graph answers in O(1).
            TensorId const tid = graph.find_tensor_id_by_ptr(handle.tensor_ptr);
            if (tid == 0) {
                continue; // this stage does not touch the tensor
            }
            auto &mutable_h = graph.tensor(tid);

            if (materialized_in_earlier_stage.count(handle.tensor_ptr)) {
                // Already materialized in an earlier stage, don't re-materialize/re-initialize
                mutable_h.alloc_state = AllocState::Materialized;
                mutable_h.init_kind   = InitKind::None;
            } else {
                // First stage using this tensor, set up materialization
                mutable_h.alloc_state    = handle.alloc_state;
                mutable_h.init_kind      = handle.init_kind;
                mutable_h.materialize_fn = handle.materialize_fn;
                mutable_h.zero_fn        = handle.zero_fn;
                mutable_h.random_fn      = handle.random_fn;
                materialized_in_earlier_stage.insert(handle.tensor_ptr);
            }
        }
    };

    for_each_stage([&](Stage &, Graph &graph, LoopNode *) { register_in_graph(graph); });

    // Now run passes on each stage.
    bool modified = false;
    for_each_stage([&](Stage &, Graph &graph, LoopNode *) { modified |= graph.apply(pm); });
    return modified;
}

EINSUMS_NAMESPACE_END(compute_graph)
