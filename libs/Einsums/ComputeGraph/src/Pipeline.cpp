//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/Optimizer.hpp>
#include <Einsums/ComputeGraph/Pipeline.hpp>
#include <Einsums/Profile/Profile.hpp>

#include <fmt/format.h>

#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace einsums::compute_graph {

Pipeline::Pipeline(std::string name) : _name(std::move(name)) {
}

Graph &Pipeline::add_stage(std::string const &name) {
    _stages.push_back(Stage{.name = name, .content = Graph{name}});
    auto &g = std::get<Graph>(_stages.back().content);
    g.set_params_ptr(_params);
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

void Pipeline::execute() {
    profile::Profiler::instance().push(fmt::format("Pipeline::execute({})", _name));

    // Propagate hierarchy metadata to child graphs for the profiler viewer
    std::string ws_name = _workspace ? _workspace->name() : "";
    for (int si = 0; std::cmp_less(si, _stages.size()); si++) {
        auto &stage    = _stages[si];
        auto  set_meta = [&](Graph &g, char const *type) {
            g.set_pipeline_name(_name);
            g.set_workspace_name(ws_name);
            g.set_stage_name(stage.name);
            g.set_stage_type(type);
            g.set_stage_index(si);
        };
        if (auto *graph = std::get_if<Graph>(&stage.content)) {
            set_meta(*graph, "graph");
        } else if (auto *loop = std::get_if<LoopNode>(&stage.content)) {
            set_meta(loop->body, "loop");
        }
    }

    for (auto &stage : _stages) {
        if (auto *graph = std::get_if<Graph>(&stage.content)) {
            profile::Profiler::instance().push(fmt::format("stage:{}", stage.name));
            graph->execute();
            profile::Profiler::instance().pop();
        } else if (auto *loop = std::get_if<LoopNode>(&stage.content)) {
            profile::Profiler::instance().push(fmt::format("loop:{}", stage.name));
            loop->last_iteration_count = 0;
            for (size_t iter = 0; iter < loop->max_iterations; iter++) {
                profile::Profiler::instance().push(fmt::format("iteration:{}", iter));
                loop->body.execute();
                profile::Profiler::instance().pop();
                loop->last_iteration_count = iter + 1;
                if (loop->condition && !loop->condition(iter)) {
                    break;
                }
            }
            profile::Profiler::instance().pop();
        }
    }

    profile::Profiler::instance().pop();
}

void Pipeline::execute(Executor &executor) {
    profile::Profiler::instance().push(fmt::format("Pipeline::execute({}, executor={})", _name, executor.name()));

    // Propagate hierarchy metadata to child graphs for the profiler viewer
    std::string ws_name = _workspace ? _workspace->name() : "";
    for (int si = 0; std::cmp_less(si, _stages.size()); si++) {
        auto &stage    = _stages[si];
        auto  set_meta = [&](Graph &g, char const *type) {
            g.set_pipeline_name(_name);
            g.set_workspace_name(ws_name);
            g.set_stage_name(stage.name);
            g.set_stage_type(type);
            g.set_stage_index(si);
        };
        if (auto *graph = std::get_if<Graph>(&stage.content)) {
            set_meta(*graph, "graph");
        } else if (auto *loop = std::get_if<LoopNode>(&stage.content)) {
            set_meta(loop->body, "loop");
        }
    }

    for (auto &stage : _stages) {
        if (auto *graph = std::get_if<Graph>(&stage.content)) {
            profile::Profiler::instance().push(fmt::format("stage:{}", stage.name));
            graph->execute(executor);
            profile::Profiler::instance().pop();
        } else if (auto *loop = std::get_if<LoopNode>(&stage.content)) {
            profile::Profiler::instance().push(fmt::format("loop:{}", stage.name));
            loop->last_iteration_count = 0;
            for (size_t iter = 0; iter < loop->max_iterations; iter++) {
                profile::Profiler::instance().push(fmt::format("iteration:{}", iter));
                loop->body.execute(executor);
                profile::Profiler::instance().pop();
                loop->last_iteration_count = iter + 1;
                if (loop->condition && !loop->condition(iter)) {
                    break;
                }
            }
            profile::Profiler::instance().pop();
        }
    }

    profile::Profiler::instance().pop();
}

bool Pipeline::apply(PassManager &pm) {
    // Pre-register pipeline-declared tensors with each stage graph.
    // This ensures that MaterializationPass can find them and their
    // materialize_fn/zero_fn/random_fn lambdas.
    // Track which tensors have been assigned a Materialize node (first use only).
    std::unordered_set<void *> materialized_in_earlier_stage;

    auto register_in_graph = [&](Graph &graph) {
        for (auto const &handle : _handles) {
            for (auto const &[tid, h] : graph.tensors_map()) {
                if (h.tensor_ptr == handle.tensor_ptr) {
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
                    break;
                }
            }
        }
    };

    for (auto &stage : _stages) {
        if (auto *graph = std::get_if<Graph>(&stage.content)) {
            register_in_graph(*graph);
        } else if (auto *loop = std::get_if<LoopNode>(&stage.content)) {
            register_in_graph(loop->body);
        }
    }

    // Now run passes on each stage.
    bool modified = false;
    for (auto &stage : _stages) {
        if (auto *graph = std::get_if<Graph>(&stage.content)) {
            modified |= graph->apply(pm);
        } else if (auto *loop = std::get_if<LoopNode>(&stage.content)) {
            modified |= loop->body.apply(pm);
        }
    }
    return modified;
}

} // namespace einsums::compute_graph
