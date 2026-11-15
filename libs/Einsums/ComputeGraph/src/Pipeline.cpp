//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/Optimizer.hpp>
#include <Einsums/ComputeGraph/Pipeline.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Profile/Profile.hpp>

#include <fmt/format.h>

#include <functional>
#include <memory>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

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

/// What of Graph a pipeline reaches past its public surface: giving a stage the pipeline's
/// parameter table, and telling it where in the pipeline it sits.
struct detail::PipelineAccess {
    static void share_params(Graph &graph, std::shared_ptr<ParamTable> const &params) { graph.set_params_ptr(params); }

    static void set_stage(Graph &graph, std::string const &pipeline, std::string const &workspace, std::string const &stage,
                          std::string const &type, int index) {
        graph.set_stage(pipeline, workspace, stage, type, index);
    }
};

Graph &Pipeline::add_stage(std::string const &name) {
    _stages.push_back(Stage{.name = name, .graph = Graph{name}});
    auto &g = _stages.back().graph;
    detail::PipelineAccess::share_params(g, _params);
    attach_scope_maps(g);
    return g;
}

Graph &Pipeline::add_loop(std::string name, size_t max_iterations, LoopCondition condition) {
    // A loop stage is a stage graph holding one Loop node, so that the loop has a parent. A body
    // that was the stage graph itself had nowhere ahead of the loop to put a tensor's lifecycle,
    // and a pipeline-declared accumulator first used in the loop was re-zeroed every iteration.
    Graph &stage = add_stage(name);
    // The body shares the stage's parameter table (Graph::add_loop sees to that), and needs the
    // scope tables too, since its manifest is built from its own handles.
    Graph &body = stage.add_loop(std::move(name), max_iterations, std::move(condition));
    attach_scope_maps(body);
    auto const &loop          = stage.nodes().back().op_data.get<LoopDescriptor>();
    _stages.back().loop_body  = loop.body;
    _stages.back().loop_state = loop.state;
    return body;
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
    for_each_stage([&](Stage &stage, Graph &g, Graph *body) {
        std::string const type = body != nullptr ? "loop" : "graph";
        detail::PipelineAccess::set_stage(g, _name, ws_name, stage.name, type, si);
        if (body != nullptr) {
            detail::PipelineAccess::set_stage(*body, _name, ws_name, stage.name, type, si);
        }
        ++si;
    });
}

void Pipeline::run_stages(std::function<void(Graph &)> const &run) {
    for_each_stage([&](Stage &stage, Graph &graph, Graph *body) {
        // The iteration zones are the Loop node's own (see build_loop).
        WAGGLE_ZONE("{}:{}", body != nullptr ? "loop" : "stage", stage.name);
        run(graph);
    });
}

void Pipeline::execute() {
    WAGGLE_ZONE("Pipeline::execute({})", _name);

    propagate_stage_metadata();
    run_stages([](Graph &graph) { graph.execute(); });
}

void Pipeline::execute(Executor &executor) {
    WAGGLE_ZONE("Pipeline::execute({}, executor={})", _name, executor.name());

    propagate_stage_metadata();

    // A Loop node runs its body with the body's own executor, not with whatever runs the graph
    // holding it, so the caller's executor is installed on each loop body for this call: a loop
    // stage ran every iteration under it before loop stages became Loop nodes, and the
    // documentation says it does. The body's own executor comes back however the call ends.
    // Non-owning: the caller's executor outlives this call, and the bodies drop it before return.
    std::shared_ptr<Executor> const borrowed(std::shared_ptr<Executor>{}, &executor);
    struct Restore {
        std::vector<std::pair<Graph *, std::shared_ptr<Executor>>> previous;
        ~Restore() {
            for (auto &[body, prior] : previous) {
                body->set_executor(std::move(prior));
            }
        }
    } restore;
    for_each_stage([&](Stage &, Graph &, Graph *body) {
        if (body != nullptr) {
            restore.previous.emplace_back(body, body->executor());
            body->set_executor(borrowed);
        }
    });

    run_stages([&executor](Graph &graph) { graph.execute(executor); });
}

bool Pipeline::apply(PassManager &pm) {
    // Pre-register pipeline-declared tensors with each stage graph.
    // This ensures that MaterializationPass can find them and how each is
    // initialized.
    // Track which tensors have been assigned a Materialize node (first use only).
    std::unordered_set<void *> materialized_in_earlier_stage;

    // A stage touches a tensor if its graph does or any loop body or branch below it does: a
    // loop stage's work is all in its Loop node's body. The holders are every graph of the stage
    // with a handle for the tensor, so the lifecycle is on the handle Materialization reads, which
    // for a tensor only a body uses is the body's, hoisted into the stage graph ahead of the loop.
    // Setup bodies are included: Materialization gives a Setup body's own deferred handle its
    // lifecycle inside that body, so it runs once per bind with the setup, and skips it when an
    // enclosing graph already placed one for the same tensor.
    std::vector<std::pair<Graph *, TensorId>> holders;
    // NOLINTNEXTLINE(misc-no-recursion): sub-graphs nest.
    std::function<void(Graph &, void const *)> collect = [&](Graph &graph, void const *ptr) {
        // The address index rather than a scan of the tensor table: a pipeline with many declared
        // tensors and stages holding many tensors made this quadratic for an O(1) lookup.
        if (TensorId const tid = graph.find_tensor_id_by_ptr(ptr); tid != 0) {
            holders.emplace_back(&graph, tid);
        }
        graph.for_each_subgraph([&](Graph &sub) { collect(sub, ptr); }, /*include_setup=*/true);
    };

    for_each_stage([&](Stage &, Graph &graph, Graph *) {
        for (auto const &handle : _handles) {
            holders.clear();
            collect(graph, handle.tensor_ptr);
            if (holders.empty()) {
                continue; // this stage does not touch the tensor
            }
            bool const earlier = materialized_in_earlier_stage.contains(handle.tensor_ptr);
            for (auto const &[holder, tid] : holders) {
                if (earlier) {
                    // Already materialized in an earlier stage, don't re-materialize/re-initialize
                    holder->mark_materialized(tid);
                } else {
                    // First stage using this tensor, set up materialization
                    holder->adopt_lifecycle(tid, handle);
                }
            }
            materialized_in_earlier_stage.insert(handle.tensor_ptr);
        }
    });

    // Now run passes on each stage.
    bool modified = false;
    for_each_stage([&](Stage &, Graph &graph, Graph *) { modified |= graph.apply(pm); });
    return modified;
}

EINSUMS_NAMESPACE_END(compute_graph)
