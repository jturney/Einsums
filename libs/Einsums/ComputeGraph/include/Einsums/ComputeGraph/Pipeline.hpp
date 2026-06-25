//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/ComputeGraph/BoundExpr.hpp>
#include <Einsums/ComputeGraph/CaptureContext.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/TensorHandle.hpp>
#include <Einsums/ComputeGraph/Workspace.hpp>
#include <Einsums/Python/Annotations.hpp>
#include <Einsums/Tensor/Tensor.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

namespace einsums::compute_graph {

class OptimizerPass;

/**
 * @brief Convergence/termination predicate for loop stages.
 *
 * Called after each iteration of a loop stage with the current iteration number
 * (0-based). Return true to continue iterating, false to break out of the loop.
 *
 * The predicate can inspect tensor values (e.g., compute energy differences,
 * density matrix RMS) to implement convergence-based early exit.
 *
 * @par Example
 * @code
 * auto condition = [&](size_t iter) -> bool {
 *     double delta = std::abs(energy - energy_old);
 *     return delta > 1e-8;  // Continue if not converged
 * };
 * @endcode
 */
using LoopCondition = std::function<bool(size_t iteration)>;

/**
 * @brief An iterative subgraph with convergence-based early exit.
 *
 * Contains a Graph (the loop body) that is executed repeatedly until either:
 * - The LoopCondition returns false (convergence)
 * - max_iterations is reached (safety limit)
 *
 * The condition is evaluated AFTER each iteration, so the body always runs at
 * least once.
 *
 * @see Pipeline::add_loop()
 */
struct LoopNode {
    Graph         body;                 ///< The subgraph to execute each iteration
    size_t        max_iterations{1000}; ///< Maximum number of iterations (safety limit)
    LoopCondition condition;            ///< Predicate: return false to stop, true to continue

    /// Set after execution: the actual number of iterations that ran.
    size_t last_iteration_count{0};
};

/**
 * @brief A pipeline of sequential stages with optional iterative loops.
 *
 * Pipeline models real computational workflows as a linear sequence of stages,
 * where each stage is either:
 * - A one-shot subgraph (e.g., setup, post-processing)
 * - An iterative loop with convergence check (e.g., SCF iterations)
 *
 * Intermediate tensors shared across stages should be declared in the outer
 * scope so they outlive the pipeline.
 *
 * @par Example: SCF-like workflow
 * @code
 * // Intermediates in outer scope
 * auto tmp = create_zero_tensor<double>("tmp", N, N);
 *
 * cg::Pipeline pipeline("scf");
 *
 * // Stage 1: Setup
 * {
 *     auto &setup = pipeline.add_stage("setup");
 *     cg::CaptureGuard guard(setup);
 *     cg::einsum(...);  // Initialize
 * }
 *
 * // Stage 2: Iterative loop
 * {
 *     auto &loop = pipeline.add_loop("scf_iter", 200,
 *         [&](size_t iter) { return !converged(); });
 *     cg::CaptureGuard guard(loop);
 *     cg::einsum(...);  // SCF step
 * }
 *
 * // Stage 3: Post-processing
 * {
 *     auto &post = pipeline.add_stage("post");
 *     cg::CaptureGuard guard(post);
 *     cg::einsum(...);  // Final computation
 * }
 *
 * // Optimize and run
 * cg::passes::ScaleAbsorption fuse;
 * pipeline.apply(pm);
 * pipeline.execute();
 * @endcode
 *
 * @see Graph for individual stage graphs
 * @see LoopNode for iterative stage configuration
 */
class APIARY_EXPOSE APIARY_MODULE("graph") APIARY_NOCOPY APIARY_NOMOVE EINSUMS_EXPORT Pipeline {
  public:
    /**
     * @brief Construct a pipeline with the given name.
     * @param[in] name Human-readable name for profiling and debugging.
     */
    APIARY_EXPOSE explicit Pipeline(std::string name);

    /**
     * @brief Add a one-shot (non-looping) stage to the pipeline.
     *
     * Returns a reference to the stage's Graph for use with CaptureGuard.
     * Stages execute in the order they are added.
     *
     * @param[in] name Human-readable name for the stage.
     * @return Reference to the stage's Graph.
     */
    APIARY_EXPOSE APIARY_RVP(reference_internal) Graph &add_stage(std::string const &name);

    /**
     * @brief Add an iterative loop stage to the pipeline.
     *
     * Returns a reference to the loop body's Graph for use with CaptureGuard.
     * The loop executes the body repeatedly until the condition returns false
     * or max_iterations is reached.
     *
     * @param[in] name Human-readable name for the loop stage.
     * @param[in] max_iterations Maximum number of iterations (safety limit).
     * @param[in] condition Predicate called after each iteration. Return false to stop.
     * @return Reference to the loop body's Graph.
     */
    APIARY_EXPOSE APIARY_RVP(reference_internal) Graph &add_loop(std::string name, size_t max_iterations, LoopCondition condition);

    /**
     * @brief Add a one-shot stage with lambda-captured body.
     *
     * @param[in] name Stage name.
     * @param[in] body_fn Lambda capturing the stage's operations.
     *
     * @code
     * pipeline.add_stage("setup", [&]() {
     *     cg::einsum("ij <- ik ; kj", &C, A, B);
     *     cg::scale(2.0, &C);
     * });
     * @endcode
     */
    template <typename BodyFn>
    void add_stage(std::string const &name, BodyFn &&body_fn) {
        auto              &stage = add_stage(name);
        CaptureGuard const g(stage);
        // Two body forms are accepted: zero-arg (the original) or
        // ``(Graph &)`` so callers can invoke ``g.declare_*_tensor(...)``
        // for stage-scoped intermediates. Selected at compile time so
        // the wrong form yields a normal overload-resolution error.
        if constexpr (std::is_invocable_v<BodyFn &, Graph &>)
            body_fn(stage);
        else
            body_fn();
    }

    /**
     * @brief Add a loop stage with lambda-captured body.
     *
     * @param[in] name Stage name.
     * @param[in] max_iterations Maximum iterations (safety limit).
     * @param[in] condition After each iteration: true → continue, false → stop.
     * @param[in] body_fn Lambda capturing the loop body operations.
     *
     * @code
     * pipeline.add_loop("scf_iter", 200,
     *     [&](size_t iter) { return delta > 1e-8; },
     *     [&]() { cg::einsum(...); cg::scale(...); }
     * );
     * @endcode
     */
    template <typename BodyFn>
    void add_loop(std::string name, size_t max_iterations, LoopCondition condition, BodyFn &&body_fn) {
        auto              &body = add_loop(std::move(name), max_iterations, std::move(condition));
        CaptureGuard const g(body);
        body_fn();
    }

    /**
     * @brief Apply a PassManager to all stages.
     *
     * Runs all passes on each one-shot stage's Graph and each loop stage's
     * body Graph. Returns true if any stage was modified.
     *
     * @code
     * auto pm = cg::PassManager::create_default();
     * pipeline.apply(pm);
     * @endcode
     */
    APIARY_EXPOSE bool apply(PassManager &pm);

    /**
     * @brief Execute all stages in order.
     *
     * One-shot stages execute once. Loop stages repeat until their condition
     * returns false or max_iterations is reached.
     */
    APIARY_EXPOSE APIARY_RELEASE_GIL void execute();

    /**
     * @brief Execute all stages using a custom executor.
     * @param[in] executor The execution backend for each stage's graph.
     */
    void execute(Executor &executor);

    /**
     * @brief Build + run: apply default passes, materialize the
     *        associated workspace, and execute in one call.
     *
     * Convenience wrapper over
     * ``apply(PassManager::create_default()) + workspace()->materialize_all() + execute()``.
     * Throws if ``workspace()`` is null, pipelines must own their
     * storage scope explicitly.
     *
     * @code
     * cg::Pipeline p("compute");
     * p.set_workspace(ws);
     * p.add_stage("multiply", [&] { cg::einsum(...); });
     * p.run();  // defaults-apply + materialize + execute
     * @endcode
     */
    void run();

    /**
     * @brief Same as ``run()`` but with a caller-supplied pass manager.
     */
    void run(PassManager &pm);

    // Note: execute() always instruments with the profiler (no separate execute_profiled variant).

    [[nodiscard]] std::string const &name() const { return _name; }                ///< Pipeline name.
    [[nodiscard]] size_t             num_stages() const { return _stages.size(); } ///< Number of stages.

    /// Access the graph for a stage by index. Returns nullptr if out of range.
    [[nodiscard]] Graph *stage_graph(size_t index) {
        if (index >= _stages.size())
            return nullptr;
        if (auto *g = std::get_if<Graph>(&_stages[index].content))
            return g;
        if (auto *l = std::get_if<LoopNode>(&_stages[index].content))
            return &l->body;
        return nullptr;
    }

    /// Access stage name by index.
    [[nodiscard]] std::string const &stage_name(size_t index) const { return _stages[index].name; }

    // ── Workspace association ───────────────────────────────────────────────

    /**
     * @brief Associate a Workspace with this pipeline.
     *
     * Workspace tensors can be used across multiple pipelines. The workspace
     * must outlive the pipeline.
     */
    void set_workspace(Workspace &ws) { _workspace = &ws; }

    /// Get the associated workspace (may be nullptr).
    [[nodiscard]] Workspace *workspace() const { return _workspace; }

    // ── Deferred tensor declaration ─────────────────────────────────────────

    /**
     * @brief Declare a pipeline-scoped tensor with deferred allocation.
     *
     * The tensor has valid dimensions but no data until MaterializationPass runs.
     * It persists across all stages of this pipeline.
     *
     * @tparam T     Element type.
     * @tparam Rank  Number of dimensions.
     * @param  name  Human-readable tensor name.
     * @param  dims  Dimensions of each rank.
     * @return Reference to the shell tensor.
     */
    template <typename T, size_t Rank, std::integral... Dims>
        requires(sizeof...(Dims) == Rank)
    Tensor<T, Rank> &declare_tensor(std::string tensor_name, Dims... dims) {
        using TensorType = Tensor<T, Rank>;
        auto *ptr        = new TensorType(typename TensorType::DeferredAlloc{}, std::move(tensor_name), dims...);
        _owned_tensors.emplace_back(ptr, [](void *p) { delete static_cast<TensorType *>(p); });

        auto handle            = make_handle(*ptr, 0);
        handle.alloc_state     = AllocState::Deferred;
        handle.is_intermediate = false;
        handle.materialize_fn  = [ptr]() { ptr->materialize(); };
        handle.zero_fn         = [ptr]() {
            ptr->materialize();
            ptr->zero();
        };
        handle.random_fn = [ptr]() {
            ptr->materialize();
            auto *data = ptr->data();
            for (size_t idx = 0; idx < ptr->size(); idx++) {
                // NOLINTNEXTLINE(misc-predictable-rand)
                data[idx] = static_cast<T>(static_cast<double>(std::rand()) / RAND_MAX * 2.0 - 1.0);
            }
        };
        _handles.push_back(std::move(handle));

        return *ptr;
    }

    /// Declare a pipeline-scoped tensor initialized to zero after materialization.
    template <typename T, size_t Rank, std::integral... Dims>
        requires(sizeof...(Dims) == Rank)
    Tensor<T, Rank> &declare_zero_tensor(std::string tensor_name, Dims... dims) {
        auto &t                   = declare_tensor<T, Rank>(tensor_name, dims...);
        _handles.back().init_kind = InitKind::Zero;
        return t;
    }

    /// Declare a pipeline-scoped tensor initialized with random values after materialization.
    template <typename T, size_t Rank, std::integral... Dims>
        requires(sizeof...(Dims) == Rank)
    Tensor<T, Rank> &declare_random_tensor(std::string tensor_name, Dims... dims) {
        auto &t                   = declare_tensor<T, Rank>(tensor_name, dims...);
        _handles.back().init_kind = InitKind::Random;
        return t;
    }

    /// Access all declared tensor handles (for passes to inspect).
    [[nodiscard]] std::vector<TensorHandle> const &declared_handles() const { return _handles; }
    [[nodiscard]] std::vector<TensorHandle>       &declared_handles() { return _handles; }

    // ── Runtime parameters ──────────────────────────────────────────────────
    //
    // Integer-valued, name-keyed scalars resolved at execute time. Used as
    // dynamic slice bounds for ``View`` ops, dynamic loop limits, etc.
    // Parameters are read every time a node executes, so updates between
    // iterations (from a ``LoopCondition``, an external callback, or a
    // ``WriteParam`` node) take effect on the next read.
    //
    // Optimization passes treat parameter values as opaque: they may inspect
    // ``BoundExpr::is_param()`` structurally but must not branch on the
    // runtime value.

    /// Set or overwrite an integer parameter.
    void set_param(std::string name, std::int64_t value) { _params->set(std::move(name), value); }

    /// Read an integer parameter. Throws if @p name is not set.
    [[nodiscard]] std::int64_t get_param(std::string const &name) const { return _params->get(name); }

    /// Read a parameter with a default.
    [[nodiscard]] std::int64_t get_param_or(std::string const &name, std::int64_t fallback) const {
        return _params->get_or(name, fallback);
    }

    /// Direct access to the parameter table, used by capture-time helpers
    /// to bake the table into executor lambdas (capture by shared_ptr).
    [[nodiscard]] std::shared_ptr<ParamTable>       &params_ptr() { return _params; }
    [[nodiscard]] std::shared_ptr<ParamTable> const &params_ptr() const { return _params; }

  private:
    /// Internal stage representation.
    struct Stage {
        std::string                   name;    ///< Stage name for profiling
        std::variant<Graph, LoopNode> content; ///< Either a one-shot Graph or a LoopNode
    };

    std::string        _name;
    std::vector<Stage> _stages;
    Workspace         *_workspace{nullptr};

    /// Pipeline-scoped tensors with deferred allocation.
    std::vector<std::unique_ptr<void, void (*)(void *)>> _owned_tensors;
    std::vector<TensorHandle>                            _handles;

    /// Mutable runtime-parameter table. Held by shared_ptr so executor
    /// lambdas (and the optional external mutator paths in LoopCondition
    /// or WriteParam nodes) can see the same instance the user mutates
    /// via set_param().
    std::shared_ptr<ParamTable> _params{std::make_shared<ParamTable>()};
};

// ═══════════════════════════════════════════════════════════════════════════════
// Lambda façade
// ═══════════════════════════════════════════════════════════════════════════════
//
// These free functions wrap the construct-set_workspace-add_stage-CaptureGuard-
// apply-materialize-execute ritual so a caller can spell the whole pipeline in
// one expression without touching every chore by hand. Workspace is always
// required: there is no default singleton; pipelines must own their storage
// scope explicitly.

/**
 * @brief Construct a Pipeline, run ``build(pipeline)`` to register its
 *        stages, and return it un-executed.
 *
 * @tparam   BuildFn   Invocable taking ``Pipeline &``.
 * @param    name      Pipeline name.
 * @param    workspace Workspace that backs pipeline-declared + workspace
 *                     tensors. Must outlive the returned pipeline.
 * @param    build     Callback that adds stages / declares tensors. Receives
 *                     the freshly-constructed Pipeline.
 * @return   The built pipeline, ready to ``apply`` + ``execute`` or to be
 *           inspected / queried further.
 *
 * @code
 * auto p = cg::make_pipeline("compute", ws, [&](cg::Pipeline &p) {
 *     auto &C = p.declare_zero_tensor<double, 2>("C", 8, 6);
 *     p.add_stage("multiply", [&] {
 *         cg::einsum("ij <- ik ; kj", &C, A, B);
 *     });
 * });
 * p.run();
 * @endcode
 */
template <typename BuildFn>
[[nodiscard]] Pipeline make_pipeline(std::string name, Workspace &workspace, BuildFn &&build) {
    Pipeline p(std::move(name));
    p.set_workspace(workspace);
    std::forward<BuildFn>(build)(p);
    return p;
}

/**
 * @brief One-shot: ``make_pipeline`` + ``Pipeline::run()``.
 *
 * Equivalent to building a pipeline via @ref make_pipeline then immediately
 * calling ``run()`` on it. The pipeline is destructed when the call returns,
 * so any result tensors the caller wants to read must be declared on the
 * workspace (which outlives the call).
 *
 * @code
 * cg::Workspace ws("example");
 * auto &A = ws.declare_random_tensor<double, 2>("A", 8, 5);
 * auto &B = ws.declare_random_tensor<double, 2>("B", 5, 6);
 * auto &C = ws.declare_zero_tensor<double, 2>("C", 8, 6);
 *
 * cg::run("compute", ws, [&](cg::Pipeline &p) {
 *     p.add_stage("multiply", [&] {
 *         cg::einsum("ij <- ik ; kj", &C, A, B);
 *     });
 * });
 * // C is now populated.
 * @endcode
 */
template <typename BuildFn>
void run(std::string name, Workspace &workspace, BuildFn &&build) {
    auto p = make_pipeline(std::move(name), workspace, std::forward<BuildFn>(build));
    p.run();
}

} // namespace einsums::compute_graph
