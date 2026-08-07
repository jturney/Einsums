//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/ComputeGraphTypes/Ids.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Python/Annotations.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

class Graph;

/**
 * @brief Abstract base class for optimization passes over a computation graph.
 *
 * An optimization pass inspects a Graph's node list and may modify it to improve
 * performance. Passes can fuse operations, eliminate redundancies, reorder nodes
 * for better memory behavior, or analyze the graph and report recommendations.
 *
 * @par Implementing a custom pass
 * @code
 * class MyPass : public OptimizerPass {
 * public:
 *     std::string name() const override { return "MyPass"; }
 *     bool run(Graph &graph) override {
 *         // Inspect and modify graph.nodes()
 *         return false;
 *     }
 * };
 * @endcode
 *
 * @note Passes that modify the graph should call graph.mark_sorted() if they
 *       produce a valid topological ordering, to prevent unnecessary re-sorting.
 */
class APIARY_EXPOSE APIARY_MODULE("graph") APIARY_HOLDER(std::shared_ptr) OptimizerPass {
  public:
    virtual ~OptimizerPass() = default;

    /// @brief Human-readable pass name, exposed so Python tests can
    ///        assert which pass they're invoking.
    APIARY_EXPOSE APIARY_GETTER("name") [[nodiscard]] virtual std::string name() const = 0;

    /**
     * @brief Run the optimization pass on the given graph.
     * @param[in,out] graph The graph to optimize.
     * @return True if the graph was modified, false if no changes were made.
     */
    virtual bool run(Graph &graph) = 0;

    /**
     * @brief Should ``PassManager`` re-invoke this pass on every sub-graph?
     *
     * Controls whether the PassManager automatically descends into
     * loop bodies and conditional branches (via
     * ``Graph::for_each_subgraph``) and calls ``run()`` again at each
     * level. Default ``false``, preserves the historical flat-graph
     * behavior.
     *
     * Passes whose semantics are *correct on a flat sub-graph* (CSE,
     * ScaleAbsorption, PermuteFusion, …) should return ``true`` once
     * verified: that lets the same per-pass tests cover bodies.
     *
     * Passes whose effect must *cross the loop boundary* (Materialization
     * hoisting allocs to the parent, FreeInsertion placing frees after
     * the loop, TransferInsertion hoisting H2D for loop-invariant inputs)
     * should keep this ``false`` and instead walk children themselves
     * inside ``run()`` via ``Graph::for_each_subgraph``. Their output
     * lands in the *parent* graph, not the child.
     *
     * Passes that need parent context to decide correctness (DNE post-loop
     * liveness, ConstantFolding iteration-variance check, Reorder
     * boundary respect) must also keep this ``false`` until they grow the
     * required cross-graph reasoning.
     */
    [[nodiscard]] virtual bool recurse_into_subgraphs() const { return false; }

    /**
     * @brief Zero this pass's per-apply statistics.
     *
     * ``PassManager::run`` calls this ONCE per ``apply()``, before descending,
     * so the counter getters report totals over the whole subgraph tree rather
     * than whatever the last-visited subgraph happened to contribute.
     *
     * Any pass that maintains counters must override this and must NOT zero
     * them inside ``run()``: with ``recurse_into_subgraphs() == true`` the
     * driver calls ``run()`` once per subgraph, so a reset there discards every
     * earlier level. That is silent for a single loop (the body is visited
     * last, so its count survives) and wrong the moment a graph has a
     * conditional -- the empty else-branch resets the counters to zero -- or
     * more than one loop. ``graph.explain()`` reads these getters.
     */
    virtual void reset_stats() {}

    /**
     * @brief Zero everything the manager owns, then the pass's own counters.
     *
     * ``PassManager::run`` calls THIS, not @ref reset_stats, so the skip-reason
     * tally is cleared once per ``apply()`` without every pass having to
     * remember to chain to a base implementation.
     */
    void reset_all_stats() {
        _skips.clear();
        reset_stats();
    }

    /**
     * @brief Why this pass declined the candidates it looked at.
     *
     * One entry per distinct reason, with the number of candidates that hit it,
     * ordered most-frequent first. Populated by @ref note_skip during
     * ``run()``; empty for a pass that never declines anything or has not been
     * taught to explain itself yet.
     *
     * This is the negative half of @ref explain: `explain()` says what a pass
     * DID, `skip_reasons()` says what it declined and why. A pipeline that
     * looks inert usually has a full skip tally - that is the useful signal,
     * and reading it should not require rebuilding with a debugger attached.
     */
    [[nodiscard]] EINSUMS_EXPORT std::vector<std::pair<std::string, std::size_t>> skip_reasons() const;

    /**
     * @brief What this pass did on its last run, for @ref PassManager::explain.
     *
     * One entry per line of the report; empty means there is nothing worth
     * saying, which is the default and what a pass with no counters wants. A
     * pass that did nothing should return empty rather than a line saying so,
     * so a quiet report means a quiet pipeline.
     *
     * Lives here, next to the counters it reads, rather than in a type switch
     * inside the manager: a pass that grows a statistic updates one file, and a
     * pass defined outside this library gets summarized like any other.
     */
    [[nodiscard]] virtual std::vector<std::string> explain() const { return {}; }

    /**
     * @brief Reads this pass intentionally redirected to a tensor's INITIAL
     *        contents by compensating the reader, exempted from the
     *        program-order validator.
     *
     * The validator in ``PassManager::run`` throws when a read that observed an
     * in-graph writer flips to observing the tensor's initial contents - the
     * writer-removed-under-a-reader bug class. A pass that removes a
     * writer and instead COMPENSATES the reader (e.g. ScaleAbsorption deleting a
     * ``scale`` and folding its factor into a downstream einsum's
     * ``ab_prefactor``, so the read is exact despite losing the writer) declares
     * the affected ``(reader NodeId, TensorId)`` pairs here; the validator skips
     * exactly those. The pass owns the compensation's correctness — its own
     * numeric tests must cover it, since the structural guard is waived. Return
     * the pairs recorded during ``run()``; empty by default (no exemptions).
     */
    [[nodiscard]] virtual std::vector<std::pair<NodeId, TensorId>> compensated_reads() const { return {}; }

    /**
     * @brief Set the pass's introspection verbosity.
     *
     * Levels: 0 = silent (default), 1 = summary (aggregate effect),
     * 2 = detail (each modification applied), 3 = trace (each candidate
     * examined, including why it was rejected). Output goes to stderr,
     * prefixed with the pass name. Usually set in bulk via
     * ``PassManager::set_verbosity`` rather than per pass.
     *
     * From level 2 up, ``PassManager::explain()`` also grows a "not applied"
     * section built from @ref skip_reasons - the aggregated form of what
     * level 3 narrates per candidate. Reach for that first when a pipeline
     * looks inert: the tally names the gate, and is a returned string rather
     * than stderr noise you have to read past.
     */
    APIARY_EXPOSE void set_verbosity(int level) { _verbosity = level; }

    /// @brief Current verbosity level (see set_verbosity).
    APIARY_EXPOSE APIARY_GETTER("verbosity") [[nodiscard]] int verbosity() const { return _verbosity; }

  protected:
    /**
     * @brief Emit ``[PassName] message`` to stderr when ``_verbosity >= level``.
     *
     * Passes call this to narrate what they see and the rewrites they make:
     * @code
     * report(3, fmt::format("examining node {} ({})", i, node.label)); // trace
     * report(2, fmt::format("folded {} contractions into {}", n, name)); // detail
     * report(1, fmt::format("folded {} groups", _num_groups));          // summary
     * @endcode
     */
    EINSUMS_EXPORT void report(int level, std::string_view message) const;

    /**
     * @brief Record that a candidate was examined and declined, and why.
     *
     * ``reason`` must be a short, *shape-independent* phrase so repeated hits
     * aggregate into one counted line - "operands are not runtime tensors",
     * not "tensor 'Wmbej' (id=17) is not a runtime tensor". Put the specifics
     * in ``detail``, which is emitted to stderr at verbosity 3 but never
     * aggregated:
     *
     * @code
     * note_skip("operands are not runtime tensors",
     *           fmt::format("output '{}' has is_runtime=false", name));
     * @endcode
     *
     * Cheap enough to call on every rejected candidate: the aggregate is a
     * small map keyed by reason, and the stderr line is formatted only when
     * verbosity warrants it.
     */
    EINSUMS_EXPORT void note_skip(std::string_view reason, std::string_view detail = {}) const;

    int _verbosity{0};

  private:
    /// Reason -> number of candidates declined for it. Mutable so a pass can
    /// record from a const analysis helper; cleared per apply() by
    /// reset_all_stats().
    mutable std::vector<std::pair<std::string, std::size_t>> _skips;
};

/**
 * @brief Optimization level for Graph::optimize(), compiler-style.
 *
 * - O0: no passes; the graph runs exactly as captured.
 * - O1: node-count cleanup only (constant folding, dead-scale removal,
 *       permute fusion, CSE, dead-node elimination, elementwise fusion) -
 *       cheap, no restructuring, no memory planning.
 * - O2: the full default pipeline (cleanup + loop hoisting + cost-model
 *       chain restructuring + batching + reorder + distribution/GPU when
 *       available + in-place merging + free insertion + the memory arena).
 */
enum class APIARY_EXPOSE OptLevel : std::uint8_t {
    O0 = 0,
    O1 = 1,
    O2 = 2,
};

/**
 * @brief Manages an ordered sequence of optimization passes.
 *
 * The PassManager collects passes and runs them in order on a graph.
 * Use add() to append individual passes, or use create_default() to
 * get a PassManager pre-loaded with all built-in passes in the
 * recommended order.
 *
 * @par Example: custom pass pipeline
 * @code
 * cg::PassManager pm;
 * pm.add<cg::passes::ConstantFolding>();
 * pm.add<cg::passes::ScaleAbsorption>();
 * pm.add<cg::passes::CSE>();
 * pm.add<cg::passes::Reorder>();
 *
 * graph.apply(pm);
 * @endcode
 *
 * @par Example: default pass pipeline
 * @code
 * auto pm = cg::PassManager::create_default();
 * graph.apply(pm);
 * @endcode
 *
 * @see Graph::apply(PassManager&)
 * @see Pipeline::apply(PassManager&)
 * @see create_for(OptLevel)
 */
class APIARY_EXPOSE APIARY_MODULE("graph") APIARY_NOCOPY APIARY_NOMOVE EINSUMS_EXPORT PassManager {
  public:
    /// Default-construct an empty PassManager. Explicit (rather than
    /// implicit) so the binding codegen has a constructor declaration to
    /// annotate with APIARY_EXPOSE.
    APIARY_EXPOSE PassManager() = default;

    // The vector-of-shared_ptr storage makes PassManager copyable; we
    // don't promise that to C++ callers (NOCOPY/NOMOVE controls the
    // Python binding), but no explicit delete is needed for the binding
    // to compile.

    /**
     * @brief Add a pass to the end of the pipeline.
     *
     * The pass is constructed in-place and owned by the PassManager.
     *
     * @tparam PassType The pass class (must derive from OptimizerPass).
     * @tparam Args Constructor argument types.
     * @param[in] args Arguments forwarded to the pass constructor.
     * @return Reference to this PassManager (for chaining).
     *
     * @code
     * pm.add<cg::passes::CSE>()
     *   .add<cg::passes::Reorder>()
     *   .add<cg::passes::MemoryPlanning>();
     * @endcode
     */
    template <typename PassType, typename... Args>
    PassManager &add(Args &&...args) {
        auto pass = std::make_shared<PassType>(std::forward<Args>(args)...);
        if (_verbosity != 0) {
            pass->set_verbosity(_verbosity);
        }
        _passes.push_back(std::move(pass));
        return *this;
    }

    /**
     * @brief Non-templated overload taking shared ownership of a pass.
     *
     * Exists so the Python binding can write ``pm.add(cg.CSE())``: the
     * templated form can't be bound directly because pybind11 has no way
     * to deduce ``PassType`` from a Python call site. C++ callers should
     * prefer ``add<PassType>(...)`` since it constructs in place. Stored
     * as ``shared_ptr`` to match the pybind11 holder type, pybind can't
     * transfer ownership of a Python-held ``unique_ptr`` across the FFI
     * boundary without ``py::smart_holder``.
     */
    // The parameter name ``optimizer_pass`` rather than ``pass`` so the
    // pyi codegen (which copies parameter names verbatim) doesn't emit
    // a method signature whose argument name collides with Python's
    // ``pass`` keyword, pyright can't parse it.
    APIARY_EXPOSE APIARY_RVP(reference_internal) PassManager &add(std::shared_ptr<OptimizerPass> optimizer_pass) {
        if (_verbosity != 0) {
            optimizer_pass->set_verbosity(_verbosity);
        }
        _passes.push_back(std::move(optimizer_pass));
        return *this;
    }

    /**
     * @brief Set introspection verbosity for every pass in the pipeline.
     *
     * Propagates @p level to all currently-registered passes and to any added
     * afterward, and makes ``run()`` print a per-pass summary line
     * (``name: MODIFIED N -> M nodes (T ms)``) to stderr. See
     * ``OptimizerPass::set_verbosity`` for the level meanings.
     *
     * @code
     * pm = cg.default_pass_manager()
     * pm.set_verbosity(2)   # narrate each modification
     * g.apply(pm)
     * @endcode
     */
    APIARY_EXPOSE void set_verbosity(int level) {
        _verbosity = level;
        for (auto &pass : _passes) {
            pass->set_verbosity(level);
        }
    }

    /**
     * @brief Run all passes in order on the given graph.
     *
     * @param[in,out] graph The graph to optimize.
     * @return True if any pass modified the graph.
     */
    APIARY_EXPOSE bool run(Graph &graph);

    /**
     * @brief Get the list of passes for inspection.
     * @return Const reference to the pass list.
     */
    [[nodiscard]] std::vector<std::shared_ptr<OptimizerPass>> const &passes() const { return _passes; }

    /**
     * @brief Number of passes in the pipeline.
     */
    APIARY_EXPOSE APIARY_GETTER("size") [[nodiscard]] size_t size() const { return _passes.size(); }

    /**
     * @brief Create a PassManager with all built-in passes in recommended order.
     *
     * Ordering rationale: lowering first (TiledExpansion, so every pass below
     * sees dense nodes), then graph-transforming cleanups (fold/absorb/fuse/
     * eliminate), then planning and fusion, then scheduling (Reorder/
     * IOPrefetch), then deferred materialization, then backend placement (GPU,
     * distributed), then memory management. The GPU and distributed blocks are
     * compile-time-gated by backend availability; the numbering below assumes
     * both are present. populate_default() in Optimizer.cpp carries the
     * per-pass ordering rationale.
     *
     *  1. TiledExpansion: lower tiled ops into per-tile dense nodes
     *  2. ConstantFolding: evaluate constant-input nodes at compile time
     *  3. ScaleAbsorption: drop a Scale(α) made dead by the next op overwriting it
     *  4. PermuteFusion: absorb leading permutes into the GEMM trans flags
     *  5. CSE: common subexpression elimination
     *  6. DeadNodeElimination: drop nodes whose outputs are unused
     *  7. SymmetrizedAccumulation: fold the r2 += s*(tmp + P(tmp)) idiom
     *  8. ElementWiseFusion: merge adjacent element-wise ops
     *  9. LinearCombinationContractionFolding: fold transpose-paired contractions
     * 10. DistributiveFactoring: factor a shared operand out of sibling contractions
     * 11. LoopInvariantHoisting: move invariant ops out of Loop bodies
     * 12. ScratchPrivatization: clone reused scratch to break false WAR/WAW chains
     * 13. ContractionPlanning: cost-model chain reassociation
     * 14. GEMMBatching: collapse compatible GEMMs into one BatchedGemm
     * 15. Reorder: memory-aware topological sort
     * 16. IOPrefetch: overlap DiskRead with compute
     * 17. DistributionPlanning: classify indices for distributed dispatch
     * 18. Materialization: resize deferred tensors to local partitions
     * 19. SymmetryPropagation: infer symmetry on graph intermediates and
     *                                 push to backing tensors for rank-2 BLAS dispatch
     * 20. StreamContractionFusion: loop-fuse sibling contractions over one big tensor
     *
     * GPU block (when a GPU backend or mock is available):
     * 21. GPUPlacement: cost-model based node-to-GPU assignment
     * 22. TransferInsertion: insert HostToDevice / DeviceToHost nodes
     * 23. TransferElimination: drop redundant transfers
     * 24. GPUDiagnostics: log placement decisions
     * 25. StreamAssignment: assign CUDA/HIP streams for overlap
     *
     * Distributed block (when MPI or its mock is available):
     * 26. InputSlicing: create per-rank views of distributed inputs
     * 27. SUMMAExpansion: expand einsums to SUMMA loops on square grids
     * 28. CommunicationInsertion: insert allreduces for replicated outputs
     * 29. CommunicationElimination: drop redundant communications
     * 30. CommunicationScheduling: split allreduce into async iallreduce + wait
     *
     * Tail (always registered):
     * 31. InplaceOptimization: merge elementwise outputs into dying inputs
     * 32. FreeInsertion: free intermediates after last consumer
     * 33. MemoryPlanning: tensor liveness, peak memory, and arena planning
     *
     * @return A fully-populated PassManager.
     */
    static PassManager create_default();

    /// Factory for a given optimization level (create_default() == O2).
    static PassManager create_for(OptLevel level);

    /**
     * @brief One human-readable report of what the last run() did.
     *
     * Harvests the per-pass statistics (nodes eliminated, chains
     * restructured with estimated speedups, buffers merged in place, Frees
     * inserted, arena size vs the buffers it hosts, batches formed and
     * profitability-gate skips) into a few lines of text. Empty until run()
     * has been called. See Graph::explain() for the graph-level entry point.
     *
     * At verbosity >= 2 the report gains a "not applied" section listing, per
     * pass, how many candidates it declined and why (see
     * ``OptimizerPass::skip_reasons``). "(no optimizations applied)" on its own
     * cannot tell you whether the graph was already optimal or whether every
     * candidate hit one satisfiable gate - and those call for opposite
     * responses - so raise the verbosity before concluding a pipeline is inert.
     */
    APIARY_EXPOSE [[nodiscard]] std::string explain() const;

    /**
     * @brief Populate this PassManager with the default pass list (in place).
     *
     * Equivalent to ``*this = create_default()`` but doesn't require the
     * class to be move-assignable. Used by the Python binding, which
     * can't bind ``add<PassType>()`` (templated) and so needs an
     * instance method to put the canonical pass list together.
     */
    APIARY_EXPOSE void populate_default();

  private:
    std::vector<std::shared_ptr<OptimizerPass>> _passes;
    int                                         _verbosity{0};
};

EINSUMS_NAMESPACE_END(compute_graph)
