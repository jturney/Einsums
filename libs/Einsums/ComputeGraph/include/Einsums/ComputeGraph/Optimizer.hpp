//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/ComputeGraphTypes/Ids.hpp>
#include <Einsums/Python/Annotations.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace einsums::compute_graph {

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
     * @brief Reads this pass intentionally redirected to a tensor's INITIAL
     *        contents by compensating the reader, exempted from the
     *        program-order validator.
     *
     * The validator in ``PassManager::run`` throws when a read that observed an
     * in-graph writer flips to observing the tensor's initial contents - the
     * writer-removed-under-a-reader bug class (bug-1012). A pass that removes a
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

    int _verbosity{0};
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
 */
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
     * Ordering rationale: graph-transforming cleanups first (fold/absorb/
     * fuse/eliminate), then fusion (GEMMBatching), then scheduling
     * (Reorder/IOPrefetch), then deferred materialization, then backend
     * placement (GPU, distributed), then memory management, then read-only
     * analyses. The GPU and distributed blocks are compile-time-gated by
     * backend availability.
     *
     *  1. ConstantFolding: evaluate constant-input nodes at compile time
     *  2. ScaleAbsorption: drop a Scale(α) made dead by the next op overwriting it
     *  3. PermuteFusion: absorb leading permutes into the GEMM trans flags
     *  4. CSE: common subexpression elimination
     *  5. DeadNodeElimination: drop nodes whose outputs are unused
     *  6. ElementWiseFusion: merge adjacent element-wise ops
     *  7. LoopInvariantHoisting: move invariant ops out of Loop bodies
     *  8. GEMMBatching: collapse compatible GEMMs into one BatchedGemm
     *  9. Reorder: memory-aware topological sort
     * 10. IOPrefetch: overlap DiskRead with compute
     * 11. DistributionPlanning: classify indices for distributed dispatch
     * 12. Materialization: resize deferred tensors to local partitions
     * 13. SymmetryPropagation: infer symmetry on graph intermediates and
     *                                 push to backing tensors for rank-2 BLAS dispatch
     *
     * GPU block (when a GPU backend or mock is available):
     * 14. GPUPlacement: cost-model based node-to-GPU assignment
     * 15. TransferInsertion: insert HostToDevice / DeviceToHost nodes
     * 16. TransferElimination: drop redundant transfers
     * 17. GPUDiagnostics: log placement decisions
     * 18. StreamAssignment: assign CUDA/HIP streams for overlap
     *
     * Distributed block (when MPI or its mock is available):
     * 19. InputSlicing: create per-rank views of distributed inputs
     * 20. SUMMAExpansion: expand einsums to SUMMA loops on square grids
     * 21. CommunicationInsertion: insert allreduces for replicated outputs
     * 22. CommunicationElimination: drop redundant communications
     * 23. CommunicationScheduling: split allreduce into async iallreduce + wait
     *
     * Tail (always registered):
     * 24. FreeInsertion: free intermediates after last consumer
     * 25. MemoryPlanning: analysis: tensor liveness + peak memory
     * 26. ContractionPlanning: analysis: per-einsum dispatch choice
     * 27. InplaceOptimization: analysis: detect in-place candidates
     *
     * Note: DistributiveFactoring is deliberately
     * not registered by default, see their own docstrings for when to opt in.
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

} // namespace einsums::compute_graph
