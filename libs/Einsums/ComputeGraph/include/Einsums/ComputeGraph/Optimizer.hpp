//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Python/Annotations.hpp>

#include <memory>
#include <string>
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
class OptimizerPass {
  public:
    virtual ~OptimizerPass() = default;

    [[nodiscard]] virtual std::string name() const = 0;

    /**
     * @brief Run the optimization pass on the given graph.
     * @param[in,out] graph The graph to optimize.
     * @return True if the graph was modified, false if no changes were made.
     */
    virtual bool run(Graph &graph) = 0;
};

/**
 * @brief Manages an ordered sequence of optimization passes.
 *
 * The PassManager collects passes and runs them in order on a graph.
 * Use add() to append individual passes, or use create_default() to
 * get a PassManager pre-loaded with all built-in passes in the
 * recommended order.
 *
 * @par Example — custom pass pipeline
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
 * @par Example — default pass pipeline
 * @code
 * auto pm = cg::PassManager::create_default();
 * graph.apply(pm);
 * @endcode
 *
 * @see Graph::apply(PassManager&)
 * @see Pipeline::apply(PassManager&)
 */
class EINSUMS_PYBIND_EXPOSE EINSUMS_PYBIND_MODULE("graph") EINSUMS_PYBIND_NOCOPY EINSUMS_PYBIND_NOMOVE EINSUMS_EXPORT PassManager {
  public:
    /// Default-construct an empty PassManager. Explicit (rather than
    /// implicit) so the binding codegen has a constructor declaration to
    /// annotate with EINSUMS_PYBIND_EXPOSE.
    EINSUMS_PYBIND_EXPOSE PassManager() = default;

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
        _passes.push_back(std::make_unique<PassType>(std::forward<Args>(args)...));
        return *this;
    }

    /**
     * @brief Run all passes in order on the given graph.
     *
     * @param[in,out] graph The graph to optimize.
     * @return True if any pass modified the graph.
     */
    EINSUMS_PYBIND_EXPOSE bool run(Graph &graph);

    /**
     * @brief Get the list of passes for inspection.
     * @return Const reference to the pass list.
     */
    [[nodiscard]] std::vector<std::unique_ptr<OptimizerPass>> const &passes() const { return _passes; }

    /**
     * @brief Number of passes in the pipeline.
     */
    EINSUMS_PYBIND_EXPOSE [[nodiscard]] size_t size() const { return _passes.size(); }

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
     *  1. ConstantFolding           — evaluate constant-input nodes at compile time
     *  2. ScaleAbsorption           — fold Scale(α) into the next op's beta
     *  3. PermuteFusion             — absorb leading permutes into the GEMM trans flags
     *  4. CSE                       — common subexpression elimination
     *  5. DeadNodeElimination       — drop nodes whose outputs are unused
     *  6. ElementWiseFusion         — merge adjacent element-wise ops
     *  7. LoopInvariantHoisting     — move invariant ops out of Loop bodies
     *  8. GEMMBatching              — collapse compatible GEMMs into one BatchedGemm
     *  9. Reorder                   — memory-aware topological sort
     * 10. IOPrefetch                — overlap DiskRead with compute
     * 11. DistributionPlanning      — classify indices for distributed dispatch
     * 12. Materialization           — resize deferred tensors to local partitions
     * 13. SymmetryPropagation       — infer symmetry on graph intermediates and
     *                                 push to backing tensors for rank-2 BLAS dispatch
     *
     * GPU block (when a GPU backend or mock is available):
     * 14. GPUPlacement              — cost-model based node-to-GPU assignment
     * 15. TransferInsertion         — insert HostToDevice / DeviceToHost nodes
     * 16. TransferElimination       — drop redundant transfers
     * 17. GPUDiagnostics            — log placement decisions
     * 18. StreamAssignment          — assign CUDA/HIP streams for overlap
     *
     * Distributed block (when MPI or its mock is available):
     * 19. InputSlicing              — create per-rank views of distributed inputs
     * 20. SUMMAExpansion            — expand einsums to SUMMA loops on square grids
     * 21. CommunicationInsertion    — insert allreduces for replicated outputs
     * 22. CommunicationElimination  — drop redundant communications
     * 23. CommunicationScheduling   — split allreduce into async iallreduce + wait
     *
     * Tail (always registered):
     * 24. FreeInsertion             — free intermediates after last consumer
     * 25. MemoryPlanning            — analysis: tensor liveness + peak memory
     * 26. ContractionPlanning       — analysis: per-einsum dispatch choice
     * 27. InplaceOptimization       — analysis: detect in-place candidates
     *
     * Note: DistributiveFactoring and ChainParenthesization are deliberately
     * not registered by default — see their own docstrings for when to opt in.
     *
     * @return A fully-populated PassManager.
     */
    static PassManager create_default();

    /**
     * @brief Populate this PassManager with the default pass list (in place).
     *
     * Equivalent to ``*this = create_default()`` but doesn't require the
     * class to be move-assignable. Used by the Python binding, which
     * can't bind ``add<PassType>()`` (templated) and so needs an
     * instance method to put the canonical pass list together.
     */
    EINSUMS_PYBIND_EXPOSE void populate_default();

  private:
    std::vector<std::unique_ptr<OptimizerPass>> _passes;
};

} // namespace einsums::compute_graph
