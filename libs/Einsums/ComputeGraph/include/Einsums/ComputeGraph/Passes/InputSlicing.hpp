//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/ComputeGraph/Optimizer.hpp>

#include <cstddef>

namespace einsums::compute_graph::passes {

/**
 * @brief Distributed-execution pass: slices pre-allocated inputs to match a distributed output.
 *
 * @rst
 * .. note::
 *
 *    Distributed (MPI) execution is a work in progress. This pass runs only on
 *    MPI-enabled (or mock) builds, and the distributed backend is not yet complete.
 * @endrst
 *
 * When an einsum (or permute) has a distributed output — say C distributed along dim 0 — any
 * pre-allocated, non-distributed input that shares that index must be presented to each rank as only
 * its local partition. This pass brackets such a node with a `begin_slice` `Custom` node before and an
 * `end_slice` `Custom` node after: `begin_slice` calls the tensor's `begin_local_view_fn` to retarget
 * the input's data pointer and dims to the local range for this rank, and `end_slice` restores the
 * original view via `end_local_view_fn`. The distributed output's `DistributionDescriptor` supplies the
 * per-rank `local_range`, and the operand index lists identify which input dims carry the distributed
 * index.
 *
 * In `create_default()` this runs inside the distributed block
 * (`if constexpr (comm::has_mpi || comm::is_mock)`), as the first distributed pass, after
 * DistributionPlanning and Materialization have assigned layouts and allocated storage. It self-guards
 * with `world_size() <= 1` and is a no-op on a single rank.
 *
 * @par Example (C++)
 * @code
 * // Run under MPI, e.g. `mpirun -np 4 ./my_app`.
 * cg::Graph graph("input_slicing");
 * {
 *     cg::CaptureGuard const capture(graph);
 *     cg::declare_tensor<double>("C", {N, N});             // deferred, distributed along i
 *     cg::einsum("ij <- ik ; kj", 0.0, &C, 1.0, A, B);     // A pre-allocated, shares index i
 * }
 * graph.apply(cg::PassManager::create_default());          // begin_slice/end_slice wrap the einsum
 * // Each rank's einsum now sees only its local i-range of A.
 * @endcode
 *
 * @par Example (Python)
 * No standalone Python entry point: not individually exposed; runs inside the distributed pipeline via
 * `g.apply(cg.default_pass_manager())` and only fires on more than one MPI rank.
 *
 * @par Limitations
 * - A no-op on a single rank; exercised only under real MPI (or the mock backend with `world_size() > 1`).
 * - Slices only pre-allocated, non-distributed (or replicated) inputs — a deferred or already
 *   block-distributed input is left alone.
 * - Handles `Einsum` and `Permute`/`Transpose` only; other kinds (including `BatchedGemm`, whose
 *   distributed form is unsupported) are skipped because they expose no index list to reason about.
 * - Depends on the tensor providing `begin_local_view_fn` / `end_local_view_fn`; without those hooks the
 *   slice nodes execute as no-ops.
 * - Matching is positional (`inp_indices[d] == dist_index`), assuming operand dims line up with the
 *   descriptor's dims.
 *
 * @par Future improvements
 * - Cache/hoist the slice view across a loop so an invariant input is not re-sliced every iteration.
 */
class EINSUMS_EXPORT InputSlicing : public OptimizerPass {
  public:
    [[nodiscard]] std::string name() const override { return "InputSlicing"; }
    bool                      run(Graph &graph) override;

    [[nodiscard]] size_t num_sliced() const { return _num_sliced; }

  private:
    size_t _num_sliced{0};
};

} // namespace einsums::compute_graph::passes
