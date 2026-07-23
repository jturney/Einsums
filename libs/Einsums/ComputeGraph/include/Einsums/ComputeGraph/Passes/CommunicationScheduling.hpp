//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/ComputeGraph/Optimizer.hpp>

namespace einsums::compute_graph::passes {

/**
 * @brief Distributed-execution pass: overlaps collective communication with computation.
 *
 * @rst
 * .. note::
 *
 *    Distributed (MPI) execution is a work in progress. This pass runs only on
 *    MPI-enabled (or mock) builds, and the distributed backend is not yet complete.
 * @endrst
 *
 * The distributed analogue of `IOPrefetch` for disk I/O. It converts each synchronous `Allreduce` into
 * a two-phase asynchronous node using the tensor handle's `iallreduce_sum_fn`: `async_start` fires a
 * non-blocking `iallreduce` (storing the `comm::Request`), and `async_finish` waits on it. The
 * synchronous `execute` is cleared, and the `DataflowExecutor` — which already supports
 * `Node::async_start` / `async_finish` — is then free to schedule independent work between the start and
 * the wait, hiding the collective's latency.
 *
 * In `create_default()` this runs inside the distributed block
 * (`if constexpr (comm::has_mpi || comm::is_mock)`), as the last distributed pass, after the Allreduces
 * have been inserted and de-duplicated. It self-guards with `world_size() <= 1` and is a no-op on a
 * single rank.
 *
 * @par Example (C++)
 * @code
 * // Run under MPI, e.g. `mpirun -np 4 ./my_app`.
 * cg::Graph graph("comm_sched");
 * {
 *     cg::CaptureGuard const capture(graph);
 *     cg::einsum("ij <- ik ; kj", 0.0, &C, 1.0, A, B);     // -> allreduce(C)
 *     cg::einsum("pq <- pr ; rq", 0.0, &D, 1.0, E, F);     // independent work to overlap
 * }
 * graph.apply(cg::PassManager::create_default());          // allreduce(C) becomes iallreduce + wait
 * @endcode
 *
 * @par Example (Python)
 * No standalone Python entry point: not individually exposed; runs inside the distributed pipeline via
 * `g.apply(cg.default_pass_manager())` and only fires on more than one MPI rank.
 *
 * @par Limitations
 * - A no-op on a single rank; exercised only under real MPI (or the mock backend with `world_size() > 1`).
 * - Splits **Allreduce** only; despite the family description, Allgather (and other collectives) are not
 *   yet converted to async form.
 * - Requires the tensor handle to provide `iallreduce_sum_fn`; nodes without it are left synchronous.
 * - Only marks the node as two-phase — actual overlap depends on the DataflowExecutor finding
 *   independent work between start and finish; it does not itself reorder nodes to create that slack.
 *
 * @par Future improvements
 * - Async-split Allgather and other collectives, not just Allreduce.
 * - Actively hoist independent computation between `async_start` and `async_finish` to widen the overlap
 *   window rather than relying on the existing schedule.
 */
class EINSUMS_EXPORT CommunicationScheduling : public OptimizerPass {
  public:
    [[nodiscard]] std::string name() const override { return "CommunicationScheduling"; }
    bool                      run(Graph &graph) override;

    [[nodiscard]] size_t num_scheduled() const { return _num_scheduled; }

  private:
    size_t _num_scheduled{0};
};

} // namespace einsums::compute_graph::passes
