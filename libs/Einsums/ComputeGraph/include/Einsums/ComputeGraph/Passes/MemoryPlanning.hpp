//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/ComputeGraph/Optimizer.hpp>

#include <cstddef>
#include <iosfwd>

namespace einsums::compute_graph::passes {

/**
 * @brief MemoryPlanning pass: liveness analysis, memory statistics, and the planned host arena.
 *
 * Analyzes the graph to determine tensor liveness intervals (from first use to last use) and computes
 * memory statistics for both host and device:
 * - **total_memory()**: Sum of all tensor sizes (upper bound if all were live simultaneously)
 * - **peak_memory()**: Maximum sum of simultaneously live tensors at any execution point
 * - **device_total_memory()**: Sum of all device-resident tensor sizes
 * - **device_peak_memory()**: Peak simultaneously live device tensors
 * - **device_reuse_savings()**: Bytes saveable by reusing non-overlapping device allocations
 *
 * With apply_arena (the default, and the default-pipeline configuration), the liveness intervals are
 * also ACTED on: graph-owned host intermediates whose lifetimes are bracketed by Materialize/Free
 * nodes get first-fit-decreasing offsets in one shared 64-byte-aligned arena per graph, sized to the
 * planned peak. Their Materialize nodes re-attach the tensor at its planned offset
 * (Tensor::materialize_into - no allocation) and their Free nodes detach without freeing, so
 * non-overlapping intermediates share storage and replays touch the same hot pages with zero allocator
 * traffic. Tensors with views, aliases, GPU placement, or unbounded lifetimes are left on their own
 * allocations, as is any graph with control flow at its level (bodies are planned at their own
 * recursion level).
 *
 * This pass is in the default pipeline as the LAST pass (with apply_arena defaulting to true, so it
 * both reports statistics and applies the arena). It runs after Materialization / FreeInsertion so the
 * Materialize/Free brackets it plans over already exist.
 *
 * @par Example (C++)
 * @code
 * cg::Graph graph("memory_planning");
 * auto     &t1 = graph.scratch<double, 2>("t1", n, n);      // graph-owned intermediates
 * auto     &t2 = graph.scratch<double, 2>("t2", n, n);
 * {
 *     cg::CaptureGuard const capture(graph);
 *     cg::einsum("ij <- ik ; kj", 0.0, &t1, 1.0, A, B);     // t1 lives, then dies
 *     cg::einsum("ij <- ik ; kj", 0.0, &t2, 1.0, t1, D);    // t2 can reuse t1's arena slot
 * }
 * graph.apply(cg::PassManager::create_default());           // MemoryPlanning (arena) runs last
 * // Non-overlapping intermediates now share one arena block; peak_memory() < total_memory().
 * @endcode
 *
 * @par Example (Python)
 * Constructible standalone for analysis (getters are properties). Applied on its own it only reports,
 * because the Materialize/Free brackets it packs are placed by earlier pipeline passes.
 * @code{.py}
 * import einsums, einsums.graph as cg
 * g = cg.Graph("memory_planning")
 * with cg.capture(g):
 *     einsums.einsum("ij <- ik ; kj", C, A, B)
 * p  = cg.MemoryPlanning()
 * pm = cg.PassManager(); pm.add(p); pm.run(g)               # or g.apply(cg.default_pass_manager())
 * # p.total_memory, p.peak_memory, p.num_planned, p.planned_arena_bytes  (properties, not methods)
 * @endcode
 *
 * @par Limitations
 * - Arena planning is HOST-only and skips any graph that has control flow at its level (Loop /
 *   Conditional) or any GPU / host<->device transfer node; those graphs get statistics only.
 * - Only graph-owned is_intermediate tensors bracketed by exactly one Materialize and one Free node,
 *   not viewed or aliased, with materialize_into_fn + release_fn and nonzero bytes, are arena-placed.
 * - The interval test uses body-LOCAL node positions and is blind to cross-iteration (loop-carried)
 *   liveness; today no in-body Materialize/Free bracket is ever reachable and a wraparound canary
 *   fires if FreeInsertion ever places one.
 * - Reported cross-loop peak is a max over graphs (a defensible lower bound), not a precise
 *   loop-carried peak; device statistics are reporting-only (no device arena is applied, and
 *   device_peak_memory / device_reuse_savings are not exposed to Python).
 *
 * @par Future improvements
 * - Precise loop-carried liveness for an exact cross-loop peak (and to unlock arena planning inside
 *   loop bodies by treating iteration-crossing tensors as always-live).
 * - A device-side arena that acts on device_reuse_savings the way the host arena acts on peak_memory.
 */
class APIARY_EXPOSE APIARY_MODULE("graph") APIARY_HOLDER(std::shared_ptr) EINSUMS_EXPORT MemoryPlanning : public OptimizerPass {
  public:
    APIARY_EXPOSE MemoryPlanning() = default;

    /// @param apply_arena When false, the pass is analysis-only (statistics
    ///        and the plan are still computed and inspectable).
    APIARY_EXPOSE explicit MemoryPlanning(bool apply_arena) : _apply_arena(apply_arena) {}

    [[nodiscard]] std::string name() const override { return "MemoryPlanning"; }
    bool                      run(Graph &graph) override;
    void                      print_report(std::ostream &os) const;

    /// Total bytes of all per-graph arenas planned in the last run.
    APIARY_EXPOSE APIARY_GETTER("planned_arena_bytes") [[nodiscard]] size_t planned_arena_bytes() const { return _planned_arena_bytes; }

    /// Number of intermediates placed at planned arena offsets.
    APIARY_EXPOSE APIARY_GETTER("num_planned") [[nodiscard]] size_t num_planned() const { return _num_planned; }

    /// Sum of the planned tensors' own buffer sizes (>= planned_arena_bytes;
    /// the difference is the storage sharing the plan achieves).
    APIARY_EXPOSE APIARY_GETTER("planned_tensor_bytes") [[nodiscard]] size_t planned_tensor_bytes() const { return _planned_tensor_bytes; }

    APIARY_EXPOSE APIARY_GETTER("total_memory") [[nodiscard]] size_t total_memory() const { return _total_memory; }
    APIARY_EXPOSE APIARY_GETTER("peak_memory") [[nodiscard]] size_t peak_memory() const { return _peak_memory; }

    /// Device memory: total across all device-resident tensors.
    APIARY_EXPOSE APIARY_GETTER("device_total_memory") [[nodiscard]] size_t device_total_memory() const { return _device_total_memory; }

    /// Device memory: peak simultaneously live.
    [[nodiscard]] size_t device_peak_memory() const { return _device_peak_memory; }

    /// Bytes saveable by reusing non-overlapping device allocations.
    [[nodiscard]] size_t device_reuse_savings() const {
        return _device_total_memory > _device_peak_memory ? _device_total_memory - _device_peak_memory : 0;
    }

  private:
    bool   _apply_arena{true};
    size_t _total_memory{0};
    size_t _peak_memory{0};
    size_t _device_total_memory{0};
    size_t _device_peak_memory{0};
    size_t _planned_arena_bytes{0};
    size_t _num_planned{0};
    size_t _planned_tensor_bytes{0};
};

} // namespace einsums::compute_graph::passes
