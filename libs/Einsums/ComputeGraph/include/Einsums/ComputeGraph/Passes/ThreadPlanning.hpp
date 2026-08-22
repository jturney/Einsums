//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/ComputeGraph/CostModel.hpp>
#include <Einsums/ComputeGraph/Optimizer.hpp>
#include <Einsums/Config/Namespace.hpp>

#include <cstddef>
#include <string>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph::passes)

/**
 * @brief ThreadPlanning pass: choose a thread width for every node.
 *
 * Writes @ref Node::thread_width and @ref Node::admission_priority. It changes
 * no structure at all: the node list, the tensors and the executors are exactly
 * what they were, so the graph computes the same thing at any width and the
 * plan is safe to run on an unoptimized graph.
 *
 * @par The heuristic
 * Critical-path-and-area moldable scheduling. Every node starts at width 1.
 * Each round measures the current critical path `CP` (the longest path through
 * the graph under `t_i(w_i)`) and the area bound `A = sum(w_i * t_i(w_i)) / P`,
 * whose maximum is the makespan any schedule of these widths can reach. It then
 * widens the ONE node, on the critical path, whose next rung buys the most time
 * per unit of area it consumes, and takes the step when it does not make the
 * estimate worse and the area stays inside the makespan the round started with.
 * A node whose next rung fails that test is frozen at its current width, so the
 * search terminates. The plan that is KEPT is the widths at the last strict
 * improvement, which is what lets the search cross a plateau - several fat
 * nodes side by side on the critical path buy nothing until the last of them is
 * widened - without leaving the graph standing on one.
 *
 * `t_i(w) = t_i(1) / s(family, bytes, w)`, with `s` from
 * @ref DeviceProfile::parallel_speedup - measured curves where the machine was
 * swept, an Amdahl fallback where it was not. Widths quantize to the rung set
 * (powers of two up to `P`, plus `P`), because every distinct width costs a hot
 * team resize and a calibration cell.
 *
 * @par Where `t_i(1)` comes from
 * The graph's own recorded timings when it has them - the median of the samples
 * the last replay wrote for that node, which is a measurement of this machine
 * running this kernel on this data and beats any model. Otherwise a
 * @ref CostModel estimate by node kind: contractions are priced as a GEMM of
 * their shape, permutes at the measured transpose bandwidth, everything else at
 * memory bandwidth over the bytes it touches. A node the model cannot price at
 * all is left at width 1 rather than guessed at.
 *
 * @par Policy vetoes
 * Independent of profitability, and applied after it:
 * - A node whose kernel cannot be told how many threads to use
 *   (@ref kernel_moldability, i.e. a BLAS route on Accelerate) is width 1: the
 *   width would be a lie the rest of the plan is built on.
 * - `GroupedDot` and `GroupedAxpby` are width 1 always. Their kernels ARE
 *   governable, but the grouped scalar ops are sequential by contract so that
 *   their accumulation order, and therefore their last bit, does not depend on
 *   the schedule.
 * - `Loop` and `Conditional` are width 1 and their bodies are planned
 *   recursively, each as though it owned the whole machine. The bodies share
 *   the one process-wide `WidthBudget` at run time, so admission composes the
 *   plans rather than the planner having to.
 * - A node whose `t_i(1)` is below @ref serial_floor_us stays at width 1
 *   forever. Forking a team costs tens of microseconds; below the floor the
 *   fork is most of the node.
 *
 * @par Kernel routes
 * A contraction has two kernels that agree to within rounding and disagree in
 * the last bit - one vendor GEMM, or the packed loops - and left to itself the
 * dispatch picks between them by asking whether the caller holds a node width.
 * That makes the last bit a function of the width, and this pass hands out
 * widths from wall-clock measurements, so two runs of one graph would not agree
 * digit for digit. So the route is settled here, before any width is: every
 * contraction the search may widen is pinned to the packed loops (@ref
 * packed_gemm::KernelRoute), which give the same bits at every width, and the
 * search then moves widths WITHIN pinned routes. A node pinned to the vendor
 * instead is frozen at width 1 in the same breath, because a threaded vendor
 * GEMM is not bit-stable across its own thread counts either.
 *
 * Nothing is pinned when the plan cannot widen anything - one thread to plan
 * for - since no width can move a route there, and a pin that changes no
 * decision would only change an answer.
 *
 * @par Not in the default pipeline
 * Widths are a tuning artifact of one machine and one process, in the sense of
 * the pass-phase rule: they are re-derived per process and never serialized.
 * Reach for @ref Graph::plan_threads, which runs this standalone and owns the
 * re-plan policy; adding it to a pipeline is for callers who want it at a
 * particular point in one.
 *
 * @par Example (C++)
 * @code
 * cg::Graph graph("cc_iteration");
 * // ... capture ...
 * graph.plan_threads();          // or: pm.add<cg::passes::ThreadPlanning>();
 * cg::DataflowExecutor df;
 * graph.execute(df);
 * @endcode
 */
class EINSUMS_EXPORT ThreadPlanning : public OptimizerPass {
  public:
    /// Plans for the machine the width budget is rationing, against the
    /// auto-detected cost model.
    ThreadPlanning();

    /// Plans for @p max_threads threads. Zero asks the width budget, as the
    /// default constructor does.
    explicit ThreadPlanning(unsigned max_threads);

    /// Plans for @p max_threads threads against an explicit cost model.
    ThreadPlanning(CostModel cost_model, unsigned max_threads);

    [[nodiscard]] std::string name() const override { return "ThreadPlanning"; }

    bool run(Graph &graph) override;
    void reset_stats() override;

    /// Recursion is done inside @ref run, not by the pass manager: a body is
    /// planned against the SAME `P` as its parent and its cost has to be
    /// reported back to the container node, neither of which the generic
    /// per-subgraph driver can do.
    [[nodiscard]] bool recurse_into_subgraphs() const override { return false; }

    [[nodiscard]] std::vector<std::string> explain() const override;

    /// Serial time below which a node is never widened, in microseconds.
    ///
    /// A fork-and-join costs 23-37 us on the machines this was measured on, so
    /// a node of about that size spends its whole width on the fork. 100 us is
    /// several forks of headroom, which is the right side to err on: the cost
    /// of leaving a 150 us node narrow is 150 us, and the cost of widening a
    /// 30 us one is a slower node plus a share of the machine nobody else can
    /// use.
    static constexpr double kSerialFloorUs = 100.0;

    /// @copydoc kSerialFloorUs
    [[nodiscard]] static constexpr double serial_floor_us() { return kSerialFloorUs; }

    /// Threads the last run planned for.
    [[nodiscard]] unsigned planned_threads() const { return _planned_threads; }

    /// Nodes given a width above 1, over the whole subgraph tree.
    [[nodiscard]] std::size_t num_widened() const { return _num_widened; }

    /// The widest width handed out.
    [[nodiscard]] unsigned max_width() const { return _max_width; }

    /// Nodes whose `t_i(1)` came from a recorded replay rather than the model.
    [[nodiscard]] std::size_t num_from_timings() const { return _num_from_timings; }

    /// Nodes whose `t_i(1)` came from the cost model.
    [[nodiscard]] std::size_t num_from_model() const { return _num_from_model; }

    /// Estimated makespan in microseconds before any node was widened.
    [[nodiscard]] double makespan_before_us() const { return _makespan_before_us; }

    /// Estimated makespan in microseconds under the final widths. This is
    /// `max(critical_path_after_us(), area_after_us())`, which is the lower
    /// bound any schedule of these widths obeys.
    [[nodiscard]] double makespan_after_us() const { return _makespan_after_us; }

    /// Longest path through the graph in microseconds, under the final widths.
    [[nodiscard]] double critical_path_after_us() const { return _critical_path_after_us; }

    /// `sum(w_i * t_i(w_i)) / P` in microseconds, under the final widths: the
    /// time the plan's work needs on `P` threads if they were never idle. The
    /// area bound the search enforces is that this never exceeds the makespan.
    [[nodiscard]] double area_after_us() const { return _area_after_us; }

    /// Contraction nodes whose kernel route this run pinned, over the whole
    /// subgraph tree.
    [[nodiscard]] std::size_t num_route_pinned() const { return _num_route_pinned; }

  private:
    /// One graph's plan. Recursion returns the body's serial cost so the
    /// container node can be priced by what it will actually run.
    struct SubPlan {
        double serial_us{0.0};      ///< sum of t_i(1) over the body, one traversal
        double before_us{0.0};      ///< max(CP, area) with every width at 1
        double cp_before_us{0.0};   ///< longest path with every width at 1
        double area_before_us{0.0}; ///< sum(t_i(1)) / P with every width at 1
        double makespan_us{0.0};    ///< max(CP, area) under the body's final widths
        double cp_us{0.0};          ///< longest path under the final widths
        double area_us{0.0};        ///< sum(w_i * t_i(w_i)) / P under the final widths
    };

    SubPlan plan_graph(Graph &graph, unsigned p);

    CostModel _cost_model;
    unsigned  _requested_threads{0};

    unsigned    _planned_threads{0};
    std::size_t _num_widened{0};
    unsigned    _max_width{1};
    std::size_t _num_from_timings{0};
    std::size_t _num_from_model{0};
    std::size_t _num_route_pinned{0};
    double      _makespan_before_us{0.0};
    double      _makespan_after_us{0.0};
    double      _critical_path_after_us{0.0};
    double      _area_after_us{0.0};
};

EINSUMS_NAMESPACE_END(compute_graph::passes)
