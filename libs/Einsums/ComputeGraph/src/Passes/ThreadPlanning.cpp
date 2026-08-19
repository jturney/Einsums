//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/CostModel.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Moldability.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Passes/ThreadPlanning.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/TaskPool/WidthBudget.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph::passes)

namespace {

/// Comparisons between two makespan estimates in microseconds. Anything below
/// this is arithmetic noise, not an improvement worth moving a width for.
constexpr double kEpsilonUs = 1e-9;

/// What one node costs the plan.
struct NodeCost {
    double       t1_us{0.0};                        ///< Serial time, measured or modeled
    KernelFamily family{KernelFamily::Elementwise}; ///< Which speedup curve prices it
    std::size_t  bytes{0};                          ///< Working set, for the size class
    bool         frozen{false};                     ///< Width is 1 and stays 1
};

/// Which speedup curve a node's kernel scales along.
KernelFamily family_for(Node const &node, std::size_t bytes, DeviceProfile const &profile, double flops) {
    switch (node.kind) {
    case OpKind::Einsum:
    case OpKind::Gemm:
    case OpKind::SymmGemm:
    case OpKind::Gemv:
    case OpKind::Ger:
    case OpKind::Dot:
        // Small and large GEMM are different code, not the same code on less
        // data, so the flop count picks between two curves rather than a size
        // class within one.
        return flops > 0.0 ? (flops < DeviceProfile::kGemmSmallFlops ? KernelFamily::GemmSmall : KernelFamily::GemmLarge)
               : profile.size_class_for_bytes(bytes) == SizeClass::L1Resident ? KernelFamily::GemmSmall
                                                                              : KernelFamily::GemmLarge;
    case OpKind::BatchedGemm:
    case OpKind::GroupedBatchedGemm:
        return KernelFamily::BatchedGemm;

    case OpKind::Permute:
    case OpKind::Transpose:
    case OpKind::HPTTPermute:
    case OpKind::TileGather:
    case OpKind::TileScatter:
        return KernelFamily::Permute;

    default:
        return KernelFamily::Elementwise;
    }
}

/// The bytes a node touches: every distinct buffer it reads or writes, counted
/// once. Views resolve to the buffer they alias, so a node working through two
/// slices of one tensor is not charged for it twice.
std::size_t working_set_bytes(Graph const &graph, Node const &node) {
    std::unordered_set<TensorId> seen;
    std::size_t                  bytes = 0;
    auto const                  &map   = graph.tensors_map();

    auto add = [&](TensorId raw) {
        TensorId const tid = graph.resolve_alias(raw);
        if (!seen.insert(tid).second) {
            return;
        }
        if (auto it = map.find(tid); it != map.end()) {
            bytes += it->second.total_bytes();
        }
    };
    for (TensorId const tid : node.inputs) {
        add(tid);
    }
    for (TensorId const tid : node.outputs) {
        add(tid);
    }
    return bytes;
}

/// Elements in a tensor, or 0 when the graph does not know the tensor.
std::size_t elems_of(Graph const &graph, TensorId raw) {
    auto const &map = graph.tensors_map();
    auto        it  = map.find(graph.resolve_alias(raw));
    return it == map.end() ? 0 : it->second.total_elems();
}

/// A GEMM shape for a contraction node, or all zeros when one cannot be derived.
///
/// A capture that matched the 2D x 2D -> 2D pattern carries the exact shape in
/// its GemmHint and that is used verbatim. Anything else is reduced to the
/// square GEMM with the same flop count: with `C = A x B` over a link of extent
/// `K`, `|A| * |B| / |C| = K^2` whatever the index topology, and `|C| = M * N`.
/// The result prices the contraction's ARITHMETIC exactly and its aspect ratio
/// only approximately, which is the right trade for a cost model whose GEMM
/// table is interpolated anyway.
struct GemmShape {
    std::size_t m{0}, n{0}, k{0};

    [[nodiscard]] bool   valid() const { return m > 0 && n > 0 && k > 0; }
    [[nodiscard]] double flops() const { return 2.0 * static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k); }
};

/// Serial time for a batched or grouped-batched GEMM node.
///
/// Sums the arithmetic of every member and charges one launch for the call.
/// Returns 0 when the node carries no descriptor the model can read, which
/// leaves it below the fork floor and so at width 1 - the same "do not guess"
/// behaviour the default branch has.
double batched_gemm_us(Node const &node, DeviceProfile const &profile) {
    auto member_us = [&profile](std::size_t m, std::size_t n, std::size_t k, double count) {
        if (m == 0 || n == 0 || k == 0 || count <= 0.0) {
            return 0.0;
        }
        double const each = profile.estimate_gemm_time_us(m, n, k, 1);
        return count * std::max(0.0, each - profile.kernel_launch_overhead_us);
    };

    double work = 0.0;
    if (auto const *b = std::get_if<BatchedGemmDescriptor>(&node.op_data); b != nullptr) {
        work = member_us(static_cast<std::size_t>(b->m), static_cast<std::size_t>(b->n), static_cast<std::size_t>(b->k),
                         static_cast<double>(b->batch_count));
    } else if (auto const *g = std::get_if<GroupedBatchedGemmDescriptor>(&node.op_data); g != nullptr) {
        for (auto const &grp : g->groups) {
            work += member_us(static_cast<std::size_t>(grp.m), static_cast<std::size_t>(grp.n), static_cast<std::size_t>(grp.k),
                              static_cast<double>(grp.count));
        }
    } else {
        return 0.0;
    }

    return work > 0.0 ? work + profile.kernel_launch_overhead_us : 0.0;
}

GemmShape gemm_shape_of(Graph const &graph, Node const &node) {
    if (auto const *ein = std::get_if<EinsumDescriptor>(&node.op_data); ein != nullptr && ein->gemm_hint) {
        auto const &hint = *ein->gemm_hint;
        if (hint.m > 0 && hint.n > 0 && hint.k > 0) {
            return {.m = static_cast<std::size_t>(hint.m), .n = static_cast<std::size_t>(hint.n), .k = static_cast<std::size_t>(hint.k)};
        }
    }
    if (node.inputs.size() != 2 || node.outputs.size() != 1) {
        return {};
    }
    double const a = static_cast<double>(elems_of(graph, node.inputs[0]));
    double const b = static_cast<double>(elems_of(graph, node.inputs[1]));
    double const c = static_cast<double>(elems_of(graph, node.outputs[0]));
    if (a <= 0.0 || b <= 0.0 || c <= 0.0) {
        return {};
    }
    auto const k    = static_cast<std::size_t>(std::llround(std::sqrt(a * b / c)));
    auto const side = static_cast<std::size_t>(std::llround(std::sqrt(c)));
    if (k == 0 || side == 0) {
        return {};
    }
    return {.m = side, .n = side, .k = k};
}

/// Median of a node's recorded samples, in microseconds. A node that ran more
/// than once in a replay (a loop body's) contributes one sample per run and the
/// median is what survives an outlier.
double median_us(std::vector<double> &samples_ms) {
    if (samples_ms.empty()) {
        return 0.0;
    }
    std::size_t const mid = samples_ms.size() / 2;
    std::nth_element(samples_ms.begin(), samples_ms.begin() + static_cast<std::ptrdiff_t>(mid), samples_ms.end());
    return samples_ms[mid] * 1e3;
}

/// The vetoes of Part 6 of the design, which override profitability entirely.
bool policy_forbids_width(Node const &node) {
    // The grouped scalar ops are SEQUENTIAL BY CONTRACT: grouped_dot and
    // grouped_axpby accumulate in entry order so that their last bit does not
    // depend on how the entries were divided. Their kernels are perfectly
    // governable - kernel_moldability says so - and they are still width 1,
    // because bit-identity is the promise the node was introduced to keep.
    if (node.kind == OpKind::GroupedDot || node.kind == OpKind::GroupedAxpby) {
        return true;
    }
    // A container's own work is a predicate and a loop counter. Its body is
    // planned separately and its nodes acquire their own widths from the same
    // process-wide budget, so giving the container a width would hand out the
    // same machine twice.
    if (node.kind == OpKind::Loop || node.kind == OpKind::Conditional) {
        return true;
    }
    // A kernel that cannot be told how many threads to use gets whatever the
    // vendor decides, so a width above 1 would be a number the rest of the plan
    // divides the machine by and nothing honors.
    return !kernel_moldability(node);
}

/// The next rung above @p width, or 0 when there is none.
unsigned next_rung(std::vector<unsigned> const &rungs, unsigned width) {
    for (unsigned const rung : rungs) {
        if (rung > width) {
            return rung;
        }
    }
    return 0;
}

} // namespace

ThreadPlanning::ThreadPlanning() : ThreadPlanning(CostModel::detect_default(), 0) {
}

ThreadPlanning::ThreadPlanning(unsigned max_threads) : ThreadPlanning(CostModel::detect_default(), max_threads) {
}

ThreadPlanning::ThreadPlanning(CostModel cost_model, unsigned max_threads)
    : _cost_model(std::move(cost_model)), _requested_threads(max_threads) {
}

void ThreadPlanning::reset_stats() {
    _planned_threads        = 0;
    _num_widened            = 0;
    _max_width              = 1;
    _num_from_timings       = 0;
    _num_from_model         = 0;
    _makespan_before_us     = 0.0;
    _makespan_after_us      = 0.0;
    _critical_path_after_us = 0.0;
    _area_after_us          = 0.0;
}

std::vector<std::string> ThreadPlanning::explain() const {
    if (_planned_threads == 0) {
        return {};
    }
    std::vector<std::string> lines;
    lines.push_back(fmt::format("planned {} node(s) wide (max width {}) for {} thread(s)", _num_widened, _max_width, _planned_threads));
    lines.push_back(
        fmt::format("serial times: {} node(s) from recorded replays, {} from the cost model", _num_from_timings, _num_from_model));
    lines.push_back(fmt::format("estimated makespan {:.1f} us -> {:.1f} us", _makespan_before_us, _makespan_after_us));
    return lines;
}

bool ThreadPlanning::run(Graph &graph) {
    unsigned p = _requested_threads;
    if (p == 0) {
        // The budget is what the executor rations against, so planning against
        // anything else would be planned stale from the start.
        auto &budget = task_pool::WidthBudget::get_singleton();
        budget.sync_machine_width();
        p = std::max(1U, budget.total());
    }
    _planned_threads = p;

    SubPlan const top       = plan_graph(graph, p);
    _makespan_before_us     = top.before_us;
    _makespan_after_us      = top.makespan_us;
    _critical_path_after_us = top.cp_us;
    _area_after_us          = top.area_us;

    if (_num_widened == 0) {
        report(1, fmt::format("no node earned a width above 1 on {} thread(s)", p));
    } else {
        report(1, fmt::format("widened {} node(s) up to width {}; estimated makespan {:.1f} us -> {:.1f} us", _num_widened, _max_width,
                              _makespan_before_us, _makespan_after_us));
    }

    // A width is metadata the executor reads; nothing structural moved, so the
    // graph is not "modified" in the sense the pass manager's validator means.
    return false;
}

ThreadPlanning::SubPlan ThreadPlanning::plan_graph(Graph &graph, unsigned p) {
    SubPlan result;

    // Positions must be topological for the two single-pass longest-path
    // sweeps below, and the dependency lists are keyed by position. Both are
    // exactly what execute() would build, and this is a no-op on a graph that
    // already has them.
    graph.topological_sort();

    auto             &nodes = graph.nodes();
    std::size_t const n     = nodes.size();
    graph.set_planned_thread_count(p);
    if (n == 0) {
        return result;
    }

    auto const          &deps    = graph.dependencies();
    DeviceProfile const &profile = _cost_model.device(Target::CPU);
    if (deps.successors.size() != n || deps.predecessors.size() != n) {
        // Nothing to plan against; leave every width alone rather than write a
        // plan from a dependency view that does not describe this node list.
        note_skip("dependency lists do not match the node list");
        return result;
    }

    // Bodies first: a container is priced by what it will run, and its body's
    // nodes need their own widths whether or not the container earns one.
    std::vector<double> body_serial_us(n, 0.0);
    for (std::size_t i = 0; i < n; i++) {
        if (auto *loop = std::get_if<LoopDescriptor>(&nodes[i].op_data); loop != nullptr && loop->body) {
            // Planned as though it owned the whole machine. It does not have to
            // share P with the parent here, because the one process-wide
            // WidthBudget composes the two at run time: the container's own
            // unit is lent back for the duration of the nested replay, so the
            // body is admitted against a budget nobody else is holding.
            body_serial_us[i] += plan_graph(*loop->body, p).serial_us;
        } else if (auto *cond = std::get_if<ConditionalDescriptor>(&nodes[i].op_data); cond != nullptr) {
            if (cond->then_branch) {
                body_serial_us[i] += plan_graph(*cond->then_branch, p).serial_us;
            }
            if (cond->else_branch) {
                // Only one branch runs, so the cost of the node is the worse of
                // the two rather than their sum.
                body_serial_us[i] = std::max(body_serial_us[i], plan_graph(*cond->else_branch, p).serial_us);
            }
        }
    }

    // Recorded medians, keyed by node id. A replay writes one sample per node
    // execution, so a node inside a loop body has as many as the loop ran.
    std::unordered_map<NodeId, std::vector<double>> samples;
    for (auto const &timing : graph.timing_report()) {
        samples[timing.id].push_back(timing.duration_ms);
    }

    std::vector<NodeCost> cost(n);
    for (std::size_t i = 0; i < n; i++) {
        Node             &node  = nodes[i];
        NodeCost         &c     = cost[i];
        std::size_t const bytes = working_set_bytes(graph, node);
        GemmShape const   shape = gemm_shape_of(graph, node);

        c.bytes  = bytes;
        c.family = family_for(node, bytes, profile, shape.valid() ? shape.flops() : 0.0);

        if (auto it = samples.find(node.id); it != samples.end()) {
            c.t1_us = median_us(it->second);
            _num_from_timings++;
        } else {
            _num_from_model++;
            switch (node.kind) {
            case OpKind::Loop:
            case OpKind::Conditional:
                // Priced by the body it runs, so a loop full of fat work sits
                // on the critical path where it belongs even though the
                // container itself is width 1.
                c.t1_us = body_serial_us[i];
                break;
            case OpKind::Einsum:
            case OpKind::Gemm:
            case OpKind::SymmGemm:
            case OpKind::Gemv:
            case OpKind::Ger:
            case OpKind::Dot:
                c.t1_us = shape.valid() ? profile.estimate_gemm_time_us(shape.m, shape.n, shape.k, 1)
                                        : (bytes > 0 ? profile.estimate_memory_time_us(bytes, 1) : 0.0);
                break;
            case OpKind::BatchedGemm:
            case OpKind::GroupedBatchedGemm:
                // A batch is arithmetic, not traffic. Without these two cases
                // both fell to the default below and were priced as the bytes
                // they move, which is the wrong dimension entirely: the flops
                // scale with the members' m*n*k while the bytes only scale with
                // the operands. family_for() above already knows these kinds -
                // it returns KernelFamily::BatchedGemm for them - so the two
                // switches disagreed about whether a batched GEMM was a
                // recognised node, and the planner priced the CC residual's
                // dominant kernel as a memcpy.
                //
                // One batched call pays ONE launch and then does every member's
                // arithmetic, so the per-member estimate has its launch term
                // removed and a single one is added back at the end. That is
                // the whole reason the batch exists.
                c.t1_us = batched_gemm_us(node, profile);
                break;

            case OpKind::Permute:
            case OpKind::Transpose:
            case OpKind::HPTTPermute: {
                auto const &map  = graph.tensors_map();
                std::size_t rank = 2;
                if (!node.inputs.empty()) {
                    if (auto it = map.find(graph.resolve_alias(node.inputs[0])); it != map.end() && it->second.rank > 0) {
                        rank = it->second.rank;
                    }
                }
                c.t1_us = bytes > 0 ? profile.estimate_permute_time_us(bytes, rank, 1) : 0.0;
                break;
            }
            default:
                // Everything else - elementwise, axpby, scale, the tiled
                // lowerings, Custom - is priced as the traffic it moves. A node
                // whose buffers the graph does not know moves no bytes it can
                // see and gets a serial time of 0, which is below the floor and
                // so stays at width 1 instead of being guessed at.
                c.t1_us = bytes > 0 ? profile.estimate_memory_time_us(bytes, 1) : 0.0;
                break;
            }
        }

        result.serial_us += c.t1_us;

        if (policy_forbids_width(node)) {
            c.frozen = true;
            note_skip("policy forbids a width above 1", fmt::format("node '{}'", node.label));
        } else if (c.t1_us < kSerialFloorUs) {
            c.frozen = true;
            note_skip("serial time below the fork floor", fmt::format("node '{}' at {:.1f} us", node.label, c.t1_us));
        }
    }

    // t_i(w). The fixed launch cost is paid whatever the width, so it comes out
    // before the division and goes back after: a plan must never be able to buy
    // its way below the floor a kernel costs to start.
    double const overhead_us = profile.kernel_launch_overhead_us;
    auto         time_at     = [&](std::size_t i, unsigned width) {
        NodeCost const &c = cost[i];
        if (width <= 1) {
            return c.t1_us;
        }
        double const scalable = std::max(0.0, c.t1_us - overhead_us);
        double const speedup  = std::max(1e-9, profile.parallel_speedup(c.family, c.bytes, width));
        return scalable / speedup + std::min(overhead_us, c.t1_us);
    };

    std::vector<unsigned> const rungs = (profile.max_threads == p) ? profile.width_rungs() : width_rungs_for(p);
    std::vector<unsigned>       width(n, 1);
    // Out of the search's running: policy-frozen and floor-frozen from the
    // start, plus whatever the search rejects. Deliberately NOT the same flag
    // as NodeCost::frozen, which would make t_i(w) forget a width the plan had
    // already taken the moment the node stopped being a candidate.
    std::vector<char> exhausted(n, 0);
    for (std::size_t i = 0; i < n; i++) {
        exhausted[i] = cost[i].frozen ? 1 : 0;
    }
    std::vector<double> times(n, 0.0);
    std::vector<double> bottom(n, 0.0);
    std::vector<double> top(n, 0.0);

    // Longest path to a sink (bottom) and from a source (top), both in
    // estimated time. Positions are topological, so one sweep each way is
    // enough, and a node lies on a critical path exactly when top + bottom
    // reaches the longest path's length.
    auto recompute = [&]() {
        double critical = 0.0;
        double area_sum = 0.0;
        for (std::size_t i = 0; i < n; i++) {
            times[i] = time_at(i, width[i]);
            area_sum += static_cast<double>(width[i]) * times[i];
        }
        for (std::size_t k = n; k-- > 0;) {
            double longest = 0.0;
            for (std::size_t const succ : deps.successors[k]) {
                longest = std::max(longest, bottom[succ]);
            }
            bottom[k] = longest + times[k];
            critical  = std::max(critical, bottom[k]);
        }
        for (std::size_t k = 0; k < n; k++) {
            double earliest = 0.0;
            for (std::size_t const pred : deps.predecessors[k]) {
                earliest = std::max(earliest, top[pred] + times[pred]);
            }
            top[k] = earliest;
        }
        return std::pair<double, double>{critical, area_sum / static_cast<double>(p)};
    };

    auto [cp, area]  = recompute();
    result.before_us = std::max(cp, area);

    // The widths at the last STRICT improvement, which is what the plan ends up
    // being. Kept separately from the widths the search is currently holding,
    // because the search is allowed to cross a plateau to reach the next
    // improvement and must not be able to leave the graph standing on one.
    std::vector<unsigned> best_widths   = width;
    double                best_makespan = std::max(cp, area);

    if (p > 1) {
        // Each round either widens a node one rung or freezes one out of the
        // running, and both are finite, so the cap is a guard against a cost
        // model that reports a gain it cannot deliver rather than a real bound.
        std::size_t const max_rounds = n * (rungs.size() + 1) + 16;
        for (std::size_t round = 0; round < max_rounds; round++) {
            double const makespan = std::max(cp, area);

            std::size_t candidate       = n;
            unsigned    candidate_width = 0;
            double      candidate_score = 0.0;
            for (std::size_t i = 0; i < n; i++) {
                if (exhausted[i] != 0 || top[i] + bottom[i] < cp - kEpsilonUs) {
                    continue; // frozen, or not on a critical path
                }
                unsigned const wider = next_rung(rungs, width[i]);
                if (wider == 0) {
                    continue;
                }
                double const t_wide = time_at(i, wider);
                double const gain   = times[i] - t_wide;
                if (gain <= kEpsilonUs) {
                    continue;
                }
                // Area the step consumes, in the same units as the makespan.
                double const added = std::max(kEpsilonUs, (static_cast<double>(wider) * t_wide - static_cast<double>(width[i]) * times[i]) /
                                                              static_cast<double>(p));
                double const score = gain / added;
                if (score > candidate_score) {
                    candidate       = i;
                    candidate_width = wider;
                    candidate_score = score;
                }
            }
            if (candidate == n) {
                break;
            }

            unsigned const previous = width[candidate];
            width[candidate]        = candidate_width;
            auto [cp2, area2]       = recompute();

            // A step is taken when it does not make the estimate worse and the
            // area stays inside the makespan this round started at - the area
            // bound of the design. A step is not REQUIRED to improve, because
            // the graph this exists for has several fat nodes side by side on
            // the critical path and widening the first of them buys nothing
            // until the last one is widened too; a strictly-improving rule
            // stops on that plateau and plans nothing at all. What is required
            // to improve is the plan that gets kept, which is the snapshot
            // below.
            if (std::max(cp2, area2) <= makespan + kEpsilonUs && area2 <= makespan + kEpsilonUs) {
                cp   = cp2;
                area = area2;
                if (std::max(cp, area) < best_makespan - kEpsilonUs) {
                    best_makespan = std::max(cp, area);
                    best_widths   = width;
                }
                continue;
            }

            width[candidate] = previous;
            auto reverted    = recompute();
            cp               = reverted.first;
            area             = reverted.second;
            // Rejected once, never reconsidered: this rung costs more area than
            // it returns, and every later round has less room, not more.
            // Freezing it is also what bounds the loop.
            exhausted[candidate] = 1;
            note_skip("widening would cost more area than it returns", fmt::format("node '{}'", nodes[candidate].label));
        }
    } else {
        note_skip("machine has one thread");
    }

    // Final widths, and the priorities that order admission under them.
    width = best_widths;
    // Not `far`: <windows.h> still defines it as an empty legacy segment
    // keyword, so a binding by that name expands to nothing there.
    auto [final_cp, final_area] = recompute();
    cp                          = final_cp;
    area                        = final_area;
    for (std::size_t i = 0; i < n; i++) {
        nodes[i].thread_width = static_cast<std::uint16_t>(std::min<unsigned>(width[i], p));
        // Nanoseconds, and never 0: 0 is how a node says it was never planned.
        nodes[i].admission_priority = std::max<std::int64_t>(1, std::llround(bottom[i] * 1e3));
        if (width[i] > 1) {
            _num_widened++;
            _max_width = std::max(_max_width, width[i]);
            report(2, fmt::format("node '{}' at width {} ({:.1f} us -> {:.1f} us)", nodes[i].label, width[i], cost[i].t1_us, times[i]));
        }
    }

    result.cp_us       = cp;
    result.area_us     = area;
    result.makespan_us = std::max(cp, area);
    return result;
}

EINSUMS_NAMESPACE_END(compute_graph::passes)
