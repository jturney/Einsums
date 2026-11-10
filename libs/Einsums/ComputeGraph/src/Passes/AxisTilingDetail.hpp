//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/ComputeGraph/CaptureContext.hpp>
#include <Einsums/ComputeGraph/CostModel.hpp>
#include <Einsums/ComputeGraph/DescriptorHooks.hpp>
#include <Einsums/ComputeGraph/DescriptorRegistry.hpp>
#include <Einsums/ComputeGraph/DestinationRead.hpp>
#include <Einsums/ComputeGraph/EscapeAnalysis.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Operations.hpp>
#include <Einsums/ComputeGraph/Options.hpp>
#include <Einsums/ComputeGraph/Passes/AxisTiling.hpp>
#include <Einsums/ComputeGraph/Passes/PassUtil.hpp>
#include <Einsums/ComputeGraph/Prefactor.hpp>
#include <Einsums/ComputeGraph/SymmetryRules.hpp>
#include <Einsums/ComputeGraph/View.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/Options/Get.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/Tensor/SymmetryOps.hpp>
#include <Einsums/TensorBase/SymmetryDescriptor.hpp>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <ranges>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "BudgetShare.hpp"
#include "LoopEmission.hpp"
#include "StorageCost.hpp"

// What the parts of the AxisTiling pass share: the plan types, the region analysis, and the
// functions one part calls in another. Private to the pass; nothing outside src/Passes includes it.

EINSUMS_NAMESPACE_BEGIN(compute_graph::passes::axis_tiling)

using detail::axis_walks;
using detail::AxisWalk;
using detail::combine_walks;
using detail::declare_walk_bounds;
using detail::einsum_spec;
using detail::permute_spec;
using detail::walk_iterations;

/// A position carrying no slice label.
constexpr int kFree = -1;

/// The most axes one decision may slice at once.
///
/// Three is not a limit of the propagation, which handles any number; it is a limit on the
/// enumeration, which is exponential in the seed's rank. Every program this pass has been
/// asked about slices one or two, and a fourth axis costs the same traffic argument twice
/// over: an axis worth slicing is one whose streaming saves bytes, and by the third the
/// saving is already the whole tensor.
constexpr std::size_t kMaxSlicedAxes = 3;

/// Whether a node's operation can be evaluated one slice at a time.
///
/// A barrier is anything whose value at a slice is not a function of its operands at that
/// slice: a decomposition, a communication, control flow, an I/O node. @ref OpKind::Gemm is a
/// barrier for a narrower reason: it names no index letters, so there is
/// nothing to propagate a labelling through, and @ref OpKind::ElementTransform for another:
/// what it applies is a callable the node carries, and a rewrite that re-emits the operation has
/// nothing to re-emit it WITH short of a registry name the anonymous arm does not have.
bool tileable_kind(OpKind kind);

/// Whether @p node is one a registered descriptor lets this pass slice: both hooks, and a node
/// that is the same computation at every slice because it reads nothing it does not list and does
/// nothing beyond what it writes.
bool registered_sliceable(Node const &node);

/// One operand of a node, in the order the kind names its operands.
///
/// Slot 0 is always the destination. A destination the node also READS appears once, as slot
/// 0: the capture layer appends it to the input list so the scheduler sees the read, and a
/// second slot for the same tensor would only be a second name for one labelling.
struct SlotRef {
    TensorId    tid{0};
    std::size_t rank{0};
};

/// The operand slots of @p node, or nothing when the node is not one this pass reads.
std::optional<std::vector<SlotRef>> slots_of(Graph const &graph, Node const &node);

/// The index lists a contraction or a permutation names its operands with, slot by slot.
///
/// Read from the LIVE index state where the node carries one, because a pass that rewrote the
/// letters wrote them there and the descriptor's own copy is the at-capture snapshot.
std::optional<std::vector<std::vector<std::string>>> letters_of(Node const &node);

/// The letters a node's permutation operators name. An axis one of them names cannot be sliced:
/// the operator sums over its permutations, and at one slice of an exchanged axis the body would
/// need the slice it is not at.
std::vector<std::string> operator_letters_of(Node const &node);

/// The link (summed) letters of a contraction, empty for every other kind.
std::vector<std::string> link_letters_of(Node const &node);

/// Where a slot's labels live: on the tensor for one the region writes, on the USE for one it
/// only reads.
///
/// The distinction is what lets a three-index integral serve both operands of ``sum_Q B[Q,i,a]
/// B[Q,j,b]``: the two uses slice the same axis of one buffer at two different loop variables,
/// which is two views rather than one contradiction. A tensor the region writes has a single
/// writer and therefore a single labelling.
struct LabelStore {
    std::unordered_map<TensorId, std::vector<int>>                  tensor;
    std::map<std::pair<std::size_t, std::size_t>, std::vector<int>> use;
};

/// The whole of a candidate decision, before it is costed.
struct Labelling {
    LabelStore store;
    bool       feasible{true};

    /// Why an infeasible candidate is infeasible, shape-independent so the tally aggregates the
    /// candidates that fail the same way into one counted line.
    std::string reason;

    /// The sliced axes of each operand EVERY one of whose axes is sliced, by label.
    ///
    /// A point slice drops its axis, so such an operand would be left with none, and no kernel
    /// takes a rank-zero view. A block keeps its axis, so the operand survives a plan that cuts
    /// at least one of these axes into blocks. Whether that holds depends on the blocks, which
    /// are chosen after the labelling, so the propagation records the sets and the plan checks
    /// them.
    std::vector<std::set<int>> fully_sliced;
};

/// One operand of one node of the region, with the labelling the decision gave it.
struct SlotPlan {
    TensorId         tid{0};
    std::vector<int> labels;
};

/// One node of the region, everything the rewrite needs to re-emit it at a slice.
struct NodePlan {
    NodeId                id{unassigned_node_id};
    OpKind                kind{OpKind::Custom};
    OpData                op_data;
    std::vector<SlotPlan> slots;
    std::string           label;
};

/// The slice request for a registered node of @p op, naming the parameter of each sliced axis
/// through @p name_of.
template <typename NameOf>
SliceRequest slice_request(NodePlan const &op, NameOf &&name_of) {
    SliceRequest request;
    for (auto const &slot : op.slots) {
        std::vector<std::optional<std::string>> axes;
        for (int const label : slot.labels) {
            axes.push_back(label == kFree ? std::nullopt : std::optional<std::string>{name_of(label)});
        }
        request.dropped.push_back(std::move(axes));
    }
    return request;
}

/// The description of one tiled region, produced by the analysis and consumed by the rewrite.
struct Plan {
    std::vector<std::size_t> extents; ///< Extent of each sliced axis, outermost first.
    std::size_t              slices{0};
    std::size_t              depth{1};
    std::size_t              iterations{0};

    /// True when some axis is cut into blocks the body keeps as an axis, false for the grouped
    /// chunk of points.
    bool range{false};

    /// The full block extent of each sliced axis, one for a point; empty for a grouped plan.
    std::vector<std::size_t> blocks;
    std::size_t              largest_before{0};
    std::size_t              largest_after{0};
    /// Built because nothing fit: the finest slicing of a candidate whose single slice is over the
    /// budget, kept in case no candidate fits it.
    bool best_effort{false};
    /// A best-effort plan whose slices fit the tensor's share of the budget, which is as far as
    /// cutting still lowers the peak.
    bool                     fits_share{false};
    long double              traffic{0.0L};
    std::vector<std::string> axis_names;
    std::vector<std::string> axis_letters;
    std::vector<std::string> streamed;
    std::vector<std::string> whole;
    std::string              accumulator;
    LabelStore               store;

    std::vector<NodePlan> ops;

    /// The replaced position the loop takes the place of: the last when the statements can all
    /// move down to it, the first when they can all move up.
    std::size_t site{0};

    /// Positions in the parent's node list the loop replaces, ascending. A statement none of
    /// whose operands carries a sliced axis is LOOP-INVARIANT and is not one of them: it stays
    /// where it is, ahead of the loop once the dependency sort has placed it, because a value
    /// the body would compute identically on every iteration belongs outside.
    std::vector<std::size_t> replaced;

    /// Graph-owned intermediates carrying a sliced axis, re-declared at slice extents inside
    /// the body: the tensor, and where its sliced axes are.
    std::vector<std::pair<TensorId, std::vector<int>>> body_owned;

    /// The tensor the reduction accumulates into across iterations, zero when the region has
    /// no such reduction.
    TensorId accumulator_id{0};

    /// Destinations a contraction sums a sliced axis into, with the contraction's destination
    /// prefactor: scaled by it once before the loop, then added to at every iteration.
    std::vector<std::pair<TensorId, PrefactorScalar>> summed_into;
    std::vector<std::string>                          summed_names;

    packed_gemm::ScalarType dtype{packed_gemm::ScalarType::Unknown};

    /// A region whose contribution to its one output is unchanged when its two sliced axes are
    /// exchanged runs only the tiles on and above their diagonal. Each tile above it computes the
    /// contribution into a scratch tile, adds it to its own block of the output, and adds the
    /// exchanged scratch, with the contribution's sign, to the mirrored block.
    struct Triangle {
        std::size_t      writer{0};     ///< Index in @ref ops of the contraction writing the output.
        SymmetryOp       sigma;         ///< On the output's positions, carrying the contribution's sign.
        std::vector<int> mirror_labels; ///< The output's labels with the two sliced axes exchanged.
        /// Declared symmetries the proof rests on, checked before the loops on every replay.
        std::vector<std::pair<TensorId, SymmetryDescriptor>> checks;
    };
    std::optional<Triangle> triangle;

    /// The cost model's predicted time for the plan, in its own microseconds.
    double price_us{0.0};
};

/// Unify two label vectors position by position, failing on a genuine disagreement.
bool unify(std::vector<int> &lhs, std::vector<int> &rhs, bool &changed);

/// A labelling no position of any tensor carries twice, which is what a slice variable
/// appearing on two axes of one tensor would mean: a diagonal read the loop cannot express.
bool labels_are_distinct(std::vector<int> const &labels);

/// Where each graph the pass has visited sits: the graph that holds it, and the position of the
/// node there that holds it.
using Enclosing = std::unordered_map<Graph const *, std::pair<Graph *, std::size_t>>;

/// What one graph's place in the tree says about the tensors it names.
///
/// A loop body names a tensor its parent declared through a handle of its own, registered at
/// capture with no ownership claim, so a parent's intermediate reaches the body with the flag
/// clear and with none of the parent's other uses visible. The tensor's storage is the identity
/// the two tables share, which is how both questions below walk out to the declaration.
struct Scope {
    Graph           &graph;
    Enclosing const &enclosing;

    /// Whether the tensor is a graph-owned intermediate, read off the declaration.
    [[nodiscard]] bool intermediate(TensorId tid) const {
        Graph const        *level  = &graph;
        TensorHandle const *handle = graph.find_tensor(tid);
        while (handle != nullptr && !handle->is_intermediate && handle->tensor_ptr != nullptr) {
            auto const hit = enclosing.find(level);
            if (hit == enclosing.end()) {
                break;
            }
            Graph *outer = hit->second.first;
            handle       = outer->find_tensor_by_ptr(handle->tensor_ptr);
            level        = outer;
        }
        return handle != nullptr && handle->is_intermediate;
    }

    /// Whether the tensor is backed, read off the declaration: a body's handle for a parent's backed
    /// tensor carries no backing of its own.
    [[nodiscard]] bool backed(TensorId tid) const { return backing_of(tid) != nullptr; }

    /// The tensor's backing, read off the declaration; null for a tensor memory holds.
    [[nodiscard]] std::shared_ptr<TensorBacking const> backing_of(TensorId tid) const {
        Graph const        *level  = &graph;
        TensorHandle const *handle = graph.find_tensor(tid);
        while (handle != nullptr && !handle->backing && handle->tensor_ptr != nullptr) {
            auto const hit = enclosing.find(level);
            if (hit == enclosing.end()) {
                break;
            }
            Graph *outer = hit->second.first;
            handle       = outer->find_tensor_by_ptr(handle->tensor_ptr);
            level        = outer;
        }
        return handle != nullptr ? handle->backing : nullptr;
    }

    /// Whether a node outside this graph, in any graph that holds it, reads or writes the tensor.
    [[nodiscard]] bool touched_beyond(TensorId tid) const {
        Graph const        *level  = &graph;
        TensorHandle const *handle = graph.find_tensor(tid);
        while (handle != nullptr && handle->tensor_ptr != nullptr) {
            auto const hit = enclosing.find(level);
            if (hit == enclosing.end()) {
                return false;
            }
            auto const [outer, owner] = hit->second;
            TensorId const outer_tid  = outer->find_tensor_id_by_ptr(handle->tensor_ptr);
            if (outer_tid == 0) {
                return false;
            }
            TensorId const buffer = outer->buffer_of(outer_tid);
            auto const    &nodes  = outer->nodes();
            for (std::size_t j = 0; j < nodes.size(); ++j) {
                if (j == owner) {
                    continue;
                }
                auto const [reads, writes] = outer->effective_io(nodes[j]);
                auto const names           = [&](std::vector<TensorId> const &ids) {
                    return std::ranges::any_of(ids, [&](TensorId x) { return outer->buffer_of(x) == buffer; });
                };
                if (names(reads) || names(writes)) {
                    return true;
                }
            }
            handle = outer->find_tensor(outer_tid);
            level  = outer;
        }
        return false;
    }
};

/// The analysis over one region: a set of tileable nodes, in the parent's order, that need not be
/// contiguous.
class RegionAnalysis {
  public:
    /// @param[in] members The region's nodes, by their position in the parent, ascending.
    /// @param[in] seed    The intermediate whose axes the candidates are taken from; zero takes the
    ///                    largest tensor the region writes.
    RegionAnalysis(Graph const &graph, Scope const &scope, std::vector<std::size_t> members, TensorId seed = 0)
        : _graph(graph), _scope(scope), _members(std::move(members)), _seed(seed), _seed_given(seed != 0) {
        for (std::size_t k = 0; k < _members.size(); ++k) {
            _ordinal.emplace(_members[k], k);
        }
    }

    /// Collect what the region reads, writes and declares. False when the run holds a node
    /// this pass cannot read at all.
    bool collect();

    [[nodiscard]] std::vector<TensorId> const &written() const noexcept { return _written; }
    [[nodiscard]] std::size_t                  largest_written_bytes() const noexcept { return _largest_written_bytes; }
    [[nodiscard]] TensorId                     seed() const noexcept { return _seed; }
    [[nodiscard]] packed_gemm::ScalarType      dtype() const noexcept { return _dtype; }
    [[nodiscard]] bool                         all_runtime() const noexcept { return _all_runtime; }

    /// Fill in what the rewrite reads: the nodes at a slice, the intermediates to re-declare,
    /// and the reduction to accumulate. False when an intermediate the body would re-declare is
    /// still wanted whole by something outside the run.
    [[nodiscard]] bool build(Labelling const &labelling, Plan &plan, std::string &reason) const;

    /// Propagate a seed labelling through the region. An infeasible candidate comes back with
    /// @ref Labelling::feasible false.
    [[nodiscard]] Labelling propagate(std::vector<std::size_t> const &seed_positions) const;

    /// Whether a labelling can be turned into a loop: every unlabelled tensor the region
    /// writes is either its reduction target or purely internal.
    [[nodiscard]] bool closes(Labelling const &labelling, std::string &accumulator, TensorId &accumulator_id,
                              std::vector<std::pair<TensorId, PrefactorScalar>> &summed_into, std::string &reason) const;

    /// What moving a box of the backed tensor @p tid costs beyond the memory traffic already counted
    /// for its bytes, @p transfers times per replay: the storage weighs each of the box's
    /// @p box_bytes more, and each of its runs adds a fixed cost. Zero for a tensor memory holds.
    /// @p extents and @p whole_axis describe the box on each declared axis.
    [[nodiscard]] long double storage_surcharge(TensorId tid, std::vector<std::size_t> const &extents, std::vector<bool> const &whole_axis,
                                                long double box_bytes, long double transfers) const;

    /// Whether node @p index is a contraction that sums one of the sliced axes: an input carries
    /// it and the destination does not.
    [[nodiscard]] bool sums_sliced(Labelling const &labelling, std::size_t index) const;

    /// Bytes per slice of the largest intermediate, the bytes every intermediate's chunk holds
    /// together, and the traffic the schedule streams. Of that traffic, @p contracted is what the
    /// loop's contractions read and write, which @ref tile_price prices with their arithmetic.
    void cost(Labelling const &labelling, std::vector<std::size_t> const &extents, std::size_t depth, std::size_t &largest,
              std::size_t &held, long double &traffic, long double &contracted) const;

    /// The largest intermediate, the bytes every block buffer of the loops holds together, and the
    /// traffic of a plan that cuts each sliced axis into blocks of @p blocks (one for a point),
    /// remainders included.
    ///
    /// Each loop of the plan declares its own buffers at its own extents and every one of them is
    /// allocated for the whole replay, so the held bytes count a remainder loop's buffers beside
    /// the full block's.
    void range_cost(Labelling const &labelling, std::vector<std::size_t> const &extents, std::vector<std::size_t> const &blocks,
                    std::size_t &largest, std::size_t &held, long double &traffic, long double &contracted) const;

    /// The bytes the intermediates the loop re-declares hold whole, before tiling.
    /// Whether a slice of @p tid is held in memory for each iteration: an intermediate the body
    /// declares, or a write-backed tensor whose box the body fills before SlabStreaming stores it.
    /// Not an update-backed one, which is produced in memory whole (SlabStreaming::stream_write).
    [[nodiscard]] bool held_per_slice(TensorId tid) const {
        auto const backing = _scope.backing_of(tid);
        return _scope.intermediate(tid) || (backing != nullptr && backing->write && !backing->update);
    }

    [[nodiscard]] std::size_t streamed_whole_bytes(Labelling const &labelling) const;

    /// The sliced axes a block would reward: those some contraction of the loop carries while one
    /// of its operands does not, so that operand is read once per block rather than once per slice.
    [[nodiscard]] std::set<int> reusable_labels(Labelling const &labelling) const;

    /// Why the loop cannot be emitted with blocks at all, empty when it can.
    [[nodiscard]] std::string_view range_blocker(Labelling const &labelling) const;

    /// Names the decision gives to each sliced axis, and the letter a contraction calls it.
    void describe(std::vector<std::size_t> const &seed_positions, std::vector<std::string> &names, std::vector<std::string> &letters) const;

    /// Which tensors stream and which are left whole under @p labelling.
    void partition(Labelling const &labelling, std::vector<std::string> &streamed, std::vector<std::string> &whole) const;

    [[nodiscard]] Graph const                    &graph() const noexcept { return _graph; }
    [[nodiscard]] std::vector<std::size_t> const &members() const noexcept { return _members; }

    /// The tensor in one operand slot of the member at parent position @p index.
    [[nodiscard]] TensorId slot_tensor(std::size_t index, std::size_t slot) const { return slots_at(index)[slot].tid; }

  private:
    [[nodiscard]] std::vector<int> *labels_for(LabelStore &store, std::size_t node_index, std::size_t slot, SlotRef const &ref) const;

    /// The labels one operand of one node carries, or nothing when the labelling gave it none.
    [[nodiscard]] std::vector<int> const *slot_labels(Labelling const &labelling, std::size_t index, std::size_t slot) const;

    /// Whether any operand of the node at @p index carries a sliced axis under @p labelling.
    [[nodiscard]] bool node_is_sliced(Labelling const &labelling, std::size_t index) const;

    /// @brief Which sliced axes one node of the region carries, by label.
    /// @param[in] labelling The candidate.
    /// @param[in] index     The node, in the parent's numbering.
    /// @return The labels its operands and destination carry, empty for a loop-invariant node.
    [[nodiscard]] std::set<int> node_labels(Labelling const &labelling, std::size_t index) const;

    /// The operand slots of the member at parent position @p index.
    [[nodiscard]] std::vector<SlotRef> const &slots_at(std::size_t index) const { return _slots[_ordinal.at(index)]; }

    Graph const                                 &_graph;
    Scope const                                 &_scope;
    std::vector<std::size_t>                     _members;
    std::unordered_map<std::size_t, std::size_t> _ordinal;

    std::vector<TensorId>             _written;
    std::unordered_set<TensorId>      _written_set;
    std::unordered_set<TensorId>      _touched_outside;
    std::vector<std::vector<SlotRef>> _slots;
    std::size_t                       _largest_written_bytes{0};
    TensorId                          _seed{0};
    bool                              _seed_given{false};
    packed_gemm::ScalarType           _dtype{packed_gemm::ScalarType::Unknown};
    bool                              _all_runtime{true};
};

/// The largest chunk that fits the cap and divides the slice count.
///
/// Divides, because a grouped node's member count is fixed when the body is captured: a ragged
/// last chunk would need a second body, and a schedule with two bodies is two schedules.
std::size_t chunk_depth(std::size_t slices, std::size_t bytes_per_slice, std::size_t cap);

/// The block extents worth trying on an axis of @p extent: for every block count, the smallest
/// block that reaches it.
///
/// Two blocks that cut the axis into the same number of pieces stream the same bytes, since what
/// a block saves is the number of times the operands it does not carry are re-read. The smaller
/// of the two holds less and leaves a remainder no larger, so it is the only one worth costing:
/// 21 in pieces of at most 8 is three blocks of 7 rather than 8, 8 and 5.
std::vector<std::size_t> balanced_blocks(std::size_t extent);

/// One costed choice of block extents.
struct BlockChoice {
    std::vector<std::size_t> blocks;
    std::size_t              largest{0};
    std::size_t              held{0};
    long double              traffic{0.0L};
    long double              contracted{0.0L};
    double                   price_us{0.0};

    [[nodiscard]] std::size_t ranged() const {
        return static_cast<std::size_t>(std::ranges::count_if(blocks, [](std::size_t b) { return b > 1; }));
    }
};

/// Whether two predicted times are the same. Two plans that do the same work, such as a block
/// covering a whole axis beside the plan that leaves the axis unsliced, sum their prices in a
/// different order and differ in the last bits; the price cannot tell them apart, so the
/// comparisons after it must.
bool same_price(double lhs, double rhs);

/// Whether @p lhs is the better of two block choices: the lower predicted time, then fewer bytes
/// streamed, then fewer ranged axes, then the smaller footprint, then blocks on the inner axes
/// rather than the outer ones, so a tie is broken the same way in every process.
bool better(BlockChoice const &lhs, BlockChoice const &rhs);

/// The block extents @p price predicts fastest among those @p fits accepts, or nothing when no
/// choice that cuts some axis into blocks fits.
///
/// Fewer bytes streamed is not the same as faster: the bytes always favour the largest block that
/// fits, while a larger tile can run its contractions more slowly, and a smaller one pays its
/// calls and packing more often. The price weighs both, so a block too large to pay is declined by
/// its price rather than by a cap on its size.
///
/// Only the axes in @p reusable are cut, and at most two of them at once; every other axis stays a
/// point. Two is the blocking a matrix product has, one block per operand the other does not
/// carry, and it is every shape this pass has been measured on; a third ranged axis multiplies
/// the search and the remainder loops for a reuse no program here has. Each operand in @p fully_sliced needs a block on one of its axes,
/// since at a point it would have none left.
template <typename Fits, typename Cost, typename Price>
std::optional<BlockChoice> choose_blocks(std::vector<std::size_t> const &extents, std::set<int> const &reusable,
                                         std::vector<std::set<int>> const &fully_sliced, Fits &&fits, Cost &&cost, Price &&price) {
    std::vector<std::size_t> const        axes(reusable.begin(), reusable.end());
    std::vector<std::vector<std::size_t>> options(extents.size());
    for (std::size_t const k : axes) {
        options[k] = balanced_blocks(extents[k]);
    }

    std::optional<BlockChoice> best;
    auto const                 consider = [&](std::vector<std::size_t> const &blocks) {
        if (std::ranges::all_of(blocks, [](std::size_t b) { return b == 1; })) {
            return;
        }
        for (auto const &operand : fully_sliced) {
            if (std::ranges::none_of(operand, [&](int k) { return blocks[static_cast<std::size_t>(k)] > 1; })) {
                return;
            }
        }
        BlockChoice choice{.blocks = blocks};
        cost(blocks, choice.largest, choice.held, choice.traffic, choice.contracted);
        if (!fits(choice.largest, choice.held)) {
            return;
        }
        choice.price_us = price(blocks, choice.traffic - choice.contracted);
        if (!best.has_value() || better(choice, *best)) {
            best = std::move(choice);
        }
    };

    std::vector<std::size_t> blocks(extents.size(), 1);
    for (std::size_t x = 0; x < axes.size(); ++x) {
        for (std::size_t const bx : options[axes[x]]) {
            blocks[axes[x]] = bx;
            consider(blocks);
            for (std::size_t y = x + 1; y < axes.size(); ++y) {
                for (std::size_t const by : options[axes[y]]) {
                    blocks[axes[y]] = by;
                    consider(blocks);
                }
                blocks[axes[y]] = 1;
            }
        }
        blocks[axes[x]] = 1;
    }
    return best;
}

/// The loops a plan emits, each as the walk it takes over every sliced axis.
///
/// A grouped plan is one loop over every point. A range plan cuts each ranged axis into its full
/// blocks and, when the block does not divide the axis, one remainder; the loops are every
/// combination of those, so a body is declared at one fixed shape and the remainder needs no
/// guard inside it.
std::vector<std::vector<AxisWalk>> loop_walks(Plan const &plan);

/// One loop of a triangle: the walk over each of the two axes, which of each walk's positions
/// every iteration is at, and whether its tiles are off the diagonal and so write a mirror.
struct TriangleLoop {
    std::vector<AxisWalk>                 walks;
    std::vector<std::vector<std::size_t>> table;
    bool                                  mirror{false};
};

/// The loops a triangle of tiles takes over two axes of extent @p extent cut in blocks of @p block.
///
/// With @c k full blocks and a remainder @c r, the tiles on and above the diagonal are the @c k
/// full diagonal ones, the @c k(k-1)/2 full ones above them, the @c k full-by-remainder ones in the
/// last column, and the remainder's own diagonal tile: up to four loops, each at one fixed shape,
/// as the grid emission has.
std::vector<TriangleLoop> triangle_loops(std::size_t extent, std::size_t block);

/// The cost model this pass prices triangles with, built once per process.
DeviceProfile const &pricing_profile();

/// What the cost model says one run of @p plan's statements costs at the tile shape @p length
/// (one entry per sliced axis), in its microseconds, plus the mirror's two passes over a scratch
/// tile when @p mirror is set.
///
/// Each contraction is priced as the engine would run it on the tile's own layout: a tile whose
/// output interleaves its operands' letters is scattered in runs its block sets, and its operands
/// are packed once per tile, so a small block pays for more than its arithmetic. The sliced axes
/// @p point marks (none when it is empty) are taken a point at a time.
double tile_price(Graph const &graph, Plan const &plan, std::vector<std::size_t> const &length, bool mirror, DeviceProfile const &profile,
                  std::vector<bool> const &point = {});

/// The predicted time of a range plan cutting the sliced axes into @p plan's blocks, in the cost
/// model's microseconds: every loop's tiles at their own shapes, each iteration's step, and the
/// @p unpriced bytes the contractions' prices leave out (the other statements' traffic and the
/// storage's surcharge) at memory bandwidth.
double range_price(Graph const &graph, Plan const &plan, long double unpriced, DeviceProfile const &profile);

/// The predicted time of a grouped plan, in the cost model's microseconds: each slice's statements
/// at a point of every sliced axis, called once per member of a chunk, each iteration's step, and
/// the @p unpriced bytes at memory bandwidth.
double grouped_price(Graph const &graph, Plan const &plan, long double unpriced, DeviceProfile const &profile);

/// The chunking of a plan in words, for the report and the explanation.
std::string describe_chunks(Plan const &plan);

/// Every subset of @p rank positions of size 1 to @ref kMaxSlicedAxes, in a fixed order.
std::vector<std::vector<std::size_t>> candidate_sets(std::size_t rank);

/// The dtype dispatch, and the only place the pass names an element type.
bool emit_tiled_loop(Graph &graph, Plan const &plan);

/// Whether @p lhs is the better of two plans that fit. A triangle beats every plan that runs all its
/// tiles, since it does about half the arithmetic whatever the bytes; otherwise the lower predicted
/// time wins, and equal prices fall to the fewer bytes streamed.
bool preferred(Plan const &lhs, Plan const &rhs);

/// The triangle a two-axis range plan can be, or the reason it cannot.
struct TriangleVerdict {
    std::optional<Plan::Triangle> triangle;
    Reason const                 *reason{nullptr};
};

/// Whether @p plan's region writes one caller's tensor whose contribution is unchanged when the two
/// sliced axes are exchanged, together with some exchange of the output's other axes.
TriangleVerdict find_triangle(Graph const &graph, Plan const &plan, std::function<bool(TensorId)> const &backed,
                              EscapeAnalysis const &escape);

/// The intermediates worth a region for its triangle alone, largest first: every operand of an
/// einsum that a sliceable statement writes, when what that einsum adds to its output is provably
/// unchanged by exchanging two of the output's axes, together with some exchange of the others.
/// That is the question @ref find_triangle asks of the region, asked here of the einsum alone so a
/// program with nothing to stream is read once per einsum rather than once per candidate region.
/// The two axes need an extent of three or more, since a triangle cuts them into two or more
/// blocks of at least two.
///
/// @pre The graph's node list is in dependency order.
std::vector<std::pair<std::size_t, TensorId>> symmetric_seeds(Graph const &graph, Scope const &scope, std::set<TensorId> const &declined);

EINSUMS_NAMESPACE_END(compute_graph::passes::axis_tiling)
