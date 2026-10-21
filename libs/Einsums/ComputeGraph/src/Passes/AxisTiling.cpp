//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

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

EINSUMS_NAMESPACE_BEGIN(compute_graph::passes)

namespace {

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
/// barrier for a narrower reason recorded in the header: it names no index letters, so there is
/// nothing to propagate a labelling through, and @ref OpKind::ElementTransform for another:
/// what it applies is a callable the node carries, and a rewrite that re-emits the operation has
/// nothing to re-emit it WITH short of a registry name the anonymous arm does not have.
bool tileable_kind(OpKind kind) {
    switch (kind) {
    case OpKind::Einsum:
    case OpKind::Permute:
    case OpKind::Axpby:
    case OpKind::Scale:
    case OpKind::DirectProduct:
    case OpKind::DirectDivision:
    case OpKind::Dot:
        return true;
    default:
        return false;
    }
}

/// Whether @p node is one a registered descriptor lets this pass slice: both hooks, and a node
/// that is the same computation at every slice because it reads nothing it does not list and does
/// nothing beyond what it writes.
bool registered_sliceable(Node const &node) {
    DescriptorHooks const *hooks = descriptor_hooks(node);
    if (hooks == nullptr || !hooks->axes || !hooks->slice) {
        return false;
    }
    NodeEffects const effects = effects_of(node);
    return effects.deterministic && !effects.external_effects;
}

/// The index letters a registered node names its operands with, the output first.
std::optional<std::vector<std::vector<std::string>>> registered_letters(Node const &node) {
    return descriptor_hooks(node)->axes(node.op_data, node);
}

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
std::optional<std::vector<SlotRef>> slots_of(Graph const &graph, Node const &node) {
    auto handle_rank = [&graph](TensorId tid) -> std::optional<std::size_t> {
        auto const *handle = graph.find_tensor(tid);
        if (handle == nullptr) {
            return std::nullopt;
        }
        return handle->rank;
    };

    std::vector<TensorId> wanted;
    switch (node.kind) {
    case OpKind::Einsum:
    case OpKind::DirectProduct:
    case OpKind::DirectDivision:
    case OpKind::Dot:
        if (node.outputs.size() != 1 || node.inputs.size() < 2) {
            return std::nullopt;
        }
        wanted = {node.outputs[0], node.inputs[0], node.inputs[1]};
        break;
    case OpKind::Permute:
    case OpKind::Axpby:
        if (node.outputs.size() != 1 || node.inputs.empty()) {
            return std::nullopt;
        }
        wanted = {node.outputs[0], node.inputs[0]};
        break;
    case OpKind::Scale:
        if (node.outputs.size() != 1) {
            return std::nullopt;
        }
        wanted = {node.outputs[0]};
        break;
    default:
        if (!registered_sliceable(node) || node.outputs.size() != 1) {
            return std::nullopt;
        }
        // The destination, then the operands it reads; a destination re-listed because the node
        // reads it is the first slot already.
        wanted = {node.outputs[0]};
        for (TensorId const tid : operand_inputs(node)) {
            wanted.push_back(tid);
        }
        break;
    }

    std::vector<SlotRef> slots;
    slots.reserve(wanted.size());
    for (TensorId const tid : wanted) {
        auto const rank = handle_rank(tid);
        if (!rank.has_value()) {
            return std::nullopt;
        }
        slots.push_back(SlotRef{.tid = tid, .rank = *rank});
    }
    return slots;
}

/// The index lists a contraction or a permutation names its operands with, slot by slot.
///
/// Read from the LIVE index state where the node carries one, because a pass that rewrote the
/// letters wrote them there and the descriptor's own copy is the at-capture snapshot.
std::optional<std::vector<std::vector<std::string>>> letters_of(Node const &node) {
    if (registered_sliceable(node)) {
        return registered_letters(node);
    }
    auto const lists = node_index_lists(node);
    if (!lists) {
        return std::nullopt;
    }
    if (lists->b == nullptr) {
        return std::vector<std::vector<std::string>>{*lists->c, *lists->a};
    }
    return std::vector<std::vector<std::string>>{*lists->c, *lists->a, *lists->b};
}

/// The letters a node's permutation operators name. An axis one of them names cannot be sliced:
/// the operator sums over its permutations, and at one slice of an exchanged axis the body would
/// need the slice it is not at.
std::vector<std::string> operator_letters_of(Node const &node) {
    std::vector<std::string> out;
    if (auto const lists = node_index_lists(node); lists && lists->operators != nullptr) {
        for (auto const &op : *lists->operators) {
            auto const letters = op.letters();
            out.insert(out.end(), letters.begin(), letters.end());
        }
    }
    return out;
}

/// The link (summed) letters of a contraction, empty for every other kind.
std::vector<std::string> link_letters_of(Node const &node) {
    // A registered node sums the letters an input names and its output does not.
    if (registered_sliceable(node)) {
        std::vector<std::string> summed;
        if (auto const letters = registered_letters(node); letters && !letters->empty()) {
            auto const &output = letters->front();
            for (std::size_t s = 1; s < letters->size(); ++s) {
                for (auto const &letter : (*letters)[s]) {
                    if (std::ranges::find(output, letter) == output.end() && std::ranges::find(summed, letter) == summed.end()) {
                        summed.push_back(letter);
                    }
                }
            }
        }
        return summed;
    }
    if (node.kind != OpKind::Einsum) {
        return {};
    }
    auto const *desc = node.op_data.get_if<EinsumDescriptor>();
    if (desc == nullptr) {
        return {};
    }
    return live_index_lists(*desc).link;
}

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

    /// The cost model's price of a triangle plan, in its own microseconds.
    double price_us{0.0};
};

/// Unify two label vectors position by position, failing on a genuine disagreement.
bool unify(std::vector<int> &lhs, std::vector<int> &rhs, bool &changed) {
    if (lhs.size() != rhs.size()) {
        return false;
    }
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        if (lhs[i] == rhs[i]) {
            continue;
        }
        if (lhs[i] == kFree) {
            lhs[i]  = rhs[i];
            changed = true;
        } else if (rhs[i] == kFree) {
            rhs[i]  = lhs[i];
            changed = true;
        } else {
            return false;
        }
    }
    return true;
}

/// A labelling no position of any tensor carries twice, which is what a slice variable
/// appearing on two axes of one tensor would mean: a diagonal read the loop cannot express.
bool labels_are_distinct(std::vector<int> const &labels) {
    for (std::size_t i = 0; i < labels.size(); ++i) {
        if (labels[i] == kFree) {
            continue;
        }
        for (std::size_t j = i + 1; j < labels.size(); ++j) {
            if (labels[i] == labels[j]) {
                return false;
            }
        }
    }
    return true;
}

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
    /// together, and the traffic the schedule streams.
    void cost(Labelling const &labelling, std::vector<std::size_t> const &extents, std::size_t depth, std::size_t &largest,
              std::size_t &held, long double &traffic) const;

    /// The largest intermediate, the bytes every block buffer of the loops holds together, and the
    /// traffic of a plan that cuts each sliced axis into blocks of @p blocks (one for a point),
    /// remainders included.
    ///
    /// Each loop of the plan declares its own buffers at its own extents and every one of them is
    /// allocated for the whole replay, so the held bytes count a remainder loop's buffers beside
    /// the full block's.
    void range_cost(Labelling const &labelling, std::vector<std::size_t> const &extents, std::vector<std::size_t> const &blocks,
                    std::size_t &largest, std::size_t &held, long double &traffic) const;

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

bool RegionAnalysis::collect() {
    auto const &nodes = _graph.nodes();
    _slots.clear();
    _written.clear();
    _written_set.clear();
    _largest_written_bytes = 0;
    if (!_seed_given) {
        _seed = 0;
    }

    for (std::size_t const i : _members) {
        auto slots = slots_of(_graph, nodes[i]);
        if (!slots.has_value()) {
            return false;
        }
        // A tiled operand carries no single buffer, so its "rank" says nothing about a slice.
        for (auto const &slot : *slots) {
            auto const *handle = _graph.find_tensor(slot.tid);
            if (handle == nullptr || handle->is_tiled || handle->is_distributed) {
                return false;
            }
            // The rewrite reaches the tensor OBJECT at optimize time, where every other pass
            // reaches only the handle, so a caller whose tensor has already been destroyed gets
            // a decline here rather than a dereference of freed storage. Execute-time
            // validation reports the same thing later; this is the same check asked earlier.
            if (handle->validator && !handle->validator()) {
                return false;
            }
            // The rewrite emits through the capture API, which is templated on the operand's
            // STATIC type, so a rank-erased operand is one it can name and a statically ranked
            // one is not.
            _all_runtime = _all_runtime && handle->is_runtime;
            if (_dtype == packed_gemm::ScalarType::Unknown) {
                _dtype = handle->dtype;
            } else if (handle->dtype != packed_gemm::ScalarType::Unknown && handle->dtype != _dtype) {
                return false;
            }
        }
        // A contraction or a permutation whose descriptor is the tiled one names no letters.
        if ((nodes[i].kind == OpKind::Einsum || nodes[i].kind == OpKind::Permute) && !letters_of(nodes[i]).has_value()) {
            return false;
        }
        TensorId const destination = (*slots)[0].tid;
        if (_written_set.insert(destination).second) {
            _written.push_back(destination);
        }
        _slots.push_back(std::move(*slots));
    }

    // What a node OUTSIDE the run touches, so an intermediate the body would re-declare at
    // slice extents is refused when something downstream still wants the whole of it.
    //
    // EFFECTIVE io rather than the node's own lists, because a control-flow node carries none:
    // a loop whose body reads the intermediate lists nothing at all, and a scan of the raw
    // lists would have declared it unread and re-declared it at one slice underneath a reader
    // that wanted the whole of it. The region fuzz found exactly that.
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        if (_ordinal.contains(i)) {
            continue;
        }
        auto const [reads, writes] = const_cast<Graph &>(_graph).effective_io(nodes[i]);
        for (TensorId const tid : reads) {
            _touched_outside.insert(_graph.buffer_of(tid));
        }
        for (TensorId const tid : writes) {
            _touched_outside.insert(_graph.buffer_of(tid));
        }
    }

    for (TensorId const tid : _written) {
        auto const *handle = _graph.find_tensor(tid);
        if (handle == nullptr) {
            continue;
        }
        std::size_t const bytes = handle->total_bytes();
        if (bytes > _largest_written_bytes || (bytes == _largest_written_bytes && _largest_written_bytes == 0)) {
            _largest_written_bytes = bytes;
            if (!_seed_given) {
                _seed = tid;
            }
        }
    }
    return _seed != 0 && _written_set.contains(_seed);
}

std::vector<int> *RegionAnalysis::labels_for(LabelStore &store, std::size_t node_index, std::size_t slot, SlotRef const &ref) const {
    if (_written_set.contains(ref.tid)) {
        auto [it, inserted] = store.tensor.try_emplace(ref.tid, std::vector<int>(ref.rank, kFree));
        return &it->second;
    }
    auto [it, inserted] = store.use.try_emplace(std::pair{node_index, slot}, std::vector<int>(ref.rank, kFree));
    return &it->second;
}

Labelling RegionAnalysis::propagate(std::vector<std::size_t> const &seed_positions) const {
    Labelling   out;
    auto const &nodes = _graph.nodes();

    auto const *seed_handle = _graph.find_tensor(_seed);
    if (seed_handle == nullptr) {
        out.feasible = false;
        return out;
    }
    std::vector<int> seed_labels(seed_handle->rank, kFree);
    for (std::size_t k = 0; k < seed_positions.size(); ++k) {
        seed_labels[seed_positions[k]] = static_cast<int>(k);
    }
    out.store.tensor[_seed] = std::move(seed_labels);

    // A fixpoint over the region. Each sweep can only replace a free position with a label, so
    // the number of sweeps is bounded by the number of positions; the bound below is that,
    // stated generously, and a sweep that changes nothing ends it.
    std::size_t const sweeps = _members.size() * (kMaxSlicedAxes + 1) + 4;
    for (std::size_t sweep = 0; sweep < sweeps; ++sweep) {
        bool changed = false;
        for (std::size_t const i : _members) {
            auto const                     &node  = nodes[i];
            auto const                     &slots = slots_at(i);
            std::vector<std::vector<int> *> labels;
            labels.reserve(slots.size());
            for (std::size_t s = 0; s < slots.size(); ++s) {
                labels.push_back(labels_for(out.store, i, s, slots[s]));
            }

            if (node.kind == OpKind::Einsum || node.kind == OpKind::Permute || registered_sliceable(node)) {
                char const *const conflict =
                    node.kind == OpKind::Permute
                        ? "a permutation exchanges two of the candidate's sliced axes, so the body would need a slice of a tensor it "
                          "is not at"
                        : "a contraction relates two of the candidate's sliced axes through one index letter";
                auto const letters = letters_of(node);
                if (!letters.has_value() || letters->size() < slots.size()) {
                    out.feasible = false;
                    out.reason   = "a node of the run names index lists this pass cannot read";
                    return out;
                }
                std::unordered_map<std::string, int> by_letter;
                for (std::size_t s = 0; s < slots.size(); ++s) {
                    if ((*letters)[s].size() != labels[s]->size()) {
                        out.feasible = false;
                        return out;
                    }
                    for (std::size_t p = 0; p < labels[s]->size(); ++p) {
                        int const label = (*labels[s])[p];
                        if (label == kFree) {
                            continue;
                        }
                        auto [it, inserted] = by_letter.try_emplace((*letters)[s][p], label);
                        if (!inserted && it->second != label) {
                            out.feasible = false;
                            out.reason   = conflict;
                            return out;
                        }
                    }
                }
                for (auto const &letter : operator_letters_of(node)) {
                    if (by_letter.contains(letter)) {
                        out.feasible = false;
                        out.reason = "a permutation operator exchanges one of the candidate's sliced axes, so the body would need a slice "
                                     "it is not at";
                        return out;
                    }
                }
                // Slicing a summed letter cuts the contraction in half. An einsum's halves add, and
                // closes() decides whether its destination can carry the sum across the loop; any
                // other kind's would cost a partial-sum buffer the schedule was supposed to save.
                for (auto const &link : node.kind == OpKind::Einsum ? std::vector<std::string>{} : link_letters_of(node)) {
                    if (by_letter.contains(link)) {
                        out.feasible = false;
                        out.reason   = "a contraction sums over one of the candidate's sliced axes, which would cut it in half";
                        return out;
                    }
                }
                for (std::size_t s = 0; s < slots.size(); ++s) {
                    for (std::size_t p = 0; p < labels[s]->size(); ++p) {
                        auto const hit = by_letter.find((*letters)[s][p]);
                        if (hit == by_letter.end()) {
                            continue;
                        }
                        if ((*labels[s])[p] == kFree) {
                            (*labels[s])[p] = hit->second;
                            changed         = true;
                        } else if ((*labels[s])[p] != hit->second) {
                            out.feasible = false;
                            out.reason   = conflict;
                            return out;
                        }
                    }
                }
            } else if (node.kind == OpKind::Dot) {
                // The two operands align position by position; the scalar carries no label,
                // which is what makes it the reduction the loop accumulates.
                if (slots.size() != 3 || !unify(*labels[1], *labels[2], changed)) {
                    out.feasible = false;
                    out.reason   = "the reduction's two operands do not carry the candidate's sliced axes in the same places";
                    return out;
                }
                if (std::ranges::any_of(*labels[0], [](int v) { return v != kFree; })) {
                    out.feasible = false;
                    out.reason   = "the reduction's destination carries a sliced axis, so it is not the accumulation the loop makes";
                    return out;
                }
            } else {
                for (std::size_t s = 1; s < slots.size(); ++s) {
                    if (!unify(*labels[0], *labels[s], changed)) {
                        out.feasible = false;
                        out.reason   = "an elementwise operation relates two of the candidate's sliced axes to one position";
                        return out;
                    }
                }
            }
        }
        if (!changed) {
            break;
        }
    }

    // Every tensor carries each slice variable at most once, and the extents of one variable
    // agree wherever it appears.
    std::vector<std::size_t> extents(seed_positions.size(), 0);
    auto                     check = [&](TensorId tid, std::vector<int> const &labels) {
        if (!labels_are_distinct(labels)) {
            return false;
        }
        auto const *handle = _graph.find_tensor(tid);
        if (handle == nullptr) {
            return false;
        }
        for (std::size_t p = 0; p < labels.size(); ++p) {
            if (labels[p] == kFree) {
                continue;
            }
            std::size_t const extent = p < handle->dims.size() ? handle->dims[p] : 0;
            auto             &slot   = extents[static_cast<std::size_t>(labels[p])];
            if (slot == 0) {
                slot = extent;
            } else if (slot != extent) {
                return false;
            }
        }
        return true;
    };
    for (auto const &[tid, labels] : out.store.tensor) {
        if (!check(tid, labels)) {
            out.feasible = false;
            out.reason   = "one tensor would carry a sliced axis twice, or two of its uses disagree about that axis's extent";
            return out;
        }
    }
    for (auto const &[key, labels] : out.store.use) {
        auto const &slots = slots_at(key.first);
        if (!check(slots[key.second].tid, labels)) {
            out.feasible = false;
            out.reason   = "one tensor would carry a sliced axis twice, or two of its uses disagree about that axis's extent";
            return out;
        }
    }
    for (std::size_t const extent : extents) {
        if (extent < 2) {
            // An axis of one slice is not a schedule, and an axis nothing carries is a
            // candidate the propagation dissolved.
            out.feasible = false;
            out.reason   = "one of the candidate's axes spans a single slice, so cutting it is not a schedule";
            return out;
        }
    }

    // A point slice drops its axis, so an operand every one of whose axes is sliced has nothing
    // left at a point. Whether a block keeps one of them is decided later, with the blocks.
    auto const record = [&out](std::vector<int> const &labels) {
        if (labels.empty() || std::ranges::any_of(labels, [](int v) { return v == kFree; })) {
            return;
        }
        std::set<int> const carried(labels.begin(), labels.end());
        if (std::ranges::find(out.fully_sliced, carried) == out.fully_sliced.end()) {
            out.fully_sliced.push_back(carried);
        }
    };
    for (auto const &[tid, labels] : out.store.tensor) {
        record(labels);
    }
    for (auto const &[key, labels] : out.store.use) {
        record(labels);
    }
    return out;
}

namespace {

/// @brief Whether a node ADDS to its destination rather than overwriting it.
///
/// The question a loop asks of every statement it carries, since a statement run once per slice
/// of an axis it does not vary over is recomputed harmlessly when it overwrites and counted
/// again when it adds. A kind this cannot read is treated as adding, which declines a schedule
/// rather than risking one.
///
/// @param[in] node The node.
/// @return True when the destination's prior contents survive the call.
bool node_accumulates(Node const &node) {
    switch (node.kind) {
    case OpKind::Einsum: {
        auto const *desc = node.op_data.get_if<EinsumDescriptor>();
        return desc == nullptr || !is_zero(live_c_prefactor(*desc));
    }
    case OpKind::Permute: {
        auto const *desc = node.op_data.get_if<PermuteDescriptor>();
        return desc == nullptr || !is_zero(desc->params ? desc->params->beta : PrefactorScalar{desc->beta});
    }
    case OpKind::Axpby: {
        auto const *desc = node.op_data.get_if<AxpbyDescriptor>();
        return desc == nullptr || !is_zero(live_beta(*desc));
    }
    case OpKind::DirectProduct:
    case OpKind::DirectDivision: {
        auto const *desc = node.op_data.get_if<ElementwiseBinaryDescriptor>();
        return desc == nullptr || !is_zero(live_beta(*desc));
    }
    case OpKind::Dot:
        return false; // it writes element zero; the LOOP is what accumulates it
    case OpKind::Custom:
        // A registered node overwrites only when its destination hook says so.
        return !pure_overwrite(node);
    default:
        return true;
    }
}

} // namespace

long double RegionAnalysis::storage_surcharge(TensorId tid, std::vector<std::size_t> const &extents, std::vector<bool> const &whole_axis,
                                              long double box_bytes, long double transfers) const {
    auto const backing = _scope.backing_of(tid);
    if (backing == nullptr || transfers <= 0.0L) {
        return 0.0L;
    }
    // The runs in the order the tensor will be stored in: the one SlabStreaming chooses when it
    // still may, which makes this box as few runs as it can be, and the stored one otherwise.
    std::size_t runs = 0;
    if (backing->order_open && backing->order_open()) {
        runs = detail::fewest_runs(extents, whole_axis);
    } else {
        std::vector<std::size_t> order = detail::declared_order(extents.size());
        if (backing->stored_order) {
            if (auto stored = backing->stored_order(); stored.size() == extents.size()) {
                order = std::move(stored);
            }
        }
        runs = detail::runs_in(order, extents, whole_axis);
    }
    long double const weight = detail::storage_byte_weight();
    return transfers * ((weight - 1.0L) * box_bytes + static_cast<long double>(runs) * detail::storage_run_bytes());
}

bool RegionAnalysis::sums_sliced(Labelling const &labelling, std::size_t index) const {
    auto const &node = _graph.nodes()[index];
    if (node.kind != OpKind::Einsum) {
        return false;
    }
    auto const letters = letters_of(node);
    if (!letters.has_value() || letters->empty()) {
        return false;
    }
    auto const &output = letters->front();
    for (std::size_t slot = 1; slot < letters->size() && slot < slots_at(index).size(); ++slot) {
        std::vector<int> const *labels = slot_labels(labelling, index, slot);
        if (labels == nullptr) {
            continue;
        }
        for (std::size_t p = 0; p < labels->size() && p < (*letters)[slot].size(); ++p) {
            if ((*labels)[p] != kFree && std::ranges::find(output, (*letters)[slot][p]) == output.end()) {
                return true;
            }
        }
    }
    return false;
}

bool RegionAnalysis::closes(Labelling const &labelling, std::string &accumulator, TensorId &accumulator_id,
                            std::vector<std::pair<TensorId, PrefactorScalar>> &summed_into, std::string &reason) const {
    auto const &nodes = _graph.nodes();
    accumulator.clear();
    accumulator_id = 0;
    summed_into.clear();

    // A contraction that sums a sliced axis adds one block's share per iteration, so its
    // destination must keep none of the sliced axes: one that kept another would have its
    // prefactor applied once per block of the summed one.
    for (std::size_t const i : _members) {
        if (!sums_sliced(labelling, i)) {
            continue;
        }
        std::vector<int> const *destination = slot_labels(labelling, i, 0);
        if (destination != nullptr && std::ranges::any_of(*destination, [](int v) { return v != kFree; })) {
            reason = "a contraction sums over one of the candidate's sliced axes, which would cut it in half, into a destination that "
                     "keeps another of them";
            return false;
        }
    }

    for (TensorId const tid : _written) {
        auto const hit = labelling.store.tensor.find(tid);
        if (hit == labelling.store.tensor.end()) {
            continue;
        }
        bool const labelled = std::ranges::any_of(hit->second, [](int v) { return v != kFree; });
        if (labelled) {
            continue;
        }
        auto const *handle = _graph.find_tensor(tid);
        if (handle == nullptr) {
            reason = "a tensor the region writes has no handle";
            return false;
        }
        // Two unlabelled destinations a loop can live with. The first is a statement none of
        // whose operands carries a sliced axis: it is loop-invariant, and it stays outside the
        // loop rather than being computed identically on every iteration.
        bool invariant = true;
        for (std::size_t const i : _members) {
            if (slots_at(i)[0].tid != tid) {
                continue;
            }
            invariant = invariant && !node_is_sliced(labelling, i);
        }
        if (invariant) {
            continue;
        }

        // The second is a reduction: written once, by a reduction to a scalar over operands
        // that do carry sliced axes, and read by nothing else inside the region.
        std::size_t writers      = 0;
        std::size_t readers      = 0;
        std::size_t writer_index = 0;
        for (std::size_t const i : _members) {
            auto const &slots = slots_at(i);
            if (slots[0].tid == tid) {
                ++writers;
                writer_index = i;
            }
            for (std::size_t s = 1; s < slots.size(); ++s) {
                if (slots[s].tid == tid) {
                    ++readers;
                }
            }
        }
        // A contraction that sums a sliced axis: each iteration adds its block's share, the
        // destination scaled by the contraction's prefactor once, before the loop. Only for a
        // destination the region does not also read, since a reader inside the loop would see a
        // partial sum.
        if (sums_sliced(labelling, writer_index)) {
            auto const *desc = nodes[writer_index].op_data.get_if<EinsumDescriptor>();
            if (writers != 1 || readers != 0 || desc == nullptr) {
                reason = "a contraction sums over one of the candidate's sliced axes, which would cut it in half, into a tensor the "
                         "region also reads or writes";
                return false;
            }
            // A backed destination would have to be held whole in memory to collect the sum, which
            // is what backing it exists to avoid.
            if (_scope.backed(tid)) {
                reason = "a contraction sums over one of the candidate's sliced axes, which would cut it in half, into a backed tensor, "
                         "which would be held whole to collect the sum";
                return false;
            }
            summed_into.emplace_back(tid, live_c_prefactor(*desc));
            continue;
        }
        if (writers != 1 || readers != 0 || nodes[writer_index].kind != OpKind::Dot) {
            reason = "a tensor the region writes carries no sliced axis and is not the reduction the loop would accumulate";
            return false;
        }
        if (!accumulator.empty()) {
            reason = "the region reduces into more than one scalar, so there is no single accumulation to carry";
            return false;
        }
        // A reduction writes element zero of its destination and the accumulation adds the
        // whole of it, so the two are the same statement only for a one-element tensor.
        if (handle->total_bytes() != handle->element_size) {
            reason = "the reduction's destination holds more than one element, so accumulating the whole of it is not what the "
                     "reduction wrote";
            return false;
        }
        accumulator    = handle->name;
        accumulator_id = tid;
    }

    // Every node that ADDS to its destination has to carry every sliced axis. One carrying none
    // of them is loop-invariant and stays outside the loop, which is the first case above; one
    // carrying some but not all runs once per slice of an axis it never varied over, and an
    // overwrite recomputed identically is only waste where an accumulation adds its contribution
    // again on every one of those iterations. The loop-carried reduction counts as an
    // accumulation whatever its own node does, since the loop is what adds it up: a reduction
    // over a tensor carrying one of two sliced axes is the shape that found this, and it read
    // exactly twice its own value.
    std::set<int> sliced;
    for (auto const &[tid, labels] : labelling.store.tensor) {
        for (int const label : labels) {
            if (label != kFree) {
                sliced.insert(label);
            }
        }
    }
    for (auto const &[site, labels] : labelling.store.use) {
        for (int const label : labels) {
            if (label != kFree) {
                sliced.insert(label);
            }
        }
    }
    for (std::size_t const i : _members) {
        bool const adds = node_accumulates(nodes[i]) || (accumulator_id != 0 && slots_at(i)[0].tid == accumulator_id) ||
                          std::ranges::any_of(summed_into, [&](auto const &entry) { return entry.first == slots_at(i)[0].tid; });
        if (!adds) {
            continue;
        }
        std::set<int> const carried = node_labels(labelling, i);
        if (!carried.empty() && carried != sliced) {
            reason = "a statement accumulates over some of the sliced axes and not all of them, so the loop would add its contribution "
                     "once per slice of an axis it does not vary over";
            return false;
        }
    }
    return true;
}

void RegionAnalysis::cost(Labelling const &labelling, std::vector<std::size_t> const &extents, std::size_t depth, std::size_t &largest,
                          std::size_t &held, long double &traffic) const {
    // A chunk holds @p depth SLICES, so its cost is one slice times the depth however many
    // axes the slice is cut on. Multiplying by the depth once per sliced axis would charge a
    // chunk of two pairs for four, which is a square where the schedule is linear.
    auto sliced_bytes = [&](TensorId tid, std::vector<int> const &labels) -> long double {
        auto const *handle = _graph.find_tensor(tid);
        if (handle == nullptr) {
            return 0.0L;
        }
        long double bytes  = static_cast<long double>(handle->element_size);
        bool        sliced = false;
        for (std::size_t p = 0; p < handle->dims.size(); ++p) {
            if (p < labels.size() && labels[p] != kFree) {
                sliced = true;
                continue;
            }
            bytes *= static_cast<long double>(handle->dims[p]);
        }
        return sliced ? bytes * static_cast<long double>(depth) : bytes;
    };

    largest = 0;
    held    = 0;
    for (auto const &[tid, labels] : labelling.store.tensor) {
        auto const *handle = _graph.find_tensor(tid);
        if (handle == nullptr || !held_per_slice(tid)) {
            continue;
        }
        auto const bytes = static_cast<std::size_t>(sliced_bytes(tid, labels));
        largest          = std::max(largest, bytes);
        held += bytes;
    }

    // Traffic is what the whole schedule streams: every read, once per iteration, times the
    // iteration count. Minimising it under the cap is the memory-constrained fusion problem;
    // minimising it alone would always prefer slicing nothing at all.
    long double per_iteration = 0.0L;
    for (std::size_t const i : _members) {
        auto const &slots = slots_at(i);
        for (std::size_t s = 0; s < slots.size(); ++s) {
            auto const *labels = _written_set.contains(slots[s].tid)
                                     ? (labelling.store.tensor.contains(slots[s].tid) ? &labelling.store.tensor.at(slots[s].tid) : nullptr)
                                     : (labelling.store.use.contains({i, s}) ? &labelling.store.use.at({i, s}) : nullptr);
            static std::vector<int> const empty;
            per_iteration += sliced_bytes(slots[s].tid, labels == nullptr ? empty : *labels);
        }
    }
    long double slices = 1.0L;
    for (std::size_t const extent : extents) {
        slices *= static_cast<long double>(extent);
    }
    long double const iterations = slices / static_cast<long double>(depth);
    traffic                      = per_iteration * iterations;

    // A backed operand is moved to or from its storage, one slice of it per member of a chunk: a
    // point of every sliced axis it carries, the whole of every other.
    for (std::size_t const i : _members) {
        auto const &slots = slots_at(i);
        for (std::size_t s = 0; s < slots.size(); ++s) {
            auto const *handle = _graph.find_tensor(slots[s].tid);
            if (handle == nullptr) {
                continue;
            }
            std::vector<int> const  *labels = slot_labels(labelling, i, s);
            std::vector<std::size_t> box;
            std::vector<bool>        whole;
            long double              bytes = static_cast<long double>(handle->element_size);
            for (std::size_t p = 0; p < handle->dims.size(); ++p) {
                bool const sliced = labels != nullptr && p < labels->size() && (*labels)[p] != kFree;
                box.push_back(sliced ? 1 : handle->dims[p]);
                whole.push_back(!sliced || handle->dims[p] == 1);
                bytes *= static_cast<long double>(box.back());
            }
            traffic += storage_surcharge(slots[s].tid, box, whole, bytes, iterations * static_cast<long double>(depth));
        }
    }
}

void RegionAnalysis::describe(std::vector<std::size_t> const &seed_positions, std::vector<std::string> &names,
                              std::vector<std::string> &letters) const {
    auto const *handle = _graph.find_tensor(_seed);
    names.clear();
    letters.assign(seed_positions.size(), std::string{});
    for (std::size_t const position : seed_positions) {
        names.push_back(fmt::format("{}[{}]", handle == nullptr ? std::string{"?"} : handle->name, position));
    }
    if (handle == nullptr) {
        return;
    }

    // A letter is what a reader recognises the axis by, and the writer of the seed is where one
    // is named. A seed no contraction writes leaves the letters empty rather than inventing one.
    auto const &nodes = _graph.nodes();
    for (std::size_t const i : _members) {
        auto const &slots = slots_at(i);
        if (slots[0].tid != _seed) {
            continue;
        }
        auto const spelled = letters_of(nodes[i]);
        if (!spelled.has_value() || (*spelled)[0].size() != handle->rank) {
            continue;
        }
        for (std::size_t k = 0; k < seed_positions.size(); ++k) {
            letters[k] = (*spelled)[0][seed_positions[k]];
        }
        break;
    }
}

void RegionAnalysis::partition(Labelling const &labelling, std::vector<std::string> &streamed, std::vector<std::string> &whole) const {
    streamed.clear();
    whole.clear();
    std::unordered_set<TensorId> seen;
    for (std::size_t const i : _members) {
        for (auto const &slot : slots_at(i)) {
            if (!seen.insert(slot.tid).second) {
                continue;
            }
            auto const *handle = _graph.find_tensor(slot.tid);
            if (handle == nullptr) {
                continue;
            }
            bool labelled = false;
            if (auto const hit = labelling.store.tensor.find(slot.tid); hit != labelling.store.tensor.end()) {
                labelled = std::ranges::any_of(hit->second, [](int v) { return v != kFree; });
            }
            if (!labelled) {
                for (auto const &[key, labels] : labelling.store.use) {
                    if (slots_at(key.first)[key.second].tid != slot.tid) {
                        continue;
                    }
                    labelled = labelled || std::ranges::any_of(labels, [](int v) { return v != kFree; });
                }
            }
            (labelled ? streamed : whole).push_back(handle->name);
        }
    }
}

std::vector<int> const *RegionAnalysis::slot_labels(Labelling const &labelling, std::size_t index, std::size_t slot) const {
    TensorId const tid = slots_at(index)[slot].tid;
    if (_written_set.contains(tid)) {
        auto const hit = labelling.store.tensor.find(tid);
        return hit == labelling.store.tensor.end() ? nullptr : &hit->second;
    }
    auto const hit = labelling.store.use.find({index, slot});
    return hit == labelling.store.use.end() ? nullptr : &hit->second;
}

std::set<int> RegionAnalysis::node_labels(Labelling const &labelling, std::size_t index) const {
    std::set<int> carried;
    auto const   &slots = slots_at(index);
    for (std::size_t slot = 0; slot < slots.size(); ++slot) {
        std::vector<int> const *labels = slot_labels(labelling, index, slot);
        if (labels == nullptr) {
            continue;
        }
        for (int const label : *labels) {
            if (label != kFree) {
                carried.insert(label);
            }
        }
    }
    return carried;
}

bool RegionAnalysis::node_is_sliced(Labelling const &labelling, std::size_t index) const {
    return !node_labels(labelling, index).empty();
}

std::set<int> RegionAnalysis::reusable_labels(Labelling const &labelling) const {
    auto const   &nodes = _graph.nodes();
    std::set<int> out;
    for (std::size_t const i : _members) {
        if (nodes[i].kind != OpKind::Einsum) {
            continue;
        }
        std::set<int> const carried = node_labels(labelling, i);
        for (std::size_t slot = 1; slot < slots_at(i).size(); ++slot) {
            std::vector<int> const *labels = slot_labels(labelling, i, slot);
            for (int const label : carried) {
                if (labels == nullptr || std::ranges::find(*labels, label) == labels->end()) {
                    out.insert(label);
                }
            }
        }
    }
    return out;
}

std::string_view RegionAnalysis::range_blocker(Labelling const &labelling) const {
    auto const &nodes = _graph.nodes();
    for (std::size_t const i : _members) {
        if (registered_sliceable(nodes[i]) && node_is_sliced(labelling, i)) {
            return "a registered node's slice hook takes one point of an axis, not a block of it, so the loop keeps grouped chunks";
        }
    }
    return {};
}

void RegionAnalysis::range_cost(Labelling const &labelling, std::vector<std::size_t> const &extents, std::vector<std::size_t> const &blocks,
                                std::size_t &largest, std::size_t &held, long double &traffic) const {
    // One tensor under the plan: the bytes one block of it holds, and the bytes every iteration
    // together reads of it. An axis the tensor carries is read once in all, block by block; an
    // axis it does not carry repeats the read once per block of that axis, which is the reuse a
    // larger block buys.
    struct Measure {
        long double block{0.0L};
        long double streamed{0.0L};
        long double every_loop{0.0L};
    };
    auto const measure = [&](TensorId tid, std::vector<int> const *labels) -> Measure {
        auto const *handle = _graph.find_tensor(tid);
        if (handle == nullptr) {
            return {};
        }
        auto              free = static_cast<long double>(handle->element_size);
        long double       kept = 1.0L;
        long double       all  = 1.0L;
        std::vector<bool> carries(extents.size(), false);
        for (std::size_t p = 0; p < handle->dims.size(); ++p) {
            int const label = (labels != nullptr && p < labels->size()) ? (*labels)[p] : kFree;
            if (label == kFree) {
                free *= static_cast<long double>(handle->dims[p]);
                continue;
            }
            auto const k = static_cast<std::size_t>(label);
            carries[k]   = true;
            kept *= static_cast<long double>(blocks[k]);
            all *= static_cast<long double>(extents[k]);
        }
        long double repeats = 1.0L;
        // One buffer per loop: along an axis with a remainder, a tensor carrying the axis has a
        // full-block buffer and a remainder one, and a tensor that does not has one per loop.
        long double every_loop = free;
        for (std::size_t k = 0; k < extents.size(); ++k) {
            std::size_t const remainder = extents[k] % blocks[k];
            if (carries[k]) {
                every_loop *= static_cast<long double>(blocks[k] + remainder);
            } else if (remainder != 0) {
                every_loop *= 2.0L;
            }
            if (!carries[k]) {
                std::size_t const pieces = (extents[k] + blocks[k] - 1) / blocks[k];
                repeats *= static_cast<long double>(pieces);
            }
        }
        return {.block = free * kept, .streamed = free * all * repeats, .every_loop = every_loop};
    };

    largest = 0;
    held    = 0;
    for (auto const &[tid, labels] : labelling.store.tensor) {
        auto const *handle = _graph.find_tensor(tid);
        if (handle == nullptr || !held_per_slice(tid)) {
            continue;
        }
        Measure const m = measure(tid, &labels);
        largest         = std::max(largest, static_cast<std::size_t>(m.block));
        held += static_cast<std::size_t>(m.every_loop);
    }

    // A statement carrying no sliced axis stays outside the loop and runs once.
    traffic = 0.0L;
    for (std::size_t const i : _members) {
        bool const  inside = node_is_sliced(labelling, i);
        auto const &slots  = slots_at(i);
        for (std::size_t s = 0; s < slots.size(); ++s) {
            std::vector<int> const *labels = inside ? slot_labels(labelling, i, s) : nullptr;
            Measure const           m      = measure(slots[s].tid, labels);
            traffic += inside ? m.streamed : m.block;

            // A backed operand is moved to or from its storage a block at a time: the block of
            // every sliced axis it carries, the whole of every other, as many times as the
            // streamed bytes are blocks.
            auto const *handle = _graph.find_tensor(slots[s].tid);
            if (handle == nullptr || m.block <= 0.0L) {
                continue;
            }
            std::vector<std::size_t> box;
            std::vector<bool>        whole;
            for (std::size_t p = 0; p < handle->dims.size(); ++p) {
                int const label = (labels != nullptr && p < labels->size()) ? (*labels)[p] : kFree;
                box.push_back(label == kFree ? handle->dims[p] : blocks[static_cast<std::size_t>(label)]);
                whole.push_back(box.back() == handle->dims[p]);
            }
            traffic += storage_surcharge(slots[s].tid, box, whole, m.block, inside ? m.streamed / m.block : 1.0L);
        }
    }
}

std::size_t RegionAnalysis::streamed_whole_bytes(Labelling const &labelling) const {
    std::size_t total = 0;
    for (auto const &[tid, labels] : labelling.store.tensor) {
        auto const *handle = _graph.find_tensor(tid);
        if (handle != nullptr && _scope.intermediate(tid)) {
            total += handle->total_bytes();
        }
    }
    return total;
}

bool RegionAnalysis::build(Labelling const &labelling, Plan &plan, std::string &reason) const {
    auto const &nodes = _graph.nodes();
    plan.ops.clear();
    plan.body_owned.clear();
    plan.dtype = _dtype;

    auto labels_of = [&](std::size_t node_index, std::size_t slot) -> std::vector<int> {
        auto const &ref = slots_at(node_index)[slot];
        if (_written_set.contains(ref.tid)) {
            auto const hit = labelling.store.tensor.find(ref.tid);
            return hit == labelling.store.tensor.end() ? std::vector<int>(ref.rank, kFree) : hit->second;
        }
        auto const hit = labelling.store.use.find({node_index, slot});
        return hit == labelling.store.use.end() ? std::vector<int>(ref.rank, kFree) : hit->second;
    };

    plan.replaced.clear();
    for (std::size_t const i : _members) {
        if (!node_is_sliced(labelling, i)) {
            continue;
        }
        NodePlan op;
        op.id      = nodes[i].id;
        op.kind    = nodes[i].kind;
        op.op_data = nodes[i].op_data;
        op.label   = nodes[i].label;
        for (std::size_t slot = 0; slot < slots_at(i).size(); ++slot) {
            op.slots.push_back(SlotPlan{.tid = slots_at(i)[slot].tid, .labels = labels_of(i, slot)});
        }
        // A registered node is re-emitted through its slice hook, so one that cannot be sliced
        // on these axes declines the candidate here rather than at emission.
        if (registered_sliceable(nodes[i]) && !descriptor_hooks(nodes[i])->slice(op.op_data, slice_request(op, [](int label) {
                                                                                     return fmt::format("axtile:{}:0", label);
                                                                                 }))) {
            reason = "a registered node's slice hook cannot slice it on the candidate's axes";
            return false;
        }
        plan.ops.push_back(std::move(op));
        plan.replaced.push_back(i);
    }
    if (plan.ops.empty()) {
        reason = "no statement of the run carries a sliced axis, so there is nothing for a loop to hold";
        return false;
    }

    for (TensorId const tid : _written) {
        auto const hit = labelling.store.tensor.find(tid);
        if (hit == labelling.store.tensor.end()) {
            continue;
        }
        if (!std::ranges::any_of(hit->second, [](int v) { return v != kFree; })) {
            continue;
        }
        auto const *handle = _graph.find_tensor(tid);
        if (handle == nullptr) {
            reason = "a tensor the region writes has no handle";
            return false;
        }
        // A caller's tensor keeps its own buffer and is written through a slice of it, and so does
        // an intermediate something outside the region still reads or writes whole. Only an
        // intermediate the region alone touches is re-declared at slice extents.
        if (!_scope.intermediate(tid) || _touched_outside.contains(_graph.buffer_of(tid)) || _scope.touched_beyond(tid)) {
            continue;
        }
        plan.body_owned.emplace_back(tid, hit->second);
    }
    return true;
}

/// The largest chunk that fits the cap and divides the slice count.
///
/// Divides, because a grouped node's member count is fixed when the body is captured: a ragged
/// last chunk would need a second body, and a schedule with two bodies is two schedules.
std::size_t chunk_depth(std::size_t slices, std::size_t bytes_per_slice, std::size_t cap) {
    if (bytes_per_slice == 0) {
        return slices;
    }
    std::size_t const ceiling = std::max<std::size_t>(1, std::min<std::size_t>(slices, cap / bytes_per_slice));
    for (std::size_t depth = ceiling; depth > 1; --depth) {
        if (slices % depth == 0) {
            return depth;
        }
    }
    return 1;
}

/// The block extents worth trying on an axis of @p extent: for every block count, the smallest
/// block that reaches it.
///
/// Two blocks that cut the axis into the same number of pieces stream the same bytes, since what
/// a block saves is the number of times the operands it does not carry are re-read. The smaller
/// of the two holds less and leaves a remainder no larger, so it is the only one worth costing:
/// 21 in pieces of at most 8 is three blocks of 7 rather than 8, 8 and 5.
std::vector<std::size_t> balanced_blocks(std::size_t extent) {
    std::vector<std::size_t> out;
    for (std::size_t count = 1; count <= extent; ++count) {
        std::size_t const block = (extent + count - 1) / count;
        if (out.empty() || out.back() != block) {
            out.push_back(block);
        }
    }
    return out;
}

/// One costed choice of block extents.
struct BlockChoice {
    std::vector<std::size_t> blocks;
    std::size_t              largest{0};
    std::size_t              held{0};
    long double              traffic{0.0L};

    [[nodiscard]] std::size_t ranged() const {
        return static_cast<std::size_t>(std::ranges::count_if(blocks, [](std::size_t b) { return b > 1; }));
    }
};

/// Whether @p lhs is the better of two block choices: fewer bytes streamed, then fewer ranged
/// axes, then the smaller footprint, then blocks on the inner axes rather than the outer ones,
/// so a tie is broken the same way in every process.
bool better(BlockChoice const &lhs, BlockChoice const &rhs) {
    if (lhs.traffic != rhs.traffic) {
        return lhs.traffic < rhs.traffic;
    }
    if (lhs.ranged() != rhs.ranged()) {
        return lhs.ranged() < rhs.ranged();
    }
    if (lhs.largest != rhs.largest) {
        return lhs.largest < rhs.largest;
    }
    return std::ranges::lexicographical_compare(std::views::reverse(rhs.blocks), std::views::reverse(lhs.blocks));
}

/// The block extents that stream the fewest bytes among those @p fits accepts, or nothing when no
/// choice that cuts some axis into blocks fits.
///
/// Only the axes in @p reusable are cut, and at most two of them at once; every other axis stays a
/// point. Two is the blocking a matrix product has, one block per operand the other does not
/// carry, and it is every shape this pass has been measured on; a third ranged axis multiplies
/// the search and the remainder loops for a reuse no program here has. Each operand in @p fully_sliced needs a block on one of its axes,
/// since at a point it would have none left.
template <typename Fits, typename Cost>
std::optional<BlockChoice> choose_blocks(std::vector<std::size_t> const &extents, std::set<int> const &reusable,
                                         std::vector<std::set<int>> const &fully_sliced, Fits &&fits, Cost &&cost) {
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
        cost(blocks, choice.largest, choice.held, choice.traffic);
        if (!fits(choice.largest, choice.held)) {
            return;
        }
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
std::vector<std::vector<AxisWalk>> loop_walks(Plan const &plan) {
    std::vector<std::vector<AxisWalk>> per_axis(plan.extents.size());
    for (std::size_t k = 0; k < plan.extents.size(); ++k) {
        per_axis[k] = axis_walks(plan.extents[k], plan.range ? plan.blocks[k] : 1);
    }
    return combine_walks(per_axis);
}

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
std::vector<TriangleLoop> triangle_loops(std::size_t extent, std::size_t block) {
    std::size_t const full = extent / block;
    std::size_t const rest = extent % block;
    AxisWalk const    whole{.count = full, .step = block, .length = block, .range = true};
    AxisWalk const    tail{.count = 1, .step = block, .offset = full * block, .length = rest, .range = true};

    std::vector<TriangleLoop> out;
    if (full > 0) {
        TriangleLoop diagonal{.walks = {whole, whole}};
        for (std::size_t b = 0; b < full; ++b) {
            diagonal.table.push_back({b, b});
        }
        out.push_back(std::move(diagonal));
    }
    if (full > 1) {
        TriangleLoop above{.walks = {whole, whole}, .mirror = true};
        for (std::size_t b = 0; b < full; ++b) {
            for (std::size_t c = b + 1; c < full; ++c) {
                above.table.push_back({b, c});
            }
        }
        out.push_back(std::move(above));
    }
    if (full > 0 && rest > 0) {
        TriangleLoop column{.walks = {whole, tail}, .mirror = true};
        for (std::size_t b = 0; b < full; ++b) {
            column.table.push_back({b, 0});
        }
        out.push_back(std::move(column));
    }
    if (rest > 0) {
        out.push_back(TriangleLoop{.walks = {tail, tail}, .table = {{0, 0}}});
    }
    return out;
}

/// The cost model this pass prices triangles with, built once per process.
DeviceProfile const &pricing_profile() {
    static CostModel const model = CostModel::detect_default();
    return model.device(Target::CPU);
}

/// The arithmetic-only price of a contraction the packed engine has no plan for (a repeated or
/// broadcast letter): its letters sorted into batch, M, N and K at the tile lengths @p length.
double gemm_price(Graph const &graph, NodePlan const &op, EinsumIndexLists const &lists, std::vector<std::size_t> const &length,
                  DeviceProfile const &profile) {
    std::map<std::string, std::size_t> extent;
    auto const                         learn = [&](std::vector<std::string> const &letters, SlotPlan const &slot) {
        auto const &dims = graph.tensor(slot.tid).dims;
        for (std::size_t p = 0; p < letters.size() && p < dims.size(); ++p) {
            bool const sliced  = p < slot.labels.size() && slot.labels[p] != kFree;
            extent[letters[p]] = sliced ? length[static_cast<std::size_t>(slot.labels[p])] : dims[p];
        }
    };
    learn(lists.c, op.slots[0]);
    learn(lists.a, op.slots[1]);
    learn(lists.b, op.slots[2]);
    auto const in = [](std::vector<std::string> const &letters, std::string const &l) {
        return std::ranges::find(letters, l) != letters.end();
    };
    std::size_t m = 1, n = 1, k = 1, batch = 1;
    for (auto const &[letter, e] : extent) {
        bool const c = in(lists.c, letter), a = in(lists.a, letter), b = in(lists.b, letter);
        if (a && b && c) {
            batch *= e;
        } else if (a && b) {
            k *= e;
        } else if (a) {
            m *= e;
        } else if (b) {
            n *= e;
        }
    }
    return static_cast<double>(batch) * profile.estimate_gemm_time_us(m, n, k);
}

/// One operand of a statement as a tile at the lengths @p length (one per sliced axis) reads it:
/// its @p letters, the tile's extent at each position, and the strides the engine sees. A slice of
/// a tensor keeps that tensor's strides. With @p dense set - an intermediate the body re-declares
/// at slice extents, or the scratch tile a mirrored tile is computed into - the tile is its own
/// buffer, laid out densely in the order of the tensor's strides.
packed_gemm::OperandLayout tile_layout(Graph const &graph, SlotPlan const &slot, std::vector<std::string> const &letters,
                                       std::vector<std::size_t> const &length, bool dense) {
    auto const                &handle = graph.tensor(slot.tid);
    packed_gemm::OperandLayout out;
    out.indices = letters;
    for (std::size_t p = 0; p < letters.size(); ++p) {
        bool const sliced = p < slot.labels.size() && slot.labels[p] != kFree;
        out.dims.push_back(static_cast<int64_t>(sliced ? length[static_cast<std::size_t>(slot.labels[p])] : handle.dims[p]));
    }
    bool const has_strides = handle.strides.size() == letters.size();
    if (!dense && has_strides) {
        for (std::size_t const s : handle.strides) {
            out.strides.push_back(static_cast<int64_t>(s));
        }
        return out;
    }
    std::vector<std::size_t> order(letters.size());
    std::iota(order.begin(), order.end(), std::size_t{0});
    if (has_strides) {
        std::ranges::stable_sort(order, {}, [&](std::size_t p) { return handle.strides[p]; });
    }
    out.strides.assign(letters.size(), 0);
    int64_t step = 1;
    for (std::size_t const p : order) {
        out.strides[p] = step;
        step *= out.dims[p];
    }
    return out;
}

/// What the cost model says one run of @p plan's statements costs at the tile shape @p length
/// (one entry per sliced axis), in its microseconds, plus the mirror's two passes over a scratch
/// tile when @p mirror is set.
///
/// Each contraction is priced as the engine would run it on the tile's own layout: a tile whose
/// output interleaves its operands' letters is scattered in runs its block sets, and its operands
/// are packed once per tile, so a small block pays for more than its arithmetic.
double tile_price(Graph const &graph, Plan const &plan, std::vector<std::size_t> const &length, bool mirror, DeviceProfile const &profile) {
    double     total = 0.0;
    auto const owned = [&](TensorId tid) {
        return std::ranges::any_of(plan.body_owned, [&](auto const &entry) { return entry.first == tid; });
    };
    for (std::size_t o = 0; o < plan.ops.size(); ++o) {
        auto const &op = plan.ops[o];
        if (op.kind != OpKind::Einsum) {
            continue;
        }
        auto const lists   = live_index_lists(op.op_data.get<EinsumDescriptor>());
        bool const scratch = mirror && plan.triangle.has_value() && o == plan.triangle->writer;
        auto const shape =
            packed_gemm::describe_packed_contraction(tile_layout(graph, op.slots[0], lists.c, length, scratch || owned(op.slots[0].tid)),
                                                     tile_layout(graph, op.slots[1], lists.a, length, owned(op.slots[1].tid)),
                                                     tile_layout(graph, op.slots[2], lists.b, length, owned(op.slots[2].tid)), plan.dtype);
        if (shape.valid) {
            total += profile.estimate_contraction_time_us(shape, 1, graph.tensor(op.slots[0].tid).element_size);
        } else {
            total += gemm_price(graph, op, lists, length, profile);
        }
        if (scratch) {
            std::size_t elements = 1;
            auto const &dims     = graph.tensor(op.slots[0].tid).dims;
            for (std::size_t p = 0; p < dims.size(); ++p) {
                bool const sliced = p < op.slots[0].labels.size() && op.slots[0].labels[p] != kFree;
                elements *= sliced ? length[static_cast<std::size_t>(op.slots[0].labels[p])] : dims[p];
            }
            // The direct add and the mirrored permute each read the scratch and read and write a block.
            total += profile.estimate_memory_time_us(6 * elements * graph.tensor(op.slots[0].tid).element_size);
        }
    }
    return total;
}

/// The chunking of a plan in words, for the report and the explanation.
std::string describe_chunks(Plan const &plan) {
    if (plan.triangle.has_value()) {
        return fmt::format("a triangle of blocks of {} in {} loop(s) of {} iteration(s) in all", fmt::join(plan.blocks, " x "),
                           triangle_loops(plan.extents[0], plan.blocks[0]).size(), plan.iterations);
    }
    if (plan.range) {
        return fmt::format("blocks of {} in {} loop(s) of {} iteration(s) in all", fmt::join(plan.blocks, " x "), loop_walks(plan).size(),
                           plan.iterations);
    }
    return fmt::format("chunks of {}, {} iteration(s)", plan.depth, plan.iterations);
}

/// Every subset of @p rank positions of size 1 to @ref kMaxSlicedAxes, in a fixed order.
std::vector<std::vector<std::size_t>> candidate_sets(std::size_t rank) {
    std::vector<std::vector<std::size_t>> out;
    if (rank == 0 || rank > 20) {
        return out;
    }
    for (std::size_t mask = 1; mask < (std::size_t{1} << rank); ++mask) {
        std::vector<std::size_t> positions;
        for (std::size_t p = 0; p < rank; ++p) {
            if ((mask & (std::size_t{1} << p)) != 0) {
                positions.push_back(p);
            }
        }
        if (positions.size() <= kMaxSlicedAxes) {
            out.push_back(std::move(positions));
        }
    }
    // Fewest axes first, then by position, so a tie between two equally cheap sets is broken
    // the same way in every process.
    std::ranges::stable_sort(
        out, [](auto const &lhs, auto const &rhs) { return lhs.size() != rhs.size() ? lhs.size() < rhs.size() : lhs < rhs; });
    return out;
}

/// Scale @p handle's tensor by @p factor in the PARENT, before the loop: the destination prefactor a
/// contraction applied once, which the loop's additions must not apply again.
template <typename T>
void scale_in_parent(Graph &graph, TensorHandle const &handle, T factor) {
    CaptureGuard const guard(graph);
    if (handle.is_tensor_view) {
        scale(factor, static_cast<RuntimeTensorView<T> *>(handle.live_ptr()));
    } else {
        scale(factor, static_cast<RuntimeTensor<T> *>(handle.live_ptr()));
    }
}

/// Zero the accumulator in the PARENT, before the loop.
///
/// The reduction the region wrote OVERWROTE its destination, and the loop accumulates into it,
/// so the two agree only if the destination starts at zero on every replay. A scale by zero
/// assigns rather than multiplies, which is what keeps a destination holding a non-finite value
/// from carrying it into the sum.
template <typename T>
void zero_accumulator(Graph &graph, TensorHandle const &handle) {
    scale_in_parent(graph, handle, T{0});
}

/// The index letters a slot keeps once its sliced axes have become loop variables.
///
/// An axis taken a point at a time is DROPPED from the operand rather than kept as an axis of
/// extent one. Both spell the same arithmetic, and the drop is what makes the body's
/// contractions the shapes a kernel recognises: at one occupied pair the four-index contraction
/// is an ordinary matrix product over the auxiliary index, which is what the hand-written pair
/// loop performs, where a rank-four contraction with two unit axes would reach the generic
/// algorithm instead. An axis cut into blocks keeps its letter, at the block's extent.
std::vector<std::string> surviving(std::vector<std::string> const &letters, std::vector<int> const &labels,
                                   std::vector<AxisWalk> const &walks) {
    std::vector<std::string> out;
    out.reserve(letters.size());
    for (std::size_t p = 0; p < letters.size(); ++p) {
        if (p < labels.size() && labels[p] != kFree && !walks[static_cast<std::size_t>(labels[p])].range) {
            continue;
        }
        out.push_back(letters[p]);
    }
    return out;
}

/// The transposes a contraction over two matrices and one summed letter needs, when it is one.
struct GemmShape {
    bool trans_a{false};
    bool trans_b{false};
};

/// Whether a reduced contraction is a plain matrix product, and with which transposes.
std::optional<GemmShape> gemm_shape(std::vector<std::string> const &c, std::vector<std::string> const &a, std::vector<std::string> const &b,
                                    std::vector<std::string> const &link) {
    if (c.size() != 2 || a.size() != 2 || b.size() != 2 || link.size() != 1) {
        return std::nullopt;
    }
    std::string const &k = link[0];
    GemmShape          shape;
    if (a[0] == k && a[1] == c[0]) {
        shape.trans_a = true;
    } else if (a[1] == k && a[0] == c[0]) {
        shape.trans_a = false;
    } else {
        return std::nullopt;
    }
    if (b[0] == k && b[1] == c[1]) {
        shape.trans_b = false;
    } else if (b[1] == k && b[0] == c[1]) {
        shape.trans_b = true;
    } else {
        return std::nullopt;
    }
    return shape;
}

/// Whether an operand is one a vendor GEMM can read: a column-major matrix, whose first axis
/// steps by one element.
///
/// The other grouped kernels do not ask, because they call the same per-member routine the
/// ungrouped form calls and that routine reads strides. A batched GEMM does not: it hands BLAS
/// a leading dimension and a base pointer, so an operand whose first axis steps by more than one
/// is not a matrix it can describe. Dropping a MIDDLE axis of a three-index tensor leaves
/// exactly that, and the region fuzz found it as a wrong answer rather than as a refusal.
template <typename T>
bool gemm_readable(RuntimeTensorView<T> const &view) {
    return view.rank() < 2 || view.stride(0) == 1;
}

/// A prefactor a grouped kernel can take, which is a real number however the tensor is typed.
std::optional<double> real_prefactor(PrefactorScalar const &value) {
    if (!is_real_valued(value)) {
        return std::nullopt;
    }
    return as_real<double>(value);
}

/// Build the loop body: the captured algebra at one slice, a chunk of slices at a time.
///
/// Every operand the body reads or writes is a VIEW. A caller's tensor is viewed at the slice
/// the member is at, or in full when it carries no sliced axis, so what the body touches is the
/// caller's own buffer and a rebind at a new geometry follows it. An intermediate carrying a
/// sliced axis is re-declared on the body at slice extents and viewed in full, which is what
/// makes every operand one type and the emission one function rather than a dispatch over which
/// of them happen to be views.
///
/// A chunk of more than one slice is emitted as the GROUPED form of the same body: the members
/// of a chunk are independent, so one node per family per chunk replaces one node per family per
/// member wherever a grouped kernel exists for that family. The accumulation is the exception
/// and is grouped anyway, because a grouped accumulation runs its entries sequentially against a
/// repeated destination, which is the same summation order the members were captured in.
///
/// A range plan emits one body per loop of @ref loop_walks, each at depth one: an axis cut into
/// blocks is read through a RANGE view and kept in every spec at the block's extent, so one
/// contraction covers the block, and @p tag keeps the parameters of one loop apart from another's.
///
/// A triangle's loops pass @p table, the block positions each iteration visits, in place of the
/// grid, and a loop over tiles off the diagonal sets @p mirror: its contraction writes a scratch
/// tile, which is added to the tile's own block of the output and, exchanged, to the mirrored one.
template <typename T>
void emit_body(Graph &parent, Graph &body, Plan const &plan, std::vector<AxisWalk> const &walks, std::string const &tag,
               std::vector<std::vector<std::size_t>> const *table = nullptr, bool mirror = false) {
    std::size_t const depth = plan.range ? 1 : std::max<std::size_t>(1, plan.depth);

    // ── Declarations, before the guard ──────────────────────────────────────
    // A declaration is not a capture, and a buffer the resource phase is to place has to exist
    // before the nodes that read it do.
    std::map<std::pair<TensorId, std::size_t>, RuntimeTensor<T> *> owned;
    for (auto const &[tid, labels] : plan.body_owned) {
        auto const              &handle = parent.tensor(tid);
        std::vector<std::size_t> dims;
        dims.reserve(handle.dims.size());
        for (std::size_t p = 0; p < handle.dims.size(); ++p) {
            if (p < labels.size() && labels[p] != kFree) {
                if (auto const &walk = walks[static_cast<std::size_t>(labels[p])]; walk.range) {
                    dims.push_back(walk.length);
                }
                continue;
            }
            dims.push_back(handle.dims[p]);
        }
        for (std::size_t member = 0; member < depth; ++member) {
            owned[{tid, member}] = &body.declare_runtime_tensor<T>(fmt::format("{}#{}", handle.name, member), dims, /*intermediate=*/true);
        }
    }
    // A mirrored tile's contribution, at the shape of the tile's block of the output.
    RuntimeTensor<T> *scratch = nullptr;
    if (mirror) {
        auto const              &out    = plan.ops[plan.triangle->writer].slots[0];
        auto const              &handle = parent.tensor(out.tid);
        std::vector<std::size_t> dims;
        dims.reserve(handle.dims.size());
        for (std::size_t p = 0; p < handle.dims.size(); ++p) {
            bool const sliced = p < out.labels.size() && out.labels[p] != kFree;
            dims.push_back(sliced ? walks[static_cast<std::size_t>(out.labels[p])].length : handle.dims[p]);
        }
        scratch = &body.declare_runtime_tensor<T>("axtile_mirror", dims, /*intermediate=*/true);
    }
    std::vector<RuntimeTensor<T> *> partials(depth, nullptr);
    if (plan.accumulator_id != 0) {
        for (std::size_t member = 0; member < depth; ++member) {
            partials[member] = &body.declare_runtime_tensor<T>(fmt::format("axtile_partial#{}", member), std::vector<std::size_t>{1},
                                                               /*intermediate=*/true);
        }
    }

    CaptureGuard const guard(body);

    // ── The slice index, one parameter per axis per member ──────────────────
    auto const  bounds   = table != nullptr ? declare_table_bounds(body, walks, *table, "axtile", tag)
                                            : declare_walk_bounds(body, depth, walks, "axtile", tag);
    auto const &index_of = bounds.start;
    auto const &end_of   = bounds.end;

    // ── The operands ────────────────────────────────────────────────────────
    std::map<std::tuple<TensorId, std::vector<int>, std::size_t>, RuntimeTensorView<T> *> views;

    auto axes_for = [&](std::vector<int> const &labels, std::size_t rank, std::size_t member) {
        std::vector<ViewAxis> out;
        out.reserve(rank);
        for (std::size_t p = 0; p < rank; ++p) {
            if (p < labels.size() && labels[p] != kFree) {
                auto const k = static_cast<std::size_t>(labels[p]);
                out.push_back(walks[k].range ? ViewAxis::range(BoundExpr(index_of[member][k]), BoundExpr(end_of[member][k]))
                                             : ViewAxis::drop(BoundExpr(index_of[member][k])));
            } else {
                out.push_back(ViewAxis::full());
            }
        }
        return out;
    };

    auto view_of = [&](TensorId tid, std::vector<int> const &labels, std::size_t member) -> RuntimeTensorView<T> & {
        auto const key = std::tuple{tid, labels, member};
        if (auto const hit = views.find(key); hit != views.end()) {
            return *hit->second;
        }
        RuntimeTensorView<T> *made = nullptr;
        if (auto const local = owned.find({tid, member}); local != owned.end()) {
            made = &view_runtime(*local->second, std::vector<ViewAxis>(local->second->rank(), ViewAxis::full()));
        } else {
            // The live object, not the caller's: the caller may have dropped its tensor once it
            // was captured, and capturing over the stand-in still names the same operand.
            auto const &handle = parent.tensor(tid);
            auto        recipe = axes_for(labels, handle.rank, member);
            made = handle.is_tensor_view ? &view_runtime(*static_cast<RuntimeTensorView<T> *>(handle.live_ptr()), std::move(recipe))
                                         : &view_runtime(*static_cast<RuntimeTensor<T> *>(handle.live_ptr()), std::move(recipe));
        }
        views.emplace(key, made);
        return *made;
    };

    // ── The algebra, one family at a time ───────────────────────────────────
    bool const chunked = depth > 1;
    for (auto const &op : plan.ops) {
        bool const accumulating = plan.accumulator_id != 0 && op.slots[0].tid == plan.accumulator_id;
        // A contraction summing a sliced axis adds its block's share to a destination the parent
        // scaled once before the loop.
        bool const summing = std::ranges::any_of(plan.summed_into, [&](auto const &entry) { return entry.first == op.slots[0].tid; });

        // [slot][member]. Collected before anything is emitted, because a grouped node takes the
        // whole chunk at once and an ungrouped one takes it a member at a time.
        std::vector<std::vector<RuntimeTensorView<T> *>> operand(op.slots.size());
        for (std::size_t s = 0; s < op.slots.size(); ++s) {
            operand[s].reserve(depth);
            for (std::size_t member = 0; member < depth; ++member) {
                if (s == 0 && accumulating) {
                    operand[s].push_back(
                        &view_runtime(*partials[member], std::vector<ViewAxis>(partials[member]->rank(), ViewAxis::full())));
                } else {
                    operand[s].push_back(&view_of(op.slots[s].tid, op.slots[s].labels, member));
                }
            }
        }
        auto sources = [&](std::size_t slot) {
            std::vector<RuntimeTensorView<T> const *> out;
            out.reserve(depth);
            for (auto *view : operand[slot]) {
                out.push_back(view);
            }
            return out;
        };

        switch (op.kind) {
        case OpKind::Einsum: {
            auto const &desc  = op.op_data.get<EinsumDescriptor>();
            auto const  lists = live_index_lists(desc);
            auto const  c     = surviving(lists.c, op.slots[0].labels, walks);
            auto const  a     = surviving(lists.a, op.slots[1].labels, walks);
            auto const  b     = surviving(lists.b, op.slots[2].labels, walks);
            auto const  alpha = real_prefactor(live_ab_prefactor(desc));
            auto const  beta  = real_prefactor(live_c_prefactor(desc));
            if (mirror && &op == &plan.ops[plan.triangle->writer]) {
                // The tile's contribution once, then its two destinations. Both adds are permutes
                // with the contraction's own destination prefactor, so a prefactor of zero
                // overwrites each block exactly as the captured contraction did, and every element
                // of the output is written by exactly one of the two.
                auto &tile = view_runtime(*scratch, std::vector<ViewAxis>(scratch->rank(), ViewAxis::full()));
                compute_graph::detail::einsum_parsed(einsum_spec(a, b, c, live_conj_a(desc), live_conj_b(desc), lists.operators), T{0},
                                                     &tile, as<T>(live_ab_prefactor(desc)), *operand[1][0], *operand[2][0]);

                T const     beta  = as<T>(live_c_prefactor(desc));
                auto const &sigma = plan.triangle->sigma;
                compute_graph::detail::permute_parsed(permute_spec(c, c), beta, operand[0][0], T{1}, tile);

                // The mirrored block at q holds the contribution at the exchanged index, which is
                // sign times the tile at apply(sigma, q).
                std::vector<std::string> exchanged;
                exchanged.reserve(c.size());
                for (std::size_t p = 0; p < c.size(); ++p) {
                    exchanged.push_back(c[static_cast<std::size_t>(static_cast<unsigned char>(sigma.permutation[p]))]);
                }
                auto &mirrored = view_of(op.slots[0].tid, plan.triangle->mirror_labels, 0);
                compute_graph::detail::permute_parsed(permute_spec(c, exchanged), beta, &mirrored, T(static_cast<double>(sigma.sign)),
                                                      tile);
                break;
            }
            auto const as_gemm  = gemm_shape(c, a, b, lists.link);
            bool       readable = true;
            for (auto const &slot : operand) {
                for (auto const *view : slot) {
                    readable = readable && gemm_readable(*view);
                }
            }
            // A GEMM computes the one term; an operator's other terms need the per-member path. A
            // summing contraction's members all add into one destination, one after another.
            if (chunked && !summing && readable && as_gemm.has_value() && alpha.has_value() && beta.has_value() && !live_conj_a(desc) &&
                !live_conj_b(desc) && lists.operators.empty()) {
                grouped_batched_gemm(*alpha, sources(1), sources(2), *beta, operand[0], as_gemm->trans_a, as_gemm->trans_b);
                break;
            }
            auto const spec = einsum_spec(a, b, c, live_conj_a(desc), live_conj_b(desc), lists.operators);
            for (std::size_t member = 0; member < depth; ++member) {
                compute_graph::detail::einsum_parsed(spec, summing ? T{1} : as<T>(live_c_prefactor(desc)), operand[0][member],
                                                     as<T>(live_ab_prefactor(desc)), *operand[1][member], *operand[2][member]);
            }
            break;
        }
        case OpKind::Permute: {
            auto const &desc = op.op_data.get<PermuteDescriptor>();
            // A permutation keeps its scalars in the params block where every other kind does,
            // and its own snapshots are plain complex doubles, so the live block is read first
            // and the snapshot only stands in for a node that has none.
            PrefactorScalar const alpha = desc.params ? desc.params->alpha : PrefactorScalar{desc.alpha};
            PrefactorScalar const beta  = desc.params ? desc.params->beta : PrefactorScalar{desc.beta};
            auto const            spec  = permute_spec(surviving(desc.c_indices, op.slots[0].labels, walks),
                                                       surviving(desc.a_indices, op.slots[1].labels, walks), desc.operators);
            auto const            re_a  = real_prefactor(alpha);
            auto const            re_c  = real_prefactor(beta);
            if (chunked && re_a.has_value() && re_c.has_value() && desc.operators.empty()) {
                compute_graph::detail::grouped_permute_parsed(spec, operand[0], sources(1), std::vector<double>(depth, *re_c),
                                                              std::vector<double>(depth, *re_a));
                break;
            }
            for (std::size_t member = 0; member < depth; ++member) {
                compute_graph::detail::permute_parsed(spec, as<T>(beta), operand[0][member], as<T>(alpha), *operand[1][member]);
            }
            break;
        }
        case OpKind::Axpby: {
            auto const &desc  = op.op_data.get<AxpbyDescriptor>();
            auto const  alpha = real_prefactor(live_alpha(desc));
            auto const  beta  = real_prefactor(live_beta(desc));
            if (chunked && alpha.has_value() && beta.has_value()) {
                grouped_axpby(std::vector<double>(depth, *alpha), sources(1), std::vector<double>(depth, *beta), operand[0]);
                break;
            }
            for (std::size_t member = 0; member < depth; ++member) {
                axpby(as<T>(live_alpha(desc)), *operand[1][member], as<T>(live_beta(desc)), operand[0][member]);
            }
            break;
        }
        case OpKind::Scale: {
            auto const &desc = op.op_data.get<ScaleDescriptor>();
            for (std::size_t member = 0; member < depth; ++member) {
                scale(as<T>(live_factor(desc)), operand[0][member]);
            }
            break;
        }
        case OpKind::DirectProduct:
        case OpKind::DirectDivision: {
            auto const &desc  = op.op_data.get<ElementwiseBinaryDescriptor>();
            T const     alpha = as<T>(live_alpha(desc));
            T const     beta  = as<T>(live_beta(desc));
            if (chunked) {
                if (op.kind == OpKind::DirectProduct) {
                    grouped_direct_product(std::vector<T>(depth, alpha), sources(1), sources(2), std::vector<T>(depth, beta), operand[0]);
                } else {
                    grouped_direct_division(std::vector<T>(depth, alpha), sources(1), sources(2), std::vector<T>(depth, beta), operand[0]);
                }
                break;
            }
            for (std::size_t member = 0; member < depth; ++member) {
                if (op.kind == OpKind::DirectProduct) {
                    direct_product(alpha, *operand[1][member], *operand[2][member], beta, operand[0][member]);
                } else {
                    direct_division(alpha, *operand[1][member], *operand[2][member], beta, operand[0][member]);
                }
            }
            break;
        }
        case OpKind::Dot: {
            if (chunked) {
                grouped_dot(operand[0], sources(1), sources(2));
            } else {
                for (std::size_t member = 0; member < depth; ++member) {
                    dot_python(operand[0][member], *operand[1][member], *operand[2][member]);
                }
            }
            if (accumulating) {
                // The loop-carried accumulation. The destination is repeated, which is what
                // makes a grouped accumulation run its entries sequentially and sum the chunk
                // in the order its members were captured in.
                auto const           &handle = parent.tensor(plan.accumulator_id);
                RuntimeTensorView<T> &into   = view_of(plan.accumulator_id, std::vector<int>(handle.rank, kFree), 0);
                if (chunked) {
                    grouped_axpby(std::vector<double>(depth, 1.0), sources(0), std::vector<double>(depth, 1.0),
                                  std::vector<RuntimeTensorView<T> *>(depth, &into));
                } else {
                    axpby(T{1}, *operand[0][0], T{1}, &into);
                }
            }
            break;
        }
        case OpKind::Custom: {
            // A registered node, re-emitted per member through its slice hook: its descriptor at
            // this member's slice, over the members' views, read and written as any other kind's.
            DescriptorHooks const &hooks = find_descriptor_codec(op.op_data.name())->hooks;
            for (std::size_t member = 0; member < depth; ++member) {
                auto  sliced = hooks.slice(op.op_data,
                                           slice_request(op, [&](int label) { return index_of[member][static_cast<std::size_t>(label)]; }));
                auto &ctx    = CaptureContext::current();
                std::vector<TensorId> inputs;
                for (std::size_t s = 1; s < operand.size(); ++s) {
                    inputs.push_back(ctx.get_slot(*operand[s][member]).first);
                }
                std::vector<TensorId> const outputs{ctx.get_slot(*operand[0][member]).first};
                ctx.record_built(OpKind::Custom, chunked ? fmt::format("{}#{}", op.label, member) : op.label, plan.dtype,
                                 operand[0][member]->rank(), std::move(*sliced), std::span<TensorId const>{inputs},
                                 std::span<TensorId const>{outputs}, inputs, outputs);
            }
            break;
        }
        default:
            break;
        }
    }
}

/// Check, in the PARENT before the loop and on every replay, that @p handle's tensor still has the
/// symmetry @p desc it was declared with. A triangle computes half its tiles from that promise, so
/// a broken one is an error here rather than a wrong result later.
template <typename T>
void check_declared_symmetry(Graph &graph, TensorHandle const &handle, SymmetryDescriptor const &desc) {
    CaptureGuard const guard(graph);
    auto const         label = fmt::format("AxisTiling: check the declared symmetry of '{}'", handle.name);
    auto const         fail  = [name = handle.name]() {
        EINSUMS_THROW_EXCEPTION(std::runtime_error,
                                "AxisTiling: '{}' was declared symmetric, and a tiling that computes half of a contraction from that "
                                "declaration relies on it, but its values no longer have the declared symmetry",
                                name);
    };
    if (handle.is_tensor_view) {
        auto *tensor = static_cast<RuntimeTensorView<T> *>(handle.live_ptr());
        custom(label, std::tie(*tensor), std::tuple<>{}, [tensor, desc, fail]() {
            if (!check_symmetry(*tensor, desc)) {
                fail();
            }
        });
    } else {
        auto *tensor = static_cast<RuntimeTensor<T> *>(handle.live_ptr());
        custom(label, std::tie(*tensor), std::tuple<>{}, [tensor, desc, fail]() {
            if (!check_symmetry(*tensor, desc)) {
                fail();
            }
        });
    }
}

/// The dtype dispatch, and the only place the pass names an element type.
bool emit_tiled_loop(Graph &graph, Plan const &plan) {
    auto emit = [&]<typename T>() {
        std::size_t prologue = 0;
        if (plan.accumulator_id != 0) {
            zero_accumulator<T>(graph, graph.tensor(plan.accumulator_id));
            ++prologue;
        }
        // A destination a contraction sums a sliced axis into takes the contraction's prefactor
        // here, once; the loop then only adds. A scale by zero assigns, so a non-finite value the
        // destination held is not carried into the sum.
        for (auto const &[tid, prefactor] : plan.summed_into) {
            if (T const factor = as<T>(prefactor); factor != T{1}) {
                scale_in_parent(graph, graph.tensor(tid), factor);
                ++prologue;
            }
        }
        if (plan.triangle.has_value()) {
            for (auto const &[tid, desc] : plan.triangle->checks) {
                check_declared_symmetry<T>(graph, graph.tensor(tid), desc);
                ++prologue;
            }
        }
        // The zeroing was appended; it belongs where the region was, ahead of the loop. One
        // splice moves it there and erases the region in the same edit, so no writer can end up
        // behind a reader that survived.
        std::vector<bool> remove(graph.nodes().size(), false);
        std::vector<Node> moved;
        for (std::size_t k = remove.size() - prologue; k < remove.size(); ++k) {
            remove[k] = true;
            moved.push_back(graph.nodes()[k]);
        }
        for (std::size_t const position : plan.replaced) {
            remove[position] = true;
        }
        // At the site the analysis chose, which every statement in between allows: the last
        // replaced position when nothing between depends on an earlier statement, the first when
        // no later statement depends on anything between.
        std::size_t const site = plan.site;
        graph.replace_nodes(remove, {{site, std::move(moved)}});

        // Where that landed: the splice shifts an insert down by the erased nodes BELOW it. The
        // loops follow the zeroing, one after another, in the order @ref loop_walks lists them.
        auto const below = static_cast<std::size_t>(std::ranges::count_if(plan.replaced, [site](std::size_t p) { return p < site; }));
        std::size_t const position = (site - below) + prologue;
        if (plan.triangle.has_value()) {
            auto const loops = triangle_loops(plan.extents[0], plan.blocks[0]);
            for (std::size_t l = 0; l < loops.size(); ++l) {
                std::size_t const iterations = loops[l].table.size();
                Graph            &body =
                    graph.add_loop_at(fmt::format("AxisTiling#{}", l), iterations,
                                      PredExpr::iteration(CmpOp::Lt, BoundExpr(static_cast<std::int64_t>(iterations) - 1)), position + l);
                emit_body<T>(graph, body, plan, loops[l].walks, fmt::format("{}", l), &loops[l].table, loops[l].mirror);
            }
            return;
        }
        auto const loops = loop_walks(plan);
        for (std::size_t l = 0; l < loops.size(); ++l) {
            std::size_t const iterations = plan.range ? walk_iterations(loops[l]) : plan.iterations;
            Graph            &body =
                graph.add_loop_at(plan.range ? fmt::format("AxisTiling#{}", l) : std::string{"AxisTiling"}, iterations,
                                  PredExpr::iteration(CmpOp::Lt, BoundExpr(static_cast<std::int64_t>(iterations) - 1)), position + l);
            emit_body<T>(graph, body, plan, loops[l], plan.range ? fmt::format("{}", l) : std::string{});
        }
    };

    switch (plan.dtype) {
    case packed_gemm::ScalarType::Float32:
        emit.template operator()<float>();
        return true;
    case packed_gemm::ScalarType::Float64:
        emit.template operator()<double>();
        return true;
    case packed_gemm::ScalarType::Complex64:
        emit.template operator()<std::complex<float>>();
        return true;
    case packed_gemm::ScalarType::Complex128:
        emit.template operator()<std::complex<double>>();
        return true;
    default:
        return false;
    }
}

/// Whether @p lhs is the better of two plans that fit. A triangle beats every plan that runs all its
/// tiles, since it does about half the arithmetic whatever the bytes; two triangles compare by
/// price, and two grids by the bytes they stream.
bool preferred(Plan const &lhs, Plan const &rhs) {
    if (lhs.triangle.has_value() != rhs.triangle.has_value()) {
        return lhs.triangle.has_value();
    }
    if (lhs.triangle.has_value()) {
        return lhs.price_us < rhs.price_us;
    }
    return lhs.traffic < rhs.traffic;
}

/// Every way to pair up the positions in @p rest so that paired positions have equal extents, each
/// written as swaps into a copy of @p current, appended to @p out with the empty pairing first.
void exchanges(std::vector<std::size_t> const &rest, std::vector<std::size_t> const &dims, std::size_t from, SymmetryOp current,
               std::vector<bool> &taken, std::vector<SymmetryOp> &out) {
    while (from < rest.size() && taken[from]) {
        ++from;
    }
    if (from == rest.size()) {
        out.push_back(current);
        return;
    }
    taken[from] = true;
    exchanges(rest, dims, from + 1, current, taken, out); // This position stays where it is.
    for (std::size_t other = from + 1; other < rest.size(); ++other) {
        if (taken[other] || dims[rest[from]] != dims[rest[other]]) {
            continue;
        }
        taken[other]                    = true;
        auto paired                     = current;
        paired.permutation[rest[from]]  = static_cast<int8_t>(rest[other]);
        paired.permutation[rest[other]] = static_cast<int8_t>(rest[from]);
        exchanges(rest, dims, from + 1, paired, taken, out);
        taken[other] = false;
    }
    taken[from] = false;
}

/// The triangle a two-axis range plan can be, or the reason it cannot.
struct TriangleVerdict {
    std::optional<Plan::Triangle> triangle;
    Reason const                 *reason{nullptr};
};

/// Whether @p plan's region writes one caller's tensor whose contribution is unchanged when the two
/// sliced axes are exchanged, together with some exchange of the output's other axes.
TriangleVerdict find_triangle(Graph const &graph, Plan const &plan, std::function<bool(TensorId)> const &backed,
                              EscapeAnalysis const &escape) {
    // Axes of different extents cannot be exchanged at all; that is not a decision worth a line.
    if (plan.extents.size() != 2 || plan.extents[0] != plan.extents[1]) {
        return {};
    }
    if (plan.accumulator_id != 0 || !plan.summed_into.empty()) {
        return {.reason = &AxisTiling::TriangleShape};
    }
    std::set<TensorId> owned;
    for (auto const &[tid, labels] : plan.body_owned) {
        owned.insert(tid);
    }
    std::optional<std::size_t> writer;
    for (std::size_t o = 0; o < plan.ops.size(); ++o) {
        if (owned.contains(plan.ops[o].slots[0].tid)) {
            continue;
        }
        if (writer.has_value()) {
            return {.reason = &AxisTiling::TriangleShape}; // Two caller's tensors written.
        }
        writer = o;
    }
    if (!writer.has_value()) {
        return {.reason = &AxisTiling::TriangleShape};
    }
    auto const &op = plan.ops[*writer];
    if (op.kind != OpKind::Einsum) {
        return {.reason = &AxisTiling::NotExchangeable};
    }
    auto const &labels = op.slots[0].labels;
    auto const  row    = std::ranges::find(labels, 0);
    auto const  col    = std::ranges::find(labels, 1);
    if (row == labels.end() || col == labels.end() || std::ranges::count(labels, 0) != 1 || std::ranges::count(labels, 1) != 1) {
        return {.reason = &AxisTiling::TriangleShape};
    }
    // The output is the writer's alone in the region: a reader would see half-mirrored blocks.
    TensorId const out = graph.buffer_of(op.slots[0].tid);
    for (std::size_t o = 0; o < plan.ops.size(); ++o) {
        for (std::size_t slot = (o == *writer ? 1 : 0); slot < plan.ops[o].slots.size(); ++slot) {
            if (graph.buffer_of(plan.ops[o].slots[slot].tid) == out) {
                return {.reason = &AxisTiling::TriangleShape};
            }
        }
    }
    if (backed(op.slots[0].tid)) {
        return {.reason = &AxisTiling::TriangleBacked};
    }

    auto const &dims = graph.tensor(op.slots[0].tid).dims;
    if (dims.size() > static_cast<std::size_t>(kMaxSymmetryRank)) {
        return {.reason = &AxisTiling::NotExchangeable};
    }
    auto const p_row        = static_cast<std::size_t>(row - labels.begin());
    auto const p_col        = static_cast<std::size_t>(col - labels.begin());
    auto       base         = SymmetryOp::identity();
    base.permutation[p_row] = static_cast<int8_t>(p_col);
    base.permutation[p_col] = static_cast<int8_t>(p_row);
    std::vector<std::size_t> rest;
    for (std::size_t p = 0; p < dims.size(); ++p) {
        if (p != p_row && p != p_col) {
            rest.push_back(p);
        }
    }
    std::vector<SymmetryOp> candidates;
    std::vector<bool>       taken(rest.size(), false);
    exchanges(rest, dims, 0, base, taken, candidates);

    for (auto const &sigma : candidates) {
        std::vector<std::pair<TensorId, SymmetryDescriptor>> relied_on;
        auto const                                           proven = contribution_symmetry(escape, op.id, sigma, &relied_on);
        if (!proven.has_value()) {
            continue;
        }
        // Every promise the proof rests on gets a check the emitted loop can run.
        bool const checkable = std::ranges::all_of(relied_on, [&](auto const &entry) {
            auto const &handle = graph.tensor(entry.first);
            return handle.is_runtime && handle.dtype == plan.dtype;
        });
        if (!checkable) {
            return {.reason = &AxisTiling::NotExchangeable};
        }
        Plan::Triangle triangle;
        triangle.writer               = *writer;
        triangle.sigma                = *proven;
        triangle.mirror_labels        = labels;
        triangle.mirror_labels[p_row] = 1;
        triangle.mirror_labels[p_col] = 0;
        triangle.checks               = std::move(relied_on);
        return {.triangle = std::move(triangle)};
    }
    return {.reason = &AxisTiling::NotExchangeable};
}

/// The intermediates worth a region for its triangle alone, largest first: every operand of an
/// einsum that a sliceable statement writes, when what that einsum adds to its output is provably
/// unchanged by exchanging two of the output's axes, together with some exchange of the others.
/// That is the question @ref find_triangle asks of the region, asked here of the einsum alone so a
/// program with nothing to stream is read once per einsum rather than once per candidate region.
/// The two axes need an extent of three or more, since a triangle cuts them into two or more
/// blocks of at least two.
///
/// @pre The graph's node list is in dependency order.
std::vector<std::pair<std::size_t, TensorId>> symmetric_seeds(Graph const &graph, Scope const &scope, std::set<TensorId> const &declined) {
    std::set<TensorId> sliceably_written;
    for (auto const &node : graph.nodes()) {
        if (tileable_kind(node.kind) || registered_sliceable(node)) {
            sliceably_written.insert(node.outputs.begin(), node.outputs.end());
        }
    }

    std::vector<std::pair<std::size_t, TensorId>> out;
    std::optional<EscapeAnalysis>                 escape;
    for (auto const &node : graph.nodes()) {
        if (node.kind != OpKind::Einsum || node.outputs.size() != 1) {
            continue;
        }
        std::vector<std::pair<std::size_t, TensorId>> operands;
        for (TensorId const tid : node.inputs) {
            auto const *handle = graph.find_tensor(tid);
            if (handle == nullptr || handle->is_tiled || handle->is_distributed || !scope.intermediate(tid) || declined.contains(tid) ||
                !sliceably_written.contains(tid)) {
                continue;
            }
            operands.emplace_back(handle->total_bytes(), tid);
        }
        auto const *output = graph.find_tensor(node.outputs[0]);
        if (operands.empty() || output == nullptr || output->dims.size() > static_cast<std::size_t>(kMaxSymmetryRank)) {
            continue;
        }

        auto const &dims   = output->dims;
        bool        proven = false;
        for (std::size_t p = 0; p < dims.size() && !proven; ++p) {
            for (std::size_t q = p + 1; q < dims.size() && !proven; ++q) {
                if (dims[p] != dims[q] || dims[p] < 3) {
                    continue;
                }
                auto base           = SymmetryOp::identity();
                base.permutation[p] = static_cast<int8_t>(q);
                base.permutation[q] = static_cast<int8_t>(p);
                std::vector<std::size_t> rest;
                for (std::size_t r = 0; r < dims.size(); ++r) {
                    if (r != p && r != q) {
                        rest.push_back(r);
                    }
                }
                std::vector<SymmetryOp> candidates;
                std::vector<bool>       taken(rest.size(), false);
                exchanges(rest, dims, 0, base, taken, candidates);
                if (!escape.has_value()) {
                    escape.emplace(EscapeAnalysis::over(graph));
                }
                proven = std::ranges::any_of(
                    candidates, [&](SymmetryOp const &sigma) { return contribution_symmetry(*escape, node.id, sigma).has_value(); });
            }
        }
        if (!proven) {
            continue;
        }
        for (auto const &entry : operands) {
            if (std::ranges::find(out, entry) == out.end()) {
                out.push_back(entry);
            }
        }
    }
    std::ranges::sort(
        out, [](auto const &lhs, auto const &rhs) { return lhs.first != rhs.first ? lhs.first > rhs.first : lhs.second < rhs.second; });
    return out;
}

} // namespace

void AxisTiling::set_block_extent(std::int64_t extent) {
    _block_pin = extent;
}

void AxisTiling::set_memory_cap(std::int64_t bytes) {
    _memory_cap   = bytes;
    _cap_explicit = true;
}

std::int64_t AxisTiling::memory_cap() const {
    return _cap_explicit ? _memory_cap : config::get(option::GraphTilingMemoryCap);
}

void AxisTiling::reset_stats() {
    _num_tiled      = 0;
    _slice_count    = 0;
    _depth          = 0;
    _iterations     = 0;
    _largest_before = 0;
    _largest_after  = 0;
    _axis_names.clear();
    _axis_letters.clear();
    _axis_extents.clear();
    _streamed.clear();
    _whole.clear();
    _held_results.clear();
    _accumulator.clear();
    _summed_into.clear();
    _emission.clear();
    _block_extents.clear();
    _loops         = 0;
    _triangle_sign = 0;
    _regions.clear();
    _enclosing.clear();
}

std::vector<std::string> AxisTiling::explain() const {
    std::vector<std::string> lines;
    lines.reserve(_regions.size() + _held_results.size() + 1);
    for (auto const &region : _regions) {
        lines.push_back(fmt::format("AxisTiling: {}", region));
    }
    for (auto const &held : _held_results) {
        lines.push_back(fmt::format("AxisTiling: {}", held));
    }
    if (!_accumulator.empty()) {
        lines.push_back(fmt::format("AxisTiling: '{}' accumulates across the first region's iterations", _accumulator));
    }
    for (auto const &name : _summed_into) {
        lines.push_back(fmt::format("AxisTiling: '{}' collects a sum over a sliced axis, one block's share per iteration", name));
    }
    return lines;
}

namespace {

/// What one node reads and writes, by the buffer underneath each tensor.
struct NodeIo {
    std::set<TensorId> reads;
    std::set<TensorId> writes;
};

/// Every node's buffers read and written. Effective rather than listed, so a loop or a
/// conditional answers for its body.
std::vector<NodeIo> buffer_io(Graph &graph) {
    std::vector<NodeIo> out;
    out.reserve(graph.nodes().size());
    for (auto const &node : graph.nodes()) {
        auto const [reads, writes] = graph.effective_io(node);
        NodeIo io;
        for (TensorId const tid : reads) {
            io.reads.insert(graph.buffer_of(tid));
        }
        for (TensorId const tid : writes) {
            io.writes.insert(graph.buffer_of(tid));
        }
        out.push_back(std::move(io));
    }
    return out;
}

bool intersects(std::set<TensorId> const &lhs, std::set<TensorId> const &rhs) {
    return std::ranges::any_of(lhs, [&rhs](TensorId tid) { return rhs.contains(tid); });
}

/// Whether @p later has to run after @p earlier: it reads what @p earlier writes, writes what
/// @p earlier reads, or writes what @p earlier writes.
bool depends(NodeIo const &earlier, NodeIo const &later) {
    return intersects(earlier.writes, later.reads) || intersects(earlier.writes, later.writes) || intersects(earlier.reads, later.writes);
}

/// Where the loop can stand in for the statements it replaces, or nothing when it cannot.
///
/// A region need not be contiguous: a statement of another term may sit between two of its
/// statements. The loop takes the place of the LAST of them when nothing in between depends on
/// an earlier one, so every earlier one can move down to it, and of the FIRST when no later one
/// depends on anything in between, so every later one can move up. A node in between that
/// depends on one statement while another depends on it allows neither.
std::optional<std::size_t> gather_site(std::vector<NodeIo> const &io, std::vector<std::size_t> const &replaced) {
    std::set<std::size_t> const moving(replaced.begin(), replaced.end());
    bool                        sink  = true;
    bool                        hoist = true;
    for (std::size_t n = replaced.front() + 1; n < replaced.back(); ++n) {
        if (moving.contains(n)) {
            continue;
        }
        for (std::size_t const m : replaced) {
            sink  = sink && !(m < n && depends(io[m], io[n]));
            hoist = hoist && !(m > n && depends(io[n], io[m]));
        }
    }
    if (sink) {
        return replaced.back();
    }
    if (hoist) {
        return replaced.front();
    }
    return std::nullopt;
}

} // namespace

AxisTiling::AxisTiling(CostModel cost_model) : _cost_model(std::move(cost_model)) {
}

DeviceProfile const &AxisTiling::pricing() const {
    return _cost_model.has_value() ? _cost_model->device(Target::CPU) : pricing_profile();
}

bool AxisTiling::transform(Graph &graph) {
    bool const moved = tile_graph(graph);

    // The graphs this one holds: the manager runs this pass on each of them next, and that run
    // walks out through this record to the declarations its tensors share with this graph.
    auto const &nodes = graph.nodes();
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        for_each_child_graph(nodes[i], [&](auto const &sub) { _enclosing[&sub] = {&graph, i}; });
    }
    return moved;
}

std::size_t AxisTiling::footprint(Graph const &graph) const {
    // The root, and on the way to it each level's node whose body holds the graph being tiled.
    std::vector<std::pair<Graph *, std::size_t>> chain;
    Graph const                                 *root = &graph;
    for (auto hit = _enclosing.find(root); hit != _enclosing.end(); hit = _enclosing.find(root)) {
        chain.push_back(hit->second);
        root = hit->second.first;
    }

    // A body that a dependency orders before or after the branch holding this graph is not running
    // while this one is, so what only it names is not live beside it. Ordered here by a direct
    // conflict between the two nodes' effective reads and writes, which is what puts an edge between
    // them; an order through other nodes is not seen, which keeps the count conservative.
    std::set<Graph const *> apart;
    Graph const            *below = &graph;
    for (auto const &[outer, own] : chain) {
        auto &nodes = outer->nodes();
        bool  holds = false;
        if (own < nodes.size()) {
            for_each_child_graph(nodes[own], [&](Graph const &child) { holds = holds || &child == below; });
        }
        below = outer;
        if (!holds) {
            continue;
        }
        auto const buffers = [&](std::vector<TensorId> const &ids) {
            std::set<TensorId> out;
            for (TensorId const tid : ids) {
                out.insert(outer->buffer_of(tid));
            }
            return out;
        };
        auto const meets = [](std::set<TensorId> const &lhs, std::set<TensorId> const &rhs) {
            return std::ranges::any_of(lhs, [&](TensorId tid) { return rhs.contains(tid); });
        };
        auto const [own_reads, own_writes] = outer->effective_io(nodes[own]);
        auto const reads_here              = buffers(own_reads);
        auto const writes_here             = buffers(own_writes);
        for (std::size_t j = 0; j < nodes.size(); ++j) {
            bool has_body = false;
            for_each_child_graph(nodes[j], [&](Graph const &) { has_body = true; });
            if (j == own || !has_body) {
                continue;
            }
            auto const [reads, writes] = outer->effective_io(nodes[j]);
            auto const reads_there     = buffers(reads);
            auto const writes_there    = buffers(writes);
            if (meets(writes_there, reads_here) || meets(writes_there, writes_here) || meets(writes_here, reads_there)) {
                for_each_child_graph(nodes[j], [&](Graph const &child) { apart.insert(&child); });
            }
        }
    }

    // A body names its parent's tensors through handles of its own over the same tensor object,
    // so one tensor is counted once by that object; a handle with no object is its graph's own.
    std::set<void const *>                       objects;
    std::set<std::pair<Graph const *, TensorId>> unbacked;
    std::size_t                                  total = 0;
    // NOLINTNEXTLINE(misc-no-recursion): sub-graphs nest.
    std::function<void(Graph const &)> count = [&](Graph const &g) {
        // Only what a node reads or writes is ever allocated: a rewrite leaves the handles of the
        // tensors it replaced registered, and a deferred one nobody names is never materialized.
        // A node with a body is counted through the body, whose own nodes name what it uses, so a
        // body set apart above is not brought back by the lists its node restates.
        std::set<TensorId> named;
        for (auto const &node : g.nodes()) {
            bool has_body = false;
            for_each_child_graph(node, [&](Graph const &) { has_body = true; });
            if (has_body) {
                continue;
            }
            for (auto const *ids : {&node.inputs, &node.outputs}) {
                for (TensorId const tid : *ids) {
                    named.insert(g.buffer_of(tid));
                }
            }
        }
        for (auto const &[tid, handle] : g.tensors_map()) {
            if (handle.is_tensor_view || g.buffer_of(tid) != tid || !named.contains(tid)) {
                continue;
            }
            // A backed tensor's bytes are on disk, and SlabStreaming reads it a slab at a time
            // when it does not fit; a body's handle for one is recognized by the same object.
            if (handle.backing && handle.tensor_ptr != nullptr) {
                objects.insert(handle.tensor_ptr);
                continue;
            }
            bool const fresh = handle.tensor_ptr != nullptr ? objects.insert(handle.tensor_ptr).second : unbacked.emplace(&g, tid).second;
            if (fresh) {
                total += handle.total_bytes();
            }
        }
        for (auto const &node : g.nodes()) {
            for_each_child_graph(node, [&](Graph const &child) {
                if (!apart.contains(&child)) {
                    count(child);
                }
            });
        }
    };
    count(*root);
    return total;
}

bool AxisTiling::tile_graph(Graph &graph) {
    PassCounter const tiled{_num_tiled};

    Scope const scope{.graph = graph, .enclosing = _enclosing};

    std::int64_t const cap = memory_cap();
    if (cap == 0) {
        note_skip(ZeroCap);
        return tiled.moved();
    }

    // A negative cap is derived: an intermediate may take what the memory budget leaves after
    // every other tensor of the tree, and no more than its share of the budget. A program that
    // fits as a whole with no intermediate over its share has nothing to stream, and is read only
    // for regions whose triangle pays on its own.
    bool const  derived = cap < 0;
    std::size_t budget  = 0;
    std::size_t share   = 0;
    if (derived) {
        budget = detail::memory_budget();
        if (budget == 0) {
            note_skip(NoBudget);
            return tiled.moved();
        }
        share = detail::budget_share();
    }

    // One region at a time. Every rewrite renumbers the nodes and replaces a region with a loop,
    // which no later region can take in, so the seeds are read afresh after each; a seed that
    // declined once declines again on the same graph and is not retried.
    std::set<TensorId> declined;
    for (bool first = true;; first = false) {
        std::size_t const total = derived ? footprint(graph) : 0;

        // A program over its budget streams because it must, and a loop it streams in may take
        // whatever the budget leaves; the share caps only what a program that fits streams for the
        // share's own sake. Held to the share as well, the CCSD tutorial's file-backed stage cut
        // its ladder into blocks of 2 x 4 under a 16 MB budget while the replay held 11.6 MB, and
        // spent more than half its iteration on the small blocks.
        bool const forced = derived && total > budget;

        // The cap one intermediate of @p bytes is held to: the fixed cap, or what the budget leaves
        // after everything else, no more than its share unless the program is over its budget.
        // Zero when the others alone exceed the budget.
        auto const limit_for = [&](std::size_t bytes) -> std::size_t {
            if (!derived) {
                return static_cast<std::size_t>(cap);
            }
            std::size_t const others = total - std::min(total, bytes);
            if (others >= budget) {
                return 0;
            }
            return forced ? budget - others : std::min(budget - others, share);
        };

        // A seed is an intermediate over its cap that a sliceable statement writes: the tensors
        // the loop exists to stream, largest first.
        //
        // A tensor the graph declares WITHOUT calling it an intermediate is a result: the caller may
        // read it after execute(), so it has to exist whole then, and the loop would leave only
        // slices of it. It is left whole however large, and the largest such one over its cap is
        // reported, because nothing in a program says which of its declared tensors the caller
        // reads afterwards and the declaration is the only place that can.
        auto const                                   &nodes   = graph.nodes();
        std::size_t                                   largest = 0;
        std::vector<std::pair<std::size_t, TensorId>> seeds;
        TensorHandle const                           *held = nullptr;
        for (auto const &node : nodes) {
            bool const sliceable = tileable_kind(node.kind) || registered_sliceable(node);
            for (TensorId const tid : node.outputs) {
                auto const *handle = graph.find_tensor(tid);
                if (handle == nullptr || handle->is_tiled || handle->is_distributed) {
                    continue;
                }
                if (!scope.intermediate(tid)) {
                    std::size_t const bytes = handle->total_bytes();
                    // A tensor the program writes to storage, over its cap, is a seed too: no one
                    // holds it whole afterwards, and a loop over its own axes lets every statement
                    // that writes it finish one box of it before SlabStreaming stores the box, where
                    // otherwise each writer but a lone overwriting one keeps it whole in memory. Not an
                    // update-backed tensor, which is produced in memory whole.
                    if (auto const backing = scope.backing_of(tid); backing != nullptr) {
                        if (derived && backing->write && !backing->update && sliceable && bytes > limit_for(bytes) &&
                            !declined.contains(tid) && std::ranges::find(seeds, std::pair{bytes, tid}) == seeds.end()) {
                            largest = std::max(largest, bytes);
                            seeds.emplace_back(bytes, tid);
                        }
                        continue;
                    }
                    // A read-backed tensor is not one: its storage holds it, and SlabStreaming moves it.
                    if (sliceable && handle->alloc_state == AllocState::Deferred && bytes > limit_for(bytes) &&
                        (held == nullptr || bytes > held->total_bytes())) {
                        held = handle;
                    }
                    continue;
                }
                std::size_t const bytes = handle->total_bytes();
                largest                 = std::max(largest, bytes);
                if (sliceable && bytes > limit_for(bytes) && !declined.contains(tid) &&
                    std::ranges::find(seeds, std::pair{bytes, tid}) == seeds.end()) {
                    seeds.emplace_back(bytes, tid);
                }
            }
        }
        if (first && held != nullptr) {
            std::size_t const bytes = held->total_bytes();
            note_skip(ResultHeld, fmt::format("'{}', {} bytes against a cap of {}", held->name, bytes, limit_for(bytes)));
            _held_results.push_back(fmt::format("held '{}' whole, {} bytes against a cap of {}, because it is declared as a result the "
                                                "caller may read after execute(); declare it as an intermediate if nothing does",
                                                held->name, bytes, limit_for(bytes)));
            note(ResultHeld, _held_results.back(),
                 Decision{.tensors = {held->name}, .facts = {Fact{.name = "bytes", .value = static_cast<double>(bytes), .unit = "B"}}}, 1);
        }
        if (first) {
            // A running maximum: the manager runs this pass once per graph of the tree, and the
            // getters report over all of them.
            _largest_before = std::max(_largest_before, largest);
            if (largest == 0) {
                // A graph that writes no intermediate, such as the parent of an iterative
                // method's loop, has nothing here to decline.
                return tiled.moved();
            }
            if (derived && total <= budget && largest <= share) {
                note_skip(FitsBudget,
                          fmt::format("{} bytes against --einsums:max-memory of {}, the largest intermediate {} against a share of {}",
                                      total, budget, largest, share));
                note(FitsBudget, fmt::format("declined: the program's tensors hold {} bytes and the memory budget is {}", total, budget),
                     Decision{.facts = {Fact{.name = "total", .value = static_cast<double>(total), .unit = "B"},
                                        Fact{.name = "budget", .value = static_cast<double>(budget), .unit = "B"}}},
                     1);
                break;
            }
            if (!derived && largest <= static_cast<std::size_t>(cap)) {
                note_skip(FitsCap, fmt::format("{} bytes against a cap of {}", largest, cap));
                note(FitsCap, fmt::format("declined: the largest intermediate is {} bytes and the cap is {}", largest, cap),
                     Decision{.facts = {Fact{.name = "largest", .value = static_cast<double>(largest), .unit = "B"},
                                        Fact{.name = "cap", .value = static_cast<double>(cap), .unit = "B"}}},
                     1);
                break;
            }
            if (seeds.empty()) {
                note_skip(NoSliceableWriter, fmt::format("{} bytes against a cap of {}", largest, limit_for(largest)));
                break;
            }
        }
        // A destination the program writes to storage first, then largest first. Its region takes in
        // every statement that writes it and, through the labelling, the intermediates they read, so
        // those are cut along the destination's own axes inside its loop; tiled after them, the
        // destination would find one of its writers already inside a loop it cannot take in.
        auto const backed_first = [&](TensorId tid) { return scope.backed(tid) ? 0 : 1; };
        std::ranges::sort(seeds, [&](auto const &lhs, auto const &rhs) {
            if (backed_first(lhs.second) != backed_first(rhs.second)) {
                return backed_first(lhs.second) < backed_first(rhs.second);
            }
            return lhs.first != rhs.first ? lhs.first > rhs.first : lhs.second < rhs.second;
        });

        graph.topological_sort();
        bool rewrote = false;
        for (auto const &[bytes, seed] : seeds) {
            Allowance const allowance{
                .cap = limit_for(bytes), .budget = derived ? budget : 0, .share = derived ? share : 0, .total = total, .forced = forced};
            if (tile_region(graph, seed, allowance)) {
                rewrote = true;
                break;
            }
            declined.insert(seed);
        }
        if (!rewrote) {
            break;
        }
    }

    // What streaming left, or all of a program with nothing to stream: a region whose contribution
    // is provably symmetric in two axes does about half its arithmetic as a triangle whether or not
    // its intermediates fit, so it is a seed of its own, tiled only when the cost model prices the
    // triangle below running the region whole. A seed already declined declines here too, since a
    // triangle is one of the plans it was offered.
    while (true) {
        graph.topological_sort();
        std::size_t const total   = derived ? footprint(graph) : 0;
        bool              rewrote = false;
        for (auto const &[bytes, seed] : symmetric_seeds(graph, scope, declined)) {
            std::size_t limit = static_cast<std::size_t>(std::max<std::int64_t>(cap, 0));
            if (derived) {
                std::size_t const others = total - std::min(total, bytes);
                limit                    = others < budget ? std::min(budget - others, share) : 0;
            }
            Allowance const allowance{
                .cap = limit, .budget = derived ? budget : 0, .share = derived ? share : 0, .total = total, .triangle_only = true};
            declined.insert(seed);
            if (tile_region(graph, seed, allowance)) {
                rewrote = true;
                break;
            }
        }
        if (!rewrote) {
            break;
        }
    }
    return tiled.moved();
}

bool AxisTiling::tile_region(Graph &graph, TensorId seed, Allowance const &allowance) {
    auto const &nodes       = graph.nodes();
    auto const  io          = buffer_io(graph);
    auto const *seed_handle = graph.find_tensor(seed);
    if (seed_handle == nullptr) {
        return false;
    }
    std::size_t const limit = allowance.cap;
    Scope const       scope{.graph = graph, .enclosing = _enclosing};

    // The statements that read or write a tensor, by the buffer underneath it.
    auto const touchers = [&](TensorId tid) {
        TensorId const           buffer = graph.buffer_of(tid);
        std::vector<std::size_t> out;
        for (std::size_t i = 0; i < nodes.size(); ++i) {
            if (io[i].reads.contains(buffer) || io[i].writes.contains(buffer)) {
                out.push_back(i);
            }
        }
        return out;
    };
    auto const sliceable = [&](std::size_t i) {
        if (!tileable_kind(nodes[i].kind) && !registered_sliceable(nodes[i])) {
            return false;
        }
        if (!understands(graph, nodes[i])) {
            note_skip(UnknownFeature, fmt::format("node '{}': {}", nodes[i].label, describe_features(features_of(graph, nodes[i]))));
            return false;
        }
        return true;
    };

    std::optional<Plan> best;
    // Built on the first two-axis candidate that could be a triangle; the graph does not change
    // until a plan is emitted.
    std::optional<EscapeAnalysis> escape;
    // The finest slicing of a candidate no slice of which fits, taken when no candidate fits at all
    // and the seed outweighs everything the loop leaves outside it, the case where cutting it
    // still brings the peak down from the seed to a slice.
    std::optional<Plan> fallback;
    std::size_t         outside_bytes = 0;
    std::size_t         infeasible    = 0;
    std::size_t         over_cap      = 0;
    // Two rounds. The first takes every region in whole, as a loop that streams all it reaches
    // holds the least. Only when no candidate gives any plan does the second run, where a region
    // that stops slicing once it grows keeps its last sliceable form instead (see fall_back); the
    // first round's declines are withdrawn, since the second examines every candidate again.
    std::size_t const first_round = num_decisions();
    for (bool const keep_small : {false, true}) {
        if (keep_small) {
            if (best.has_value() || fallback.has_value()) {
                break;
            }
            retract_declines(first_round);
            outside_bytes = 0;
            infeasible    = 0;
            over_cap      = 0;
        }
        for (auto const &positions : candidate_sets(seed_handle->rank)) {
            if (allowance.triangle_only && (positions.size() != 2 || seed_handle->dims[positions[0]] != seed_handle->dims[positions[1]])) {
                continue;
            }
            auto const detail = fmt::format("seed '{}', candidate axes {}", seed_handle->name, fmt::join(positions, ","));

            // The region: the seed's writers and readers, and then every writer and reader of any
            // other intermediate over the cap that the labelling reaches, since the loop has to
            // stream that one too. A tensor the labelling reaches that fits the cap is not followed:
            // it stays whole, written and read through slices of its own buffer, so the statements of
            // other terms that touch it stay outside the loop.
            std::vector<std::size_t>      members = touchers(seed);
            std::optional<RegionAnalysis> analysis;
            Labelling                     labelling;
            std::string                   why;
            std::string                   barrier_label;
            // The last region the labelling went through, before it grew. Streaming an intermediate the
            // labelling reaches is a choice, not a requirement: left out, it stays whole and is written
            // and read through slices of its own buffer. So a region that stops slicing once it takes
            // such a tensor in falls back to the region before it, which streams the seed and holds the
            // rest whole, rather than streaming nothing. A budget the program's other tensors nearly
            // fill leaves the seed a cap of a few kilobytes, every intermediate is over it, and growing
            // the region to take them all in drew the whole CCSD update cycle into the ladder's loop,
            // where no labelling holds.
            std::vector<std::size_t> sliced_members;
            auto const               fall_back = [&](std::string reason) {
                if (!keep_small || sliced_members.empty()) {
                    why = std::move(reason);
                    return;
                }
                note_skip(RegionKeptSmall, fmt::format("{}; {}; the region streams the seed and holds the rest whole", detail, reason));
                members = sliced_members;
                analysis.emplace(graph, scope, members, seed);
                analysis->collect();
                labelling = analysis->propagate(positions);
            };
            while (true) {
                if (auto const barrier = std::ranges::find_if_not(members, sliceable); barrier != members.end()) {
                    barrier_label = sliced_members.empty() ? fmt::format("; statement '{}'", nodes[*barrier].label) : std::string{};
                    fall_back("a statement that reads or writes an intermediate the loop has to stream is one this pass cannot slice");
                    break;
                }
                analysis.emplace(graph, scope, members, seed);
                if (!analysis->collect()) {
                    fall_back("a statement of the region names operands this pass cannot read as a dense, single-buffer slice");
                    break;
                }
                if (!analysis->all_runtime()) {
                    fall_back(
                        "the region names an operand this pass cannot slice, because a rank-erased operand is one the emitted body can "
                        "name and a statically ranked one is not");
                    break;
                }
                labelling = analysis->propagate(positions);
                if (!labelling.feasible) {
                    fall_back(labelling.reason);
                    break;
                }
                sliced_members = members;
                // A tensor the region WRITES is completed by the loop, so every statement that touches
                // it has to be in it. A tensor the region only READS is an input: its writers and
                // other readers join when all of them can, which streams it too, and otherwise it is
                // read through slices of its whole buffer, as any caller's tensor is.
                std::set<std::size_t> grown(members.begin(), members.end());
                auto const            follow = [&](TensorId tid, std::vector<int> const &labels, bool written) {
                    auto const *handle = graph.find_tensor(tid);
                    if (handle == nullptr || !scope.intermediate(tid) || handle->total_bytes() <= limit ||
                        std::ranges::all_of(labels, [](int v) { return v == kFree; })) {
                        return;
                    }
                    auto const others = touchers(tid);
                    if (!written && !std::ranges::all_of(others, [&](std::size_t i) {
                            return grown.contains(i) ||
                                   ((tileable_kind(nodes[i].kind) || registered_sliceable(nodes[i])) && understands(graph, nodes[i]));
                        })) {
                        return;
                    }
                    grown.insert(others.begin(), others.end());
                };
                for (auto const &[tid, labels] : labelling.store.tensor) {
                    follow(tid, labels, true);
                }
                for (auto const &[site, labels] : labelling.store.use) {
                    follow(analysis->slot_tensor(site.first, site.second), labels, false);
                }
                if (grown.size() == members.size()) {
                    break;
                }
                members.assign(grown.begin(), grown.end());
            }
            if (!why.empty()) {
                ++infeasible;
                note_skip(why, detail + barrier_label);
                continue;
            }

            std::string                                       accumulator;
            TensorId                                          accumulator_id = 0;
            std::vector<std::pair<TensorId, PrefactorScalar>> summed_into;
            std::string                                       reason;
            if (!analysis->closes(labelling, accumulator, accumulator_id, summed_into, reason)) {
                note_skip(reason, detail);
                continue;
            }

            std::vector<std::size_t> extents;
            std::size_t              slices = 1;
            for (std::size_t const position : positions) {
                extents.push_back(seed_handle->dims[position]);
                slices *= seed_handle->dims[position];
            }

            // Against a fixed cap every intermediate is held to it alone. Against the memory budget
            // the loop's buffers are held to it together, beside every tensor the loop does not
            // re-declare, and to their share of it.
            bool const  joint       = allowance.budget != 0;
            std::size_t room        = limit;
            bool        best_effort = false;
            if (joint) {
                std::size_t const outside = allowance.total - std::min(allowance.total, analysis->streamed_whole_bytes(labelling));
                outside_bytes             = std::max(outside_bytes, outside);
                if (outside >= allowance.budget) {
                    note_skip(
                        "the program's other tensors alone exceed the memory budget, so no slice of the region fits beside them",
                        fmt::format("{}; {} bytes outside the loop against --einsums:max-memory of {}", detail, outside, allowance.budget));
                    // Nothing can meet the budget; a slice within the tensor's share is as far as cutting
                    // the seed still lowers the peak, and finer slices only add iterations.
                    best_effort = true;
                    room        = allowance.share;
                } else {
                    room = allowance.forced ? allowance.budget - outside : std::min(allowance.budget - outside, allowance.share);
                }
            }
            auto const fits = [joint, room](std::size_t largest, std::size_t held) { return (joint ? held : largest) <= room; };

            std::size_t unit      = 0;
            std::size_t unit_held = 0;
            long double traffic   = 0.0L;
            analysis->cost(labelling, extents, 1, unit, unit_held, traffic);
            bool const slice_fits = fits(unit, unit_held);
            if (!slice_fits) {
                ++over_cap;
                best_effort = true;
            }

            // Blocks where a contraction of the loop rewards them, a grouped chunk of points
            // otherwise.
            // A slice that fits nothing is cut as finely as it goes: blocks are chosen to fit, and none would.
            std::optional<BlockChoice> blocks;
            if (auto const reusable = analysis->reusable_labels(labelling); slice_fits && !reusable.empty()) {
                if (std::string_view const blocker = analysis->range_blocker(labelling); !blocker.empty()) {
                    note_skip(blocker, detail);
                } else {
                    blocks =
                        choose_blocks(extents, reusable, labelling.fully_sliced, fits,
                                      [&](std::vector<std::size_t> const &b, std::size_t &largest, std::size_t &held,
                                          long double &streamed) { analysis->range_cost(labelling, extents, b, largest, held, streamed); });
                }
            }
            // A region that writes a tensor to storage is one loop. Blocks that leave a remainder add a
            // loop per remainder, and SlabStreaming streams each loop's writes only when no other loop
            // writes the tensor in memory, which two loops writing it never satisfy; the grouped plan's
            // chunks divide the slices, so it is always one loop.
            bool const writes_backed = std::ranges::any_of(labelling.store.tensor, [&](auto const &entry) {
                auto const backing = scope.backing_of(entry.first);
                return backing != nullptr && backing->write && !backing->update;
            });
            if (blocks.has_value() && writes_backed) {
                for (std::size_t k = 0; k < extents.size(); ++k) {
                    if (extents[k] % blocks->blocks[k] != 0) {
                        blocks.reset();
                        break;
                    }
                }
            }
            if (!blocks.has_value() && !labelling.fully_sliced.empty()) {
                note_skip(OperandWithoutAxes, detail);
                continue;
            }

            Plan plan;
            plan.extents        = extents;
            plan.slices         = slices;
            plan.largest_before = seed_handle->total_bytes();
            if (blocks.has_value()) {
                plan.range  = true;
                plan.blocks = blocks->blocks;
                plan.depth  = 1;
                for (std::size_t const b : plan.blocks) {
                    plan.depth *= b;
                }
                plan.iterations = 0;
                for (auto const &walks : loop_walks(plan)) {
                    plan.iterations += walk_iterations(walks);
                }
                plan.largest_after = blocks->largest;
                plan.traffic       = blocks->traffic;
            } else {
                std::size_t const depth   = slice_fits ? chunk_depth(slices, joint ? unit_held : unit, room) : 1;
                std::size_t       largest = 0;
                std::size_t       held    = 0;
                analysis->cost(labelling, extents, depth, largest, held, traffic);
                plan.depth         = depth;
                plan.iterations    = slices / depth;
                plan.largest_after = largest;
                plan.traffic       = traffic;
            }
            plan.best_effort    = best_effort;
            plan.fits_share     = slice_fits;
            plan.accumulator    = accumulator;
            plan.accumulator_id = accumulator_id;
            plan.summed_into    = summed_into;
            for (auto const &[tid, prefactor] : summed_into) {
                plan.summed_names.push_back(graph.tensor(tid).name);
            }
            plan.store = labelling.store;
            if (!analysis->build(labelling, plan, reason)) {
                note_skip(reason, detail);
                continue;
            }
            if (auto const site = gather_site(io, plan.replaced); site.has_value()) {
                plan.site = *site;
            } else {
                note_skip(StatementBetween, detail);
                continue;
            }
            analysis->describe(positions, plan.axis_names, plan.axis_letters);
            analysis->partition(labelling, plan.streamed, plan.whole);

            // Two exchangeable axes run only the tiles on and above their diagonal. The block is the one
            // the cost model prices cheapest among those that fit, since halving the tiles changes the
            // kernels' shapes as well as their count, which streamed bytes do not see.
            // Whether or not the grid search found blocks worth cutting: the triangle chooses its own.
            if (positions.size() == 2) {
                if (!escape.has_value()) {
                    escape.emplace(EscapeAnalysis::over(graph));
                }
                auto verdict = find_triangle(graph, plan, [&](TensorId tid) { return scope.backing_of(tid) != nullptr; }, *escape);
                if (!verdict.triangle.has_value()) {
                    if (verdict.reason != nullptr) {
                        note_skip(*verdict.reason, detail);
                    }
                } else {
                    Plan candidate                  = plan;
                    candidate.triangle              = std::move(verdict.triangle);
                    std::size_t const        extent = extents[0];
                    std::vector<std::size_t> options;
                    if (_block_pin > 0) {
                        options.push_back(static_cast<std::size_t>(_block_pin));
                    } else {
                        for (std::size_t const b : balanced_blocks(extent)) {
                            options.push_back(b);
                        }
                    }
                    std::optional<std::size_t> chosen;
                    for (std::size_t const b : options) {
                        // Two or more blocks, or the triangle is the whole square and saves nothing. A
                        // region that writes storage is one loop per shape, so no remainder.
                        if (b < 2 || b >= extent || (writes_backed && extent % b != 0)) {
                            continue;
                        }
                        std::size_t largest = 0;
                        std::size_t held    = 0;
                        long double moved   = 0.0L;
                        analysis->range_cost(labelling, extents, {b, b}, largest, held, moved);
                        // The scratch a mirrored tile is computed into, beside the loop's other buffers.
                        std::size_t scratch = graph.tensor(candidate.ops[candidate.triangle->writer].slots[0].tid).element_size;
                        for (std::size_t const d : graph.tensor(candidate.ops[candidate.triangle->writer].slots[0].tid).dims) {
                            scratch *= d;
                        }
                        scratch = scratch / (extent * extent) * (b * b);
                        if (!fits(std::max(largest, scratch), held + scratch)) {
                            continue;
                        }
                        candidate.blocks = {b, b};
                        double price     = 0.0;
                        for (auto const &loop : triangle_loops(extent, b)) {
                            price += static_cast<double>(loop.table.size()) *
                                     tile_price(graph, candidate, {loop.walks[0].length, loop.walks[1].length}, loop.mirror, pricing());
                        }
                        report(3, fmt::format("triangle of blocks of {} priced at {:.1f} us", b, price));
                        if (!chosen.has_value() || price < candidate.price_us) {
                            chosen                  = b;
                            candidate.price_us      = price;
                            candidate.largest_after = std::max(largest, scratch);
                            candidate.traffic       = moved;
                        }
                    }
                    // A region with nothing to stream is tiled for the triangle alone, so the triangle
                    // has to beat the region run whole, one tile of the full extents.
                    double const whole = allowance.triangle_only ? tile_price(graph, plan, {extent, extent}, false, pricing()) : 0.0;
                    if (!chosen.has_value()) {
                        note_skip(_block_pin > 0 ? BlockPinned : TriangleFit, detail);
                    } else if (allowance.triangle_only && candidate.price_us >= whole) {
                        note_skip(TrianglePrice, fmt::format("{}; {:.1f} us as a triangle of blocks of {} against {:.1f} us whole", detail,
                                                             candidate.price_us, *chosen, whole));
                    } else {
                        candidate.range      = true;
                        candidate.blocks     = {*chosen, *chosen};
                        candidate.depth      = *chosen * *chosen;
                        candidate.iterations = 0;
                        for (auto const &loop : triangle_loops(extent, *chosen)) {
                            candidate.iterations += loop.table.size();
                        }
                        plan = std::move(candidate);
                    }
                }
            }

            if (allowance.triangle_only && (!plan.triangle.has_value() || best_effort)) {
                continue;
            }
            report(3,
                   fmt::format("candidate {} streams {:.0f} bytes {}", fmt::join(plan.axis_names, " x "), static_cast<double>(plan.traffic),
                               plan.range ? fmt::format("in blocks of {}", fmt::join(plan.blocks, " x "))
                                          : fmt::format("at depth {}", plan.depth)));
            if (best_effort) {
                // One whose slices fit the share, by traffic as any plan; failing that the smallest slice.
                bool const better =
                    !fallback.has_value() || (plan.fits_share && !fallback->fits_share) ||
                    (plan.fits_share == fallback->fits_share &&
                     (plan.fits_share ? plan.traffic < fallback->traffic
                                      : (plan.largest_after < fallback->largest_after ||
                                         (plan.largest_after == fallback->largest_after && plan.traffic < fallback->traffic))));
                if (better) {
                    fallback = std::move(plan);
                }
            } else if (!best.has_value() || preferred(plan, *best)) {
                best = std::move(plan);
            }
        }
    }

    // Nothing fits the budget. Where the seed outweighs everything outside the loop it is what
    // makes the peak, and its smallest slices still bring that down by the seed's size; where the
    // tensors outside outweigh it, cutting it would shrink nothing that matters, and it is left.
    // Only against the memory budget: a cap the caller set is a bound to meet, and one that cannot
    // be met is declined, as it always was.
    if (!best.has_value() && fallback.has_value() && allowance.budget != 0 && fallback->largest_before > outside_bytes &&
        fallback->largest_after < fallback->largest_before) {
        note_skip(BestEffort, fmt::format("seed '{}': {} bytes, {} outside the loop, {} per slice", seed_handle->name,
                                          fallback->largest_before, outside_bytes, fallback->largest_after));
        best = std::move(fallback);
    }

    if (!best.has_value() && allowance.triangle_only) {
        return false; // Each candidate noted why it is no triangle, or why its triangle does not pay.
    }
    if (!best.has_value()) {
        note_skip(NoAxisSet, fmt::format("seed '{}': {} set(s) the program cannot carry a slice labelling for, {} above the cap",
                                         seed_handle->name, infeasible, over_cap));
        return false;
    }

    if (!emit_tiled_loop(graph, *best)) {
        note_skip(ElementType, fmt::format("dtype {}", static_cast<int>(best->dtype)));
        return false;
    }

    // Every region's: each names a destination of its own.
    _summed_into.insert(_summed_into.end(), best->summed_names.begin(), best->summed_names.end());

    // The getters describe the first region, which is the one around the largest seed.
    if (_num_tiled == 0) {
        _slice_count   = best->slices;
        _depth         = best->depth;
        _iterations    = best->iterations;
        _largest_after = best->largest_after;
        _axis_names    = best->axis_names;
        _axis_letters  = best->axis_letters;
        _axis_extents.assign(best->extents.begin(), best->extents.end());
        _streamed    = best->streamed;
        _whole       = best->whole;
        _accumulator = best->accumulator;
        _emission    = best->triangle.has_value() ? "triangle" : best->range ? "range" : "grouped";
        _block_extents.assign(best->blocks.begin(), best->blocks.end());
        _loops         = best->triangle.has_value() ? triangle_loops(best->extents[0], best->blocks[0]).size() : loop_walks(*best).size();
        _triangle_sign = best->triangle.has_value() ? best->triangle->sigma.sign : 0;
    }
    ++_num_tiled;
    graph.note_structural_change();

    std::string const line =
        fmt::format("slicing {} into {} slice(s), {}, seed {} bytes -> largest intermediate {} bytes{}", fmt::join(best->axis_names, " x "),
                    best->slices, describe_chunks(*best), best->largest_before, best->largest_after,
                    best->best_effort ? " (none fits the budget beside the tensors outside the loop)" : "");
    _regions.push_back(line);
    if (best->triangle.has_value()) {
        std::size_t const tiles = (best->extents[0] + best->blocks[0] - 1) / best->blocks[0];
        applied(Triangle,
                fmt::format("runs {} of {} tile(s) of {} x {}, writing each off-diagonal tile's mirror with sign {:+d}",
                            tiles * (tiles + 1) / 2, tiles * tiles, best->blocks[0], best->blocks[1], int{best->triangle->sigma.sign}),
                Decision{.detail  = fmt::format("axes {}", fmt::join(best->axis_names, " x ")),
                         .graph   = graph.name(),
                         .tensors = {seed_handle->name}},
                1);
    }
    applied(Sliced, line,
            Decision{.detail  = fmt::format("axes {}", fmt::join(best->axis_names, " x ")),
                     .graph   = graph.name(),
                     .tensors = {seed_handle->name},
                     .facts   = {Fact{.name = "slices", .value = static_cast<double>(best->slices)},
                                 Fact{.name = "largest_before", .value = static_cast<double>(best->largest_before), .unit = "B"},
                                 Fact{.name = "largest_after", .value = static_cast<double>(best->largest_after), .unit = "B"}}},
            1);
    return true;
}

EINSUMS_NAMESPACE_END(compute_graph::passes)
