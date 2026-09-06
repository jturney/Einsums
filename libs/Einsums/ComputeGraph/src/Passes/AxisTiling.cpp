//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/CaptureContext.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Operations.hpp>
#include <Einsums/ComputeGraph/Options.hpp>
#include <Einsums/ComputeGraph/Passes/AxisTiling.hpp>
#include <Einsums/ComputeGraph/Prefactor.hpp>
#include <Einsums/ComputeGraph/View.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Options/Get.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph::passes)

namespace {

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
        return std::nullopt;
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
    if (node.kind == OpKind::Einsum) {
        auto const *desc = std::get_if<EinsumDescriptor>(&node.op_data);
        if (desc == nullptr) {
            return std::nullopt;
        }
        if (desc->indices) {
            return std::vector<std::vector<std::string>>{desc->indices->spec.c_indices, desc->indices->spec.a_indices,
                                                         desc->indices->spec.b_indices};
        }
        return std::vector<std::vector<std::string>>{desc->spec.c_indices, desc->spec.a_indices, desc->spec.b_indices};
    }
    if (node.kind == OpKind::Permute) {
        auto const *desc = std::get_if<PermuteDescriptor>(&node.op_data);
        if (desc == nullptr) {
            return std::nullopt;
        }
        return std::vector<std::vector<std::string>>{desc->c_indices, desc->a_indices};
    }
    return std::nullopt;
}

/// The link (summed) letters of a contraction, empty for every other kind.
std::vector<std::string> link_letters_of(Node const &node) {
    if (node.kind != OpKind::Einsum) {
        return {};
    }
    auto const *desc = std::get_if<EinsumDescriptor>(&node.op_data);
    if (desc == nullptr) {
        return {};
    }
    return desc->indices ? desc->indices->link_indices : desc->spec.link_indices;
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
};

/// One operand of one node of the region, with the labelling the decision gave it.
struct SlotPlan {
    TensorId         tid{0};
    std::vector<int> labels;
};

/// One node of the region, everything the rewrite needs to re-emit it at a slice.
struct NodePlan {
    OpKind                kind{OpKind::Custom};
    OpData                op_data;
    std::vector<SlotPlan> slots;
};

/// The description of one tiled region, produced by the analysis and consumed by the rewrite.
struct Plan {
    std::size_t              first{0}; ///< First node of the region, in the parent's numbering.
    std::size_t              last{0};  ///< One past the last.
    std::vector<std::size_t> extents;  ///< Extent of each sliced axis, outermost first.
    std::size_t              slices{0};
    std::size_t              depth{1};
    std::size_t              iterations{0};
    std::size_t              largest_before{0};
    std::size_t              largest_after{0};
    long double              traffic{0.0L};
    std::vector<std::string> axis_names;
    std::vector<std::string> axis_letters;
    std::vector<std::string> streamed;
    std::vector<std::string> whole;
    std::string              accumulator;
    LabelStore               store;

    std::vector<NodePlan> ops;

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

    packed_gemm::ScalarType dtype{packed_gemm::ScalarType::Unknown};
};

/// How far one view bound has advanced through the slice sequence.
///
/// One cursor per bound rather than one shared by the whole body. Every bound's node runs
/// exactly once per iteration, so independent cursors stay in lockstep whatever order the
/// schedule puts them in, where a shared counter would depend on which node read it first.
/// A replay starts the sweep over rather than continuing it without the counter being reset,
/// because the decode below is periodic in the slice count: the second replay's counts are the
/// first replay's plus a whole sweep, and a whole sweep is what the decode divides out.
struct TileCursor {
    std::size_t calls{0};
};

/// The bound one axis of one chunk member takes, resolved afresh on every iteration.
std::function<std::int64_t()> slice_bound(std::shared_ptr<TileCursor> cursor, std::size_t depth, std::size_t member, std::size_t stride,
                                          std::size_t extent, std::int64_t bias) {
    return [cursor = std::move(cursor), depth, member, stride, extent, bias]() -> std::int64_t {
        std::size_t const slice = (cursor->calls++ * depth) + member;
        return static_cast<std::int64_t>((slice / stride) % extent) + bias;
    };
}

/// The canonical spelling of a contraction's index lists, which is what the capture entry point
/// parses back into the same lists.
std::string einsum_spec_text(std::vector<std::string> const &a, std::vector<std::string> const &b, std::vector<std::string> const &c,
                             bool conj_a, bool conj_b) {
    auto side = [](std::vector<std::string> const &indices, bool conjugated) {
        std::string const joined = fmt::format("{}", fmt::join(indices, ","));
        return conjugated ? fmt::format("conj({})", joined) : joined;
    };
    return fmt::format("{} ; {} -> {}", side(a, conj_a), side(b, conj_b), fmt::format("{}", fmt::join(c, ",")));
}

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

/// The analysis over one contiguous run of tileable nodes.
class RegionAnalysis {
  public:
    RegionAnalysis(Graph const &graph, std::size_t first, std::size_t last) : _graph(graph), _first(first), _last(last) {}

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
    [[nodiscard]] bool closes(Labelling const &labelling, std::string &accumulator, TensorId &accumulator_id, std::string &reason) const;

    /// Bytes per slice of the largest intermediate, and the traffic the schedule streams.
    void cost(Labelling const &labelling, std::vector<std::size_t> const &extents, std::size_t depth, std::size_t &largest,
              long double &traffic) const;

    /// Names the decision gives to each sliced axis, and the letter a contraction calls it.
    void describe(std::vector<std::size_t> const &seed_positions, std::vector<std::string> &names, std::vector<std::string> &letters) const;

    /// Which tensors stream and which are left whole under @p labelling.
    void partition(Labelling const &labelling, std::vector<std::string> &streamed, std::vector<std::string> &whole) const;

    [[nodiscard]] Graph const &graph() const noexcept { return _graph; }
    [[nodiscard]] std::size_t  first() const noexcept { return _first; }
    [[nodiscard]] std::size_t  last() const noexcept { return _last; }

  private:
    [[nodiscard]] std::vector<int> *labels_for(LabelStore &store, std::size_t node_index, std::size_t slot, SlotRef const &ref) const;

    /// Whether any operand of the node at @p index carries a sliced axis under @p labelling.
    [[nodiscard]] bool node_is_sliced(Labelling const &labelling, std::size_t index) const;

    /// @brief Which sliced axes one node of the region carries, by label.
    /// @param[in] labelling The candidate.
    /// @param[in] index     The node, in the parent's numbering.
    /// @return The labels its operands and destination carry, empty for a loop-invariant node.
    [[nodiscard]] std::set<int> node_labels(Labelling const &labelling, std::size_t index) const;

    Graph const &_graph;
    std::size_t  _first{0};
    std::size_t  _last{0};

    std::vector<TensorId>             _written;
    std::unordered_set<TensorId>      _written_set;
    std::unordered_set<TensorId>      _touched_outside;
    std::vector<std::vector<SlotRef>> _slots;
    std::size_t                       _largest_written_bytes{0};
    TensorId                          _seed{0};
    packed_gemm::ScalarType           _dtype{packed_gemm::ScalarType::Unknown};
    bool                              _all_runtime{true};
};

bool RegionAnalysis::collect() {
    auto const &nodes = _graph.nodes();
    _slots.clear();
    _written.clear();
    _written_set.clear();
    _largest_written_bytes = 0;
    _seed                  = 0;

    for (std::size_t i = _first; i < _last; ++i) {
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
        if (i >= _first && i < _last) {
            continue;
        }
        auto const [reads, writes] = const_cast<Graph &>(_graph).effective_io(nodes[i]);
        for (TensorId const tid : reads) {
            _touched_outside.insert(_graph.resolve_alias(tid));
        }
        for (TensorId const tid : writes) {
            _touched_outside.insert(_graph.resolve_alias(tid));
        }
    }

    for (TensorId const tid : _written) {
        auto const *handle = _graph.find_tensor(tid);
        if (handle == nullptr) {
            continue;
        }
        std::size_t const bytes = handle->total_bytes();
        if (bytes > _largest_written_bytes || (bytes == _largest_written_bytes && _seed == 0)) {
            _largest_written_bytes = bytes;
            _seed                  = tid;
        }
    }
    return _seed != 0;
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
    std::size_t const sweeps = (_last - _first) * (kMaxSlicedAxes + 1) + 4;
    for (std::size_t sweep = 0; sweep < sweeps; ++sweep) {
        bool changed = false;
        for (std::size_t i = _first; i < _last; ++i) {
            auto const                     &node  = nodes[i];
            auto const                     &slots = _slots[i - _first];
            std::vector<std::vector<int> *> labels;
            labels.reserve(slots.size());
            for (std::size_t s = 0; s < slots.size(); ++s) {
                labels.push_back(labels_for(out.store, i, s, slots[s]));
            }

            if (node.kind == OpKind::Einsum || node.kind == OpKind::Permute) {
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
                // Slicing a summed letter would cut the contraction in half and cost a
                // partial-sum buffer the schedule was supposed to save.
                for (auto const &link : link_letters_of(node)) {
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
        auto const &slots = _slots[key.first - _first];
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

    // A sliced axis becomes a loop variable and leaves the operand, so an operand every one of
    // whose axes is sliced has nothing left to be. The emitted body would name a rank-zero
    // view, which is not an operand any kernel takes.
    auto survives = [](std::vector<int> const &labels) { return std::ranges::any_of(labels, [](int v) { return v == kFree; }); };
    for (auto const &[tid, labels] : out.store.tensor) {
        if (!labels.empty() && !survives(labels)) {
            out.feasible = false;
            out.reason   = "every axis of one operand is sliced, so the body would name an operand with no axes left";
            return out;
        }
    }
    for (auto const &[key, labels] : out.store.use) {
        if (!labels.empty() && !survives(labels)) {
            out.feasible = false;
            out.reason   = "every axis of one operand is sliced, so the body would name an operand with no axes left";
            return out;
        }
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
        auto const *desc = std::get_if<EinsumDescriptor>(&node.op_data);
        return desc == nullptr || !is_zero(live_c_prefactor(*desc));
    }
    case OpKind::Permute: {
        auto const *desc = std::get_if<PermuteDescriptor>(&node.op_data);
        return desc == nullptr || !is_zero(desc->params ? desc->params->beta : PrefactorScalar{desc->beta});
    }
    case OpKind::Axpby: {
        auto const *desc = std::get_if<AxpbyDescriptor>(&node.op_data);
        return desc == nullptr || !is_zero(live_beta(*desc));
    }
    case OpKind::DirectProduct:
    case OpKind::DirectDivision: {
        auto const *desc = std::get_if<ElementwiseBinaryDescriptor>(&node.op_data);
        return desc == nullptr || !is_zero(live_beta(*desc));
    }
    case OpKind::Dot:
        return false; // it writes element zero; the LOOP is what accumulates it
    default:
        return true;
    }
}

} // namespace

bool RegionAnalysis::closes(Labelling const &labelling, std::string &accumulator, TensorId &accumulator_id, std::string &reason) const {
    auto const &nodes = _graph.nodes();
    accumulator.clear();
    accumulator_id = 0;

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
        for (std::size_t i = _first; i < _last; ++i) {
            if (_slots[i - _first][0].tid != tid) {
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
        for (std::size_t i = _first; i < _last; ++i) {
            auto const &slots = _slots[i - _first];
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
    for (std::size_t i = _first; i < _last; ++i) {
        bool const adds = node_accumulates(nodes[i]) || (accumulator_id != 0 && _slots[i - _first][0].tid == accumulator_id);
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
                          long double &traffic) const {
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
    for (auto const &[tid, labels] : labelling.store.tensor) {
        auto const *handle = _graph.find_tensor(tid);
        if (handle == nullptr || !handle->is_intermediate) {
            continue;
        }
        largest = std::max(largest, static_cast<std::size_t>(sliced_bytes(tid, labels)));
    }

    // Traffic is what the whole schedule streams: every read, once per iteration, times the
    // iteration count. Minimising it under the cap is the memory-constrained fusion problem;
    // minimising it alone would always prefer slicing nothing at all.
    long double per_iteration = 0.0L;
    for (std::size_t i = _first; i < _last; ++i) {
        auto const &slots = _slots[i - _first];
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
    traffic = per_iteration * (slices / static_cast<long double>(depth));
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
    for (std::size_t i = _first; i < _last; ++i) {
        auto const &slots = _slots[i - _first];
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
    for (std::size_t i = _first; i < _last; ++i) {
        for (auto const &slot : _slots[i - _first]) {
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
                    if (_slots[key.first - _first][key.second].tid != slot.tid) {
                        continue;
                    }
                    labelled = labelled || std::ranges::any_of(labels, [](int v) { return v != kFree; });
                }
            }
            (labelled ? streamed : whole).push_back(handle->name);
        }
    }
}

std::set<int> RegionAnalysis::node_labels(Labelling const &labelling, std::size_t index) const {
    std::set<int> carried;
    auto const   &slots = _slots[index - _first];
    for (std::size_t slot = 0; slot < slots.size(); ++slot) {
        std::vector<int> const *labels = nullptr;
        if (_written_set.contains(slots[slot].tid)) {
            auto const hit = labelling.store.tensor.find(slots[slot].tid);
            labels         = hit == labelling.store.tensor.end() ? nullptr : &hit->second;
        } else {
            auto const hit = labelling.store.use.find({index, slot});
            labels         = hit == labelling.store.use.end() ? nullptr : &hit->second;
        }
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

bool RegionAnalysis::build(Labelling const &labelling, Plan &plan, std::string &reason) const {
    auto const &nodes = _graph.nodes();
    plan.ops.clear();
    plan.body_owned.clear();
    plan.dtype = _dtype;

    auto labels_of = [&](std::size_t node_index, std::size_t slot) -> std::vector<int> {
        auto const &ref = _slots[node_index - _first][slot];
        if (_written_set.contains(ref.tid)) {
            auto const hit = labelling.store.tensor.find(ref.tid);
            return hit == labelling.store.tensor.end() ? std::vector<int>(ref.rank, kFree) : hit->second;
        }
        auto const hit = labelling.store.use.find({node_index, slot});
        return hit == labelling.store.use.end() ? std::vector<int>(ref.rank, kFree) : hit->second;
    };

    plan.replaced.clear();
    for (std::size_t i = _first; i < _last; ++i) {
        if (!node_is_sliced(labelling, i)) {
            continue;
        }
        NodePlan op;
        op.kind    = nodes[i].kind;
        op.op_data = nodes[i].op_data;
        for (std::size_t slot = 0; slot < _slots[i - _first].size(); ++slot) {
            op.slots.push_back(SlotPlan{.tid = _slots[i - _first][slot].tid, .labels = labels_of(i, slot)});
        }
        plan.ops.push_back(std::move(op));
        plan.replaced.push_back(i);
    }
    if (plan.ops.empty()) {
        reason = "no statement of the run carries a sliced axis, so there is nothing for a loop to hold";
        return false;
    }
    plan.first = plan.replaced.front();

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
        // A caller's tensor keeps its own buffer and is written through a slice of it; only a
        // graph-owned intermediate is re-declared at slice extents, and only when nothing
        // outside the run still wants the whole of it.
        if (!handle->is_intermediate) {
            continue;
        }
        if (_touched_outside.contains(_graph.resolve_alias(tid))) {
            reason = "an intermediate the loop would declare at slice extents is read outside the run, where the whole of it is wanted";
            return false;
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

/// Zero the accumulator in the PARENT, before the loop.
///
/// The reduction the region wrote OVERWROTE its destination, and the loop accumulates into it,
/// so the two agree only if the destination starts at zero on every replay. A scale by zero
/// assigns rather than multiplies, which is what keeps a destination holding a non-finite value
/// from carrying it into the sum.
template <typename T>
void zero_accumulator(Graph &graph, TensorHandle const &handle) {
    CaptureGuard const guard(graph);
    if (handle.is_tensor_view) {
        scale(T{0}, static_cast<RuntimeTensorView<T> *>(handle.tensor_ptr));
    } else {
        scale(T{0}, static_cast<RuntimeTensor<T> *>(handle.tensor_ptr));
    }
}

/// The index letters a slot keeps once its sliced axes have become loop variables.
///
/// A sliced axis is DROPPED from the operand rather than kept as an axis of extent one. Both
/// spell the same arithmetic, and the drop is what makes the body's contractions the shapes a
/// kernel recognises: at one occupied pair the four-index contraction is an ordinary matrix
/// product over the auxiliary index, which is what the hand-written pair loop performs, where a
/// rank-four contraction with two unit axes would reach the generic algorithm instead.
std::vector<std::string> surviving(std::vector<std::string> const &letters, std::vector<int> const &labels) {
    std::vector<std::string> out;
    out.reserve(letters.size());
    for (std::size_t p = 0; p < letters.size(); ++p) {
        if (p < labels.size() && labels[p] != kFree) {
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
template <typename T>
void emit_body(Graph &parent, Graph &body, Plan const &plan) {
    std::size_t const depth = std::max<std::size_t>(1, plan.depth);
    std::size_t const axes  = plan.extents.size();

    // Row-major strides over the sliced axes, so one integer names one slice.
    std::vector<std::size_t> stride(axes, 1);
    for (std::size_t k = axes; k-- > 0;) {
        stride[k] = (k + 1 < axes) ? stride[k + 1] * plan.extents[k + 1] : 1;
    }

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
                continue;
            }
            dims.push_back(handle.dims[p]);
        }
        for (std::size_t member = 0; member < depth; ++member) {
            owned[{tid, member}] = &body.declare_runtime_tensor<T>(fmt::format("{}#{}", handle.name, member), dims, /*intermediate=*/true);
        }
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
    std::vector<std::vector<std::string>> index_of(depth);
    for (std::size_t member = 0; member < depth; ++member) {
        for (std::size_t k = 0; k < axes; ++k) {
            auto const name = fmt::format("axtile:{}:{}", k, member);
            write_param(name, slice_bound(std::make_shared<TileCursor>(), depth, member, stride[k], plan.extents[k], 0));
            // Seed the table with the first iteration's index, so a view recorded below carries
            // the SLICE's shape rather than the parent's and the capture-time shape checks see
            // the program that will run. The write above overwrites it on every iteration.
            body.params_ptr()->set(name, static_cast<std::int64_t>((member / stride[k]) % plan.extents[k]));
            index_of[member].push_back(name);
        }
    }

    // ── The operands ────────────────────────────────────────────────────────
    std::map<std::tuple<TensorId, std::vector<int>, std::size_t>, RuntimeTensorView<T> *> views;

    auto axes_for = [&](std::vector<int> const &labels, std::size_t rank, std::size_t member) {
        std::vector<ViewAxis> out;
        out.reserve(rank);
        for (std::size_t p = 0; p < rank; ++p) {
            if (p < labels.size() && labels[p] != kFree) {
                out.push_back(ViewAxis::drop(BoundExpr(index_of[member][static_cast<std::size_t>(labels[p])])));
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
            auto const &handle = parent.tensor(tid);
            auto        recipe = axes_for(labels, handle.rank, member);
            made = handle.is_tensor_view ? &view_runtime(*static_cast<RuntimeTensorView<T> *>(handle.tensor_ptr), std::move(recipe))
                                         : &view_runtime(*static_cast<RuntimeTensor<T> *>(handle.tensor_ptr), std::move(recipe));
        }
        views.emplace(key, made);
        return *made;
    };

    // ── The algebra, one family at a time ───────────────────────────────────
    bool const chunked = depth > 1;
    for (auto const &op : plan.ops) {
        bool const accumulating = plan.accumulator_id != 0 && op.slots[0].tid == plan.accumulator_id;

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
            auto const &desc = std::get<EinsumDescriptor>(op.op_data);
            auto const &spec =
                desc.indices ? desc.indices->spec : ParsedEinsumSpec{desc.spec.c_indices, desc.spec.a_indices, desc.spec.b_indices};
            auto const c        = surviving(spec.c_indices, op.slots[0].labels);
            auto const a        = surviving(spec.a_indices, op.slots[1].labels);
            auto const b        = surviving(spec.b_indices, op.slots[2].labels);
            auto const alpha    = real_prefactor(live_ab_prefactor(desc));
            auto const beta     = real_prefactor(live_c_prefactor(desc));
            auto const as_gemm  = gemm_shape(c, a, b, desc.indices ? desc.indices->link_indices : desc.spec.link_indices);
            bool       readable = true;
            for (auto const &slot : operand) {
                for (auto const *view : slot) {
                    readable = readable && gemm_readable(*view);
                }
            }
            if (chunked && readable && as_gemm.has_value() && alpha.has_value() && beta.has_value() && !live_conj_a(desc) &&
                !live_conj_b(desc)) {
                grouped_batched_gemm(*alpha, sources(1), sources(2), *beta, operand[0], as_gemm->trans_a, as_gemm->trans_b);
                break;
            }
            auto const text = einsum_spec_text(a, b, c, live_conj_a(desc), live_conj_b(desc));
            for (std::size_t member = 0; member < depth; ++member) {
                einsum(EinsumFormatString(text), as<T>(live_c_prefactor(desc)), operand[0][member], as<T>(live_ab_prefactor(desc)),
                       *operand[1][member], *operand[2][member]);
            }
            break;
        }
        case OpKind::Permute: {
            auto const &desc = std::get<PermuteDescriptor>(op.op_data);
            // A permutation keeps its scalars in the params block where every other kind does,
            // and its own snapshots are plain complex doubles, so the live block is read first
            // and the snapshot only stands in for a node that has none.
            PrefactorScalar const alpha = desc.params ? desc.params->alpha : PrefactorScalar{desc.alpha};
            PrefactorScalar const beta  = desc.params ? desc.params->beta : PrefactorScalar{desc.beta};
            auto const            text  = fmt::format("{} <- {}", fmt::join(surviving(desc.c_indices, op.slots[0].labels), ","),
                                                      fmt::join(surviving(desc.a_indices, op.slots[1].labels), ","));
            auto const            re_a  = real_prefactor(alpha);
            auto const            re_c  = real_prefactor(beta);
            if (chunked && re_a.has_value() && re_c.has_value()) {
                grouped_permute(text, operand[0], sources(1), std::vector<double>(depth, *re_c), std::vector<double>(depth, *re_a));
                break;
            }
            for (std::size_t member = 0; member < depth; ++member) {
                permute(PermuteFormatString(text), as<T>(beta), operand[0][member], as<T>(alpha), *operand[1][member]);
            }
            break;
        }
        case OpKind::Axpby: {
            auto const &desc  = std::get<AxpbyDescriptor>(op.op_data);
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
            auto const &desc = std::get<ScaleDescriptor>(op.op_data);
            for (std::size_t member = 0; member < depth; ++member) {
                scale(as<T>(live_factor(desc)), operand[0][member]);
            }
            break;
        }
        case OpKind::DirectProduct:
        case OpKind::DirectDivision: {
            auto const &desc  = std::get<ElementwiseBinaryDescriptor>(op.op_data);
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
        default:
            break;
        }
    }
}

/// The dtype dispatch, and the only place the pass names an element type.
bool emit_tiled_loop(Graph &graph, Plan const &plan) {
    auto emit = [&]<typename T>() {
        if (plan.accumulator_id != 0) {
            zero_accumulator<T>(graph, graph.tensor(plan.accumulator_id));
        }
        // The zeroing was appended; it belongs where the region was, ahead of the loop. One
        // splice moves it there and erases the region in the same edit, so no writer can end up
        // behind a reader that survived.
        std::vector<bool> remove(graph.nodes().size(), false);
        std::vector<Node> moved;
        if (plan.accumulator_id != 0) {
            remove.back() = true;
            moved.push_back(graph.nodes().back());
        }
        for (std::size_t const position : plan.replaced) {
            remove[position] = true;
        }
        // At the LAST replaced position rather than the first. A statement none of whose
        // operands carries a sliced axis stays where it is, and one of those may sit between two
        // sliced statements: the loop reads what it writes, and a loop node carries no operand
        // lists of its own for the dependency sort to order it by. The graph was sorted before
        // the rewrite, so every such writer precedes the sliced statement that reads it, and
        // therefore precedes the last of them.
        std::size_t const site = plan.replaced.back();
        graph.replace_nodes(remove, {{site, std::move(moved)}});

        // Where that landed: the splice shifts an insert down by the erased nodes BELOW it, and
        // every replaced node but the last is below the site. The loop follows the zeroing.
        std::size_t const position = (site - (plan.replaced.size() - 1)) + (plan.accumulator_id != 0 ? 1 : 0);
        Graph            &body =
            graph.add_loop_at("AxisTiling", plan.iterations,
                              PredExpr::iteration(CmpOp::Lt, BoundExpr(static_cast<std::int64_t>(plan.iterations) - 1)), position);
        emit_body<T>(graph, body, plan);
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

} // namespace

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
    _accumulator.clear();
}

std::vector<std::string> AxisTiling::explain() const {
    if (_slice_count == 0) {
        return {};
    }
    std::vector<std::string> lines{fmt::format("AxisTiling: sliced {} over {} slice(s) in chunks of {}, {} iteration(s)",
                                               fmt::join(_axis_names, " x "), _slice_count, _depth, _iterations)};
    lines.push_back(fmt::format("AxisTiling: largest intermediate {} bytes -> {} bytes", _largest_before, _largest_after));
    if (!_accumulator.empty()) {
        lines.push_back(fmt::format("AxisTiling: '{}' accumulates across iterations", _accumulator));
    }
    return lines;
}

bool AxisTiling::run(Graph &graph) {
    PassCounter const tiled{_num_tiled};

    std::int64_t const cap = memory_cap();
    if (cap <= 0) {
        note_skip("the memory cap is zero, which is how a caller asks for the captured schedule");
        return tiled.moved();
    }

    graph.topological_sort();
    auto const &nodes = graph.nodes();

    // The longest run of nodes whose value at a slice is a function of their operands at that
    // slice. Anything else terminates the run, which is the same escape rule the region
    // framework uses spelled for a schedule rather than for an expression.
    std::size_t best_first = 0;
    std::size_t best_last  = 0;
    std::size_t run_first  = 0;
    for (std::size_t i = 0; i <= nodes.size(); ++i) {
        bool const ok = i < nodes.size() && tileable_kind(nodes[i].kind);
        if (!ok) {
            if (i - run_first > best_last - best_first) {
                best_first = run_first;
                best_last  = i;
            }
            run_first = i + 1;
        }
    }
    if (best_last - best_first < 2) {
        note_skip("the graph holds no run of two or more nodes whose value at a slice is a function of their operands at that slice");
        return tiled.moved();
    }

    RegionAnalysis analysis(graph, best_first, best_last);
    if (!analysis.collect()) {
        note_skip("a node of the run names operands this pass cannot read as a dense, single-buffer slice");
        return tiled.moved();
    }
    if (!analysis.all_runtime()) {
        note_skip("the run names an operand this pass cannot slice, because a rank-erased operand is one the emitted body can name "
                  "and a statically ranked one is not");
        return tiled.moved();
    }
    _largest_before = analysis.largest_written_bytes();

    if (_largest_before <= static_cast<std::size_t>(cap)) {
        note_skip("the largest intermediate already fits the cap, so there is nothing to stream",
                  fmt::format("{} bytes against a cap of {}", _largest_before, cap));
        report(1, fmt::format("declined: the largest intermediate is {} bytes and the cap is {}", _largest_before, cap));
        return tiled.moved();
    }

    auto const *seed_handle = graph.find_tensor(analysis.seed());
    if (seed_handle == nullptr) {
        note_skip("the largest intermediate of the run has no handle to take candidate axes from");
        return tiled.moved();
    }

    std::optional<Plan> best;
    std::size_t         infeasible = 0;
    std::size_t         over_cap   = 0;
    for (auto const &positions : candidate_sets(seed_handle->rank)) {
        auto const labelling = analysis.propagate(positions);
        if (!labelling.feasible) {
            ++infeasible;
            note_skip(labelling.reason, fmt::format("candidate axes {}", fmt::join(positions, ",")));
            continue;
        }
        std::string accumulator;
        TensorId    accumulator_id = 0;
        std::string reason;
        if (!analysis.closes(labelling, accumulator, accumulator_id, reason)) {
            note_skip(reason, fmt::format("candidate axes {}", fmt::join(positions, ",")));
            continue;
        }

        std::vector<std::size_t> extents;
        std::size_t              slices = 1;
        for (std::size_t const position : positions) {
            extents.push_back(seed_handle->dims[position]);
            slices *= seed_handle->dims[position];
        }

        std::size_t unit    = 0;
        long double traffic = 0.0L;
        analysis.cost(labelling, extents, 1, unit, traffic);
        if (unit > static_cast<std::size_t>(cap)) {
            ++over_cap;
            continue;
        }
        std::size_t const depth   = chunk_depth(slices, unit, static_cast<std::size_t>(cap));
        std::size_t       largest = 0;
        analysis.cost(labelling, extents, depth, largest, traffic);

        Plan plan;
        plan.first          = best_first;
        plan.last           = best_last;
        plan.extents        = extents;
        plan.slices         = slices;
        plan.depth          = depth;
        plan.iterations     = slices / depth;
        plan.largest_before = _largest_before;
        plan.largest_after  = largest;
        plan.traffic        = traffic;
        plan.accumulator    = accumulator;
        plan.accumulator_id = accumulator_id;
        plan.store          = labelling.store;
        if (!analysis.build(labelling, plan, reason)) {
            note_skip(reason, fmt::format("candidate axes {}", fmt::join(positions, ",")));
            continue;
        }
        analysis.describe(positions, plan.axis_names, plan.axis_letters);
        analysis.partition(labelling, plan.streamed, plan.whole);

        report(3, fmt::format("candidate {} streams {:.0f} bytes at depth {}", fmt::join(plan.axis_names, " x "),
                              static_cast<double>(traffic), depth));
        if (!best.has_value() || traffic < best->traffic) {
            best = std::move(plan);
        }
    }

    if (!best.has_value()) {
        note_skip("no candidate axis set brings the largest intermediate under the cap",
                  fmt::format("{} set(s) the program cannot carry a slice labelling for, {} above the cap", infeasible, over_cap));
        return tiled.moved();
    }

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

    if (!emit_tiled_loop(graph, *best)) {
        note_skip("the run's element type is not one the emitted body can name", fmt::format("dtype {}", static_cast<int>(best->dtype)));
        return tiled.moved();
    }
    ++_num_tiled;
    graph.note_structural_change();
    graph.topological_sort();

    report(1, fmt::format("slicing {} into {} slice(s), chunks of {}, largest intermediate {} -> {} bytes", fmt::join(_axis_names, " x "),
                          _slice_count, _depth, _largest_before, _largest_after));
    return tiled.moved();
}

EINSUMS_NAMESPACE_END(compute_graph::passes)
