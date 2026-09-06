//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Options.hpp>
#include <Einsums/ComputeGraph/Passes/AxisTiling.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Options/Get.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
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
/// nothing to propagate a labelling through.
bool tileable_kind(OpKind kind) {
    switch (kind) {
    case OpKind::Einsum:
    case OpKind::Permute:
    case OpKind::Axpby:
    case OpKind::Scale:
    case OpKind::DirectProduct:
    case OpKind::DirectDivision:
    case OpKind::Dot:
    case OpKind::ElementTransform:
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
    case OpKind::ElementTransform:
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

    /// Propagate a seed labelling through the region. An infeasible candidate comes back with
    /// @ref Labelling::feasible false.
    [[nodiscard]] Labelling propagate(std::vector<std::size_t> const &seed_positions) const;

    /// Whether a labelling can be turned into a loop: every unlabelled tensor the region
    /// writes is either its reduction target or purely internal.
    [[nodiscard]] bool closes(Labelling const &labelling, std::string &accumulator, std::string &reason) const;

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

    Graph const &_graph;
    std::size_t  _first{0};
    std::size_t  _last{0};

    std::vector<TensorId>             _written;
    std::unordered_set<TensorId>      _written_set;
    std::vector<std::vector<SlotRef>> _slots;
    std::size_t                       _largest_written_bytes{0};
    TensorId                          _seed{0};
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
                auto const letters = letters_of(node);
                if (!letters.has_value() || letters->size() < slots.size()) {
                    out.feasible = false;
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
                            return out;
                        }
                    }
                }
                // Slicing a summed letter would cut the contraction in half and cost a
                // partial-sum buffer the schedule was supposed to save.
                for (auto const &link : link_letters_of(node)) {
                    if (by_letter.contains(link)) {
                        out.feasible = false;
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
                            return out;
                        }
                    }
                }
            } else if (node.kind == OpKind::Dot) {
                // The two operands align position by position; the scalar carries no label,
                // which is what makes it the reduction the loop accumulates.
                if (slots.size() != 3 || !unify(*labels[1], *labels[2], changed)) {
                    out.feasible = false;
                    return out;
                }
                if (std::ranges::any_of(*labels[0], [](int v) { return v != kFree; })) {
                    out.feasible = false;
                    return out;
                }
            } else {
                for (std::size_t s = 1; s < slots.size(); ++s) {
                    if (!unify(*labels[0], *labels[s], changed)) {
                        out.feasible = false;
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
            return out;
        }
    }
    for (auto const &[key, labels] : out.store.use) {
        auto const &slots = _slots[key.first - _first];
        if (!check(slots[key.second].tid, labels)) {
            out.feasible = false;
            return out;
        }
    }
    for (std::size_t const extent : extents) {
        if (extent < 2) {
            // An axis of one slice is not a schedule, and an axis nothing carries is a
            // candidate the propagation dissolved.
            out.feasible = false;
            return out;
        }
    }
    return out;
}

bool RegionAnalysis::closes(Labelling const &labelling, std::string &accumulator, std::string &reason) const {
    auto const &nodes = _graph.nodes();
    accumulator.clear();

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
        // The one unlabelled destination a loop can produce is a reduction: written once, by a
        // reduction to a scalar over operands that do carry sliced axes, and read by nothing
        // else inside the region.
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
        accumulator = handle->name;
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
            continue;
        }
        std::string accumulator;
        std::string reason;
        if (!analysis.closes(labelling, accumulator, reason)) {
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
        plan.store          = labelling.store;
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

    report(1, fmt::format("slicing {} into {} slice(s), chunks of {}, largest intermediate {} -> {} bytes", fmt::join(_axis_names, " x "),
                          _slice_count, _depth, _largest_before, _largest_after));
    return tiled.moved();
}

EINSUMS_NAMESPACE_END(compute_graph::passes)
