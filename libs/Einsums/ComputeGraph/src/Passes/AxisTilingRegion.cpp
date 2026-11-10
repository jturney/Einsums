//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// AxisTiling: the analysis of one region, from collecting its tensors to the plan the rewrite reads.

#include <Einsums/Config/Namespace.hpp>

#include "AxisTilingDetail.hpp"

EINSUMS_NAMESPACE_BEGIN(compute_graph::passes::axis_tiling)

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
                          std::size_t &held, long double &traffic, long double &contracted) const {
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
    long double per_iteration   = 0.0L;
    long double per_contraction = 0.0L;
    for (std::size_t const i : _members) {
        auto const &slots  = slots_at(i);
        bool const  priced = _graph.nodes()[i].kind == OpKind::Einsum && node_is_sliced(labelling, i);
        for (std::size_t s = 0; s < slots.size(); ++s) {
            auto const *labels = _written_set.contains(slots[s].tid)
                                     ? (labelling.store.tensor.contains(slots[s].tid) ? &labelling.store.tensor.at(slots[s].tid) : nullptr)
                                     : (labelling.store.use.contains({i, s}) ? &labelling.store.use.at({i, s}) : nullptr);
            static std::vector<int> const empty;
            long double const             bytes = sliced_bytes(slots[s].tid, labels == nullptr ? empty : *labels);
            per_iteration += bytes;
            if (priced) {
                per_contraction += bytes;
            }
        }
    }
    long double slices = 1.0L;
    for (std::size_t const extent : extents) {
        slices *= static_cast<long double>(extent);
    }
    long double const iterations = slices / static_cast<long double>(depth);
    traffic                      = per_iteration * iterations;
    contracted                   = per_contraction * iterations;

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
                                std::size_t &largest, std::size_t &held, long double &traffic, long double &contracted) const {
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
    traffic    = 0.0L;
    contracted = 0.0L;
    for (std::size_t const i : _members) {
        bool const  inside = node_is_sliced(labelling, i);
        auto const &slots  = slots_at(i);
        for (std::size_t s = 0; s < slots.size(); ++s) {
            std::vector<int> const *labels = inside ? slot_labels(labelling, i, s) : nullptr;
            Measure const           m      = measure(slots[s].tid, labels);
            traffic += inside ? m.streamed : m.block;
            if (inside && _graph.nodes()[i].kind == OpKind::Einsum) {
                contracted += m.streamed;
            }

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

EINSUMS_NAMESPACE_END(compute_graph::passes::axis_tiling)
