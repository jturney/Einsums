//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/BufferAllocator/Options.hpp>
#include <Einsums/ComputeGraph/CaptureContext.hpp>
#include <Einsums/ComputeGraph/Detail/BackedWrites.hpp>
#include <Einsums/ComputeGraph/Detail/ScalarDispatch.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Operations.hpp>
#include <Einsums/ComputeGraph/Options.hpp>
#include <Einsums/ComputeGraph/Passes/SlabStreaming.hpp>
#include <Einsums/ComputeGraph/Prefactor.hpp>
#include <Einsums/ComputeGraph/View.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Options/Get.hpp>
#include <Einsums/StringUtil/MemoryString.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <ranges>
#include <string>
#include <utility>
#include <vector>

#include "BackingFill.hpp"
#include "LoopEmission.hpp"

EINSUMS_NAMESPACE_BEGIN(compute_graph::passes)

namespace {

using detail::AxisWalk;
using detail::declare_walk_bounds;
using detail::einsum_spec_text;
using detail::slice_bound;
using detail::TileCursor;
using detail::walk_iterations;

/// An operand of a contraction that reads a backed tensor: the tensor itself, or a full,
/// unpermuted view of it.
struct BackedOperand {
    std::size_t                          slot{0};    ///< 0 for the destination, 1 for A, 2 for B.
    TensorId                             operand{0}; ///< The id the node names.
    std::optional<std::size_t>           view_node;  ///< The View node that made it, for a view.
    std::string                          name;       ///< The backed tensor's name.
    std::shared_ptr<TensorBacking const> backing;
    void const                          *object{nullptr};        ///< The backed tensor object.
    bool                                 written_outside{false}; ///< See SlabStreaming::BackedEntry.
    bool                                 full{true};             ///< False for a view that slices or permutes.
};

/// Where the View node that produced @p tid sits, if a View node did.
std::optional<std::size_t> producing_view(Graph const &graph, TensorId tid) {
    auto const &nodes = graph.nodes();
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        if (nodes[i].kind == OpKind::View && std::ranges::find(nodes[i].outputs, tid) != nodes[i].outputs.end()) {
            return i;
        }
    }
    return std::nullopt;
}

/// The backed tensor @p tid reads, if it reads one.
template <typename Backed>
std::optional<BackedOperand> backed_operand(Graph const &graph, Backed const &backed, TensorId tid, std::size_t slot) {
    TensorHandle const *handle = graph.find_tensor(tid);
    if (handle == nullptr) {
        return std::nullopt;
    }
    TensorHandle const *owner = graph.find_tensor(graph.buffer_of(tid));
    if (owner == nullptr || owner->tensor_ptr == nullptr) {
        return std::nullopt;
    }
    auto const hit = backed.find(owner->tensor_ptr);
    if (hit == backed.end()) {
        return std::nullopt;
    }
    BackedOperand out{.slot            = slot,
                      .operand         = tid,
                      .name            = hit->second.name,
                      .backing         = hit->second.backing,
                      .object          = owner->tensor_ptr,
                      .written_outside = hit->second.written_outside};
    if (handle->is_tensor_view) {
        out.view_node    = producing_view(graph, tid);
        auto const *view = out.view_node ? graph.nodes()[*out.view_node].op_data.get_if<ViewDescriptor>() : nullptr;
        out.full         = view != nullptr && graph.buffer_of(view->parent_id) == graph.buffer_of(tid) && view->permutation.empty() &&
                           std::ranges::all_of(view->axes, [](ViewAxis const &axis) { return axis.kind == ViewAxis::Kind::Full; });
    }
    return out;
}

/// How many times @p letter occurs in @p letters.
std::size_t occurrences(std::vector<std::string> const &letters, std::string const &letter) {
    return static_cast<std::size_t>(std::ranges::count(letters, letter));
}

/// The positions of @p letter in @p letters.
std::vector<std::size_t> positions_of(std::vector<std::string> const &letters, std::string const &letter) {
    std::vector<std::size_t> out;
    for (std::size_t p = 0; p < letters.size(); ++p) {
        if (letters[p] == letter) {
            out.push_back(p);
        }
    }
    return out;
}

/// What the body needs to know about an operand that is not the backed one: the tensor to view,
/// the view of it the node read (every axis full for the tensor itself), and which of that
/// tensor's axes carry the streamed letter.
///
/// An operand that is itself a view is viewed through to its parent with its own axes rather than
/// viewed again, so the loop holds one view per operand whatever produced the original.
struct OtherOperand {
    /// The tensor's live object (@ref TensorHandle::live_ptr), which is what the loop captures
    /// over: the caller may have dropped its own tensor once it was captured.
    void                      *object{nullptr};
    bool                       is_view{false};
    std::vector<ViewAxis>      base;
    std::vector<std::size_t>   streamed;
    std::optional<std::size_t> view_node; ///< The View node that made the operand, when one did.
};

/// The operand @p tid as a view over the tensor underneath it, with the axes carrying the
/// streamed letter (@p positions, in the operand's numbering) found in that tensor's numbering;
/// nothing, with the reason in @p why, when the operand cannot take a block of the letter.
std::optional<OtherOperand> underlying(Graph const &graph, TensorId tid, std::vector<std::size_t> const &positions, std::string &why) {
    TensorHandle const *handle = graph.find_tensor(tid);
    if (handle == nullptr || handle->tensor_ptr == nullptr || !handle->is_runtime) {
        why = "an operand is not a rank-erased tensor";
        return std::nullopt;
    }
    OtherOperand out;
    if (!handle->is_tensor_view) {
        out.object   = handle->live_ptr();
        out.base     = std::vector<ViewAxis>(handle->rank, ViewAxis::full());
        out.streamed = positions;
        return out;
    }
    out.view_node              = producing_view(graph, tid);
    auto const         *view   = out.view_node ? graph.nodes()[*out.view_node].op_data.get_if<ViewDescriptor>() : nullptr;
    TensorHandle const *parent = view != nullptr ? graph.find_tensor(view->parent_id) : nullptr;
    if (view == nullptr || parent == nullptr || parent->tensor_ptr == nullptr || !parent->is_runtime || !view->permutation.empty()) {
        why = "an operand is a view this pass cannot see through";
        return std::nullopt;
    }
    // The view's axis k is the k-th of the parent's axes that the view does not drop.
    std::vector<std::size_t> kept;
    for (std::size_t p = 0; p < view->axes.size(); ++p) {
        if (view->axes[p].kind != ViewAxis::Kind::Drop) {
            kept.push_back(p);
        }
    }
    for (std::size_t const k : positions) {
        if (k >= kept.size() || view->axes[kept[k]].kind != ViewAxis::Kind::Full) {
            why = "an operand is a view that already slices the streamed letter";
            return std::nullopt;
        }
        out.streamed.push_back(kept[k]);
    }
    out.object  = parent->live_ptr();
    out.is_view = parent->is_tensor_view;
    out.base    = view->axes;
    return out;
}

/// Everything a streamed contraction's loops are built from, copied out of the graph before the
/// node is removed.
struct Rewrite {
    BackedOperand            backed;
    std::vector<std::string> a;
    std::vector<std::string> b;
    std::vector<std::string> c;
    bool                     conj_a{false};
    bool                     conj_b{false};
    PrefactorScalar          c_prefactor;
    PrefactorScalar          ab_prefactor;
    OtherOperand             other;
    OtherOperand             destination;
    std::vector<std::size_t> dims;
    std::size_t              block{1};
    bool                     summed{false};
    std::string              label;
    std::string              prefix;
};

/// The walks over the streamed axis, every one a RANGE: the slab keeps the axis at the block's
/// extent, a block of one included. AxisTiling's walks make a block of one a point, which drops
/// the axis, and a body built on that has no end bound for its views to stop at.
///
/// A summed letter's first block is a loop of its own: it is the one that applies the
/// destination's prefactor, and every later block adds.
std::vector<AxisWalk> streamed_walks(std::size_t extent, std::size_t block, bool summed) {
    std::vector<AxisWalk> out;
    std::size_t const     full = extent / block;
    if (full > 0) {
        if (summed && full > 1) {
            out.push_back(AxisWalk{.count = 1, .step = block, .offset = 0, .length = block, .range = true});
            out.push_back(AxisWalk{.count = full - 1, .step = block, .offset = block, .length = block, .range = true});
        } else {
            out.push_back(AxisWalk{.count = full, .step = block, .offset = 0, .length = block, .range = true});
        }
    }
    if (extent % block != 0) {
        out.push_back(AxisWalk{.count = 1, .step = block, .offset = full * block, .length = extent % block, .range = true});
    }
    return out;
}

template <typename T>
RuntimeTensorView<T> &view_over(OtherOperand const &operand, std::string const &lo, std::string const &hi) {
    std::vector<ViewAxis> recipe = operand.base;
    for (std::size_t const p : operand.streamed) {
        recipe[p] = ViewAxis::range(BoundExpr(lo), BoundExpr(hi));
    }
    return operand.is_view ? view_runtime(*static_cast<RuntimeTensorView<T> *>(operand.object), std::move(recipe))
                           : view_runtime(*static_cast<RuntimeTensor<T> *>(operand.object), std::move(recipe));
}

/// One loop's body: read the block's slab, then run the contraction on it.
template <typename T>
void emit_body(Graph &body, Rewrite const &rw, AxisWalk const &walk, std::size_t loop) {
    std::vector<std::size_t> slab_dims = rw.dims;
    slab_dims.back()                   = walk.length;
    auto &slab = body.declare_runtime_tensor<T>(fmt::format("{}#slab{}", rw.backed.name, loop), slab_dims, /*intermediate=*/true);

    CaptureGuard const guard(body);
    auto const         bounds = declare_walk_bounds(body, 1, {walk}, rw.prefix, fmt::format("{}", loop));
    std::string const &lo     = bounds.start[0][0];
    std::string const &hi     = bounds.end[0][0];

    // The read follows the loop with cursors of its own rather than through the parameters,
    // which order a parameter-bound VIEW after its write and nothing else.
    auto       &ctx     = CaptureContext::current();
    auto const  slab_id = ctx.get_or_register(slab);
    auto        first = slice_bound(std::make_shared<TileCursor>(), 1, 0, 1, walk.count, walk.step, static_cast<std::int64_t>(walk.offset));
    auto        last  = slice_bound(std::make_shared<TileCursor>(), 1, 0, 1, walk.count, walk.step,
                                    static_cast<std::int64_t>(walk.offset) + static_cast<std::int64_t>(walk.length));
    std::size_t bytes = sizeof(T);
    for (std::size_t const extent : slab_dims) {
        bytes *= extent;
    }
    ctx.record(
        OpKind::DiskRead, fmt::format("read_slab({})", rw.backed.name), {}, {slab_id},
        [backing = rw.backed.backing, dims = rw.dims, first = std::move(first), last = std::move(last), target = &slab]() {
            std::vector<std::pair<std::size_t, std::size_t>> ranges;
            ranges.reserve(dims.size());
            for (std::size_t p = 0; p + 1 < dims.size(); ++p) {
                ranges.emplace_back(0, dims[p]);
            }
            auto const start = static_cast<std::size_t>(first());
            auto const end   = static_cast<std::size_t>(last());
            ranges.emplace_back(start, end);
            detail::fill_from_backing(*backing, target->data(), target->impl(), ranges);
        },
        compute_graph::detail::backing_io(*rw.backed.backing, slab_id, bytes));

    auto const &slab_view   = view_runtime(slab, std::vector<ViewAxis>(slab.rank(), ViewAxis::full()));
    auto const &other       = view_over<T>(rw.other, lo, hi);
    auto       &destination = view_over<T>(rw.destination, lo, hi);
    auto const &a           = rw.backed.slot == 1 ? slab_view : other;
    auto const &b           = rw.backed.slot == 1 ? other : slab_view;

    // Every block of a free letter writes its own slice, so each applies the prefactor; a summed
    // letter's blocks add into one destination, which only the first may scale.
    T const    c_prefactor = (!rw.summed || loop == 0) ? as<T>(rw.c_prefactor) : T{1};
    auto const text        = einsum_spec_text(rw.a, rw.b, rw.c, rw.conj_a, rw.conj_b, {});
    einsum(EinsumFormatString(text), c_prefactor, &destination, as<T>(rw.ab_prefactor), a, b);
}

/// Everything a streamed producer's loops are built from, copied out of the graph before the
/// node is removed.
struct WriteRewrite {
    BackedOperand            backed;
    std::vector<std::string> a;
    std::vector<std::string> b;
    std::vector<std::string> c;
    bool                     conj_a{false};
    bool                     conj_b{false};
    PrefactorScalar          ab_prefactor;
    OtherOperand             in_a;
    OtherOperand             in_b;
    std::vector<std::size_t> dims;
    std::size_t              block{1};
    std::string              label;
    std::string              prefix;
};

/// One loop's body: compute the block's slab of the destination, then write it to the storage.
template <typename T>
void emit_write_body(Graph &body, WriteRewrite const &rw, AxisWalk const &walk, std::size_t loop) {
    std::vector<std::size_t> slab_dims = rw.dims;
    slab_dims.back()                   = walk.length;
    auto &slab = body.declare_runtime_tensor<T>(fmt::format("{}#slab{}", rw.backed.name, loop), slab_dims, /*intermediate=*/true);

    CaptureGuard const guard(body);
    auto const         bounds = declare_walk_bounds(body, 1, {walk}, rw.prefix, fmt::format("{}", loop));
    std::string const &lo     = bounds.start[0][0];
    std::string const &hi     = bounds.end[0][0];

    auto       &slab_view = view_runtime(slab, std::vector<ViewAxis>(slab.rank(), ViewAxis::full()));
    auto const &a         = view_over<T>(rw.in_a, lo, hi);
    auto const &b         = view_over<T>(rw.in_b, lo, hi);
    auto const  text      = einsum_spec_text(rw.a, rw.b, rw.c, rw.conj_a, rw.conj_b, {});
    einsum(EinsumFormatString(text), T{0}, &slab_view, as<T>(rw.ab_prefactor), a, b);

    // The write follows the loop with cursors of its own, as a read does in emit_body.
    auto       &ctx     = CaptureContext::current();
    auto const  slab_id = ctx.get_or_register(slab);
    auto        first = slice_bound(std::make_shared<TileCursor>(), 1, 0, 1, walk.count, walk.step, static_cast<std::int64_t>(walk.offset));
    auto        last  = slice_bound(std::make_shared<TileCursor>(), 1, 0, 1, walk.count, walk.step,
                                    static_cast<std::int64_t>(walk.offset) + static_cast<std::int64_t>(walk.length));
    std::size_t bytes = sizeof(T);
    for (std::size_t const extent : slab_dims) {
        bytes *= extent;
    }
    ctx.record(
        OpKind::DiskWrite, fmt::format("write_slab({})", rw.backed.name), {slab_id}, {},
        [backing = rw.backed.backing, dims = rw.dims, first = std::move(first), last = std::move(last), source = &slab]() {
            std::vector<std::pair<std::size_t, std::size_t>> ranges;
            ranges.reserve(dims.size());
            for (std::size_t p = 0; p + 1 < dims.size(); ++p) {
                ranges.emplace_back(0, dims[p]);
            }
            auto const start = static_cast<std::size_t>(first());
            auto const end   = static_cast<std::size_t>(last());
            ranges.emplace_back(start, end);
            detail::store_to_backing(*backing, source->data(), source->impl(), ranges);
        },
        compute_graph::detail::backing_io(*rw.backed.backing, slab_id, bytes));
}

/// Mark for removal the View node @p view_node that made @p view, unless a node other than
/// @p node or the view itself uses it.
///
/// Through the EFFECTIVE io, because a loop or a branch lists none of the tensors its body uses
/// and a view read only there would otherwise look unused.
void drop_view_if_unused(Graph &graph, std::vector<bool> &remove, std::size_t node, std::optional<std::size_t> view_node, TensorId view) {
    if (!view_node) {
        return;
    }
    auto const &nodes  = graph.nodes();
    bool const  shared = std::ranges::any_of(std::views::iota(std::size_t{0}, nodes.size()), [&](std::size_t j) {
        if (j == node || j == *view_node) {
            return false;
        }
        auto const [reads, writes] = graph.effective_io(nodes[j]);
        return std::ranges::find(reads, view) != reads.end() || std::ranges::find(writes, view) != writes.end();
    });
    if (!shared) {
        remove[*view_node] = true;
    }
}

} // namespace

std::size_t SlabStreaming::budget() const {
    if (_budget_override >= 0) {
        return static_cast<std::size_t>(_budget_override);
    }
    std::size_t const budget = string_util::memory_string(config::get(option::MaxMemory));
    if (budget == 0) {
        return 0;
    }
    double const fraction = config::get(option::GraphTilingBudgetShare);
    return fraction > 0.0 && fraction < 1.0 ? static_cast<std::size_t>(fraction * static_cast<double>(budget)) : budget;
}

bool SlabStreaming::run(Graph &graph) {
    // What the enclosing graph saw, then this graph's own backed tensors: a body's handle for a
    // parent's backed tensor carries no backing, so the object is what identifies it.
    Backed backed;
    if (auto const hit = _backed_of.find(&graph); hit != _backed_of.end()) {
        backed = hit->second;
    } else {
        // Only a body is handed a map, so the graph without one is the one the apply started at.
        _root = &graph;
    }
    for (auto const &[tid, handle] : graph.tensors_map()) {
        if (handle.backing && handle.tensor_ptr != nullptr) {
            backed.insert_or_assign(handle.tensor_ptr, BackedEntry{.name = handle.name, .backing = handle.backing});
        }
    }

    bool const moved = !backed.empty() && stream_graph(graph, backed);

    // Each body learns whether a write-backed tensor is written in memory anywhere OUTSIDE it:
    // above it, or in a sibling body it cannot see. A writer left in this graph's own nodes, or in
    // a body processed later, counts, which keeps the answer conservative and the program correct.
    for (auto const &node : graph.nodes()) {
        for_each_child_graph(node, [&](Graph const &sub) {
            Backed inherited = backed;
            for (auto &[object, entry] : inherited) {
                if (entry.backing->write) {
                    entry.written_outside = entry.written_outside || compute_graph::detail::written_in_memory(graph, object, &sub);
                }
            }
            _backed_of[&sub] = std::move(inherited);
        });
    }
    return moved;
}

bool SlabStreaming::stream_graph(Graph &graph, Backed const &backed) {
    std::size_t const cap = budget();
    if (cap == 0) {
        note_skip("no slab budget: --einsums:max-memory is zero or the pass was set off, so backed tensors are loaded whole");
        return false;
    }

    bool moved = false;
    // One rewrite at a time: each renumbers the nodes, and a rewritten contraction's loop reads a
    // slab, not a backed tensor, so it is never a candidate again.
    for (bool again = true; again;) {
        again       = false;
        auto &nodes = graph.nodes();
        for (std::size_t i = 0; i < nodes.size() && !again; ++i) {
            Node const &node = nodes[i];
            if (node.kind != OpKind::Einsum || node.outputs.size() != 1 || node.inputs.size() < 2) {
                continue;
            }
            auto const *desc = node.op_data.get_if<EinsumDescriptor>();
            if (desc == nullptr) {
                continue;
            }
            // A contraction that produces a write-backed tensor is a write candidate, whatever it reads.
            if (auto const out = backed_operand(graph, backed, node.outputs[0], 0); out && out->backing->write) {
                if (stream_write(graph, backed, i, cap)) {
                    moved = true;
                    again = true;
                }
                continue;
            }
            auto const in_a = backed_operand(graph, backed, node.inputs[0], 1);
            auto const in_b = backed_operand(graph, backed, node.inputs[1], 2);
            if (!in_a && !in_b) {
                continue;
            }
            auto const where = fmt::format("node '{}'", node.label);
            if (in_a && in_b) {
                note_skip("both operands of the contraction are backed, so it is loaded whole", where);
                continue;
            }
            BackedOperand const &use     = in_a ? *in_a : *in_b;
            auto const          &backing = *use.backing;
            // A write-backed tensor's storage holds it only once it has been produced there. Produced
            // in memory anywhere, it is read from memory, and stored after its last writer.
            if (backing.write && (use.written_outside || compute_graph::detail::written_in_memory(graph, use.object))) {
                note_skip("the tensor is produced in memory, so its readers read it there and it is stored after its last writer", where);
                continue;
            }
            if (!use.full) {
                note_skip("the backed tensor is read through a view that slices or permutes it, so it is loaded whole", where);
                continue;
            }

            std::size_t const element = [&] {
                return compute_graph::detail::dispatch_scalar_type(backing.dtype, []<typename T>(T) { return sizeof(T); });
            }();
            std::size_t whole = element;
            for (std::size_t const extent : backing.dims) {
                whole *= extent;
            }
            if (whole <= cap) {
                note_skip("the backed tensor fits its share of the budget, so it is loaded once rather than read per contraction",
                          fmt::format("{}: '{}' is {} bytes against {}", where, use.name, whole, cap));
                continue;
            }

            auto const lists = live_index_lists(*desc);
            if (!lists.operators.empty()) {
                note_skip("the contraction applies permutation operators, so its backed operand is loaded whole", where);
                continue;
            }
            auto const &mine = use.slot == 1 ? lists.a : lists.b;
            if (mine.empty() || backing.dims.empty() || backing.dims.back() == 0) {
                note_skip("the backed tensor has no last axis to read in blocks", where);
                continue;
            }
            std::string const letter = mine.back();
            auto const       &theirs = use.slot == 1 ? lists.b : lists.a;
            if (occurrences(mine, letter) != 1 || occurrences(theirs, letter) > 1 || occurrences(lists.c, letter) > 1) {
                note_skip("the letter of the backed tensor's last axis repeats, which a block of it cannot express", where);
                continue;
            }

            TensorId const      other_id    = use.slot == 1 ? node.inputs[1] : node.inputs[0];
            TensorHandle const *destination = graph.find_tensor(node.outputs[0]);
            TensorHandle const *other       = graph.find_tensor(other_id);
            if (destination == nullptr || other == nullptr || destination->dtype != backing.dtype || other->dtype != backing.dtype) {
                note_skip("an operand is not of the backed tensor's element type", where);
                continue;
            }
            // An empty operand leaves nothing to compute, and the captured contraction applies the
            // destination's prefactor on its own.
            auto const empty = [](TensorHandle const &handle) {
                return std::ranges::find(handle.dims, std::size_t{0}) != handle.dims.end();
            };
            if (empty(*destination) || empty(*other)) {
                note_skip("an operand has no elements, so there is nothing to stream", where);
                continue;
            }
            std::string why;
            auto const  other_view = underlying(graph, other_id, positions_of(theirs, letter), why);
            auto const  destination_view =
                other_view ? underlying(graph, node.outputs[0], positions_of(lists.c, letter), why) : std::nullopt;
            if (!other_view || !destination_view) {
                note_skip(why + ", so the backed operand is loaded whole", where);
                continue;
            }

            Rewrite rw;
            rw.backed                   = use;
            rw.a                        = lists.a;
            rw.b                        = lists.b;
            rw.c                        = lists.c;
            rw.conj_a                   = live_conj_a(*desc);
            rw.conj_b                   = live_conj_b(*desc);
            rw.c_prefactor              = live_c_prefactor(*desc);
            rw.ab_prefactor             = live_ab_prefactor(*desc);
            rw.other                    = *other_view;
            rw.destination              = *destination_view;
            rw.dims                     = backing.dims;
            rw.summed                   = rw.destination.streamed.empty();
            std::size_t const per_index = whole / backing.dims.back();
            rw.block                    = std::clamp<std::size_t>(cap / std::max<std::size_t>(per_index, 1), 1, backing.dims.back());
            rw.label                    = node.label;
            rw.prefix                   = fmt::format("slab:{}", _num_streamed);

            // The node goes, and with it every view it read or wrote that nothing else uses: the
            // loop views the tensors underneath directly, and a view left behind of the backed
            // tensor would name it and have it loaded whole after all.
            std::vector<bool> remove(nodes.size(), false);
            remove[i] = true;
            drop_view_if_unused(graph, remove, i, use.view_node, use.operand);
            drop_view_if_unused(graph, remove, i, rw.other.view_node, other_id);
            drop_view_if_unused(graph, remove, i, rw.destination.view_node, node.outputs[0]);
            auto const below = static_cast<std::size_t>(std::count(remove.begin(), remove.begin() + static_cast<std::ptrdiff_t>(i), true));

            // A rough count for the report: two flops per term of the full contraction.
            double flops = 2.0;
            {
                std::map<std::string, std::size_t> extent_of;
                auto const note_extents = [&](std::vector<std::string> const &letters, std::vector<std::size_t> const &dims) {
                    for (std::size_t p = 0; p < letters.size() && p < dims.size(); ++p) {
                        extent_of[letters[p]] = dims[p];
                    }
                };
                note_extents(mine, backing.dims);
                note_extents(theirs, other->dims);
                note_extents(lists.c, destination->dims);
                for (auto const &[name, extent] : extent_of) {
                    flops *= static_cast<double>(extent);
                }
            }

            graph.replace_nodes(remove, {});
            std::size_t const position = i - below;
            auto const        walks    = streamed_walks(backing.dims.back(), rw.block, rw.summed);
            compute_graph::detail::dispatch_scalar_type(backing.dtype, [&]<typename T>(T) {
                for (std::size_t l = 0; l < walks.size(); ++l) {
                    std::size_t const iterations = walk_iterations({walks[l]});
                    Graph &body = graph.add_loop_at(fmt::format("SlabStreaming#{}", l), iterations,
                                                    PredExpr::iteration(CmpOp::Lt, BoundExpr(static_cast<std::int64_t>(iterations) - 1)),
                                                    position + l);
                    emit_body<T>(body, rw, walks[l], l);
                }
            });

            _records.push_back(Record{.tensor = use.name,
                                      .label  = rw.label,
                                      .extent = backing.dims.back(),
                                      .block  = rw.block,
                                      .bytes  = whole,
                                      .flops  = flops,
                                      .summed = rw.summed});
            _streamed.push_back(use.name);
            _blocks.push_back(static_cast<std::int64_t>(rw.block));
            ++_num_streamed;
            report(1, fmt::format("streamed '{}' into '{}' in blocks of {} along its last axis ({} {})", use.name, rw.label, rw.block,
                                  rw.summed ? "summed," : "kept by the output,",
                                  walks.size() == 1 ? "one loop" : fmt::format("{} loops", walks.size())));
            moved = true;
            again = true;
        }
    }
    return moved;
}

bool SlabStreaming::stream_write(Graph &graph, Backed const &backed, std::size_t index, std::size_t cap) {
    auto       &nodes = graph.nodes();
    Node const &node  = nodes[index];
    auto const *desc  = node.op_data.get_if<EinsumDescriptor>();
    auto const  out   = backed_operand(graph, backed, node.outputs[0], 0);
    if (desc == nullptr || !out) {
        return false;
    }
    auto const        where   = fmt::format("node '{}'", node.label);
    auto const       &backing = *out->backing;
    std::string const decline = "so it is computed in memory and stored after its last writer";

    // The storage then holds the tensor only if this contraction is all that produces it: it
    // overwrites every element, and nothing else writes the tensor in memory.
    if (!is_zero(live_c_prefactor(*desc))) {
        note_skip(fmt::format("the contraction adds into the write-backed tensor rather than overwriting it, {}", decline), where);
        return false;
    }
    if (out->written_outside || compute_graph::detail::written_in_memory(graph, out->object, nullptr, &node)) {
        note_skip(fmt::format("another operation writes the write-backed tensor in memory, {}", decline), where);
        return false;
    }
    if (!out->full) {
        note_skip(fmt::format("the contraction writes a view that slices or permutes the write-backed tensor, {}", decline), where);
        return false;
    }
    if (backed_operand(graph, backed, node.inputs[0], 1) || backed_operand(graph, backed, node.inputs[1], 2)) {
        note_skip(fmt::format("the contraction also reads a backed tensor, {}", decline), where);
        return false;
    }

    std::size_t const element = compute_graph::detail::dispatch_scalar_type(backing.dtype, []<typename T>(T) { return sizeof(T); });
    std::size_t       whole   = element;
    for (std::size_t const extent : backing.dims) {
        whole *= extent;
    }
    if (whole <= cap) {
        note_skip(fmt::format("the write-backed tensor fits its share of the budget, {}", decline),
                  fmt::format("{}: '{}' is {} bytes against {}", where, out->name, whole, cap));
        return false;
    }
    auto const lists = live_index_lists(*desc);
    if (!lists.operators.empty()) {
        note_skip(fmt::format("the contraction applies permutation operators, {}", decline), where);
        return false;
    }
    if (lists.c.empty() || backing.dims.empty() || backing.dims.back() == 0) {
        note_skip(fmt::format("the write-backed tensor has no last axis to write in blocks, {}", decline), where);
        return false;
    }
    std::string const letter = lists.c.back();
    if (occurrences(lists.c, letter) != 1 || occurrences(lists.a, letter) > 1 || occurrences(lists.b, letter) > 1) {
        note_skip(fmt::format("the letter of the write-backed tensor's last axis repeats, which a block of it cannot express, {}", decline),
                  where);
        return false;
    }
    TensorHandle const *a = graph.find_tensor(node.inputs[0]);
    TensorHandle const *b = graph.find_tensor(node.inputs[1]);
    if (a == nullptr || b == nullptr || a->dtype != backing.dtype || b->dtype != backing.dtype) {
        note_skip(fmt::format("an operand is not of the write-backed tensor's element type, {}", decline), where);
        return false;
    }
    auto const empty = [](TensorHandle const &handle) { return std::ranges::find(handle.dims, std::size_t{0}) != handle.dims.end(); };
    if (empty(*a) || empty(*b)) {
        note_skip(fmt::format("an operand has no elements, {}", decline), where);
        return false;
    }
    std::string why;
    auto const  in_a = underlying(graph, node.inputs[0], positions_of(lists.a, letter), why);
    auto const  in_b = in_a ? underlying(graph, node.inputs[1], positions_of(lists.b, letter), why) : std::nullopt;
    if (!in_a || !in_b) {
        note_skip(fmt::format("{}, {}", why, decline), where);
        return false;
    }

    WriteRewrite rw;
    rw.backed                   = *out;
    rw.a                        = lists.a;
    rw.b                        = lists.b;
    rw.c                        = lists.c;
    rw.conj_a                   = live_conj_a(*desc);
    rw.conj_b                   = live_conj_b(*desc);
    rw.ab_prefactor             = live_ab_prefactor(*desc);
    rw.in_a                     = *in_a;
    rw.in_b                     = *in_b;
    rw.dims                     = backing.dims;
    std::size_t const per_index = whole / backing.dims.back();
    rw.block                    = std::clamp<std::size_t>(cap / std::max<std::size_t>(per_index, 1), 1, backing.dims.back());
    rw.label                    = node.label;
    rw.prefix                   = fmt::format("slab:{}", _num_streamed);

    double flops = 2.0;
    {
        std::map<std::string, std::size_t> extent_of;
        auto const note_extents = [&](std::vector<std::string> const &letters, std::vector<std::size_t> const &dims) {
            for (std::size_t p = 0; p < letters.size() && p < dims.size(); ++p) {
                extent_of[letters[p]] = dims[p];
            }
        };
        note_extents(lists.a, a->dims);
        note_extents(lists.b, b->dims);
        note_extents(lists.c, backing.dims);
        for (auto const &[name, extent] : extent_of) {
            flops *= static_cast<double>(extent);
        }
    }

    // The node goes, with every view it used that nothing else does: the loops view the inputs'
    // tensors directly, and the destination exists only in storage.
    std::vector<bool> remove(nodes.size(), false);
    remove[index] = true;
    drop_view_if_unused(graph, remove, index, out->view_node, node.outputs[0]);
    drop_view_if_unused(graph, remove, index, rw.in_a.view_node, node.inputs[0]);
    drop_view_if_unused(graph, remove, index, rw.in_b.view_node, node.inputs[1]);
    auto const below = static_cast<std::size_t>(std::count(remove.begin(), remove.begin() + static_cast<std::ptrdiff_t>(index), true));

    graph.replace_nodes(remove, {});
    std::size_t const position = index - below;
    auto const        walks    = streamed_walks(backing.dims.back(), rw.block, /*summed=*/false);
    compute_graph::detail::dispatch_scalar_type(backing.dtype, [&]<typename T>(T) {
        for (std::size_t l = 0; l < walks.size(); ++l) {
            std::size_t const iterations = walk_iterations({walks[l]});
            Graph            &body =
                graph.add_loop_at(fmt::format("SlabStreaming#{}", l), iterations,
                                  PredExpr::iteration(CmpOp::Lt, BoundExpr(static_cast<std::int64_t>(iterations) - 1)), position + l);
            emit_write_body<T>(body, rw, walks[l], l);
        }
    });

    _records.push_back(Record{.tensor = out->name,
                              .label  = rw.label,
                              .extent = backing.dims.back(),
                              .block  = rw.block,
                              .bytes  = whole,
                              .flops  = flops,
                              .writes = true});
    // What still reads the tensor whole now reads it from the storage, filled after the writes.
    // The program-order check looks at the top-level graph only, whose ids these are.
    if (&graph == _root) {
        for (auto const &reader : graph.nodes()) {
            for (TensorId const tid : reader.inputs) {
                TensorHandle const *owner = graph.find_tensor(graph.buffer_of(tid));
                if (owner != nullptr && owner->tensor_ptr == out->object) {
                    _compensated.emplace_back(reader.id, tid);
                }
            }
        }
    }
    _streamed.push_back(out->name);
    _blocks.push_back(static_cast<std::int64_t>(rw.block));
    ++_num_streamed;
    ++_num_written;
    report(1, fmt::format("streamed '{}' out of '{}' in blocks of {} along its last axis ({})", out->name, rw.label, rw.block,
                          walks.size() == 1 ? "one loop" : fmt::format("{} loops", walks.size())));
    return true;
}

void SlabStreaming::reset_stats() {
    _records.clear();
    _num_streamed = 0;
    _num_written  = 0;
    // Per apply, not per run: the manager reads compensated_reads() once, after the descent, and
    // a graph address from an earlier apply may now be a different graph.
    _backed_of.clear();
    _root = nullptr;
    _compensated.clear();
    _streamed.clear();
    _blocks.clear();
}

std::vector<std::string> SlabStreaming::explain() const {
    std::vector<std::string> out;
    for (auto const &record : _records) {
        double const per_byte = record.bytes > 0 ? record.flops / static_cast<double>(record.bytes) : 0.0;
        if (record.writes) {
            out.push_back(fmt::format("SlabStreaming: '{}' writes '{}' in blocks of {} of {} along its last axis; {} bytes written per "
                                      "execution, about {:.0f} flop per byte written",
                                      record.label, record.tensor, record.block, record.extent, record.bytes, per_byte));
            continue;
        }
        out.push_back(fmt::format("SlabStreaming: '{}' reads '{}' in blocks of {} of {} along its last axis ({}); {} bytes read per "
                                  "execution, about {:.0f} flop per byte read",
                                  record.label, record.tensor, record.block, record.extent,
                                  record.summed ? "a summed letter, so the blocks add"
                                                : "a letter the output keeps, so each block writes "
                                                  "its own slice",
                                  record.bytes, per_byte));
    }
    return out;
}

EINSUMS_NAMESPACE_END(compute_graph::passes)
