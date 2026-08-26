//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/Detail/ScalarDispatch.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/InterfaceManifest.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraphTypes/Descriptors.hpp>
#include <Einsums/ComputeGraphTypes/Enums.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/TensorImpl/TensorImpl.hpp>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <ranges>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

// Part 3.7 of the algebraic-optimizer design: symbolic extents. ``rebind`` rejects any
// dim mismatch, which is right for a same-problem pointer swap and fatal for the
// cross-problem reuse a saved graph exists to allow. The manifest therefore declares each
// dimension as literal, as a symbol, or as ragged over a space; ``bind`` solves the
// symbols from the tensors it is handed, checks every slot that names one, re-derives the
// graph's own intermediates, and only then lets the repointing stand.

EINSUMS_NAMESPACE_BEGIN(compute_graph)

namespace {

/// Index letters bound to the extents this graph currently holds for them.
using LetterExtents = std::vector<std::pair<std::string, std::size_t>>;

/// A letter that two slots of one node disagree about, for the diagnostic.
struct LetterConflict {
    std::string letter;
    std::size_t held{0};
    std::size_t given{0};
};

/// Bind one operand's letters to @p dims. A letter repeated within the operand is a
/// diagonal access and must name one extent, which falls out of the same check.
/// @return False, filling @p conflict, when a letter would take two extents.
bool bind_letters(std::vector<std::string> const &indices, std::vector<std::size_t> const &dims, LetterExtents &bound,
                  LetterConflict &conflict) {
    if (dims.size() != indices.size()) {
        return true; // Rank disagreement is the node check's business, not this one's.
    }
    for (std::size_t slot = 0; slot < indices.size(); ++slot) {
        auto const existing = std::ranges::find_if(bound, [&](auto const &entry) { return entry.first == indices[slot]; });
        if (existing == bound.end()) {
            bound.emplace_back(indices[slot], dims[slot]);
        } else if (existing->second != dims[slot]) {
            conflict = LetterConflict{.letter = indices[slot], .held = existing->second, .given = dims[slot]};
            return false;
        }
    }
    return true;
}

/// Read the extents @p indices names out of @p bound.
/// @return False when some letter is unbound, so nothing is derivable.
bool extents_from_letters(std::vector<std::string> const &indices, LetterExtents const &bound, std::vector<std::size_t> &out) {
    out.clear();
    out.reserve(indices.size());
    for (auto const &letter : indices) {
        auto const it = std::ranges::find_if(bound, [&](auto const &entry) { return entry.first == letter; });
        if (it == bound.end()) {
            return false;
        }
        out.push_back(it->second);
    }
    return true;
}

/// The index lists a node contracts over, when it has any this walk understands.
struct NodeIndices {
    std::vector<std::string> const *a{nullptr};
    std::vector<std::string> const *b{nullptr};
    std::vector<std::string> const *c{nullptr};
};

/// The letter lists of @p node, or an empty result for a kind with none.
NodeIndices node_indices(Node const &node) {
    if (node.kind == OpKind::Einsum) {
        if (auto const *desc = std::get_if<EinsumDescriptor>(&node.op_data); desc != nullptr) {
            return NodeIndices{.a = &desc->spec.a_indices, .b = &desc->spec.b_indices, .c = &desc->spec.c_indices};
        }
        return {};
    }
    if (node.kind == OpKind::Permute || node.kind == OpKind::Transpose) {
        if (auto const *desc = std::get_if<PermuteDescriptor>(&node.op_data); desc != nullptr) {
            return NodeIndices{.a = &desc->a_indices, .b = nullptr, .c = &desc->c_indices};
        }
        return {};
    }
    return {};
}

/// Whether @p kind writes an output whose extents equal every input's, slot for slot.
bool is_extent_preserving(OpKind kind) {
    return kind == OpKind::Scale || kind == OpKind::Axpby;
}

/// The live extents and strides of @p handle, read through its rank-erased impl rather
/// than off the snapshot fields. The point of reading them is to catch the snapshot and
/// the tensor disagreeing, so reading the snapshot would answer the wrong question.
/// @return False when the handle exposes no impl (a tile-wise sparse tensor).
bool live_impl_geometry(TensorHandle const &handle, std::vector<std::size_t> &dims, std::vector<std::size_t> &strides) {
    if (!handle.impl_fn || handle.dtype == packed_gemm::ScalarType::Unknown) {
        return false;
    }
    void *raw = handle.impl_fn();
    if (raw == nullptr) {
        return false;
    }
    detail::dispatch_scalar_type(handle.dtype, [&]<typename T>(T /*tag*/) {
        auto const *impl = static_cast<::einsums::detail::TensorImpl<T> const *>(raw);
        dims.assign(impl->dims().begin(), impl->dims().end());
        strides.assign(impl->strides().begin(), impl->strides().end());
    });
    return true;
}

} // namespace

// ── Declaration ────────────────────────────────────────────────────────────

void Graph::record_symbol_space_ties(TensorHandle const &handle) {
    if (handle.dim_symbols.empty() || handle.spaces.empty()) {
        return;
    }
    std::size_t const axes = std::min(handle.dim_symbols.size(), handle.spaces.size());
    for (std::size_t axis = 0; axis < axes; ++axis) {
        std::string const &symbol = handle.dim_symbols[axis];
        SpaceId const      space  = handle.spaces[axis];
        if (symbol.empty() || is_ragged_symbol(symbol) || !space.valid()) {
            continue;
        }
        auto const [it, inserted] = _symbol_spaces.emplace(symbol, space);
        if (!inserted && it->second != space) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                    "Graph '{}': dim symbol '{}' is tied to index space '{}' on tensor '{}' axis {}, and to '{}' "
                                    "elsewhere in this graph; one symbol names one extent, so it cannot range over two sets",
                                    _name, symbol, space_registry().space(space).name, handle.name, axis,
                                    space_registry().space(it->second).name);
        }
    }
}

// ── Space-typed dimensions ─────────────────────────────────────────────────

void Graph::learn_space_extents(TensorHandle const &handle) {
    if (handle.spaces.empty() || handle.dims.empty()) {
        return;
    }
    std::size_t const axes = std::min(handle.spaces.size(), handle.dims.size());
    for (std::size_t axis = 0; axis < axes; ++axis) {
        SpaceId const space = handle.spaces[axis];
        if (!space.valid()) {
            continue;
        }
        auto const [it, inserted] = _space_extents.try_emplace(space.value(), handle.dims[axis], true);
        if (!inserted && it->second.first != handle.dims[axis]) {
            // Not an error. Two axes over one space with different extents is exactly what a
            // ragged family is, and the design says so: PNO domains have a different virtual
            // extent per pair. What it costs is the ability to SIZE an axis from the space,
            // so the space stops being usable that way and says so if anyone asks.
            it->second.second = false;
        }
    }
}

std::optional<std::size_t> Graph::space_extent(SpaceId space) const noexcept {
    if (!space.valid()) {
        return std::nullopt;
    }
    auto const it = _space_extents.find(space.value());
    if (it == _space_extents.end() || !it->second.second) {
        return std::nullopt;
    }
    return it->second.first;
}

void Graph::pin_space_extent(SpaceId space, std::size_t extent) {
    if (!space.valid()) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "Graph '{}': pin_space_extent: the id does not name a space", _name);
    }
    if (extent == 0) {
        _space_extents.erase(space.value());
        return;
    }
    _space_extents[space.value()] = {extent, true};
}

std::optional<std::vector<int>> Graph::space_tiling(SpaceId space) const {
    if (!space.valid()) {
        return std::nullopt;
    }
    auto const it = _space_tiles.find(space.value());
    if (it == _space_tiles.end() || !it->second.second) {
        return std::nullopt;
    }
    return it->second.first;
}

void Graph::pin_space_tiling(SpaceId space, std::vector<int> tile_sizes) {
    if (!space.valid()) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "Graph '{}': pin_space_tiling: the id does not name a space", _name);
    }
    if (tile_sizes.empty()) {
        _space_tiles.erase(space.value());
        return;
    }

    std::string const name = space_registry().space(space).name;

    std::size_t total = 0;
    for (std::size_t tile = 0; tile < tile_sizes.size(); ++tile) {
        if (tile_sizes[tile] <= 0) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                    "Graph '{}': pin_space_tiling index space '{}': tile {} has size {}; a tile is a non-empty piece "
                                    "of the space",
                                    _name, name, tile, tile_sizes[tile]);
        }
        total += static_cast<std::size_t>(tile_sizes[tile]);
    }

    // A partition states an extent, so the two must not be able to drift apart.
    if (auto const known = space_extent(space); known.has_value() && *known != total) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                "Graph '{}': pin_space_tiling index space '{}': the tiles sum to {}, but this graph already has it "
                                "measuring {}",
                                _name, name, total, *known);
    }

    auto const [it, inserted] = _space_tiles.try_emplace(space.value(), tile_sizes, true);
    if (!inserted && it->second.first != tile_sizes) {
        // Mirrors the extent rule: a second, different statement is not an error but it does
        // leave the space without a canonical answer, so an axis has to bring its own.
        it->second.second = false;
    }
    _space_extents[space.value()] = {total, true};
}

Graph::ResolvedTiledShape Graph::resolve_tiled_shape(std::vector<SpaceTiling> const &shape, std::string const &name) const {
    ResolvedTiledShape resolved;
    resolved.tile_sizes.reserve(shape.size());
    resolved.spaces.reserve(shape.size());
    resolved.symbols.reserve(shape.size());

    for (std::size_t axis = 0; axis < shape.size(); ++axis) {
        SpaceTiling const &entry = shape[axis];

        if (!entry.space.valid()) {
            // tiles({...}): a partition with no meaning. No space, no symbol, and a bind may
            // not move it, exactly as fixed() behaves for a dense axis.
            if (entry.tile_sizes.empty()) {
                EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                        "Graph '{}': tensor '{}' axis {} names no index space and carries no tiles, so nothing says "
                                        "how big it is or how it is cut up",
                                        _name, name, axis);
            }
            resolved.tile_sizes.push_back(entry.tile_sizes);
            resolved.spaces.emplace_back();
            resolved.symbols.emplace_back();
            continue;
        }

        std::string const space_name = space_registry().space(entry.space).name;

        std::vector<int> axis_tiles = entry.tile_sizes;
        if (axis_tiles.empty()) {
            auto const canonical = space_tiling(entry.space);
            if (!canonical.has_value()) {
                auto const it = _space_tiles.find(entry.space.value());
                if (it != _space_tiles.end() && !it->second.second) {
                    EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                            "Graph '{}': tensor '{}' axis {} takes its tiling from index space '{}', which has been "
                                            "given two different ones; give this axis the partition it wants",
                                            _name, name, axis, space_name);
                }
                EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                        "Graph '{}': tensor '{}' axis {} takes its tiling from index space '{}', but nothing has said "
                                        "how that space is cut up; call pin_space_tiling, or give this axis its own partition",
                                        _name, name, axis, space_name);
            }
            axis_tiles = *canonical;
        } else {
            std::size_t total = 0;
            for (int const tile : axis_tiles) {
                if (tile <= 0) {
                    EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                            "Graph '{}': tensor '{}' axis {}: a tile has size {}; a tile is a non-empty piece of the "
                                            "space",
                                            _name, name, axis, tile);
                }
                total += static_cast<std::size_t>(tile);
            }
            if (auto const known = space_extent(entry.space); known.has_value() && *known != total) {
                EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                        "Graph '{}': tensor '{}' axis {} tiles index space '{}' into pieces summing to {}, but this "
                                        "graph has that space measuring {}",
                                        _name, name, axis, space_name, total, *known);
            }
        }

        resolved.tile_sizes.push_back(std::move(axis_tiles));
        resolved.spaces.push_back(entry.space);
        // A PLAIN symbol, not a ragged one: the space fixes the axis TOTAL, and how that total
        // is cut up is a layout choice two tensors may differ on without the space being ragged.
        resolved.symbols.push_back(space_name);
        resolved.any_space = true;
    }

    return resolved;
}

Graph::ResolvedSpaceShape Graph::resolve_space_shape(std::vector<SpaceDim> const &shape, std::string const &name) const {
    ResolvedSpaceShape resolved;
    resolved.dims.reserve(shape.size());
    resolved.spaces.reserve(shape.size());
    resolved.symbols.reserve(shape.size());

    for (std::size_t axis = 0; axis < shape.size(); ++axis) {
        SpaceDim const &dim = shape[axis];
        if (!dim.space.valid()) {
            // A number is a number: literal extent, no space, no symbol.
            resolved.dims.push_back(dim.extent);
            resolved.spaces.emplace_back();
            resolved.symbols.emplace_back();
            continue;
        }

        auto const extent = space_extent(dim.space);
        if (!extent.has_value()) {
            SpaceRegistry const &registry = space_registry();
            std::string const    space    = registry.space(dim.space).name;
            auto const           it       = _space_extents.find(dim.space.value());
            if (it != _space_extents.end() && !it->second.second) {
                EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                        "Graph '{}': tensor '{}' axis {} is shaped by index space '{}', whose extent differs between "
                                        "the tensors already annotated with it; that is a ragged family and has no single size to "
                                        "take, so give the axis a number or declare it with annotate_ragged_dim",
                                        _name, name, axis, space);
            }
            EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                    "Graph '{}': tensor '{}' axis {} is shaped by index space '{}', but nothing annotated with that "
                                    "space has told this graph how big it is; annotate an operand over it first, or say so with "
                                    "pin_space_extent",
                                    _name, name, axis, space);
        }

        resolved.dims.push_back(*extent);
        resolved.spaces.push_back(dim.space);
        // The space's own name as the dim symbol. It keeps the (symbol, space) tie trivially
        // consistent, and it adds no field to IndexSpace and so nothing to the saved schema.
        resolved.symbols.push_back(space_registry().space(dim.space).name);
        resolved.any_space = true;
    }

    return resolved;
}

void Graph::apply_space_shape(TensorId id, ResolvedSpaceShape const &resolved) {
    if (!resolved.any_space) {
        // Every axis was a number. Annotating nothing is the honest record of that, and it
        // keeps a fully literal shape indistinguishable from the dims-based overload.
        return;
    }

    bool const every_axis = std::none_of(resolved.spaces.begin(), resolved.spaces.end(), [](SpaceId s) { return !s.valid(); });
    if (every_axis) {
        annotate_spaces(id, resolved.spaces);
    } else {
        // A mixed shape has a hole in it by construction, and the vector form refuses one.
        for (std::size_t axis = 0; axis < resolved.spaces.size(); ++axis) {
            if (resolved.spaces[axis].valid()) {
                annotate_space_axis(id, axis, resolved.spaces[axis]);
            }
        }
    }

    // Dim symbols carry the hole natively: the empty string IS a literal axis.
    annotate_dims(id, resolved.symbols);
}

void Graph::annotate_dims(TensorId id, std::vector<std::string> symbols) {
    auto &handle = tensor(id);

    if (!symbols.empty() && symbols.size() != handle.rank) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "Graph '{}': annotate_dims tensor '{}': got {} symbols for a rank-{} tensor", _name,
                                handle.name, symbols.size(), handle.rank);
    }

    // A ragged symbol names a space, and a space name that does not resolve would surface
    // later as a table nothing can be matched against. Check it here, once.
    for (std::size_t axis = 0; axis < symbols.size(); ++axis) {
        if (!is_ragged_symbol(symbols[axis])) {
            continue;
        }
        std::string_view const space = ragged_symbol_space(symbols[axis]);
        if (!space_registry().find(space).has_value()) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                    "Graph '{}': annotate_dims tensor '{}': axis {} is declared ragged over index space '{}', which "
                                    "this graph's registry does not hold",
                                    _name, handle.name, axis, space);
        }
    }

    // Ties are derived from the annotation as a whole, so they are recorded from a COPY of
    // the finished state: a throw below must leave the handle exactly as it was.
    TensorHandle probe;
    probe.name        = handle.name;
    probe.dim_symbols = symbols;
    probe.spaces      = handle.spaces;
    record_symbol_space_ties(probe);

    handle.dim_symbols = std::move(symbols);
}

void Graph::annotate_ragged_dim(TensorId id, std::size_t axis, std::string_view space_name) {
    auto &handle = tensor(id);

    if (axis >= handle.rank) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "Graph '{}': annotate_ragged_dim tensor '{}': axis {} is past its rank of {}", _name,
                                handle.name, axis, handle.rank);
    }

    std::vector<std::string> symbols = handle.dim_symbols;
    if (symbols.empty()) {
        symbols.assign(handle.rank, std::string{});
    }
    symbols[axis] = make_ragged_symbol(space_name);
    annotate_dims(id, std::move(symbols));
}

std::vector<std::string> const &Graph::tensor_dim_symbols(TensorId id) const {
    return tensor(id).dim_symbols;
}

// ── The bind solver ────────────────────────────────────────────────────────

void Graph::solve_bind_dims(DimSolution &solution, ManifestEntry const &entry, std::vector<std::size_t> const &dims) const {
    if (entry.dim_symbols.empty()) {
        return; // Every axis literal; validate_bind_shape already required an exact match.
    }
    solution.any_symbolic = true;

    std::size_t const axes = std::min({entry.dim_symbols.size(), entry.dims.size(), dims.size()});
    for (std::size_t axis = 0; axis < axes; ++axis) {
        std::string const &symbol = entry.dim_symbols[axis];
        if (symbol.empty()) {
            continue; // Literal: checked exactly, elsewhere.
        }
        if (dims[axis] != entry.dims[axis]) {
            solution.extents_changed = true;
        }
        if (is_ragged_symbol(symbol)) {
            continue; // Constrains nothing across slots; each instance carries its own.
        }

        auto const [it, inserted] = solution.values.emplace(symbol, dims[axis]);
        if (inserted) {
            solution.witness.emplace(symbol, entry.name);
            continue;
        }
        if (it->second != dims[axis]) {
            auto const witness = solution.witness.find(symbol);
            EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                    "Graph '{}': bind('{}'): axis {} declares dim symbol '{}' and the given tensor makes it {}, but "
                                    "'{}' already made it {} in this same bind; one symbol is one extent",
                                    _name, entry.name, axis, symbol, dims[axis],
                                    witness != solution.witness.end() ? witness->second : std::string{"an earlier slot"}, it->second);
        }
    }
}

void Graph::prepare_bind_solution(DimSolution const &solution) const {
    if (!solution.extents_changed) {
        return;
    }

    // Batching is a decision over ONE set of shapes. It is never saved, the load path
    // re-runs it, and a saved graph holds the pre-resource algebraic form for exactly that
    // reason - so re-forming these nodes under new extents belongs to the resource phase
    // and not here. Refuse rather than rebind a batch whose group table describes the old
    // problem: this runs before anything is repointed, so a refusal leaves the graph
    // untouched.
    for (auto const &node : _nodes) {
        if (node.kind == OpKind::BatchedGemm || node.kind == OpKind::GroupedBatchedGemm) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                    "Graph '{}': this bind changes an extent, but node '{}' is a {} whose group shapes were planned "
                                    "for the previous problem. Batching is a resource decision, never saved and re-derived on load: "
                                    "bind before apply(), or re-run the batching passes after binding",
                                    _name, node.label, op_kind_name(node.kind));
        }
    }
}

void Graph::finish_bind_solution(DimSolution const &solution) {
    if (!solution.extents_changed) {
        return;
    }
    rederive_owned_extents();
    validate_node_extents();
}

// ── Intermediate re-derivation ─────────────────────────────────────────────

void Graph::resize_derived_extent(TensorId id, std::vector<std::size_t> const &derived, std::string_view producer) {
    TensorHandle *handle = find_tensor(id);
    if (handle == nullptr || handle->rank != derived.size() || handle->dims == derived) {
        return;
    }

    // Only storage the GRAPH owns may be reshaped. A caller's tensor that no longer fits
    // is a bind that under-supplied the interface, and saying so is validate_node_extents'
    // job - it can name the node, which is what makes the mistake findable.
    bool const graph_owned = handle->is_intermediate || _owned_tensor_ptrs.contains(handle->tensor_ptr);
    if (!graph_owned) {
        return;
    }

    if (handle->alloc_state != AllocState::Deferred || !handle->resize_deferred_fn) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                "Graph '{}': binding these extents makes intermediate '{}' (written by {}) [{}] instead of [{}], but "
                                "its storage is already materialized and cannot be reshaped. Re-bind before materialization, or "
                                "create the graph with deferred intermediates (Graph::scratch)",
                                _name, handle->name, producer, fmt::join(derived, ", "), fmt::join(handle->dims, ", "));
    }

    // A deferred intermediate that has been executed once holds a buffer sized for the
    // previous problem. Its lifecycle already says that buffer is disposable - a
    // Materialize node re-attaches one at the right position - so give it back rather than
    // refuse a second bind.
    if (handle->is_materialized_fn && handle->is_materialized_fn()) {
        if (!handle->release_fn) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                    "Graph '{}': binding these extents makes intermediate '{}' (written by {}) [{}] instead of [{}], "
                                    "but it holds storage it cannot release. Re-bind before materialization, or create the graph with "
                                    "deferred intermediates (Graph::scratch)",
                                    _name, handle->name, producer, fmt::join(derived, ", "), fmt::join(handle->dims, ", "));
        }
        handle->release_fn();
    }

    handle->resize_deferred_fn(derived);

    // Snapshots. ``dims`` lives in three places - here, on the slot, and inside the live
    // impl - and the two snapshots are what every extent check and the alias derivation
    // read, so a resize that updated only the tensor would be invisible to both.
    std::vector<std::size_t> live_dims;
    std::vector<std::size_t> live_strides;
    if (live_impl_geometry(*handle, live_dims, live_strides)) {
        if (live_dims != derived) {
            EINSUMS_THROW_EXCEPTION(std::runtime_error,
                                    "Graph '{}': resizing intermediate '{}' to [{}] left its live geometry at [{}]; the resize hook "
                                    "and the tensor disagree",
                                    _name, handle->name, fmt::join(derived, ", "), fmt::join(live_dims, ", "));
        }
        handle->strides = std::move(live_strides);
    }
    handle->dims = derived;
    // Deferred again: no address until the Materialize node runs, which is the state
    // make_handle records for a shell.
    handle->data_ptr = nullptr;

    if (TensorSlot *slot = find_slot(id); slot != nullptr) {
        slot->dims = derived;
    }

    // The buffer moved and changed size, so whatever the alias relation said about it is
    // about a tensor that no longer exists.
    _aliases_linked = false;
    _deps_valid     = false;
}

void Graph::rederive_owned_extents() {
    // Topological order makes one sweep a fixpoint: a node is visited after everything
    // that writes its inputs, so a chain of intermediates resolves end to end in one pass.
    // The same traversal shape SpacePropagation uses, carrying extents instead of spaces.
    topological_sort();

    for (auto const &node : _nodes) {
        if (node.outputs.size() != 1) {
            continue;
        }
        TensorId const out = node.outputs[0];

        if (is_extent_preserving(node.kind)) {
            for (TensorId const in : node.inputs) {
                TensorHandle const *src = find_tensor(in);
                if (src != nullptr && in != out && !src->dims.empty()) {
                    resize_derived_extent(out, src->dims, node.label);
                    break;
                }
            }
            continue;
        }

        NodeIndices const indices = node_indices(node);
        if (indices.c == nullptr) {
            continue;
        }

        LetterExtents  bound;
        LetterConflict conflict;
        bool           usable = true;
        // Only the first two inputs are operands; a third is the destination re-listed
        // because the node accumulates onto it, and its extents are what we are deriving.
        std::size_t const operands = std::min<std::size_t>(node.inputs.size(), indices.b != nullptr ? 2 : 1);
        for (std::size_t operand = 0; operand < operands; ++operand) {
            auto const *list = operand == 0 ? indices.a : indices.b;
            if (list == nullptr) {
                continue;
            }
            TensorHandle const *src = find_tensor(node.inputs[operand]);
            if (src == nullptr) {
                usable = false;
                break;
            }
            if (!bind_letters(*list, src->dims, bound, conflict)) {
                usable = false; // Reported by validate_node_extents, which names the node.
                break;
            }
        }
        if (!usable) {
            continue;
        }

        std::vector<std::size_t> derived;
        if (extents_from_letters(*indices.c, bound, derived)) {
            resize_derived_extent(out, derived, node.label);
        }
    }
}

void Graph::validate_node_extents() const {
    for (auto const &node : _nodes) {
        if (is_extent_preserving(node.kind)) {
            std::vector<std::size_t> const *reference = nullptr;
            std::string                     reference_name;
            for (TensorId const tid : node.inputs) {
                TensorHandle const *operand = find_tensor(tid);
                if (operand == nullptr || operand->dims.empty()) {
                    continue;
                }
                if (reference == nullptr) {
                    reference      = &operand->dims;
                    reference_name = operand->name;
                    continue;
                }
                if (operand->dims.size() == reference->size() && operand->dims != *reference) {
                    EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                            "Graph '{}': after binding, node '{}' reads '{}' at [{}] and '{}' at [{}]; an elementwise "
                                            "combination needs one shape. Declare the axes that move together with annotate_dims",
                                            _name, node.label, reference_name, fmt::join(*reference, ", "), operand->name,
                                            fmt::join(operand->dims, ", "));
                }
            }
            continue;
        }

        NodeIndices const indices = node_indices(node);
        if (indices.c == nullptr || node.outputs.size() != 1) {
            continue;
        }

        LetterExtents  bound;
        LetterConflict conflict;
        // Destination first, so a conflict message reads "the output says N, this operand
        // says M" rather than the other way round.
        std::array<std::pair<std::vector<std::string> const *, TensorId>, 3> const slots{
            std::pair{indices.c, node.outputs[0]},
            std::pair{indices.a, node.inputs.empty() ? TensorId{0} : node.inputs[0]},
            std::pair{indices.b, node.inputs.size() < 2 ? TensorId{0} : node.inputs[1]},
        };
        for (auto const &[list, tid] : slots) {
            if (list == nullptr || tid == 0) {
                continue;
            }
            TensorHandle const *operand = find_tensor(tid);
            if (operand == nullptr || operand->dims.size() != list->size()) {
                continue;
            }
            if (!bind_letters(*list, operand->dims, bound, conflict)) {
                EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                        "Graph '{}': after binding, node '{}' binds index '{}' to extent {} and to {} through '{}'; "
                                        "the operands describe two different problems. Declare the axes that move together with "
                                        "annotate_dims",
                                        _name, node.label, conflict.letter, conflict.held, conflict.given, operand->name);
            }
        }
    }
}

// ── Ragged extent tables ───────────────────────────────────────────────────

void Graph::bind_ragged_extents(std::string const &name, std::size_t axis, std::vector<std::size_t> extents) {
    InterfaceManifest const contract = manifest();
    ManifestEntry const    &entry    = lookup_manifest_entry(contract, name);

    if (axis >= entry.rank || axis >= entry.dim_symbols.size() || !is_ragged_symbol(entry.dim_symbols[axis])) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                "Graph '{}': bind_ragged_extents('{}', {}): that axis is not declared ragged. Declare it with "
                                "annotate_ragged_dim before supplying a per-instance extent table",
                                _name, name, axis);
    }
    if (extents.empty()) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                "Graph '{}': bind_ragged_extents('{}', {}): the table is empty, so it names no instance at all", _name,
                                name, axis);
    }

    std::string const space(ragged_symbol_space(entry.dim_symbols[axis]));

    // The table's length is the instance COUNT, which belongs to the space (how many pairs
    // there are) rather than to one operand. Two tables over one space disagreeing about it
    // describes nothing, so it is the arity this validates.
    for (auto const &held : _ragged_extents) {
        if (held.space == space && held.extents.size() != extents.size()) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                    "Graph '{}': bind_ragged_extents('{}', {}) supplies {} instances of index space '{}', but the "
                                    "table already held for '{}' axis {} supplies {}; the instance count is a property of the space",
                                    _name, name, axis, extents.size(), space, held.name, held.axis, held.extents.size());
        }
    }

    auto const existing =
        std::ranges::find_if(_ragged_extents, [&](RaggedExtentTable const &held) { return held.id == entry.id && held.axis == axis; });
    if (existing != _ragged_extents.end()) {
        existing->name    = entry.name;
        existing->space   = space;
        existing->extents = std::move(extents);
        return;
    }

    _ragged_extents.push_back(
        RaggedExtentTable{.id = entry.id, .name = entry.name, .axis = axis, .space = space, .extents = std::move(extents)});
}

RaggedExtentTable const *Graph::find_ragged_extents(std::string_view name, std::size_t axis) const noexcept {
    auto const it =
        std::ranges::find_if(_ragged_extents, [&](RaggedExtentTable const &held) { return held.name == name && held.axis == axis; });
    return it == _ragged_extents.end() ? nullptr : &*it;
}

// ── Rebind staleness ───────────────────────────────────────────────────────

void Graph::note_rebind_geometry(TensorHandle &handle, void *new_data, std::vector<std::size_t> const &new_dims,
                                 std::vector<std::size_t> const &new_strides) {
    bool const moved = handle.data_ptr != new_data || handle.dims != new_dims;

    handle.data_ptr = new_data;
    handle.dims     = new_dims;
    handle.strides  = new_strides;

    if (!moved) {
        return;
    }

    // The alias relation is derived from the address and the extents, so the link this
    // handle carries is about the tensor it USED to name. Drop the derived part only - a
    // declaration is an input to the derivation, not an output of it, and survives - and
    // let the next link pass re-derive against what is there now. Whole-table and
    // amortized, so one rebind of one slot costs one flag rather than a scan.
    if (!_declared_aliases.contains(handle.id)) {
        handle.aliases = 0;
        handle.alias_box.clear();
    }
    _aliases_linked = false;
    _deps_valid     = false;
}

EINSUMS_NAMESPACE_END(compute_graph)
