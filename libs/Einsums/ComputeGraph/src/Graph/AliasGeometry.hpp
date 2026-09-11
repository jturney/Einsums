//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/// @file AliasGeometry.hpp
/// @brief Whether two tensor handles share memory, and where.
///
/// Private to the `Graph` sources, and shared by exactly two of them because the
/// question has two askers. `Alias.cpp` asks it in order to LINK handles, so that
/// a view and its parent are known to be one buffer. `Schedule.cpp` asks it in
/// order to order nodes, because two nodes that touch one region cannot run at
/// the same time.
///
/// Two derivations live here and they answer for different graphs. The
/// pointer-derived one (@ref handle_byte_span, @ref derive_alias_box) reads
/// addresses and strides, so it needs tensors that have been allocated; its
/// bodies are in `AliasGeometry.cpp`. The structural one
/// (@ref StructuralAliasResolver) reads ``View`` nodes and their descriptors and
/// consults no data pointer anywhere, which is what a graph loaded from a file
/// has instead: such a graph has allocated nothing, and the tensors bound to it
/// afterwards are not the tensors that were captured.
///
/// Both derivations are conservative in the same direction. An ambiguity yields
/// the WIDER region, because a box that is too wide costs hazard edges and only
/// a box that is too narrow loses one.

#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/Config/Namespace.hpp>

#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph::alias_geometry)

/// Per-axis half-open interval list, the representation both derivations speak.
using AliasBox = std::vector<std::pair<std::int64_t, std::int64_t>>;

/// Half-open byte span of a handle's storage, or false when it has none that
/// can be reasoned about (deferred allocation, tiled layout, zero extent).
///
/// A tiled handle has no single buffer, so it is refused here rather than in the
/// shared span helper, which knows only about one strided allocation.
bool handle_byte_span(TensorHandle const &h, char const *&lo, char const *&hi);

/// How well a handle's strides describe a lattice, which is what decides
/// whether an offset may be decoded into per-axis indices at all.
enum class LayoutFit {
    None,   ///< Overlapping or degenerate; nothing may be decoded from an offset.
    Nested, ///< Injective: each traversed axis starts at or past the end of the one below it.
    Packed  ///< Nested with no gaps, so every offset under the total is an index.
};

/// Classify @p h's layout and hand back its traversed axes, largest stride
/// first, which is the order both derivations peel an offset in.
///
/// Nesting is the property that makes an offset decode to ONE index: sort the
/// axes holding more than one element by stride, and require each one to start
/// at or past the end of everything below it. Without it a layout can be
/// non-injective - dims (4, 4) with strides (1, 2) reaches offset 4 as both
/// (0, 2) and (2, 1) - and two boxes that do not intersect in the axis space
/// can still name the same memory, which is the one way a box can be too
/// narrow. Packing additionally forbids gaps, which is what makes every offset
/// in a range decodable and so is what the span derivation needs.
///
/// Axes holding one element are left out of @p order: their only index is 0,
/// and a dense layout is free to give them any stride at all.
///
/// @param[in]  h     Handle to classify.
/// @param[out] order Its traversed axes, largest stride first.
/// @param[out] total Number of elements, meaningful for a Packed layout.
LayoutFit layout_axis_order(TensorHandle const &h, std::vector<size_t> &order, size_t &total);

/// @p child's region in @p parent's axis space: the exact per-axis match where
/// the layouts allow it, the looser offset-span bound where they do not, and no
/// box at all (a conservatively whole-tensor access) where neither is provable.
///
/// A box covering the whole parent is reported as NO box, which is the same
/// statement - both overlap every access and cover every access - in the form
/// the hazard scan reasons about better. A whole-tensor write there retires
/// every writer before it, because it dominates them; a full-cover BOX is not
/// recognized as dominating anything, so the writer list grows without bound
/// and each later access emits an edge against all of it. The two spellings
/// schedule identically and the quadratic one is worth avoiding: adding the
/// span bound WITHOUT this normalization put 103,000 extra edges on the
/// DLPNO merged iteration, every one of them from a reshaped view that covers
/// its whole parent.
bool derive_alias_box(TensorHandle const &parent, TensorHandle const &child, std::vector<std::pair<std::int64_t, std::int64_t>> &box);

/// True when @p box is every element of a parent with extents @p dims, which is
/// the shape `derive_alias_box` normalizes away. See its comment for why: a
/// full-cover BOX is not recognized as dominating anything, so the writer list
/// grows without bound, while the same statement spelled as NO box retires
/// every writer it covers.
bool whole_cover(AliasBox const &box, std::vector<size_t> const &dims);

/// What a structural walk knows about one tensor's relation to its alias root.
///
/// @ref box and @ref axis_map are only meaningful under their respective flags;
/// @ref root is always meaningful, because a walk that cannot describe a region
/// still knows which buffer the tensor is part of, and that is the half whose
/// absence races.
struct StructuralAlias {
    /// The tensor at the end of the alias chain. Equal to the tensor itself when
    /// it owns its storage.
    TensorId root{0};
    /// The region the tensor covers, in @ref root's axis space. Valid only when
    /// @ref box_known.
    AliasBox box;
    /// This tensor's axis @c r maps onto @ref root's axis ``axis_map[r]``. One
    /// entry per axis of THIS tensor, so a Drop axis contributes none. Valid only
    /// when @ref map_known, and needed by a child that composes through it.
    std::vector<size_t> axis_map;
    bool                box_known{false}; ///< Whether @ref box describes the region.
    bool                map_known{false}; ///< Whether @ref axis_map describes the axis correspondence.
};

/// Alias discovery with NO addresses: the relation `(root, region)` derived from
/// ``View`` nodes, their @ref ViewDescriptor axes, and the alias links already on
/// the handles, with no data pointer consulted anywhere.
///
/// This is the counterpart of the pointer-derived containment search in
/// @ref Graph::link_alias_storage and it is what a loaded graph has instead of
/// one, because a graph read from a file has allocated nothing and the tensors
/// bound to it afterwards are not the tensors that were captured. It is also
/// STRICTLY more general than the View-node scan the hazard pass used to run
/// inline, which refused a permuted view and a view of a view; both compose here.
///
/// **Composition.** A view's descriptor gives, per parent axis, the interval that
/// axis is restricted to (a whole axis, a constant Range, or the single index a
/// Drop pins), and its permutation says which parent axis each RESULT axis reads.
/// Composing a chain is therefore: take the parent's own region in the root's
/// axis space, and place the child's per-parent-axis intervals inside it by
/// offsetting each one by where the parent's corresponding axis starts. The axis
/// map is what makes that placement possible past one hop, and it is why a view
/// of a permuted view is describable at all.
///
/// **Where it declines.** A non-constant bound (a Param or a Callback, whose
/// value is not known until execute) yields no box, which reads as the whole
/// parent - the same conservative answer the pointer path gives for a region it
/// cannot prove. So does a root whose own strides are not injective: such a
/// layout reaches one element through more than one index tuple, so two boxes
/// that share no index can still name the same memory, and trusting the axis
/// space there would DROP a real hazard edge. That fence is checked against the
/// root's registration-time strides when it has them; a loaded graph carries
/// none yet, and its axis space is trusted because nothing else can be.
class StructuralAliasResolver {
  public:
    /// Index the graph's ``View`` nodes. One pass; the walk itself is memoized.
    explicit StructuralAliasResolver(Graph const &graph) : _graph(&graph) {
        for (auto const &nd : graph.nodes()) {
            if (nd.kind != OpKind::View || nd.outputs.size() != 1) {
                continue;
            }
            auto const *vd = std::get_if<ViewDescriptor>(&nd.op_data);
            if (vd == nullptr) {
                continue;
            }
            // First View node writing a tensor wins. A second one would be a
            // re-description of the same slice; taking the first keeps the
            // derivation independent of node order.
            _views.emplace(nd.outputs[0], vd);
        }
    }

    /// Whether @p id is the output of a ``View`` node, i.e. whether this
    /// derivation has anything to say about it that the handle does not.
    [[nodiscard]] bool is_view(TensorId id) const { return _views.contains(id); }

    /// @return The relation, memoized.
    ///
    /// Walks UP the view chain collecting what this answer depends on, then
    /// composes back DOWN. Iterative rather than recursive on purpose: a chain
    /// is bounded only by the tensor count (DLPNO-MP2 registers ~13k), and a
    /// stack frame per hop is the wrong thing to bound it with. It also makes
    /// the cycle case a plain visited-set test rather than a depth guard - a
    /// cycle is not constructible today, and if one ever is, the tensor is
    /// reported conservatively instead of looped on.
    StructuralAlias const &resolve(TensorId id) {
        if (auto const it = _memo.find(id); it != _memo.end()) {
            return it->second;
        }

        std::vector<TensorId>        chain;
        std::unordered_set<TensorId> on_chain;
        for (TensorId current = id;;) {
            if (_memo.contains(current)) {
                break; // an answer already exists to compose against
            }
            auto const vit = _views.find(current);
            if (vit == _views.end()) {
                _memo.emplace(current, self(current)); // chain ends at a tensor no View node describes
                break;
            }
            if (!on_chain.insert(current).second) {
                _memo.emplace(current, StructuralAlias{.root = current}); // cycle: conservative
                break;
            }
            chain.push_back(current);
            TensorId const parent = vit->second->parent_id;
            if (parent == current || _graph->find_tensor(parent) == nullptr) {
                chain.pop_back();
                _memo.emplace(current, self(current)); // a View node naming nothing usable
                break;
            }
            current = parent;
        }

        // Compose downward: every entry's parent is already answered.
        for (TensorId const tid : std::views::reverse(chain)) {
            _memo.emplace(tid, compose(tid));
        }
        return _memo.at(id);
    }

  private:
    /// The answer for a tensor that no ``View`` node describes: it owns its
    /// storage, or its handle already names an alias parent whose axis space
    /// this derivation cannot recover (a pointer-linked slice, a manifest
    /// declaration). Either way the ROOT is known and the region is not.
    [[nodiscard]] StructuralAlias self(TensorId id) const {
        StructuralAlias res;
        res.root                   = id;
        TensorHandle const *handle = _graph->find_tensor(id);
        if (handle == nullptr) {
            return res;
        }
        if (handle->aliases != 0) {
            // resolve_alias throws only on a cycle, which every writer of
            // ``aliases`` makes unconstructible; a conservative catch here would
            // hide exactly the corruption it is there to report.
            res.root = _graph->resolve_alias(handle->aliases);
            return res; // region unknown: a box on the handle lives in a space this walk did not build
        }
        if (handle->is_tiled) {
            return res; // no single axis space to place a region in
        }
        // The injectivity fence. Only checked when the root carries the strides
        // to check it with; see the class comment.
        if (handle->strides.size() == handle->dims.size() && !handle->strides.empty()) {
            std::vector<size_t> order;
            size_t              total = 0;
            if (layout_axis_order(*handle, order, total) == LayoutFit::None) {
                return res;
            }
        }
        res.box.reserve(handle->dims.size());
        res.axis_map.reserve(handle->dims.size());
        for (size_t d = 0; d < handle->dims.size(); ++d) {
            res.box.emplace_back(0, static_cast<std::int64_t>(handle->dims[d]));
            res.axis_map.push_back(d);
        }
        res.box_known = true;
        res.map_known = true;
        return res;
    }

    /// Place @p id's own slice inside the region its parent already occupies.
    /// Only called by @ref resolve, and only once the parent is answered.
    [[nodiscard]] StructuralAlias compose(TensorId id) const {
        ViewDescriptor const  *vd     = _views.at(id);
        TensorHandle const    *ph     = _graph->find_tensor(vd->parent_id);
        StructuralAlias const &parent = _memo.at(vd->parent_id);
        StructuralAlias        res;
        res.root = parent.root;
        if (ph == nullptr) {
            return res;
        }

        size_t const prank = ph->dims.size();
        if (!parent.box_known || !parent.map_known || parent.axis_map.size() != prank || vd->axes.size() != prank) {
            return res; // the parent's own region is not placeable, so neither is this one
        }

        // The child's region in the PARENT's axis space, plus which parent axis
        // each of the child's own axes reads.
        AliasBox            local(prank, {0, 0});
        std::vector<size_t> local_map;
        std::vector<bool>   touched(prank, false);
        local_map.reserve(prank);
        for (size_t i = 0; i < prank; ++i) {
            // Result axis i slices parent axis p. Empty permutation is identity;
            // a permutation is a bijection of [0, prank), so every parent axis is
            // named exactly once and the box has no hole.
            size_t const p = vd->permutation.empty() ? i : vd->permutation[i];
            if (p >= prank || touched[p]) {
                return res;
            }
            touched[p]          = true;
            ViewAxis const &ax  = vd->axes[i];
            auto const      dim = static_cast<std::int64_t>(ph->dims[p]);
            switch (ax.kind) {
            case ViewAxis::Kind::Full:
                local[p] = {0, dim};
                local_map.push_back(p);
                break;
            case ViewAxis::Kind::Range:
                if (!ax.lo.is_const() || !ax.hi.is_const()) {
                    return res; // a runtime bound conflicts as the whole parent
                }
                local[p] = {ax.lo.const_value(), ax.hi.const_value()};
                local_map.push_back(p);
                break;
            case ViewAxis::Kind::Drop:
                if (!ax.lo.is_const()) {
                    return res;
                }
                local[p] = {ax.lo.const_value(), ax.lo.const_value() + 1};
                break; // a dropped axis contributes its collapsed index and no result axis
            }
            if (local[p].first < 0 || local[p].first > local[p].second || local[p].second > dim) {
                return res; // out of the parent, so not describable in its axes
            }
        }

        // Place the local intervals inside the parent's own region: parent axis p
        // is root axis q, and its index 0 sits at the root index the parent's box
        // starts at.
        res.box = parent.box;
        for (size_t p = 0; p < prank; ++p) {
            size_t const q = parent.axis_map[p];
            if (q >= res.box.size()) {
                return {.root = parent.root};
            }
            std::int64_t const base = parent.box[q].first;
            res.box[q]              = {base + local[p].first, base + local[p].second};
            if (res.box[q].second > parent.box[q].second) {
                return {.root = parent.root};
            }
        }
        res.axis_map.reserve(local_map.size());
        for (size_t const p : local_map) {
            res.axis_map.push_back(parent.axis_map[p]);
        }
        res.box_known = true;
        res.map_known = true;
        return res;
    }

    Graph const                                         *_graph;
    std::unordered_map<TensorId, ViewDescriptor const *> _views;
    std::unordered_map<TensorId, StructuralAlias>        _memo;
};

EINSUMS_NAMESPACE_END(compute_graph::alias_geometry)
