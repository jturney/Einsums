//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file Alias.cpp
/// @brief Which of a graph's tensors are views onto one another.
///
/// An alias link says that two handles name one buffer, and it is what every
/// later pass reads instead of comparing pointers itself. There are three ways
/// one gets established and this file is all three.
///
/// @ref Graph::declare_alias is the caller saying so outright. @ref
/// Graph::link_alias_structural derives links from ``View`` nodes and their
/// descriptors, so a graph loaded from a file has them before anything is
/// allocated. @ref Graph::link_alias_storage derives them from addresses, which
/// recovers the links a capture never declared because the caller handed in a
/// `TensorView` whose parent the graph never saw.
///
/// The geometry the last two ask about lives in `AliasGeometry.hpp`.

#include <Einsums/CXX23/Expected.hpp>
#include <Einsums/ComputeGraph/CaptureContext.hpp>
#include <Einsums/ComputeGraph/Detail/ScalarDispatch.hpp>
#include <Einsums/ComputeGraph/EinsumSpec.hpp>
#include <Einsums/ComputeGraph/Error.hpp>
#include <Einsums/ComputeGraph/ExecutorBuilder.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Options.hpp>
#include <Einsums/ComputeGraph/SpaceRegistryAccess.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/Profile/Profile.hpp>
#include <Einsums/Tensor/Tensor.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <optional>
#include <ostream>
#include <queue>
#include <ranges>
#include <set>
#include <span>
#include <unordered_set>
#include <utility>

#include "AliasGeometry.hpp"

EINSUMS_NAMESPACE_BEGIN(compute_graph)

using namespace alias_geometry;

void Graph::declare_alias(TensorId child, TensorId parent) {
    if (child == 0 || parent == 0 || child == parent) {
        return;
    }
    // An unknown id is a caller's mistake, and ignoring it would drop the hazard edges the
    // declaration exists to add: two tensors sharing storage would then be scheduled as if they
    // did not.
    auto *child_handle = find_tensor(child);
    if (child_handle == nullptr || !_tensors.contains(parent)) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "Graph '{}': declare_alias({}, {}) names a tensor this graph does not have: {}",
                                _name, child, parent, child_handle == nullptr ? child : parent);
    }
    // A cycle would turn every later resolve_alias into a throw, and the message
    // there is about a corrupt link rather than about the declaration that made
    // it. Refuse here, where the two names are still in hand: a saved graph
    // claiming A is part of B and B part of A describes nothing.
    for (TensorId walk = parent, hops = 0; walk != 0 && hops <= _tensors.size(); ++hops) {
        if (walk == child) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                    "Graph '{}': declare_alias({}, {}) would make the two tensors each other's alias parent; an "
                                    "alias declaration has to name a containing buffer, and containment is not symmetric",
                                    _name, child, parent);
        }
        auto const *link = find_tensor(walk);
        if (link == nullptr) {
            break;
        }
        walk = link->aliases;
    }
    _declared_aliases.insert_or_assign(child, parent);
    child_handle->aliases = parent;
    // A declaration says WHICH buffer, never WHICH REGION - the manifest schema
    // carries no box - so the declared alias conflicts as the whole parent.
    child_handle->alias_box.clear();
    // The hazard relation just changed, so anything derived from it is stale.
    _deps_valid = false;
}

void Graph::clear_alias_links() noexcept {
    for (auto &[id, handle] : _tensors) {
        handle.aliases = 0;
        handle.alias_box.clear();
    }
    _aliases_linked = false;
    _deps_valid     = false;
    // Owner-keyed analyses were keyed on the relation this just dropped; see link_alias_storage.
    _analysis_version++;
}

void Graph::apply_declared_aliases() {
    for (auto const &[child, parent] : _declared_aliases) {
        auto *handle = find_tensor(child);
        if (handle == nullptr || child == parent || !_tensors.contains(parent)) {
            continue;
        }
        handle->aliases = parent;
        handle->alias_box.clear();
    }
}

void Graph::link_alias_structural() {
    // Declarations first, and they are what the pointer path cannot see: two
    // caller-supplied tensors that happen to share storage have no View node
    // recording it, and in a loaded graph they have no addresses to compare
    // either. Applying them first also means a view OF a declared alias
    // composes down onto the declared root below.
    apply_declared_aliases();

    StructuralAliasResolver resolver(*this);

    // Derive first, write second. The walk reads ``aliases`` for the handles no
    // View node describes, so rewriting a handle mid-walk would let a later
    // tensor resolve against a half-updated relation.
    struct Update {
        TensorId tid;
        TensorId root;
        AliasBox box;
    };
    std::vector<Update> updates;
    for (auto const &[id, handle] : _tensors) {
        if (!resolver.is_view(id)) {
            continue;
        }
        StructuralAlias const &res = resolver.resolve(id);
        if (res.root == 0 || res.root == id) {
            continue;
        }
        AliasBox box;
        if (res.box_known) {
            TensorHandle const *root = find_tensor(res.root);
            // A box covering the whole root is reported as NO box, exactly as
            // the pointer path's derive_alias_box does and for the same
            // dominance reason. Keeping the two normalizations identical is
            // what makes the derivations comparable at all.
            if (root != nullptr && !whole_cover(res.box, root->dims)) {
                box = res.box;
            }
        }
        updates.push_back({.tid = id, .root = res.root, .box = std::move(box)});
    }
    for (auto &update : updates) {
        auto *handle = find_tensor(update.tid);
        if (handle == nullptr) {
            continue;
        }
        handle->aliases   = update.root;
        handle->alias_box = std::move(update.box);
    }

    // Deliberately NOT marking the pointer derivation as done. Structural adds
    // what the ``View`` nodes and the declarations say and claims nothing about
    // the relations only an address can reveal, so a graph that HAS addresses
    // still gets its containment search - which is also what makes "run one, then
    // the other, and nothing moves" an invariant every scheduling test enforces
    // for free rather than a property one test remembers to check.
    // Graph::link_alias_storage sets the flag itself before delegating here, so
    // the loaded-graph path does not re-enter.

    // A Loop body and a Conditional branch are separate graphs with their own
    // handles, and fourteen passes rewrite them; a body left unlinked answers
    // "this view aliases nothing" and its hazard edges vanish. The pointer path
    // is recursed by Graph::apply for that reason, and this recurses itself so
    // a direct call (the loader's, and the tests') gets the same tree.
    for_each_subgraph([](Graph &sub) { sub.link_alias_structural(); });
}

void Graph::link_alias_storage() {
    if (_aliases_linked) {
        return;
    }
    _aliases_linked = true;

    // A DERIVATION IS ABOUT TO RUN, so anything stamped with analysis_version is stale. That
    // counter reads as a node-list version and UsageAnalysis is keyed by alias OWNER, so a
    // relation this call is about to change is as much an input to it as the node list is.
    // Missed, the table stays keyed on the ids the previous derivation resolved to and every
    // lookup of a newly linked handle asks for an owner nobody recorded: an interface tensor
    // that a first manifest() reported vanished from the second, since the mint that made the
    // link happens inside the very analysis the link then invalidates.
    _analysis_version++;

    // A declaration is authoritative and an address coincidence is not, so
    // declarations go on first and the containment search below leaves them
    // alone (it skips any handle that already names a parent).
    apply_declared_aliases();

    // The loaded-graph state: structure was read from a file, nothing is
    // allocated, and no handle has an address to compare. Linking nothing here
    // is not a safe default - it is the exact shape of the full-cover alias bug,
    // a silently incomplete relation that surfaces as a race - so the derivation
    // switches to the structural one, which needs no addresses at all.
    //
    // MIXED graphs are the ordinary case, not an error: a deferred shell has no
    // address until it is materialized, and TensorHandle::data_ptr is a
    // registration-time snapshot nothing refreshes. Refusing to mix would refuse
    // most real graphs. The rule is therefore the honest one - the pointer
    // derivation runs whenever ANY handle carries an address, unchanged from
    // before, and it simply cannot see a null-address handle; the structural
    // derivation takes the whole graph only when NO handle carries one. What
    // spans the two modes is the declaration, which is applied above in both and
    // is the only way an alias between two address-less operands is expressible.
    if (!_tensors.empty() && std::ranges::none_of(_tensors, [](auto const &kv) { return kv.second.data_ptr != nullptr; })) {
        link_alias_structural();
        return;
    }

    struct Entry {
        char const *lo;
        char const *hi;
        TensorId    id;
    };
    std::vector<Entry> spans;
    spans.reserve(_tensors.size());
    for (auto const &[id, h] : _tensors) {
        char const *lo = nullptr;
        char const *hi = nullptr;
        if (handle_byte_span(h, lo, hi)) {
            spans.push_back({.lo = lo, .hi = hi, .id = id});
        }
    }
    if (spans.size() < 2) {
        return;
    }
    // The backward walk below only visits spans sorted at or before its own,
    // so a span must sort after every span that can own it: wider spans first
    // among equal starts, and among IDENTICAL spans the designated owner
    // (lower id, see below) first.
    std::ranges::sort(spans, [](Entry const &a, Entry const &b) {
        if (a.lo != b.lo) {
            return a.lo < b.lo;
        }
        if (a.hi != b.hi) {
            return a.hi > b.hi;
        }
        return a.id < b.id;
    });

    // Running max of hi over the prefix. Only a span starting at or before this
    // one can contain it, so once the best hi in that prefix falls short of our
    // end there is nothing left to find and the backward walk stops. Without it
    // the walk is the O(n^2) scan this replaces.
    std::vector<char const *> prefix_max_hi(spans.size());
    char const               *running = nullptr;
    for (size_t i = 0; i < spans.size(); ++i) {
        running          = (running == nullptr || spans[i].hi > running) ? spans[i].hi : running;
        prefix_max_hi[i] = running;
    }

    // Ownership is the strict order (more elements, then lower id): the id
    // tie-break keeps the relation acyclic when two handles cover the SAME
    // bytes, which is exactly what a view that spans its whole parent does.
    // Requiring strictly fewer elements instead left such a view linked to
    // nothing, so the hazard scan saw two unrelated tensors on one buffer and
    // emitted no edge between their accesses - DLPNO's singleton shape
    // classes hit this (`_W_pair` covers `_W` when a class has one pair).
    for (size_t i = 0; i < spans.size(); ++i) {
        auto self = _tensors.find(spans[i].id);
        if (self == _tensors.end() || self->second.aliases != 0) {
            continue;
        }
        // Last span starting at or before this one.
        size_t j = i;
        while (j > 0 && spans[j].lo > spans[i].lo) {
            --j;
        }
        for (;; --j) {
            if (prefix_max_hi[j] < spans[i].hi) {
                break; // nothing at or before j reaches far enough
            }
            if (spans[j].id != spans[i].id && spans[j].lo <= spans[i].lo && spans[i].hi <= spans[j].hi) {
                auto const owner = _tensors.find(spans[j].id);
                if (owner != _tensors.end() &&
                    (owner->second.total_elems() > self->second.total_elems() ||
                     (owner->second.total_elems() == self->second.total_elems() && owner->first < self->first))) {
                    // Link to the owner's ROOT, not to the owner. Containment
                    // is transitive, so both are correct answers to "who owns
                    // this", but only one of them keeps the chain short - and
                    // resolve_alias walks the chain on every hazard-scan
                    // lookup and gives up after a bounded number of hops.
                    //
                    // Without this, N handles covering the SAME bytes (the
                    // tie-break below the containment test links each to the
                    // one before it) form a chain of depth N, and past the hop
                    // limit resolve_alias returns a different mid-chain id for
                    // each of them. The hazard scan then keys their accesses
                    // under different owners and emits no edge between any of
                    // them, which under a threading executor is a silent data
                    // race - DLPNO-(T0) hit exactly this with one scratch
                    // buffer shared by 40 triplets, and the schedule's widest
                    // level came out at exactly (triplets - hop limit).
                    //
                    // Owners sort before their aliases and the outer loop runs
                    // in sorted order, so the owner's own link is already final
                    // here and one resolve gives the true root.
                    TensorId const root_id = resolve_alias(owner->first);
                    auto const    *root    = find_tensor(root_id);
                    if (root == nullptr) {
                        break;
                    }
                    self->second.aliases = root_id;
                    // The box has to live in the axis space of whatever
                    // ``aliases`` names, which is now the root rather than the
                    // immediate container.
                    if (!derive_alias_box(*root, self->second, self->second.alias_box)) {
                        self->second.alias_box.clear(); // unknown box reads as the whole parent
                    }
                    break;
                }
            }
            if (j == 0) {
                break;
            }
        }
    }

    // Second pass: PARTIAL overlap, which containment cannot express.
    //
    // Two handles that overlap without either containing the other - sliding
    // windows over a buffer no node names, so the common parent is never
    // registered - both come out of the loop above unlinked. The hazard scan
    // then sees two unrelated tensors sharing bytes and orders nothing between
    // their accesses, which under a threading executor is a silent data race.
    //
    // The fix is to give each run of mutually overlapping spans ONE root, so
    // the scan keys their accesses together. It cannot be exact: `aliases`
    // names a container, and a run like this has none, so the members' regions
    // are no longer expressible in the root's axis space. They lose their boxes
    // and conflict conservatively - which is the honest answer for a relation
    // this model does not describe, and still far better than no edge at all.
    //
    // A run whose members ALREADY share one root is left completely alone. That
    // is the common case and the important one: a registered parent and its
    // slices form a single run, they already resolve to the parent, and
    // relinking anything there would strip the slices of the boxes that keep
    // provably disjoint ones running in parallel.
    {
        size_t run_begin = 0;
        while (run_begin < spans.size()) {
            size_t      run_end = run_begin;
            char const *reach   = spans[run_begin].hi;
            while (run_end + 1 < spans.size() && spans[run_end + 1].lo < reach) {
                ++run_end;
                reach = std::max(reach, spans[run_end].hi);
            }

            // The canonical root is the widest span's, not the lowest id's: in
            // a run that mixes a real container with a partial overlapper, the
            // container is the one whose axis space the other members' boxes
            // are already written in, so choosing it keeps those boxes valid.
            // The id breaks ties so the choice does not depend on map order.
            TensorId canonical  = 0;
            size_t   widest     = 0;
            TensorId first_root = 0;
            bool     mixed      = false;
            for (size_t k = run_begin; k <= run_end; ++k) {
                TensorId const root = resolve_alias(spans[k].id);
                if (first_root == 0) {
                    first_root = root;
                } else if (root != first_root) {
                    mixed = true;
                }
                auto const *handle = find_tensor(root);
                if (handle == nullptr) {
                    continue;
                }
                size_t const extent = handle->total_elems();
                if (canonical == 0 || extent > widest || (extent == widest && root < canonical)) {
                    canonical = root;
                    widest    = extent;
                }
            }
            if (mixed && canonical != 0) {
                for (size_t k = run_begin; k <= run_end; ++k) {
                    TensorId const root = resolve_alias(spans[k].id);
                    if (root == canonical) {
                        continue;
                    }
                    auto *handle = find_tensor(root);
                    if (handle == nullptr) {
                        continue;
                    }
                    // Only ever a root, and only ever onto a DIFFERENT root of
                    // the same run, so the relation stays acyclic and every
                    // member of the run resolves to `canonical` from here.
                    handle->aliases = canonical;
                    handle->alias_box.clear();
                }
            }
            run_begin = run_end + 1;
        }
    }
}

EINSUMS_NAMESPACE_END(compute_graph)
