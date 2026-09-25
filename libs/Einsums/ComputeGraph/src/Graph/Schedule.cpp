//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file Schedule.cpp
/// @brief What has to happen before what, and how wide each node runs.
///
/// The dependence edges, the level assignment they induce, the thread plan laid
/// over that, and the topological order an executor finally walks.
///
/// @ref Graph::for_each_hazard_edge is the whole of the first part. Two nodes
/// get an edge when a region one writes overlaps a region the other touches, so
/// the interesting work is deciding what a node's effective inputs and outputs
/// ARE: a conditional or a loop stands for its whole subtree, and a view stands
/// for the region of its parent it covers. The overlap test itself is in
/// `AliasGeometry.hpp`, and it is conservative, so a region this file cannot
/// prove disjoint gets an edge it may not have needed.
///
/// @ref Graph::verify_level_independence is the assertion that the result is
/// right, and it is deliberately a separate walk rather than a check folded into
/// the construction: a bug that drops an edge cannot also hide itself from the
/// verifier.

#include <Einsums/CXX23/Expected.hpp>
#include <Einsums/ComputeGraph/CaptureContext.hpp>
#include <Einsums/ComputeGraph/Detail/ScalarDispatch.hpp>
#include <Einsums/ComputeGraph/EinsumSpec.hpp>
#include <Einsums/ComputeGraph/Error.hpp>
#include <Einsums/ComputeGraph/ExecutorBuilder.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Optimizer.hpp> // For OptimizerPass and PassManager
#include <Einsums/ComputeGraph/Options.hpp>
#include <Einsums/ComputeGraph/Passes/ThreadPlanning.hpp>
#include <Einsums/ComputeGraph/SpaceRegistryAccess.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/Profile/Profile.hpp>
#include <Einsums/TaskPool/WidthBudget.hpp>
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

namespace {

/// A view's box in its alias root's axis space: per axis, the half-open range [lo, hi).
using Box = std::vector<std::pair<std::int64_t, std::int64_t>>;

/// Whether two boxes might share an element. A missing box, or two of different rank, is
/// unprovable and so answers yes; one axis with an empty intersection makes them disjoint.
bool may_overlap(Box const *a, Box const *b) {
    if (a == nullptr || b == nullptr || a->size() != b->size()) {
        return true;
    }
    for (size_t d = 0; d < a->size(); ++d) {
        if (std::max((*a)[d].first, (*b)[d].first) >= std::min((*a)[d].second, (*b)[d].second)) {
            return false;
        }
    }
    return true;
}

/// The byte range a tensor's elements occupy, [lo, hi).
struct ByteSpan {
    char const *lo;
    char const *hi;
};

/// The byte span of every tensor whose storage can be reasoned about; a deferred or tiled tensor
/// has none and is left out.
std::unordered_map<TensorId, ByteSpan> byte_spans(std::unordered_map<TensorId, TensorHandle> const &tensors) {
    std::unordered_map<TensorId, ByteSpan> out;
    for (auto const &[id, handle] : tensors) {
        char const *lo = nullptr;
        char const *hi = nullptr;
        if (alias_geometry::handle_byte_span(handle, lo, hi)) {
            out.emplace(id, ByteSpan{.lo = lo, .hi = hi});
        }
    }
    return out;
}

} // namespace

using namespace alias_geometry;

void Graph::collect_subtree_referenced_ptrs(std::unordered_set<void const *> &out, bool *saw_unresolved) const {
    // Insert the tensor pointers referenced (read or written) by one graph's
    // own nodes. Resolves each TensorId through that graph's own map.
    auto collect_own = [saw_unresolved](Graph const &g, std::unordered_set<void const *> &acc) {
        auto add = [&](TensorId tid) {
            auto const *handle = g.find_tensor(tid);
            if (handle != nullptr && handle->tensor_ptr != nullptr) {
                acc.insert(handle->tensor_ptr);
                return;
            }
            // Either this graph's map has no entry for the id or the handle is an
            // unattached shell. The reference is real either way; what is missing is
            // the pointer that would let a caller recognise it, so say so rather than
            // let the omission read as "nothing here".
            if (saw_unresolved != nullptr) {
                *saw_unresolved = true;
            }
        };
        for (auto const &node : g._nodes) {
            for (auto tid : node.inputs) {
                add(tid);
            }
            for (auto tid : node.outputs) {
                add(tid);
            }
        }
    };

    for_each_subgraph([&](Graph const &sub) {
        collect_own(sub, out);                                    // sub's own references
        sub.collect_subtree_referenced_ptrs(out, saw_unresolved); // and sub's descendants
    });
}

std::pair<std::vector<TensorId>, std::vector<TensorId>> Graph::subtree_io(Node const &node) {
    // Walk the node's subtree (body / branches, recursively) and collect the
    // buffer pointers it reads and writes, keeping one representative handle per
    // buffer. Each sub-graph resolves its own TensorIds, so we key on the stable
    // tensor_ptr; the handle lets us register the buffer in the parent below if
    // it isn't already known there.
    std::set<void const *>                         read_ptrs;
    std::set<void const *>                         write_ptrs;
    std::unordered_map<void const *, TensorHandle> rep_handle;
    std::function<void(Graph const &)>             collect = [&](Graph const &sub) {
        for (auto const &nd : sub._nodes) {
            if (is_lifecycle(nd.kind)) {
                continue;
            }
            auto note = [&](TensorId tid, std::set<void const *> &dst) {
                // Resolve view aliases to the owning buffer so a read/write
                // through a view inside the subtree is attributed to the parent
                // tensor: otherwise an op outside the control-flow node that
                // touches the owner sees no dependency and can be misordered.
                auto const *handle = sub.find_tensor(sub.resolve_alias(tid));
                if (handle != nullptr && handle->tensor_ptr != nullptr) {
                    dst.insert(handle->tensor_ptr);
                    rep_handle.emplace(handle->tensor_ptr, *handle);
                }
            };
            for (auto tid : nd.inputs) {
                note(tid, read_ptrs);
            }
            for (auto tid : nd.outputs) {
                note(tid, write_ptrs);
            }
        }
        sub.for_each_subgraph(collect);
    };

    for_each_child_graph(node, collect);

    // Map subtree buffer pointers back to this graph's TensorIds. A buffer used
    // only inside sub-graphs has no parent TensorId yet; register one (a stable
    // shared id) so two control-flow nodes touching the same buffer resolve to
    // the same id and a dependency edge forms between them.
    //
    // That reuse-or-mint rule is find_or_register_tensor_ptr, and going through it rather
    // than through a private scan of the tensor table is what keeps this agreeing with the
    // hoisting and lifecycle passes, which mint their parent ids the same way. It also
    // settles an ambiguity a scan has: a capture can leave two handles naming one address
    // (a caller's wrapper dies and the next allocation lands there), and a scan returns
    // whichever the table iterates first, which is the stale one under the MSVC STL.
    auto resolve = [&](void const *ptr) -> TensorId { return find_or_register_tensor_ptr(rep_handle.at(ptr)); };

    // Collect the mapped TensorIds into ordered sets so the appended order is
    // deterministic (the dependency edge set is order-independent, but a stable
    // order keeps builds and any downstream iteration reproducible).
    std::set<TensorId> add_in;
    std::set<TensorId> add_out;
    for (auto const *p : read_ptrs) {
        add_in.insert(resolve(p));
    }
    for (auto const *p : write_ptrs) {
        add_out.insert(resolve(p));
    }

    return {std::vector<TensorId>(add_in.begin(), add_in.end()), std::vector<TensorId>(add_out.begin(), add_out.end())};
}

std::pair<std::vector<TensorId>, std::vector<TensorId>> Graph::effective_io(Node const &node) {
    std::vector<TensorId> ins  = node.inputs;
    std::vector<TensorId> outs = node.outputs;

    if (!is_control_flow(node.kind)) {
        return {ins, outs};
    }

    auto [add_in, add_out] = subtree_io(node);

    std::unordered_set<TensorId> have_in(ins.begin(), ins.end());
    std::unordered_set<TensorId> have_out(outs.begin(), outs.end());
    for (TensorId const tid : add_in) {
        if (have_in.insert(tid).second) {
            ins.push_back(tid);
        }
    }
    for (TensorId const tid : add_out) {
        if (have_out.insert(tid).second) {
            outs.push_back(tid);
        }
    }

    return {ins, outs};
}

// NOLINTNEXTLINE(misc-no-recursion): sub-graphs nest, so the refresh does too.
void Graph::refresh_setup_io() {
    // Recursively, because a fitting emitted into a loop body puts its setup there and the
    // body is the graph whose sort has to place it.
    for_each_subgraph([](Graph &sub) { sub.refresh_setup_io(); });

    for (auto &node : _nodes) {
        if (node.kind != OpKind::Setup) {
            continue;
        }
        auto [ins, outs] = subtree_io(node);
        node.inputs      = std::move(ins);
        node.outputs     = std::move(outs);
    }
}

std::pair<std::span<TensorId const>, std::span<TensorId const>> Graph::effective_io_cached(Node const &node, EffectiveIoCache &cache) {
    if (!is_control_flow(node.kind)) {
        // Ordinary nodes: their own I/O lists ARE the effective lists; hand
        // out views instead of heap-copying two vectors per node per scan.
        return {node.inputs, node.outputs};
    }
    // Control-flow nodes: the subtree walk is expensive, memoize it for the
    // duration of one sort (both hazard scans). The cache must NOT outlive
    // the call: passes like LoopInvariantHoisting move nodes across loop-body
    // boundaries, which changes the subtree I/O between sorts.
    auto it = cache.find(node.id);
    if (it == cache.end()) {
        it = cache.emplace(node.id, effective_io(node)).first;
    }
    return {it->second.first, it->second.second};
}

template <typename F>
void Graph::for_each_hazard_edge(EffectiveIoCache &cache, F &&emit) {
    // Storage-level aliasing must be resolved before anything reasons about
    // which buffer a node touches; cheap and idempotent after the first call.
    link_alias_storage();
    // Owner-resolved (resolve_alias), subtree-aware (effective_io_cached)
    // RAW/WAW/WAR scan. Every emitted edge points from an earlier to a later
    // position, so program order remains a valid topological order.
    //
    // Accesses through views with STATICALLY DISJOINT extents do not conflict:
    // per-slice writes like the CCSD ladder's ``r2[i,j] += ...`` touch
    // provably different elements of one parent, and serializing them (the old
    // owner-only scan) chained every slice of a tensor behind every other,
    // leaving parallel executors no width. Each describable view gets a
    // per-ROOT-axis interval box; two accesses conflict only when their boxes
    // may overlap. Anything unprovable - a runtime bound, a non-injective
    // parent layout, a whole-tensor access - keeps a null box, which overlaps
    // everything (the previous behavior). Element-disjoint writes commute
    // bitwise, so relaxing the order cannot change results.
    //
    // The box comes from StructuralAliasResolver, the SAME derivation
    // link_alias_structural writes onto the handles, and deliberately so: two
    // derivations of one alias relation disagreeing is the shape of both the
    // full-cover bug and the 32-hop cap. Sharing it also widened what is
    // describable here, since the walk composes chains - a permuted view and a
    // view of a view were both refused outright by the scan this replaces.
    std::unordered_map<TensorId, Box>      view_box;    // view tid -> box in root axis space
    std::unordered_map<TensorId, TensorId> view_parent; // view tid -> alias ROOT tid
    StructuralAliasResolver                resolver(*this);
    for (auto const &nd : _nodes) {
        if (nd.kind != OpKind::View || nd.outputs.size() != 1 || !nd.op_data.holds<ViewDescriptor>()) {
            continue;
        }
        TensorId const         vid = nd.outputs[0];
        StructuralAlias const &res = resolver.resolve(vid);
        if (!res.box_known || res.root == vid || view_box.contains(vid)) {
            continue;
        }
        TensorHandle const *root = find_tensor(res.root);
        // A box covering the whole root is the same statement as NO box and
        // schedules better: a whole-tensor write dominates and retires the
        // writers before it, while a full-cover box is not recognized as
        // dominating anything and every later access takes an edge against all
        // of them.
        if (root == nullptr || whole_cover(res.box, root->dims)) {
            continue;
        }
        view_parent.emplace(vid, res.root);
        view_box.emplace(vid, res.box);
    }

    // Views that reached the graph without a View node (sliced outside a
    // capture, linked by storage containment at registration; or declared by a
    // manifest and linked at bind) carry their box on the handle instead.
    // Without this they would still be ordered against the parent correctly, but
    // as whole-tensor accesses, chaining every slice behind every other and
    // costing the parallel executors their width.
    for (auto const &[tid, h] : _tensors) {
        if (h.aliases == 0 || h.alias_box.empty() || view_box.contains(tid)) {
            continue;
        }
        if (resolve_alias(h.aliases) != h.aliases) {
            continue; // box lives in the immediate parent's axis space
        }
        view_parent.emplace(tid, h.aliases);
        view_box.emplace(tid, Box(h.alias_box.begin(), h.alias_box.end()));
    }

    // a fully inside b. A retired reader may only be dropped when the write
    // COVERS it: an overlapped-but-uncovered reader still needs WAR edges
    // against later writers that touch its uncovered part.
    auto const covered_by = [](Box const *a, Box const *b) {
        if (b == nullptr) {
            return true; // whole-tensor write covers everything
        }
        if (a == nullptr || a->size() != b->size()) {
            return false;
        }
        for (size_t d = 0; d < a->size(); ++d) {
            if ((*a)[d].first < (*b)[d].first || (*a)[d].second > (*b)[d].second) {
                return false;
            }
        }
        return true;
    };

    // Box of an access through @p raw against owner @p tid; null = whole tensor.
    // A View node's own read/write of its parent is the METADATA rebind of the
    // slice it describes, so it carries that slice's box - a whole-tensor read
    // here would re-serialize every consumer of every other slice through it.
    auto const box_of = [&](Node const &nd, TensorId raw, TensorId tid) -> Box const * {
        if (auto it = view_box.find(raw); it != view_box.end() && resolve_alias(view_parent.at(raw)) == tid) {
            return &it->second;
        }
        if (nd.kind == OpKind::View && nd.outputs.size() == 1) {
            // ``view_parent`` names the alias ROOT, which for a view of a view
            // is not @p raw (the immediate parent). Comparing against the
            // resolved owner is what keeps a chained view's own metadata
            // rebind boxed rather than widening to the whole buffer.
            if (auto it = view_box.find(nd.outputs[0]); it != view_box.end() && view_parent.at(nd.outputs[0]) == tid) {
                return &it->second;
            }
        }
        return nullptr;
    };

    struct Access {
        size_t     pos;
        Box const *box;
    };
    std::unordered_map<TensorId, std::vector<Access>> writers;
    std::unordered_map<TensorId, std::vector<Access>> readers;

    // The same three hazards over the PARAMETER table. A WriteParam's effect
    // and a parameter-bound View's dependence on it never touch a tensor, so
    // the owner-resolved scan above emits no edge between them and the two are
    // free to be reordered - which silently freezes the slice at whatever the
    // table happened to hold. Keyed by parameter name; see param_writes /
    // param_reads in Node.hpp.
    std::unordered_map<std::string, std::vector<size_t>> param_writers;
    std::unordered_map<std::string, std::vector<size_t>> param_readers;

    size_t const n = _nodes.size();
    for (size_t i = 0; i < n; i++) {
        auto [eff_in, eff_out] = effective_io_cached(_nodes[i], cache);
        for (auto raw : eff_in) {
            TensorId const tid = resolve_alias(raw);
            Box const     *box = box_of(_nodes[i], raw, tid);
            for (auto const &w : writers[tid]) {
                if (w.pos != i && may_overlap(w.box, box)) {
                    emit(w.pos, i); // RAW: writer -> reader
                }
            }
            readers[tid].push_back({.pos = i, .box = box});
        }
        for (auto raw : eff_out) {
            TensorId const tid = resolve_alias(raw);
            Box const     *box = box_of(_nodes[i], raw, tid);
            auto          &wl  = writers[tid];
            for (auto const &w : wl) {
                if (w.pos != i && may_overlap(w.box, box)) {
                    emit(w.pos, i); // WAW: prior writer -> this writer
                }
            }
            auto &rl = readers[tid];
            for (auto const &r : rl) {
                if (r.pos != i && may_overlap(r.box, box)) {
                    emit(r.pos, i); // WAR: prior reader -> this writer
                }
            }
            // Ordered readers are consumed; anything not fully covered must
            // stay visible to future writers of its uncovered part.
            std::erase_if(rl, [&](Access const &r) { return covered_by(r.box, box); });
            // A COVERED prior writer is consumed by the same argument, and for
            // the same reason it has to be covered rather than merely
            // overlapped. Anything later that reaches the covered writer
            // reaches this one too, so it takes an edge from this one, which
            // already carries an edge from the covered writer: the order
            // survives as a path. What that saves is quadratic - a buffer
            // written n times in a row used to keep all n writers and emit an
            // edge from every one of them to every later access - and it used
            // to be spelled as a whole-tensor special case, which stopped
            // applying the moment the write carried a box.
            std::erase_if(wl, [&](Access const &w) { return covered_by(w.box, box); });
            wl.push_back({.pos = i, .box = box});
        }

        for (auto const &pname : param_reads(_nodes[i])) {
            for (size_t const w : param_writers[pname]) {
                if (w != i) {
                    emit(w, i); // RAW: parameter write -> slice that resolves it
                }
            }
            param_readers[pname].push_back(i);
        }
        for (auto const &pname : param_writes(_nodes[i])) {
            for (size_t const w : param_writers[pname]) {
                if (w != i) {
                    emit(w, i); // WAW: the later write must win
                }
            }
            for (size_t const r : param_readers[pname]) {
                if (r != i) {
                    emit(r, i); // WAR: readers of the old value must go first
                }
            }
            param_readers[pname].clear();
            param_writers[pname].clear();
            param_writers[pname].push_back(i);
        }
    }
}

void Graph::rebuild_deps(EffectiveIoCache &cache) {
    // Position-keyed dependency lists for the current node order.
    size_t const n = _nodes.size();
    _deps.successors.assign(n, {});
    _deps.predecessors.assign(n, {});

    for_each_hazard_edge(cache, [&](size_t producer, size_t consumer) {
        _deps.successors[producer].push_back(consumer);
        _deps.predecessors[consumer].push_back(producer);
    });

    rebuild_levels();
}

std::vector<std::string> Graph::unjustified_hazard_edges() {
    topological_sort();

    // The alias relation the edges were derived from is the one under test, so the justification
    // is rebuilt from the graph's own View NODES instead. A view's parent is its first input,
    // which is what the node records and what no pass rewrites.
    std::unordered_map<TensorId, TensorId> view_parent;
    for (auto const &node : _nodes) {
        if (node.kind == OpKind::View && node.outputs.size() == 1 && !node.inputs.empty()) {
            view_parent.emplace(node.outputs[0], node.inputs[0]);
        }
    }
    auto const structural_root = [&](TensorId tid) {
        for (std::size_t hop = 0; hop <= _tensors.size(); ++hop) {
            auto const it = view_parent.find(tid);
            if (it == view_parent.end() || it->second == tid) {
                break;
            }
            tid = it->second;
        }
        return tid;
    };

    auto const span = byte_spans(_tensors);

    auto const may_share = [&](TensorId a, TensorId b) {
        if (a == b) {
            return true;
        }
        TensorId const ra = structural_root(a);
        TensorId const rb = structural_root(b);
        if (ra == rb) {
            return true;
        }
        auto const ia = span.find(ra);
        auto const ib = span.find(rb);
        if (ia == span.end() || ib == span.end()) {
            // No address on one of them, so nothing but the structure above could relate them.
            return false;
        }
        return ia->second.lo < ib->second.hi && ib->second.lo < ia->second.hi;
    };

    EffectiveIoCache                                    cache;
    std::vector<std::vector<std::pair<TensorId, bool>>> touched(_nodes.size());
    std::vector<std::unordered_set<std::string>>        params(_nodes.size());
    for (std::size_t i = 0; i < _nodes.size(); ++i) {
        auto [eff_in, eff_out] = effective_io_cached(_nodes[i], cache);
        for (TensorId const tid : eff_in) {
            touched[i].emplace_back(tid, false);
        }
        for (TensorId const tid : eff_out) {
            touched[i].emplace_back(tid, true);
        }
        // The other half of what the hazard scan orders: a parameter write and the slice that
        // resolves against it touch no tensor at all.
        for (auto const &name : param_reads(_nodes[i])) {
            params[i].insert(name);
        }
        for (auto const &name : param_writes(_nodes[i])) {
            params[i].insert(name);
        }
    }

    std::vector<std::string> out;
    for (std::size_t from = 0; from < _deps.successors.size(); ++from) {
        for (std::size_t const to : _deps.successors[from]) {
            bool justified = false;
            for (auto const &[tu, wu] : touched[from]) {
                for (auto const &[tv, wv] : touched[to]) {
                    if ((!wu && !wv) || !may_share(tu, tv)) {
                        continue;
                    }
                    justified = true;
                    break;
                }
                if (justified) {
                    break;
                }
            }
            if (!justified) {
                for (auto const &name : params[from]) {
                    if (params[to].contains(name)) {
                        justified = true;
                        break;
                    }
                }
            }
            if (!justified) {
                out.push_back(fmt::format("{} ({}) -> {} ({})", from, _nodes[from].label, to, _nodes[to].label));
            }
        }
    }
    return out;
}

void Graph::verify_level_independence() const {
    // Byte span per registered tensor. A tensor with no span that can be
    // reasoned about (deferred allocation, tiled layout) is skipped entirely
    // rather than guessed at; the hazard scan skips it too, so a conflict
    // through one is out of scope for both.
    auto const span = byte_spans(_tensors);
    if (span.size() < 2) {
        return;
    }

    // Group tensors whose byte ranges overlap, by merging sorted intervals.
    // This is the independence that matters: it never consults `aliases`, so a
    // defect in the alias links cannot hide a conflict from this check.
    std::vector<std::pair<TensorId, ByteSpan>> ordered(span.begin(), span.end());
    std::ranges::sort(ordered, [](auto const &a, auto const &b) {
        return a.second.lo != b.second.lo ? a.second.lo < b.second.lo : a.second.hi > b.second.hi;
    });
    std::unordered_map<TensorId, size_t> group_of;
    std::vector<char const *>            group_end;
    for (auto const &[id, s] : ordered) {
        if (!group_end.empty() && s.lo < group_end.back()) {
            group_of[id]     = group_end.size() - 1;
            group_end.back() = std::max(group_end.back(), s.hi);
        } else {
            group_of[id] = group_end.size();
            group_end.push_back(s.hi);
        }
    }

    // Per group, the widest member, which is the only candidate for an axis
    // space every other member's region can be expressed in. A group with no
    // single container keeps a null root and every access in it is treated as
    // whole-group, which is conservative.
    std::vector<TensorId> root(group_end.size(), 0);
    for (auto const &[id, s] : ordered) {
        size_t const g = group_of[id];
        if (root[g] == 0) {
            root[g] = id; // sorted: first member of a group starts earliest and spans furthest
        } else {
            auto const &r = span.at(root[g]);
            if (s.lo < r.lo || s.hi > r.hi) {
                root[g] = 0; // no single container; fall back to conservative
                // Keep the group; a zero root simply disables the box test.
            }
        }
    }

    struct Access {
        size_t   pos;
        TensorId tid;
        bool     is_write;
    };

    // A region's box in its group root's axis space, derived from the two
    // handles alone. Memoized: a tensor read by many nodes derives once.
    std::unordered_map<TensorId, Box> box_cache;
    std::unordered_set<TensorId>      box_absent;
    auto const                        box_for = [&](TensorId tid, size_t g) -> Box const                        *{
        if (root[g] == 0 || box_absent.contains(tid)) {
            return nullptr;
        }
        if (auto it = box_cache.find(tid); it != box_cache.end()) {
            return &it->second;
        }
        auto const *self  = find_tensor(tid);
        auto const *owner = find_tensor(root[g]);
        if (self == nullptr || owner == nullptr) {
            box_absent.insert(tid);
            return nullptr;
        }
        Box derived;
        if (tid == root[g]) {
            derived.reserve(owner->dims.size());
            for (size_t const d : owner->dims) {
                derived.emplace_back(0, static_cast<std::int64_t>(d));
            }
        } else if (!derive_alias_box(*owner, *self, derived)) {
            box_absent.insert(tid);
            return nullptr;
        }
        return &box_cache.emplace(tid, std::move(derived)).first->second;
    };

    EffectiveIoCache cache;
    for (size_t level_index = 0; level_index < _deps.levels.size(); ++level_index) {
        auto const &level = _deps.levels[level_index];
        if (level.size() < 2) {
            continue;
        }
        // Only nodes sharing a storage group can conflict, so the pairwise
        // test runs per group rather than over the level.
        std::unordered_map<size_t, std::vector<Access>> touched;
        // A View node's effect is the handle it binds, so its hazards are
        // against the OTHER nodes that name that handle rather than against
        // the storage: bound = who binds a handle, named = who else names it.
        std::unordered_map<TensorId, std::vector<size_t>> bound;
        std::unordered_map<TensorId, std::vector<size_t>> named;
        for (size_t const pos : level) {
            auto const &node       = _nodes[pos];
            auto [eff_in, eff_out] = const_cast<Graph *>(this)->effective_io_cached(node, cache);
            // A View node writes its slice handle's dims, strides and data
            // pointer, and no element of the parent: the executor re-emplaces
            // the handle from the parent's own pointer and strides. So it is
            // NOT a writer of the storage it spans, and two of them are
            // independent however their slices overlap - which is what every
            // graph that records more than one view of one buffer depends on.
            //
            // What it does read is the parent's pointer and extents, so it
            // stays a READER of the region it describes: a node that moves the
            // parent's data or its allocation out from under it on the same
            // level is still a conflict. The read is attributed to the SLICE
            // rather than to the parent it names as an input, because the
            // parent's whole extent would conflict with every disjoint slice
            // written on the level.
            if (node.kind == OpKind::View && node.outputs.size() == 1) {
                TensorId const slice = node.outputs[0];
                if (auto it = group_of.find(slice); it != group_of.end()) {
                    touched[it->second].push_back({.pos = pos, .tid = slice, .is_write = false});
                }
                bound[slice].push_back(pos);
                for (auto const tid : eff_in) {
                    named[tid].push_back(pos); // a view of a view reads the parent handle
                }
                continue;
            }
            for (auto const tid : eff_in) {
                named[tid].push_back(pos);
                if (auto it = group_of.find(tid); it != group_of.end()) {
                    touched[it->second].push_back({.pos = pos, .tid = tid, .is_write = false});
                }
            }
            for (auto const tid : eff_out) {
                named[tid].push_back(pos);
                if (auto it = group_of.find(tid); it != group_of.end()) {
                    touched[it->second].push_back({.pos = pos, .tid = tid, .is_write = true});
                }
            }
        }

        // The hazard a metadata write still carries. Everything that reads or
        // writes a slice reads the handle the View node binds, so it may not
        // run alongside it; and two View nodes binding the SAME handle are a
        // write-write on that handle.
        for (auto const &[tid, binders] : bound) {
            size_t const self  = binders.front();
            size_t       other = self;
            if (binders.size() > 1) {
                other = binders[1];
            } else if (auto const it = named.find(tid); it != named.end()) {
                for (size_t const pos : it->second) {
                    if (pos != self) {
                        other = pos;
                        break;
                    }
                }
            }
            if (other == self) {
                continue;
            }
            EINSUMS_THROW_EXCEPTION(std::runtime_error,
                                    "Graph '{}': nodes {} ({}) and {} ({}) share execution level {} but one binds tensor {}'s view "
                                    "metadata while the other uses it. A level-scheduling executor launches them together, so this is "
                                    "a data race on the handle; the hazard scan should have ordered them.",
                                    _name, self, _nodes[self].label, other, _nodes[other].label, level_index, tid);
        }

        for (auto const &[g, accesses] : touched) {
            for (size_t a = 0; a < accesses.size(); ++a) {
                for (size_t b = a + 1; b < accesses.size(); ++b) {
                    if (accesses[a].pos == accesses[b].pos || (!accesses[a].is_write && !accesses[b].is_write)) {
                        continue;
                    }
                    if (!may_overlap(box_for(accesses[a].tid, g), box_for(accesses[b].tid, g))) {
                        continue;
                    }
                    EINSUMS_THROW_EXCEPTION(std::runtime_error,
                                            "Graph '{}': nodes {} ({}) and {} ({}) share execution level {} but both touch overlapping "
                                            "storage (tensors {} and {}, at least one written). A level-scheduling executor launches them "
                                            "together, so this is a data race; the hazard scan should have ordered them.",
                                            _name, accesses[a].pos, _nodes[accesses[a].pos].label, accesses[b].pos,
                                            _nodes[accesses[b].pos].label, level_index, accesses[a].tid, accesses[b].tid);
                }
            }
        }
    }
}

void Graph::rebuild_levels() {
    // Level partition for level-scheduling executors. Edges always point
    // from earlier to later positions (the hazard scan links prior
    // writers/readers to the current node), so one forward pass suffices.
    size_t const        n = _deps.predecessors.size();
    std::vector<size_t> level(n, 0);
    size_t              max_level = 0;
    for (size_t i = 0; i < n; i++) {
        for (size_t const pred : _deps.predecessors[i]) {
            level[i] = std::max(level[i], level[pred] + 1);
        }
        max_level = std::max(max_level, level[i]);
    }
    _deps.levels.assign(max_level + 1, {});
    for (size_t i = 0; i < n; i++) {
        _deps.levels[level[i]].push_back(i);
    }
}

bool Graph::run_thread_planner(unsigned threads) {
    passes::ThreadPlanning planner(threads);
    // This path builds its pass directly rather than through a PassManager, so
    // nothing else would ever give the planner a verbosity and every report it
    // makes would be unreachable from a normal run. A thread plan is the one
    // result here that cannot be inferred from the outside: whether it widened
    // anything, and if not which gate declined, is otherwise invisible.
    planner.set_verbosity(static_cast<int>(config::get(option::PassVerbosity)));
    planner.run(*this);
    _planned_thread_count = static_cast<std::uint16_t>(threads);
    return planner.num_widened() > 0;
}

bool Graph::plan_threads(bool freeze) {
    auto &budget = task_pool::WidthBudget::get_singleton();
    // The budget is what admission rations against, so it is also what a plan
    // has to be made for; asking the hardware directly could disagree with it.
    budget.sync_machine_width();
    unsigned const threads = std::max(1U, budget.total());

    // Timings are cleared at the start of every execute() and written during
    // it, so a non-empty set means this graph has completed at least one
    // replay and the planner has measurements instead of a model.
    bool have_timings = false;
    {
        std::scoped_lock const lock(*_content_mutex);
        have_timings = !_timing_samples.empty();
    }

    bool const widened = run_thread_planner(threads);

    // Cold plans are model plans, and the model is the part of this that is
    // guessing. What the first real timings buy is a TRIAL, not a decree: a
    // re-planned candidate has to beat the cold plan on the wall clock before
    // it may replace it (see finish_replay_thread_plan for why estimates
    // cannot referee that contest).
    _plan_trial = (!freeze && !have_timings) ? ThreadPlanTrial::Armed : ThreadPlanTrial::None;
    _plan_incumbent.clear();
    _plan_candidate.clear();
    return widened;
}

namespace {

using PlanSnapshot = std::vector<std::pair<std::uint16_t, std::int64_t>>;

/// Record every node's planned width and admission priority, container bodies
/// included, in one deterministic walk order shared with apply_thread_plan.
///
/// Setup bodies are deliberately left out of both walks: the widths these two
/// snapshot and restore are the ones ThreadPlanning::plan_graph writes, and that
/// planner descends into loop bodies and conditional branches only.
void collect_thread_plan(Graph &graph, PlanSnapshot &out) {
    for (auto &node : graph.nodes()) {
        out.emplace_back(node.thread_width, node.admission_priority);
        for_each_child_graph(node, [&out](Graph &sub) { collect_thread_plan(sub, out); }, /*include_setup=*/false);
    }
}

void apply_thread_plan(Graph &graph, PlanSnapshot const &plan, size_t &pos) {
    for (auto &node : graph.nodes()) {
        if (pos >= plan.size()) {
            return; // structure changed under the trial; leave the rest alone
        }
        node.thread_width       = plan[pos].first;
        node.admission_priority = plan[pos].second;
        pos++;
        for_each_child_graph(node, [&](Graph &sub) { apply_thread_plan(sub, plan, pos); }, /*include_setup=*/false);
    }
}

[[nodiscard]] bool same_widths(PlanSnapshot const &a, PlanSnapshot const &b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); i++) {
        if (a[i].first != b[i].first) {
            return false;
        }
    }
    return true;
}

/// How much faster the candidate's replay must be before it replaces the cold
/// plan. Consecutive replays of one graph differ by a few percent on a machine
/// doing anything else, and the trial reads exactly one replay of each side,
/// so a candidate inside this band is indistinguishable from noise - and the
/// incumbent is the plan a calibrated model chose on purpose.
constexpr double kThreadPlanTrialMargin = 0.05;

} // namespace

void Graph::finish_replay_thread_plan(double replay_ms) {
    switch (_plan_trial) {
    case ThreadPlanTrial::None:
        return;

    case ThreadPlanTrial::Armed: {
        // Disarmed BEFORE planning, so a planner that throws does not leave the
        // graph re-planning at the end of every replay from here on.
        _plan_trial = ThreadPlanTrial::None;

        _plan_incumbent.clear();
        collect_thread_plan(*this, _plan_incumbent);

        auto &budget = task_pool::WidthBudget::get_singleton();
        budget.sync_machine_width();
        run_thread_planner(std::max(1U, budget.total()));

        _plan_candidate.clear();
        collect_thread_plan(*this, _plan_candidate);
        if (same_widths(_plan_candidate, _plan_incumbent)) {
            // The timings agree with the model about the widths, so there is
            // nothing to referee; the re-plan's measured admission priorities
            // are kept and the plan is final.
            _plan_incumbent.clear();
            _plan_candidate.clear();
            return;
        }
        // The candidate's widths are live now; the next replay times them.
        _plan_trial = ThreadPlanTrial::Candidate;
        return;
    }

    case ThreadPlanTrial::Candidate: {
        _plan_candidate_ms = replay_ms;
        size_t pos         = 0;
        apply_thread_plan(*this, _plan_incumbent, pos);
        _plan_trial = ThreadPlanTrial::Incumbent;
        return;
    }

    case ThreadPlanTrial::Incumbent: {
        _plan_trial          = ThreadPlanTrial::None;
        bool const candidate = _plan_candidate_ms < (1.0 - kThreadPlanTrialMargin) * replay_ms;
        if (candidate) {
            size_t pos = 0;
            apply_thread_plan(*this, _plan_candidate, pos);
        }
        if (config::get(option::PassVerbosity) >= 1) {
            fmt::print(stderr, "[ThreadPlanning] trial on '{}': candidate replay {:.1f} ms vs incumbent replay {:.1f} ms; keeping the {}\n",
                       _name, _plan_candidate_ms, replay_ms, candidate ? "re-planned widths" : "cold plan");
        }
        _plan_incumbent.clear();
        _plan_incumbent.shrink_to_fit();
        _plan_candidate.clear();
        _plan_candidate.shrink_to_fit();
        return;
    }
    }
}

size_t Graph::schedule_edge_count() {
    topological_sort();
    size_t edges = 0;
    for (auto const &succ : _deps.successors) {
        edges += succ.size();
    }
    return edges;
}

std::vector<size_t> Graph::schedule_level_sizes() {
    topological_sort();
    std::vector<size_t> sizes;
    sizes.reserve(_deps.levels.size());
    for (auto const &level : _deps.levels) {
        sizes.push_back(level.size());
    }
    return sizes;
}

void Graph::topological_sort() {
    std::scoped_lock const lock(*_content_mutex);
    assign_node_ids();
    // Defense in depth: a pass that mutates the node list without declaring
    // it (mark_sorted / add_node) leaves stale flags. A count mismatch is the
    // detectable symptom; downgrade to a full re-sort instead of letting a
    // consumer index _deps out of range.
    if (_deps.successors.size() != _nodes.size()) {
        _deps_valid = false;
    }

    if (_sorted && _deps_valid) {
        // Node order and dependency lists both current; the ~20-pass default
        // pipeline hits this on every pass that follows a non-mutating one.
        return;
    }

    if (_nodes.empty()) {
        _deps.successors.clear();
        _deps.predecessors.clear();
        _deps.levels.clear();
        _sorted     = true;
        _deps_valid = true;
        return;
    }

    // A setup body captured since the last sort makes the node's lists stale, and the sort is
    // the first consumer of them. Idempotent, and a no-op on the graphs that hold no setup.
    refresh_setup_io();

    if (_sorted) {
        // A pass rebuilt or filtered the node list and vouched for the order
        // via mark_sorted(); only the position-keyed _deps are stale. This
        // also means a pass-chosen order (e.g. Reorder's memory-aware
        // schedule) survives instead of being re-derived by a fresh Kahn.
        EffectiveIoCache cache;
        rebuild_deps(cache);
        _deps_valid = true;
        return;
    }

    // Build adjacency from data dependencies:
    // If node A writes tensor T and node B reads tensor T (and B comes after A),
    // then A → B (A must execute before B).

    size_t const n = _nodes.size();

    // Track dependencies: read-after-write, write-after-write, write-after-read.
    // Keyed by *owner* TensorId (resolve_alias), so reads/writes through a view
    // register against the parent tensor. Without this, the scheduler would
    // treat ``GEMM(C_occ, …)`` and ``Syev(C, …)`` as independent, they're not,
    // since C_occ aliases C.
    std::vector<std::vector<size_t>> adj(n);
    std::vector<size_t>              in_degree(n, 0);

    // eff_cache memoizes effective I/O across this scan and the rebuild_deps
    // call further below; keyed by NodeId so it survives the move of nodes into
    // their sorted positions.
    EffectiveIoCache eff_cache;

    // The position-keyed dependency lists come out of THIS scan rather than a
    // second one. They are the same edges, and the scan is what this function
    // costs: it walks every node's effective I/O and intersects view boxes
    // pairwise, which on a DLPNO-MP2 iteration body (nodes carrying ~1000
    // operands each) is 28 ms of a 83 ms graph build - paid twice. They are
    // only valid if the sort leaves the nodes where they are, which is why the
    // Kahn loop below reports whether anything moved.
    _deps.successors.assign(n, {});
    _deps.predecessors.assign(n, {});

    for_each_hazard_edge(eff_cache, [&](size_t producer, size_t consumer) {
        adj[producer].push_back(consumer);
        in_degree[consumer]++;
        _deps.successors[producer].push_back(consumer);
        _deps.predecessors[consumer].push_back(producer);
    });

    // Kahn's algorithm, taking the smallest ready POSITION rather than FIFO.
    // Hazard edges always point from an earlier to a later position, so
    // program order is itself a valid topological order and this reproduces
    // it exactly. A FIFO queue does not: a zero-in-degree node late in
    // program order pops ahead of an earlier node that waits on any edge, so
    // an edge the hazard scan missed became a REORDER that broke even serial
    // replay, instead of staying harmless there.
    std::priority_queue<size_t, std::vector<size_t>, std::greater<>> ready;
    for (size_t i = 0; i < n; i++) {
        if (in_degree[i] == 0) {
            ready.push(i);
        }
    }

    std::vector<Node> sorted;
    sorted.reserve(n);

    bool reordered = false;
    while (!ready.empty()) {
        size_t const idx = ready.top();
        ready.pop();
        if (idx != sorted.size()) {
            reordered = true;
        }
        sorted.push_back(std::move(_nodes[idx]));

        for (size_t const succ : adj[idx]) {
            if (--in_degree[succ] == 0) {
                ready.push(succ);
            }
        }
    }

    if (sorted.size() != n) {
        EINSUMS_THROW_EXCEPTION(std::runtime_error, "Graph '{}': topological sort failed — cycle detected", _name);
    }

    _nodes = std::move(sorted);

    if (reordered) {
        // Positions moved, so the lists built during the scan are keyed to the
        // wrong slots and there is nothing to do but scan again.
        rebuild_deps(eff_cache);
    } else {
        // Every node stayed put, so the lists are already right and only the
        // level partition is missing. This is the normal outcome rather than a
        // lucky one: every hazard edge points from an earlier position to a
        // later one, so program order is itself a topological order, and the
        // priority queue above takes the smallest ready position, which
        // reproduces it. The `reordered` flag is what keeps that an observation
        // instead of an assumption - if the invariant ever breaks, this falls
        // back to the full rebuild rather than to stale lists.
        rebuild_levels();
    }

    _sorted     = true;
    _deps_valid = true;
    // Positions changed but the node COUNT did not, so cached position-keyed
    // analyses (UsageAnalysis) must be invalidated explicitly - the count
    // defense in usage() cannot see a same-size reorder.
    _analysis_version++;
}

EINSUMS_NAMESPACE_END(compute_graph)
