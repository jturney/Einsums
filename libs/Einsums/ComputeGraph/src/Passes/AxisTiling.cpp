//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config/Namespace.hpp>

#include "AxisTilingDetail.hpp"

EINSUMS_NAMESPACE_BEGIN(compute_graph::passes)

using namespace axis_tiling;

void AxisTiling::set_block_extent(std::int64_t extent) {
    _block_pin = extent;
}

void AxisTiling::use_reference_cost_model() {
    _cost_model = CostModel{};
}

void AxisTiling::set_memory_cap(std::int64_t bytes) {
    _memory_cap   = bytes;
    _cap_explicit = true;
}

std::int64_t AxisTiling::memory_cap() const {
    return _cap_explicit ? _memory_cap : config::get(option::GraphTilingMemoryCap);
}

std::vector<std::string> AxisTiling::explain() const {
    std::vector<std::string> lines;
    lines.reserve(_stats.regions.size() + _stats.held_results.size() + 1);
    for (auto const &region : _stats.regions) {
        lines.push_back(fmt::format("AxisTiling: {}", region));
    }
    for (auto const &held : _stats.held_results) {
        lines.push_back(fmt::format("AxisTiling: {}", held));
    }
    if (!_stats.accumulator.empty()) {
        lines.push_back(fmt::format("AxisTiling: '{}' accumulates across the first region's iterations", _stats.accumulator));
    }
    for (auto const &name : _stats.summed_into) {
        lines.push_back(fmt::format("AxisTiling: '{}' collects a sum over a sliced axis, one block's share per iteration", name));
    }
    return lines;
}

namespace {

/// The first and last position of @p io that reads or writes @p tid's buffer in @p graph, or every
/// position when none does.
std::pair<std::size_t, std::size_t> held_span(Graph const &graph, std::vector<BufferIo> const &io, TensorId tid) {
    TensorId const buffer = tid == 0 ? 0 : graph.buffer_of(tid);
    std::size_t    first  = io.size();
    std::size_t    last   = 0;
    for (std::size_t i = 0; i < io.size(); ++i) {
        if (buffer != 0 && (io[i].reads.contains(buffer) || io[i].writes.contains(buffer))) {
            first = std::min(first, i);
            last  = i;
        }
    }
    if (first == io.size()) {
        return {0, io.empty() ? 0 : io.size() - 1};
    }
    return {first, last};
}

/// Where the loop can stand in for the statements it replaces, or nothing when it cannot.
///
/// A region need not be contiguous: a statement of another term may sit between two of its
/// statements. The loop takes the place of the LAST of them when nothing in between depends on
/// an earlier one, so every earlier one can move down to it, and of the FIRST when no later one
/// depends on anything in between, so every later one can move up. A node in between that
/// depends on one statement while another depends on it allows neither.
std::optional<std::size_t> gather_site(std::vector<BufferIo> const &io, std::vector<std::size_t> const &replaced) {
    std::set<std::size_t> const moving(replaced.begin(), replaced.end());
    bool                        sink  = true;
    bool                        hoist = true;
    for (std::size_t n = replaced.front() + 1; n < replaced.back(); ++n) {
        if (moving.contains(n)) {
            continue;
        }
        for (std::size_t const m : replaced) {
            sink  = sink && !(m < n && conflicts(io[m], io[n]));
            hoist = hoist && !(m > n && conflicts(io[n], io[m]));
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
        for_each_child_graph(nodes[i], [&](auto const &sub) { _stats.enclosing[&sub] = {&graph, i}; });
    }
    return moved;
}

std::size_t AxisTiling::Footprint::beside(std::size_t lo, std::size_t hi) const {
    std::size_t out = fixed;
    for (auto const &life : lives) {
        if (life.first <= hi && lo <= life.last) {
            out += life.bytes;
        }
    }
    return out;
}

std::size_t AxisTiling::Footprint::peak() const {
    // The most held at once is held while some intermediate is first written, so only those
    // positions need summing.
    std::size_t out = fixed;
    for (auto const &at : lives) {
        out = std::max(out, beside(at.first, at.first));
    }
    return out;
}

AxisTiling::Footprint AxisTiling::footprint(Graph &graph) const {
    // The root, and on the way to it each level's node whose body holds the graph being tiled.
    std::vector<std::pair<Graph *, std::size_t>> chain;
    Graph const                                 *root = &graph;
    for (auto hit = _stats.enclosing.find(root); hit != _stats.enclosing.end(); hit = _stats.enclosing.find(root)) {
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
        BufferIo const here = buffer_io(*outer, nodes[own]);
        for (std::size_t j = 0; j < nodes.size(); ++j) {
            bool has_body = false;
            for_each_child_graph(nodes[j], [&](Graph const &) { has_body = true; });
            if (j == own || !has_body) {
                continue;
            }
            if (conflicts(here, buffer_io(*outer, nodes[j]))) {
                for_each_child_graph(nodes[j], [&](Graph const &child) { apart.insert(&child); });
            }
        }
    }

    // A body names its parent's tensors through handles of its own over the same tensor object,
    // so one tensor is counted once by that object; a handle with no object is its graph's own.
    // The span of this graph's statements each buffer is touched in, and whether the first touch
    // writes it.
    struct Touch {
        std::size_t first{0};
        std::size_t last{0};
        bool        written_first{false};
    };
    std::map<TensorId, Touch> touches;
    {
        auto const io = buffer_io(graph);
        for (std::size_t i = 0; i < io.size(); ++i) {
            for (auto const *buffers : {&io[i].reads, &io[i].writes}) {
                for (TensorId const buffer : *buffers) {
                    auto [at, added] = touches.try_emplace(buffer, Touch{.first = i, .last = i});
                    at->second.last  = i;
                    if (added) {
                        at->second.written_first = io[i].writes.contains(buffer) && !io[i].reads.contains(buffer);
                    }
                }
            }
        }
    }

    Footprint                                    out;
    std::set<void const *>                       objects;
    std::set<std::pair<Graph const *, TensorId>> unbacked;
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
            if (!fresh) {
                continue;
            }
            std::size_t const bytes = handle.total_bytes();
            out.total += bytes;
            // An intermediate of the graph being tiled that a statement writes before any reads it
            // is allocated before that writer and freed after its last reader.
            if (&g == &graph && handle.is_intermediate) {
                if (auto const hit = touches.find(tid); hit != touches.end() && hit->second.written_first) {
                    out.lives.push_back(Footprint::Life{.bytes = bytes, .first = hit->second.first, .last = hit->second.last});
                    continue;
                }
            }
            out.fixed += bytes;
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
    return out;
}

bool AxisTiling::tile_graph(Graph &graph) {
    PassCounter const tiled{_stats.num_tiled};

    Scope const scope{.graph = graph, .enclosing = _stats.enclosing};

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
        // The spans the footprint records are positions in this order, and so are the regions'.
        graph.topological_sort();
        Footprint const   fp   = derived ? footprint(graph) : Footprint{};
        std::size_t const peak = fp.peak();
        auto const        io   = derived ? buffer_io(graph) : std::vector<BufferIo>{};

        // A program over its budget streams because it must, and a loop it streams in may take
        // whatever the budget leaves; the share caps only what a program that fits streams for the
        // share's own sake. Held to the share as well, the CCSD tutorial's file-backed stage cut
        // its ladder into blocks of 2 x 4 under a 16 MB budget while the replay held 11.6 MB, and
        // spent more than half its iteration on the small blocks.
        bool const forced = derived && peak > budget;

        // The cap one intermediate @p tid of @p bytes is held to: the fixed cap, or what the budget
        // leaves after everything else held while it is, no more than its share unless the program
        // is over its budget. Zero when the others alone exceed the budget.
        auto const limit_for = [&](TensorId tid, std::size_t bytes) -> std::size_t {
            if (!derived) {
                return static_cast<std::size_t>(cap);
            }
            auto const [lo, hi]      = held_span(graph, io, tid);
            std::size_t const beside = fp.beside(lo, hi);
            std::size_t const others = beside - std::min(beside, bytes);
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
                        if (derived && backing->write && !backing->update && sliceable && bytes > limit_for(tid, bytes) &&
                            !declined.contains(tid) && std::ranges::find(seeds, std::pair{bytes, tid}) == seeds.end()) {
                            largest = std::max(largest, bytes);
                            seeds.emplace_back(bytes, tid);
                        }
                        continue;
                    }
                    // A read-backed tensor is not one: its storage holds it, and SlabStreaming moves it.
                    if (sliceable && handle->alloc_state == AllocState::Deferred && bytes > limit_for(tid, bytes) &&
                        (held == nullptr || bytes > held->total_bytes())) {
                        held = handle;
                    }
                    continue;
                }
                std::size_t const bytes = handle->total_bytes();
                largest                 = std::max(largest, bytes);
                if (sliceable && bytes > limit_for(tid, bytes) && !declined.contains(tid) &&
                    std::ranges::find(seeds, std::pair{bytes, tid}) == seeds.end()) {
                    seeds.emplace_back(bytes, tid);
                }
            }
        }
        if (first && held != nullptr) {
            std::size_t const bytes   = held->total_bytes();
            TensorId const    held_id = graph.find_tensor_id_by_ptr(held->tensor_ptr);
            note_skip(ResultHeld, fmt::format("'{}', {} bytes against a cap of {}", held->name, bytes, limit_for(held_id, bytes)));
            _stats.held_results.push_back(
                fmt::format("held '{}' whole, {} bytes against a cap of {}, because it is declared as a result the "
                            "caller may read after execute(); declare it as an intermediate if nothing does",
                            held->name, bytes, limit_for(held_id, bytes)));
            note(ResultHeld, _stats.held_results.back(),
                 Decision{.tensors = {held->name}, .facts = {Fact{.name = "bytes", .value = static_cast<double>(bytes), .unit = "B"}}}, 1);
        }
        if (first) {
            // A running maximum: the manager runs this pass once per graph of the tree, and the
            // getters report over all of them.
            _stats.largest_before = std::max(_stats.largest_before, largest);
            if (largest == 0) {
                // A graph that writes no intermediate, such as the parent of an iterative
                // method's loop, has nothing here to decline.
                return tiled.moved();
            }
            if (derived && peak <= budget && largest <= share) {
                note_skip(FitsBudget,
                          fmt::format("{} bytes at the program's peak against --einsums:max-memory of {}, the largest intermediate {} "
                                      "against a share of {}",
                                      peak, budget, largest, share));
                note(FitsBudget,
                     fmt::format("declined: the program's tensors hold {} bytes at its peak and the memory budget is {}", peak, budget),
                     Decision{.facts = {Fact{.name = "total", .value = static_cast<double>(peak), .unit = "B"},
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
                note_skip(NoSliceableWriter, fmt::format("{} bytes against a cap of {}", largest, limit_for(0, largest)));
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

        bool rewrote = false;
        for (auto const &[bytes, seed] : seeds) {
            Allowance const allowance{.cap       = limit_for(seed, bytes),
                                      .budget    = derived ? budget : 0,
                                      .share     = derived ? share : 0,
                                      .footprint = derived ? &fp : nullptr,
                                      .forced    = forced};
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
        Footprint const fp      = derived ? footprint(graph) : Footprint{};
        auto const      io      = derived ? buffer_io(graph) : std::vector<BufferIo>{};
        bool            rewrote = false;
        for (auto const &[bytes, seed] : symmetric_seeds(graph, scope, declined)) {
            std::size_t limit = static_cast<std::size_t>(std::max<std::int64_t>(cap, 0));
            if (derived) {
                auto const [lo, hi]      = held_span(graph, io, seed);
                std::size_t const beside = fp.beside(lo, hi);
                std::size_t const others = beside - std::min(beside, bytes);
                limit                    = others < budget ? std::min(budget - others, share) : 0;
            }
            Allowance const allowance{.cap           = limit,
                                      .budget        = derived ? budget : 0,
                                      .share         = derived ? share : 0,
                                      .footprint     = derived ? &fp : nullptr,
                                      .triangle_only = true};
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
    Scope const       scope{.graph = graph, .enclosing = _stats.enclosing};

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
        return admits(graph, nodes[i]);
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
                auto const [lo, hi]       = std::ranges::minmax(analysis->members());
                std::size_t const held    = allowance.footprint->beside(lo, hi);
                std::size_t const outside = held - std::min(held, analysis->streamed_whole_bytes(labelling));
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

            std::size_t unit       = 0;
            std::size_t unit_held  = 0;
            long double traffic    = 0.0L;
            long double contracted = 0.0L;
            analysis->cost(labelling, extents, 1, unit, unit_held, traffic, contracted);

            // The statements the loop would run, which do not depend on the blocks, so each candidate
            // can be priced at its own tile shapes before one is chosen.
            Plan probe;
            probe.extents = extents;
            probe.slices  = slices;
            std::string probe_note;
            bool const  priced     = analysis->build(labelling, probe, probe_note);
            bool const  slice_fits = fits(unit, unit_held);
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
                    blocks = choose_blocks(
                        extents, reusable, labelling.fully_sliced, fits,
                        [&](std::vector<std::size_t> const &b, std::size_t &largest, std::size_t &held, long double &streamed,
                            long double &in_contractions) {
                            analysis->range_cost(labelling, extents, b, largest, held, streamed, in_contractions);
                        },
                        [&](std::vector<std::size_t> const &b, long double unpriced) {
                            if (!priced) {
                                return 0.0;
                            }
                            probe.range        = true;
                            probe.blocks       = b;
                            double const price = range_price(graph, probe, unpriced, pricing());
                            report(3, fmt::format("blocks of {} priced at {:.1f} us", fmt::join(b, " x "), price));
                            return price;
                        });
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
                plan.price_us      = blocks->price_us;
            } else {
                std::size_t const depth   = slice_fits ? chunk_depth(slices, joint ? unit_held : unit, room) : 1;
                std::size_t       largest = 0;
                std::size_t       held    = 0;
                analysis->cost(labelling, extents, depth, largest, held, traffic, contracted);
                plan.depth         = depth;
                plan.iterations    = slices / depth;
                plan.largest_after = largest;
                plan.traffic       = traffic;
                if (priced) {
                    probe.depth      = depth;
                    probe.iterations = plan.iterations;
                    plan.price_us    = grouped_price(graph, probe, traffic - contracted, pricing());
                }
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
                        long double inside  = 0.0L;
                        analysis->range_cost(labelling, extents, {b, b}, largest, held, moved, inside);
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
            report(3, fmt::format("candidate {} streams {:.0f} bytes {}, priced at {:.1f} us", fmt::join(plan.axis_names, " x "),
                                  static_cast<double>(plan.traffic),
                                  plan.range ? fmt::format("in blocks of {}", fmt::join(plan.blocks, " x "))
                                             : fmt::format("at depth {}", plan.depth),
                                  plan.price_us));
            if (best_effort) {
                // One whose slices fit the share, by price as any plan; failing that the smallest slice.
                bool const better =
                    !fallback.has_value() || (plan.fits_share && !fallback->fits_share) ||
                    (plan.fits_share == fallback->fits_share &&
                     (plan.fits_share ? preferred(plan, *fallback)
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
    _stats.summed_into.insert(_stats.summed_into.end(), best->summed_names.begin(), best->summed_names.end());

    // The getters describe the first region, which is the one around the largest seed.
    if (_stats.num_tiled == 0) {
        _stats.slice_count   = best->slices;
        _stats.depth         = best->depth;
        _stats.iterations    = best->iterations;
        _stats.largest_after = best->largest_after;
        _stats.axis_names    = best->axis_names;
        _stats.axis_letters  = best->axis_letters;
        _stats.axis_extents.assign(best->extents.begin(), best->extents.end());
        _stats.streamed    = best->streamed;
        _stats.whole       = best->whole;
        _stats.accumulator = best->accumulator;
        _stats.emission    = best->triangle.has_value() ? "triangle" : best->range ? "range" : "grouped";
        _stats.block_extents.assign(best->blocks.begin(), best->blocks.end());
        _stats.loops = best->triangle.has_value() ? triangle_loops(best->extents[0], best->blocks[0]).size() : loop_walks(*best).size();
        _stats.triangle_sign = best->triangle.has_value() ? best->triangle->sigma.sign : 0;
    }
    ++_stats.num_tiled;
    graph.note_structural_change();

    std::string const line =
        fmt::format("slicing {} into {} slice(s), {}, seed {} bytes -> largest intermediate {} bytes{}", fmt::join(best->axis_names, " x "),
                    best->slices, describe_chunks(*best), best->largest_before, best->largest_after,
                    best->best_effort ? " (none fits the budget beside the tensors outside the loop)" : "");
    _stats.regions.push_back(line);
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
