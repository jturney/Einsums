//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file Graph.cpp
/// @brief A graph as a container: its lifetime, its nodes, and its tensors.
///
/// The core of `Graph`, and the part every other `Graph/*.cpp` builds on. What
/// it holds is a node list, a tensor table, and the annotations hung off that
/// table; what it offers is the operations that keep those three consistent with
/// each other.
///
/// @ref Graph::move_members_from is the "don't forget a member" discipline: a
/// new member added to the class has to be handled there, and whether it also
/// travels in a saved file is written down in `GraphIR.cpp`. The two comments
/// point at each other on purpose.
///
/// The rest of the implementation, by subject:
///   - `Graph/Alias.cpp`         which tensors are views onto which
///   - `Graph/AliasGeometry.hpp` whether two handles share memory, and where
///   - `Graph/Schedule.cpp`      dependence edges, levels, threads, order
///   - `Graph/ControlFlow.cpp`   conditionals, loops, setup, accuracy budgets
///   - `Graph/Operations.cpp`    node and executor factories, operand validation
///   - `Graph/Execute.cpp`       replay, and the optimizer entry points
///   - `Graph/GpuDispatch.hpp`   the GPU BLAS fast paths replay tries first
///   - `Graph/Report.cpp`        timings, DOT, summaries, JSON, the registry

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
#include <Einsums/ComputeGraph/StringDispatch.hpp>
#include <Einsums/ComputeGraphTypes/GraphData.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/GPU/BLAS.hpp>
#include <Einsums/LinearAlgebra.hpp>
#include <Einsums/Profile/Profile.hpp>
#include <Einsums/TaskPool/WidthBudget.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TypeSupport/JsonEscape.hpp>

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

EINSUMS_NAMESPACE_BEGIN(compute_graph)

Graph::Graph(std::string name) : _name(std::move(name)) {
}

Graph::~Graph() {
    // Run cleanups in reverse order so later-adopted objects (which may
    // depend on earlier ones) tear down first.
    while (!_adopted_cleanups.empty()) {
        auto fn = std::move(_adopted_cleanups.back());
        _adopted_cleanups.pop_back();
        if (fn)
            fn();
    }
    unregister_graph(this);
}

void Graph::adopt(std::function<void()> deleter) {
    if (deleter)
        _adopted_cleanups.push_back(std::move(deleter));
}

/// @note This function and ``GraphIR.cpp``'s member walk are the two places a
///       newly added Graph member has to be considered, and they ask different
///       questions of it. Here the question is "does the member travel with a
///       move", and the answer is yes for everything that is not a fresh
///       per-object resource (the content mutex). There the question is "is the
///       member STRUCTURE, and therefore part of what a saved file carries and a
///       content hash covers", and the answer is deliberately no for most of it:
///       thread widths, admission priorities, stream ids, timings, estimated
///       flops and bytes, and the planned thread count are all tuning artifacts
///       of one machine, which is the structure/tuning rule made concrete.
///       ``GraphIR.cpp`` states the verdict member by member; keep the two in
///       step when adding one.
void Graph::move_members_from(Graph &&other) noexcept {
    _name             = std::move(other._name);
    _space_registry   = other._space_registry;
    _pipeline_name    = std::move(other._pipeline_name);
    _workspace_name   = std::move(other._workspace_name);
    _stage_name       = std::move(other._stage_name);
    _stage_type       = std::move(other._stage_type);
    _stage_index      = other._stage_index;
    _nodes            = std::move(other._nodes);
    _tensors          = std::move(other._tensors);
    _next_node_id     = other._next_node_id;
    _next_tensor_id   = other._next_tensor_id;
    _sorted           = other._sorted;
    _executed         = other._executed;
    _deps             = std::move(other._deps);
    _owned_tensors    = std::move(other._owned_tensors);
    _adopted_cleanups = std::move(other._adopted_cleanups);
    _params           = std::move(other._params);
    _scope_maps       = std::move(other._scope_maps);
    _bound_operands   = std::move(other._bound_operands);
    _declared_aliases = std::move(other._declared_aliases);
    _interface_names  = std::move(other._interface_names);
    _symbol_spaces    = std::move(other._symbol_spaces);
    _space_extents    = std::move(other._space_extents);
    _space_tiles      = std::move(other._space_tiles);
    _ragged_extents   = std::move(other._ragged_extents);
    _named_gate_flags = std::move(other._named_gate_flags);
    _slot_map         = std::move(other._slot_map);
    // Seven members that used to be dropped by a move. Each is state a moved-to
    // graph genuinely needs, and the omission was latent only because nothing
    // moved a graph and then used it: `load_graph` returns one by value, so
    // every load exercises this path.
    //
    // `_aliases_linked` is the sharpest of them. Its default is TRUE, meaning
    // "the relation is up to date", so a graph moved out of a state that needed
    // relinking arrived claiming it did not - a silently incomplete alias
    // relation, which is the shape of both alias bugs this module has had.
    _ptr_index             = std::move(other._ptr_index);
    _owned_tensor_ptrs     = std::move(other._owned_tensor_ptrs);
    _device_shadows        = std::move(other._device_shadows);
    _executor              = std::move(other._executor);
    _aliases_linked        = other._aliases_linked;
    _slots_validated       = other._slots_validated;
    _timing_samples        = std::move(other._timing_samples);
    _timing_report         = std::move(other._timing_report);
    _timing_report_valid   = other._timing_report_valid;
    _slot_redirects        = std::move(other._slot_redirects);
    _deps_valid            = other._deps_valid;
    _profile_strings       = std::move(other._profile_strings);
    _profile_strings_valid = other._profile_strings_valid;
    _exec_zone_name        = std::move(other._exec_zone_name);
    _exec_zone_id          = other._exec_zone_id;
    _last_optimize_report  = std::move(other._last_optimize_report);
    _analysis_version      = other._analysis_version;
    _structure_version     = other._structure_version;
    _structural_passes     = std::move(other._structural_passes);
    _setup_key             = std::move(other._setup_key);
    _approximations        = std::move(other._approximations);
    _accuracy_budget       = other._accuracy_budget;
    _usage_version         = other._usage_version;
    _usage                 = std::move(other._usage);
    // The widths themselves ride along inside _nodes, so the count they were
    // planned for has to travel with them or the staleness check compares
    // against a zero and lets a foreign plan run.
    _planned_thread_count = other._planned_thread_count;
    _plan_trial           = other._plan_trial;
    _plan_incumbent       = std::move(other._plan_incumbent);
    _plan_candidate       = std::move(other._plan_candidate);
    _plan_candidate_ms    = other._plan_candidate_ms;
}

Graph::Graph(Graph &&other) noexcept {
    move_members_from(std::move(other));
    // Invalidate moved-from so its destructor doesn't unregister
    other._executed = false;
    // Transfer registration from old address to new
    unregister_graph(&other);
    if (_executed) {
        register_graph(this);
    }
}

Graph &Graph::operator=(Graph &&other) noexcept {
    if (this != &other) {
        unregister_graph(this);
        move_members_from(std::move(other));

        // Invalidate moved-from so its destructor doesn't unregister
        other._executed = false;
        unregister_graph(&other);
        if (_executed) {
            register_graph(this);
        }
    }
    return *this;
}

NodeId Graph::add_node(Node node) {
    std::scoped_lock const lock(*_content_mutex);
    node.id         = _next_node_id++;
    NodeId const id = node.id;
    _nodes.push_back(std::move(node));
    _sorted                = false;
    _deps_valid            = false;
    _profile_strings_valid = false;
    _executed              = false;
    _analysis_version++;
    _structure_version++;
    return id;
}

size_t Graph::erase_nodes(std::vector<bool> const &remove) {
    std::scoped_lock const lock(*_content_mutex);
    std::vector<Node>      filtered;
    filtered.reserve(_nodes.size());
    size_t removed = 0;
    for (size_t i = 0; i < _nodes.size(); ++i) {
        if (i < remove.size() && remove[i]) {
            ++removed;
            continue;
        }
        filtered.push_back(std::move(_nodes[i]));
    }
    _nodes = std::move(filtered);
    if (removed != 0) {
        _structure_version++;
    }
    return removed;
}

void Graph::insert_node_groups(std::vector<std::pair<std::size_t, std::vector<Node>>> groups) {
    std::scoped_lock const lock(*_content_mutex);
    // Splice in descending position order so an earlier insertion doesn't shift
    // the indices of later ones (positions are given in the original numbering).
    //
    // Two groups can legitimately share a position: when a caller replaces a run of
    // adjacent nodes, every position between them is erased and they collapse onto
    // the same index. Ties must then splice the LATER group first, so the earlier
    // one lands in front of it. Without the tiebreak the groups come out reversed,
    // which for a producer followed by its consumer means the consumer runs first
    // and reads unwritten storage.
    std::vector<std::size_t> order(groups.size());
    std::iota(order.begin(), order.end(), 0);
    std::ranges::sort(order, [&groups](std::size_t a, std::size_t b) {
        if (groups[a].first != groups[b].first) {
            return groups[a].first > groups[b].first;
        }
        return a > b;
    });
    for (auto idx : order) {
        auto &[at, nodes] = groups[idx];
        if (nodes.empty()) {
            continue;
        }
        _structure_version++;
        _nodes.insert(_nodes.begin() + static_cast<std::ptrdiff_t>(at), std::make_move_iterator(nodes.begin()),
                      std::make_move_iterator(nodes.end()));
    }
    mark_sorted();
}

size_t Graph::replace_nodes(std::vector<bool> const &remove, std::vector<std::pair<std::size_t, std::vector<Node>>> inserts) {
    size_t const removed = erase_nodes(remove);

    // The positions in `inserts` are in the PRE-ERASE numbering, so each one has
    // to come down by the number of nodes erased below it. A prefix-sum table
    // rather than a per-position count, because a pass can record one group per
    // subsumed node and the quadratic version shows up on long chains.
    std::vector<size_t> erased_below(remove.size() + 1, 0);
    for (size_t i = 0; i < remove.size(); ++i) {
        erased_below[i + 1] = erased_below[i] + (remove[i] ? 1 : 0);
    }
    for (auto &[position, group] : inserts) {
        position -= erased_below[std::min(position, remove.size())];
    }

    insert_node_groups(std::move(inserts));
    return removed;
}

TensorId Graph::register_tensor(TensorHandle handle) {
    std::scoped_lock const lock(*_content_mutex);
    TensorId               id = _next_tensor_id++;
    handle.id                 = id;
    // A tensor another scope declared arrives here as a fresh, default handle
    // (make_handle knows nothing about workspaces), so the scope tables the
    // declaring Workspace/Pipeline published are what recover its ownership.
    // An intermediate is graph-owned by construction and is never looked up.
    if (!handle.is_intermediate && !_scope_maps.empty()) {
        handle.ownership = scope_for_ptr(handle.tensor_ptr);
    }
    auto const &stored = _tensors.emplace(id, std::move(handle)).first->second;
    if (stored.tensor_ptr != nullptr) {
        // insert_or_assign, not emplace: an address freed during a capture can
        // be reused by a different tensor, and the index has to name the tensor
        // that lives there NOW. The stale-entry case is caught upstream by the
        // liveness-token check in CaptureContext::get_or_register, which is
        // what routes a recycled address here for re-registration.
        _ptr_index.insert_or_assign(stored.tensor_ptr, id);
    }
    // Deliberately not linked here: containment is resolved in one amortized
    // pass (see link_alias_storage), because doing it per registration is
    // O(n) each and quadratic overall. A DLPNO-MP2 capture registers ~13k
    // tensors and paid 0.3s for it.
    _aliases_linked = false;
    return id;
}

TensorId Graph::find_or_register_tensor_ptr(TensorHandle const &handle) {
    if (handle.tensor_ptr != nullptr) {
        if (TensorId const id = find_tensor_id_by_ptr(handle.tensor_ptr); id != 0) {
            return id;
        }
    }
    return register_tensor(handle);
}

TensorHandle &Graph::tensor(TensorId id) {
    auto it = _tensors.find(id);
    if (it == _tensors.end()) {
        EINSUMS_THROW_EXCEPTION(std::out_of_range, "Graph '{}': no tensor with id {}", _name, id);
    }
    return it->second;
}

TensorHandle *Graph::find_tensor(TensorId id) noexcept {
    auto it = _tensors.find(id);
    return it == _tensors.end() ? nullptr : &it->second;
}

TensorHandle const *Graph::find_tensor(TensorId id) const noexcept {
    auto it = _tensors.find(id);
    return it == _tensors.end() ? nullptr : &it->second;
}

TensorHandle const &Graph::tensor(TensorId id) const {
    auto it = _tensors.find(id);
    if (it == _tensors.end()) {
        EINSUMS_THROW_EXCEPTION(std::out_of_range, "Graph '{}': no tensor with id {}", _name, id);
    }
    return it->second;
}

SpaceRegistry &Graph::space_registry() const noexcept {
    return _space_registry != nullptr ? *_space_registry : global_space_registry();
}

void Graph::set_space_registry(SpaceRegistry &registry) noexcept {
    _space_registry = &registry;
}

void Graph::note_structural_pass(std::string pass_name) {
    // Once per pass, however many times it ran. The list says what shaped this graph; a count
    // of applies would say something about the caller's pipeline instead, and the two get
    // confused the moment anyone applies a manager twice.
    if (std::ranges::find(_structural_passes, pass_name) != _structural_passes.end()) {
        return;
    }
    _structural_passes.push_back(std::move(pass_name));
}

void Graph::annotate_tag(TensorId id, ProvenanceTag tag) {
    auto &handle = tensor(id);

    // Sorted on the way in, so two tags built by setting the same keys in a different order
    // compare equal and a saved graph's bytes do not depend on the order a caller happened to
    // use. Stable, so a caller who set one key twice keeps the LAST value rather than an
    // arbitrary one; the duplicate is then removed, since a tag carrying two values for one key
    // has no meaning and every reader would have to pick.
    std::ranges::stable_sort(tag.attributes, [](auto const &lhs, auto const &rhs) { return lhs.first < rhs.first; });
    auto const duplicates = std::ranges::unique(tag.attributes, [](auto const &lhs, auto const &rhs) { return lhs.first == rhs.first; });
    tag.attributes.erase(duplicates.begin(), duplicates.end());

    handle.tag = std::move(tag);
}

ProvenanceTag const &Graph::tensor_tag(TensorId id) const {
    return tensor(id).tag;
}

void Graph::annotate_spaces(TensorId id, std::vector<SpaceId> spaces) {
    auto &handle = tensor(id);

    if (!spaces.empty() && spaces.size() != handle.rank) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "Graph '{}': annotate_spaces tensor '{}': got {} spaces for a rank-{} tensor", _name,
                                handle.name, spaces.size(), handle.rank);
    }

    // Ids are validated against the registry the graph reads them back through, because a
    // SpaceId is meaningless against any other registry and an id that silently fails to
    // resolve later would surface as a missing annotation rather than as this mistake.
    SpaceRegistry const &registry = space_registry();
    std::size_t const    known    = registry.size();
    for (std::size_t axis = 0; axis < spaces.size(); ++axis) {
        SpaceId const space = spaces[axis];
        if (!space.valid() || space.value() >= known) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                    "Graph '{}': annotate_spaces tensor '{}': axis {} names a space that does not resolve in this "
                                    "graph's registry",
                                    _name, handle.name, axis);
        }
    }

    // A symbol tied to two different spaces is a contradiction, and it is reachable from
    // this side too: annotate the dims first, the spaces second. Checked against a copy of
    // the finished state so a throw leaves the handle exactly as it was.
    {
        TensorHandle probe;
        probe.name        = handle.name;
        probe.dim_symbols = handle.dim_symbols;
        probe.spaces      = spaces;
        record_symbol_space_ties(probe);
    }

    // A declaration is authoritative, so it also clears the inferred flag: whatever capture or the
    // propagation pass guessed for these axes, the user has now said what they are, and nothing
    // downstream may overwrite that.
    handle.spaces          = std::move(spaces);
    handle.spaces_inferred = false;

    // A declaration over a tensor with concrete dims also says how big those spaces are on
    // this problem, which is what lets a later tensor be shaped in spaces instead of numbers.
    learn_space_extents(handle);
}

void Graph::annotate_space_axis(TensorId id, std::size_t axis, SpaceId space) {
    auto &handle = tensor(id);

    if (axis >= handle.rank) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "Graph '{}': annotate_space_axis tensor '{}': axis {} is past its rank of {}", _name,
                                handle.name, axis, handle.rank);
    }
    if (!space.valid() || space.value() >= space_registry().size()) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                "Graph '{}': annotate_space_axis tensor '{}': axis {} names a space that does not resolve in this "
                                "graph's registry",
                                _name, handle.name, axis);
    }

    if (handle.spaces.size() != handle.rank) {
        // Axes nobody has spoken for stay default-constructed, which is what "this axis has no
        // space" looks like everywhere that reads the annotation per axis.
        handle.spaces.resize(handle.rank);
    }
    handle.spaces[axis]    = space;
    handle.spaces_inferred = false;

    {
        TensorHandle probe;
        probe.name        = handle.name;
        probe.dim_symbols = handle.dim_symbols;
        probe.spaces      = handle.spaces;
        record_symbol_space_ties(probe);
    }

    learn_space_extents(handle);
}

std::vector<SpaceId> const &Graph::tensor_spaces(TensorId id) const {
    return tensor(id).spaces;
}

void Graph::for_each_subgraph(std::function<void(Graph &)> const &visitor) {
    for (auto &node : _nodes) {
        for_each_child_graph(node, visitor);
    }
}

void Graph::for_each_subgraph(std::function<void(Graph const &)> const &visitor) const {
    for (auto const &node : _nodes) {
        for_each_child_graph(node, visitor);
    }
}

EINSUMS_NAMESPACE_END(compute_graph)

EINSUMS_NAMESPACE_BEGIN(compute_graph)

SpaceTiling tiles(std::vector<int> sizes) {
    SpaceTiling axis{SpaceId{}};
    axis.tile_sizes = std::move(sizes);
    return axis;
}

TensorId Graph::find_tensor_id_by_ptr(void const *ptr) const noexcept {
    auto const it = _ptr_index.find(ptr);
    return it == _ptr_index.end() ? TensorId{0} : it->second;
}

TensorId Graph::live_tensor_id_by_ptr(void const *ptr, std::weak_ptr<void> const &token) const noexcept {
    TensorId const id = find_tensor_id_by_ptr(ptr);
    if (id == 0) {
        return 0;
    }
    TensorHandle const *handle = find_tensor(id);
    if (handle == nullptr || !detail::same_tensor(handle->caller_token, token)) {
        return 0;
    }
    return id;
}

void *Graph::live_tensor_ptr(TensorId id) const noexcept {
    auto const *handle = find_tensor(id);
    if (handle == nullptr) {
        return nullptr;
    }
    return handle->live_ptr();
}

void Graph::record_node_timing(NodeId id, OpKind kind, double duration_ms, unsigned width) {
    std::scoped_lock const lock(*_content_mutex);
    _timing_samples.push_back({.id = id, .kind = kind, .duration_ms = duration_ms, .width = width});
    _timing_report_valid = false;
}

void Graph::record_node_timing(NodeId id, std::string const & /*label*/, OpKind kind, double duration_ms) {
    record_node_timing(id, kind, duration_ms);
}

void Graph::record_node_timings(std::vector<NodeTimingSample> &&samples) {
    std::scoped_lock const lock(*_content_mutex);
    if (_timing_samples.empty()) {
        _timing_samples = std::move(samples);
    } else {
        _timing_samples.insert(_timing_samples.end(), samples.begin(), samples.end());
    }
    _timing_report_valid = false;
}

void Graph::clear_timing_report() {
    _timing_samples.clear();
    _timing_report.clear();
    _timing_report_valid = true;
}

std::vector<std::pair<std::string, std::shared_ptr<std::vector<std::uint8_t>>>> const &Graph::named_gate_flags() const noexcept {
    return _named_gate_flags;
}

TensorId Graph::resolve_alias(TensorId id) const {
    for (size_t hops = 0; hops <= _tensors.size(); ++hops) {
        auto it = _tensors.find(id);
        if (it == _tensors.end() || it->second.aliases == 0) {
            return id;
        }
        id = it->second.aliases;
    }
    EINSUMS_THROW_EXCEPTION(std::runtime_error,
                            "Graph '{}': alias chain from tensor {} exceeds the tensor count ({}), which means a "
                            "cycle in the alias links; the hazard scan cannot order accesses to it",
                            _name, id, _tensors.size());
}

void Graph::mark_sorted() {
    _sorted   = true;
    _executed = false;
    // The caller vouches for the node ORDER, but node positions changed,
    // so the position-keyed _deps lists must be rebuilt on next demand.
    _deps_valid = false;
    // Passes also rewrite labels/descriptors; refresh cached profiler
    // payloads on next execute.
    _profile_strings_valid = false;
    // ... and slot pointers (arena slices, CSE redirects).
    _slots_validated = false;
    // Position-keyed analyses (UsageAnalysis) are stale too.
    _analysis_version++;
}

double Graph::accuracy_budget_value() const noexcept {
    return _accuracy_budget.has_value() ? _accuracy_budget->second : -1.0;
}

unsigned Graph::planned_thread_count() const {
    return _planned_thread_count;
}

bool Graph::thread_replan_armed() const {
    return _plan_trial != ThreadPlanTrial::None;
}

void Graph::free_tensor(TensorId id, std::string name, size_t size_bytes) {
    AllocDescriptor desc;
    desc.tensor_id   = id;
    desc.size_bytes  = size_bytes;
    desc.tensor_name = std::move(name);

    Node node;
    ProfileMemFree(size_bytes);

    node.kind    = OpKind::Free;
    node.label   = fmt::format("free({})", desc.tensor_name);
    node.execute = []() {}; // No-op: graph still owns the memory
    node.inputs  = {id};
    node.op_data = std::move(desc);

    add_node(std::move(node));
}

TensorSlot *Graph::find_slot(TensorId id) {
    auto it = _slot_map.find(id);
    return it != _slot_map.end() ? it->second.get() : nullptr;
}

void Graph::redirect_slot(TensorId from, TensorId to) {
    // Collapse chains so every recorded redirect points at a terminal id.
    for (auto it = _slot_redirects.find(to); it != _slot_redirects.end(); it = _slot_redirects.find(to)) {
        to = it->second;
    }
    if (from == to) {
        return;
    }
    if (TensorHandle const *fh = find_tensor(from), *th = find_tensor(to);
        fh != nullptr && th != nullptr && fh->dtype != packed_gemm::ScalarType::Unknown && th->dtype != packed_gemm::ScalarType::Unknown &&
        fh->dtype != th->dtype) {
        EINSUMS_THROW_EXCEPTION(
            std::logic_error, "Graph '{}': cannot redirect tensor {} to tensor {}, which holds a different element type", _name, from, to);
    }
    TensorSlot const *to_slot   = find_slot(to);
    TensorSlot       *from_slot = find_slot(from);
    if (to_slot == nullptr || from_slot == nullptr) {
        return;
    }
    // The geometry accessor travels with the pointer. @p from's own
    // accessor was baked for @p from's static type, and the object behind
    // the redirect is @p to's, so keeping the old one would decode a
    // different type's layout.
    from_slot->ptr        = to_slot->ptr;
    from_slot->impl_of    = to_slot->impl_of;
    from_slot->resync_of  = to_slot->resync_of;
    _slot_redirects[from] = to;
    _slots_validated      = false;
    // Anything already redirected to `from` now follows the same terminal.
    for (auto &[f, t] : _slot_redirects) {
        if (t == from) {
            t = to;
            if (auto *fs = find_slot(f)) {
                fs->ptr       = to_slot->ptr;
                fs->impl_of   = to_slot->impl_of;
                fs->resync_of = to_slot->resync_of;
            }
        }
    }
}

void Graph::bind_commit() {
    // The slots are taken off the graph BEFORE the transaction runs, which is what
    // clears the pending list whatever happens: a refused transaction must not leak
    // into the next one.
    run_bind(std::exchange(_pending_binds, {}));
}

void Graph::rederive_intermediate_extents() {
    rederive_owned_extents();
    validate_node_extents();
}

void Graph::resize_intermediate(TensorId id, std::vector<std::size_t> const &dims, std::string_view producer) {
    resize_derived_extent(id, dims, producer);
}

void Graph::clear_bindings() noexcept {
    _bound_operands.clear();
    _ragged_extents.clear();
}

bool Graph::BoundSpan::overlaps(BoundSpan const &other) const noexcept {
    return lo != nullptr && other.lo != nullptr && lo < other.hi && other.lo < hi;
}

std::size_t Graph::BoundSpan::overlap_bytes(BoundSpan const &other) const noexcept {
    if (!overlaps(other)) {
        return 0;
    }
    return static_cast<std::size_t>(std::min(hi, other.hi) - std::max(lo, other.lo));
}

void Graph::add_alloc_node(TensorId id, std::string const &name, size_t size_bytes) {
    AllocDescriptor desc;
    desc.tensor_id   = id;
    desc.size_bytes  = size_bytes;
    desc.tensor_name = name;

    Node node;
    ProfileMemAlloc(desc.size_bytes);

    node.kind    = OpKind::Alloc;
    node.label   = fmt::format("alloc({})", name);
    node.execute = []() {};
    node.outputs = {id};
    node.op_data = std::move(desc);

    add_node(std::move(node));
}

void Graph::run_bind(std::vector<PendingBind> const &pending) {
    InterfaceManifest const contract = manifest();

    DimSolution solution;
    for (auto const &slot : pending) {
        slot.collect(contract, solution);
    }
    prepare_bind_solution(solution);
    for (auto const &slot : pending) {
        slot.apply(contract, solution);
    }
    finish_bind_solution(solution);
}

EINSUMS_NAMESPACE_END(compute_graph)

EINSUMS_NAMESPACE_BEGIN(compute_graph)
namespace detail {

std::vector<std::pair<std::string, SpaceId>> bind_einsum_spaces(Graph &graph, TensorId a_id, TensorId b_id, TensorId c_id,
                                                                std::vector<std::string> const &a_indices,
                                                                std::vector<std::string> const &b_indices,
                                                                std::vector<std::string> const &c_indices, std::string_view context) {
    TensorHandle const *a = graph.find_tensor(a_id);
    TensorHandle const *b = graph.find_tensor(b_id);
    TensorHandle const *c = graph.find_tensor(c_id);

    std::array<LetterSpaceOperand, 3> const operands{
        LetterSpaceOperand{.label = "A", .indices = &a_indices, .spaces = a != nullptr ? &a->spaces : nullptr},
        LetterSpaceOperand{.label = "B", .indices = &b_indices, .spaces = b != nullptr ? &b->spaces : nullptr},
        LetterSpaceOperand{.label = "C", .indices = &c_indices, .spaces = c != nullptr ? &c->spaces : nullptr},
    };

    auto letters = build_letter_spaces(std::span<LetterSpaceOperand const>{operands}, &graph.space_registry(), context);

    if (auto *output = graph.find_tensor(c_id); output != nullptr && output->is_intermediate && output->spaces.empty()) {
        auto inferred = spaces_from_letters(c_indices, letters);
        if (inferred.size() == output->rank) {
            output->spaces          = std::move(inferred);
            output->spaces_inferred = true;
        }
    }

    return letters;
}

} // namespace detail
EINSUMS_NAMESPACE_END(compute_graph)

EINSUMS_NAMESPACE_BEGIN(compute_graph)

#define EINSUMS_INSTANTIATE_GRAPH_TENSOR_MEMBERS(...) EINSUMS_GRAPH_TENSOR_MEMBERS(template EINSUMS_EXPORT, __VA_ARGS__)
EINSUMS_CG_COMMON_TENSOR_TYPES(EINSUMS_INSTANTIATE_GRAPH_TENSOR_MEMBERS)
#undef EINSUMS_INSTANTIATE_GRAPH_TENSOR_MEMBERS

EINSUMS_NAMESPACE_END(compute_graph)

EINSUMS_NAMESPACE_BEGIN(compute_graph)

#define EINSUMS_INSTANTIATE_GRAPH_ELEMENT_MEMBERS(T) EINSUMS_GRAPH_ELEMENT_MEMBERS(template EINSUMS_EXPORT, T)
EINSUMS_CG_ELEMENT_TYPES(EINSUMS_INSTANTIATE_GRAPH_ELEMENT_MEMBERS)
#undef EINSUMS_INSTANTIATE_GRAPH_ELEMENT_MEMBERS

EINSUMS_NAMESPACE_END(compute_graph)
