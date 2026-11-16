//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file NodeRunner.cpp
/// @brief Running one node of a replay: see NodeRunner.hpp.
///
/// Per node, the GPU is offered the work first through `GpuDispatch.hpp` and the
/// CPU closure runs whenever it declines.

#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/NodeRunner.hpp>
#include <Einsums/ComputeGraph/Options.hpp>
#include <Einsums/ComputeGraph/PlacementPolicy.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/GPU/Runtime.hpp>
#include <Einsums/Logging.hpp>
#include <Einsums/Profile/Profile.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <exception>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../BodyReplay.hpp"
#include "../Passes/NodeCost.hpp"
#include "GpuDispatch.hpp"

EINSUMS_NAMESPACE_BEGIN(compute_graph)

/// What of Graph the runner reaches past its public surface: the replay state it records into,
/// the no-bump residency write a CPU fallback makes from a worker thread, and the timings a
/// replay hands back as it ends.
struct detail::RunnerAccess {
    static ReplayState &replay(Graph &graph) noexcept { return graph._replay; }

    static void mark_host_resident(Graph &graph, TensorId id) noexcept { graph._registry.mark_host_resident(id); }

    static void record_node_timings(Graph &graph, std::vector<NodeTimingSample> &&samples) {
        graph.record_node_timings(std::move(samples));
    }
};

using detail::RunnerAccess;
using namespace gpu_dispatch;

namespace {

// The call site every node zone reports. The NAMES are per graph and per node and come
// pre-interned from the graph's profile strings; a site supplies the file, function and line, the
// same every time. An accessor, not a function-local static in run(): a static there is built, and
// the profiler with it, even by a replay that records nothing.
waggle::ZoneSite const &graph_node_site() {
    static waggle::ZoneSite const site{"ComputeGraph::node", __FILE__, __LINE__, "execute", WAGGLE_CURRENT_DOMAIN};
    return site;
}

/// Copy @p bytes of @p handle's live host data up into @p shadow.
/// @return Whether a copy happened: a missing shadow (allocation failed) or host buffer (not
///         materialized yet) leaves nothing to copy, and the device copy stays as it was.
bool upload(TensorHandle const *handle, void *shadow, std::size_t bytes, detail::TransferTally &tally) {
    void *host = handle != nullptr ? live_host_ptr(*handle) : nullptr;
    if (shadow == nullptr || host == nullptr) {
        return false;
    }
    gpu::memcpy_host_to_device(shadow, host, bytes);
    tally.uploads.fetch_add(1, std::memory_order_relaxed);
    tally.upload_bytes.fetch_add(bytes, std::memory_order_relaxed);
    return true;
}

/// Copy @p bytes of @p shadow down into @p handle's live host data.
/// @return Whether a copy happened; see upload.
bool download(TensorHandle const *handle, void const *shadow, std::size_t bytes, detail::TransferTally &tally) {
    void *host = handle != nullptr ? live_host_ptr(*handle) : nullptr;
    if (shadow == nullptr || host == nullptr) {
        return false;
    }
    gpu::memcpy_device_to_host(host, shadow, bytes);
    tally.downloads.fetch_add(1, std::memory_order_relaxed);
    tally.download_bytes.fetch_add(bytes, std::memory_order_relaxed);
    return true;
}

/// The executor running the node in progress on this thread, or null outside any node.
thread_local Executor *t_enclosing_executor = nullptr;

/// The placement policy of the node in progress on this thread, or null outside any node.
thread_local PlacementPolicy const *t_enclosing_placement = nullptr;

/// How a diagnostic names a place.
std::string place_name(Place place) {
    return place.is_host() ? std::string{"host"} : "device " + std::to_string(place.ordinal());
}

/// Whether @p node runs a transfer, which is a coherence request and not work to place.
bool is_transfer(Node const &node) noexcept {
    return node.kind == OpKind::HostToDevice || node.kind == OpKind::DeviceToHost;
}

/// The policy a replay of @p graph asks: the graph's own, else the replay enclosing it, else the
/// option's.
PlacementPolicy const &resolve_policy(Graph const &graph) {
    if (auto const &installed = graph.placement_policy(); installed != nullptr) {
        return *installed;
    }
    if (t_enclosing_placement != nullptr) {
        return *t_enclosing_placement;
    }
    return builtin_placement_policy(config::get(option::GraphPlacement));
}

} // namespace

detail::EnclosingPlacement::EnclosingPlacement(PlacementPolicy const &policy) noexcept : _previous(t_enclosing_placement) {
    t_enclosing_placement = &policy;
}

detail::EnclosingPlacement::~EnclosingPlacement() {
    t_enclosing_placement = _previous;
}

detail::EnclosingExecutor::EnclosingExecutor(Executor &executor) noexcept : _previous(t_enclosing_executor) {
    t_enclosing_executor = &executor;
}

detail::EnclosingExecutor::~EnclosingExecutor() {
    t_enclosing_executor = _previous;
}

void detail::replay_body(Graph &body) {
    // The body's own installed executor, or with none the built-in walk, is what execute() runs.
    if (body.executor() == nullptr && t_enclosing_executor != nullptr) {
        body.execute(*t_enclosing_executor);
    } else {
        body.execute();
    }
}

NodeRunner::NodeRunner(Graph &graph, Executor &executor, bool recording)
    : _graph(graph), _executor(executor), _recording(recording), _policy(&resolve_policy(graph)) {
    // assign() reuses the slots' storage, so a replay of an unchanged graph allocates nothing.
    RunnerAccess::replay(_graph)._run_slots.assign(_graph.nodes().size(), detail::NodeRunSlot{});
    RunnerAccess::replay(_graph)._transfers.reset();
    _asks = !_policy->follows_plan();
    if (_asks) {
        prepare_choices();
    }
    if constexpr (!gpu::has_unified_memory) {
        for (std::size_t i = 0; i < _graph.nodes().size(); ++i) {
            Node const &node = _graph.nodes()[i];
            if (node.target == Target::GPU || is_transfer(node) || (_asks && _choices[i].offered && _choices[i].eligible.any_device())) {
                prepare_device_work();
                break;
            }
        }
    }
    if (_asks && _policy->cost_model() != nullptr) {
        price_choices();
    }
}

void NodeRunner::prepare_choices() {
    // A view and its parent are separate tensor ids, so each would get its own shadow and its own
    // coherence entry, and a device write through one would not reach the other. A node touching
    // either is therefore not offered: it runs at its plan, exactly as it did before a policy
    // could ask.
    std::unordered_set<TensorId> entangled;
    for (auto const &[id, handle] : _graph.tensors_map()) {
        TensorId const buffer = _graph.buffer_of(id);
        if (buffer != id || handle.is_tensor_view) {
            entangled.insert(id);
            entangled.insert(buffer);
        }
    }

    auto const &nodes = _graph.nodes();
    _choices.assign(nodes.size(), NodeChoice{});
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        Node const &node   = nodes[i];
        NodeChoice &choice = _choices[i];

        choice.eligible = node.eligible;
        if (node.target == Target::GPU) {
            // The plan is a pass's statement that the device can run it.
            choice.eligible.add_device(Place::of(node.target).ordinal());
        }

        bool offered = !is_transfer(node);
        for (auto const *ids : {&node.inputs, &node.outputs}) {
            for (TensorId const id : *ids) {
                if (entangled.contains(id)) {
                    offered = false;
                }
            }
        }
        if (!offered && !is_transfer(node) && node.eligible.any_device()) {
            EINSUMS_LOG_DEBUG(
                "Graph::execute: node {} ({}) not offered to placement policy '{}': it touches a view or a tensor a view aliases", node.id,
                node.label, _policy->name());
        }
        choice.offered = offered;
    }
}

void NodeRunner::price_choices() {
    CostModel const &model = *_policy->cost_model();
    auto const      &nodes = _graph.nodes();

    // The nodes a run can hold: offered, and with a device place left after storage was settled.
    std::vector<std::size_t> members;
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        if (_choices[i].offered && _choices[i].eligible.any_device()) {
            members.push_back(i);
        }
    }
    if (members.empty()) {
        return;
    }

    detail::ContractionPriceMemo host_memo;
    detail::ContractionPriceMemo device_memo;
    for (std::size_t const i : members) {
        _choices[i].host_us = detail::model_cost(_graph, nodes[i], model.cpu, &host_memo).t1_us;
        if (model.has_gpu()) {
            _choices[i].device_us = detail::model_cost(_graph, nodes[i], model.gpu, &device_memo).t1_us;
        }
    }

    // Runs: members joined through the buffers one writes and another reads. Union-find over the
    // position in `members`.
    std::vector<std::size_t> parent(members.size());
    for (std::size_t m = 0; m < parent.size(); ++m) {
        parent[m] = m;
    }
    auto const find = [&](std::size_t m) {
        while (parent[m] != m) {
            parent[m] = parent[parent[m]];
            m         = parent[m];
        }
        return m;
    };
    struct Access {
        std::vector<std::size_t> writers; // positions in `members`
        std::vector<std::size_t> readers;
    };
    std::unordered_map<TensorId, Access> accessed;
    for (std::size_t m = 0; m < members.size(); ++m) {
        Node const &node = nodes[members[m]];
        for (TensorId const id : node.inputs) {
            accessed[_graph.buffer_of(id)].readers.push_back(m);
        }
        for (TensorId const id : node.outputs) {
            accessed[_graph.buffer_of(id)].writers.push_back(m);
        }
    }
    for (auto const &[buffer, access] : accessed) {
        if (access.writers.empty()) {
            continue;
        }
        std::size_t const root = find(access.writers.front());
        for (auto const *group : {&access.writers, &access.readers}) {
            for (std::size_t const m : *group) {
                parent[find(m)] = find(root);
            }
        }
    }

    // Which run (root position) each node belongs to, or none; and who reads each buffer.
    constexpr std::size_t    kNone = static_cast<std::size_t>(-1);
    std::vector<std::size_t> run_of(nodes.size(), kNone);
    for (std::size_t m = 0; m < members.size(); ++m) {
        run_of[members[m]] = find(m);
    }
    std::unordered_map<TensorId, std::vector<std::size_t>> read_by; // buffer to reader node positions
    for (std::size_t j = 0; j < nodes.size(); ++j) {
        if (auto const io = _control_io.find(j); io != _control_io.end()) {
            for (TensorId const id : io->second.first) {
                read_by[_graph.buffer_of(id)].push_back(j);
            }
        } else if (is_control_flow(nodes[j].kind)) {
            for (TensorId const id : _graph.effective_io(nodes[j]).first) {
                read_by[_graph.buffer_of(id)].push_back(j);
            }
        } else {
            for (TensorId const id : nodes[j].inputs) {
                read_by[_graph.buffer_of(id)].push_back(j);
            }
        }
    }

    struct Run {
        double      gain{0.0};
        bool        priced{false};
        std::size_t bytes{0};
    };
    std::unordered_map<std::size_t, Run> runs;
    auto const                           bytes_of = [&](TensorId buffer) {
        TensorHandle const *handle = _graph.find_tensor(buffer);
        return handle != nullptr ? handle->total_bytes() : std::size_t{0};
    };
    for (std::size_t m = 0; m < members.size(); ++m) {
        NodeChoice const &choice = _choices[members[m]];
        Run              &run    = runs[find(m)];
        if (choice.host_us && choice.device_us) {
            run.gain += *choice.host_us - *choice.device_us;
            run.priced = true;
        }
    }
    for (auto const &[buffer, access] : accessed) {
        // Read by a member, produced by none of the run's members: brought in. A buffer's readers
        // and writers share one run when it has a writer, so the run is the first reader's or writer's.
        if (access.writers.empty()) {
            std::unordered_set<std::size_t> seen;
            for (std::size_t const m : access.readers) {
                std::size_t const root = find(m);
                if (seen.insert(root).second) {
                    Run &run = runs[root];
                    run.bytes += bytes_of(buffer);
                }
            }
            continue;
        }
        // Written by a member: sent out when a node beyond the run reads it or it is a result.
        std::size_t const   root   = find(access.writers.front());
        Run                &run    = runs[root];
        TensorHandle const *handle = _graph.find_tensor(buffer);
        bool                leaves = handle == nullptr || !handle->is_intermediate;
        if (!leaves) {
            if (auto const it = read_by.find(buffer); it != read_by.end()) {
                leaves = std::ranges::any_of(it->second, [&](std::size_t j) { return run_of[j] != root; });
            }
        }
        if (leaves) {
            run.bytes += bytes_of(buffer);
        }
    }

    for (std::size_t m = 0; m < members.size(); ++m) {
        Run const  &run    = runs[find(m)];
        NodeChoice &choice = _choices[members[m]];
        if (run.priced) {
            choice.run_gain_us = run.gain;
        }
        choice.run_boundary_bytes = run.bytes;
    }
}

Place NodeRunner::choose_place(std::size_t index) const {
    Node const &node    = _graph.nodes()[index];
    Place const planned = Place::of(node.target);
    if (!_asks || !_choices[index].offered) {
        return planned;
    }
    NodeChoice const &choice = _choices[index];

    std::vector<PlacementOperand> operands;
    operands.reserve(node.inputs.size() + node.outputs.size());
    auto const describe = [&](TensorId id) {
        TensorHandle const *handle = _graph.find_tensor(id);
        Validity            valid  = Validity::Host;
        if constexpr (!gpu::has_unified_memory) {
            switch (_coherence.state(id)) {
            case detail::DeviceCoherence::State::HostOnly:
                break;
            case detail::DeviceCoherence::State::Both:
                valid = Validity::Both;
                break;
            case detail::DeviceCoherence::State::DeviceOnly:
                valid = Validity::Device;
                break;
            }
        }
        operands.push_back({.id = id, .bytes = handle != nullptr ? handle->total_bytes() : 0, .valid = valid});
    };
    for (TensorId const id : node.inputs) {
        describe(id);
    }
    for (TensorId const id : node.outputs) {
        describe(id);
    }
    std::span<PlacementOperand const> const all(operands);

    PlacementQuery const query{.node               = node,
                               .planned            = planned,
                               .eligible           = choice.eligible,
                               .inputs             = all.first(node.inputs.size()),
                               .outputs            = all.subspan(node.inputs.size()),
                               .host_us            = choice.host_us,
                               .device_us          = choice.device_us,
                               .run_gain_us        = choice.run_gain_us,
                               .run_boundary_bytes = choice.run_boundary_bytes};

    Place const chosen = _policy->choose(query);
    if (!choice.eligible.contains(chosen) && chosen != planned) {
        EINSUMS_THROW_EXCEPTION(std::logic_error, "placement policy '{}' chose {} for node {} ({}), which cannot run there",
                                _policy->name(), place_name(chosen), node.id, node.label);
    }
    return chosen;
}

void NodeRunner::prepare_device_work() {
    _device_work = true;

    // Every shadow is allocated here, before any node runs, so the shadow map is only read while
    // nodes run, from whichever threads run them.
    std::vector<TensorId> tracked;
    auto const            track = [&](TensorId id, std::size_t bytes) {
        tracked.push_back(id);
        RunnerAccess::replay(_graph)._device_shadows.ensure(id, bytes);
    };
    for (std::size_t i = 0; i < _graph.nodes().size(); ++i) {
        Node const &node = _graph.nodes()[i];
        if (node.kind == OpKind::HostToDevice || node.kind == OpKind::DeviceToHost) {
            if (auto const *tdesc = node.op_data.get_if<TransferDescriptor>()) {
                track(tdesc->tensor_id, tdesc->size_bytes);
            }
        } else if (node.target == Target::GPU || (_asks && _choices[i].offered && _choices[i].eligible.any_device())) {
            for (auto const *ids : {&node.inputs, &node.outputs}) {
                for (TensorId const id : *ids) {
                    if (auto const *handle = _graph.find_tensor(id)) {
                        track(id, handle->total_bytes());
                    }
                }
            }
        }
    }
    _coherence.reset(tracked);

    // A node whose operands could not all be given device storage is host only for this replay,
    // which is what a planned device node already falls back to when it finds a shadow missing.
    if (_asks) {
        for (std::size_t i = 0; i < _graph.nodes().size(); ++i) {
            Node const &node = _graph.nodes()[i];
            if (!_choices[i].offered || !_choices[i].eligible.any_device()) {
                continue;
            }
            for (auto const *ids : {&node.inputs, &node.outputs}) {
                for (TensorId const id : *ids) {
                    if (RunnerAccess::replay(_graph)._device_shadows.get(id) == nullptr) {
                        _choices[i].eligible.clear_devices();
                    }
                }
            }
        }
    }

    // A body replays through its own runner and its own shadows, so what it touches must be
    // current on the host before it runs. effective_io can register a handle, which is why it is
    // asked here, before the executor starts, and never while nodes run.
    for (std::size_t i = 0; i < _graph.nodes().size(); ++i) {
        if (is_control_flow(_graph.nodes()[i].kind)) {
            _control_io.emplace(i, _graph.effective_io(_graph.nodes()[i]));
        }
    }
}

void NodeRunner::host_current(TensorId id) {
    TensorHandle const *handle = _graph.find_tensor(id);
    _coherence.need_on_host(id, [&]() {
        return download(handle, RunnerAccess::replay(_graph)._device_shadows.get(id), handle != nullptr ? handle->total_bytes() : 0,
                        RunnerAccess::replay(_graph)._transfers);
    });
}

void NodeRunner::open_zone(std::optional<waggle::ScopedZone> &zone, std::size_t idx, Place place) const {
    if (!_recording) {
        return;
    }
    Node const &node = _graph.nodes()[idx];

    // A node whose cache entry does not match falls back to its bare label
    // (defensive: an undeclared mutation after the profile strings were built).
    static detail::ReplayState::NodeProfileStrings const kEmptyEntry{};
    detail::ReplayState::NodeProfileStrings const       &ps = (idx < RunnerAccess::replay(_graph)._profile_strings.size() &&
                                                               RunnerAccess::replay(_graph)._profile_strings[idx].node_id == node.id)
                                                                  ? RunnerAccess::replay(_graph)._profile_strings[idx]
                                                                  : kEmptyEntry;

    if (ps.zone_id != 0) {
        zone.emplace(graph_node_site(), ps.zone_id);
    } else {
        zone.emplace(node.label);
    }

    // These two are static strings, interned once per process.
    waggle::annotate("op_kind", op_kind_name(node.kind));
    waggle::annotate("device", place.is_device() ? "GPU" : "CPU");

    for (auto const &[key, value] : ps.texts) {
        waggle::annotate_interned(key, value);
    }
    for (auto const &[key, value] : ps.numbers) {
        waggle::annotate_interned(key, value);
    }
    for (auto const &[key, value] : ps.reals) {
        waggle::annotate_interned(key, value);
    }
}

bool NodeRunner::is_async(std::size_t idx) const {
    Node const &node = _graph.nodes()[idx];
    return node.async_start && node.async_finish;
}

void NodeRunner::start_async(std::size_t idx) {
    Node const                       &node = _graph.nodes()[idx];
    std::optional<waggle::ScopedZone> zone;
    open_zone(zone, idx, Place::of(node.target));
    // The two halves are separate calls, possibly on two threads: the observer hears the node
    // begin here and end where finish_async completes.
    detail::notify_node(*RunnerAccess::replay(_graph)._run_context, EventKind::NodeStart, _graph, node, 0.0);
    if (_device_work) {
        for (auto const *ids : {&node.inputs, &node.outputs}) {
            for (TensorId const id : *ids) {
                host_current(id);
            }
        }
    }
    node.async_start();
}

void NodeRunner::finish_async(std::size_t idx) {
    Node const                       &node = _graph.nodes()[idx];
    std::optional<waggle::ScopedZone> zone;
    open_zone(zone, idx, Place::of(node.target));
    auto const t_start = std::chrono::steady_clock::now();
    node.async_finish();
    if (_device_work) {
        for (TensorId const id : node.outputs) {
            _coherence.wrote_on_host(id);
        }
    }
    double const ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_start).count();
    RunnerAccess::replay(_graph)._run_slots[idx].ms = ms;
    detail::notify_node(*RunnerAccess::replay(_graph)._run_context, EventKind::NodeEnd, _graph, node, ms);
}

void NodeRunner::run(std::size_t idx, unsigned width) {
    Node const &node = _graph.nodes()[idx];
    // A body this node replays runs under this replay's executor unless it has its own.
    detail::EnclosingExecutor const  enclosing(_executor);
    detail::EnclosingPlacement const enclosing_policy(*_policy);

    // Where the node runs, which is the plan unless the policy is asked and says otherwise.
    Place const place     = choose_place(idx);
    bool const  on_device = place.is_device();

    std::optional<waggle::ScopedZone> node_zone;
    open_zone(node_zone, idx, place);

    // An attached observer hears the node begin here and end however this call ends.
    detail::NodeObservation const observed(*RunnerAccess::replay(_graph)._run_context, _graph, node);
    auto                          t_start = std::chrono::steady_clock::now();

    bool device_kernel = false; // whether a device kernel ran the node, for Graph::ran_on_device

    if (node.kind == OpKind::HostToDevice) {
        // H2D transfer. On unified memory there is nothing to copy: the GPU reads host memory.
        if constexpr (!gpu::has_unified_memory) {
            if (auto const *tdesc = node.op_data.get_if<TransferDescriptor>(); tdesc != nullptr && _device_work) {
                // A tensor whose storage has not been materialized yet has no live host buffer -
                // graphs built inside a Setup closure reach their transfer nodes before the
                // allocation happens - and then there is nothing to copy: the device buffer is
                // exactly as defined as the host one.
                TensorId const      id     = tdesc->tensor_id;
                TensorHandle const *handle = _graph.find_tensor(id);
                _coherence.need_on_device(id, [&]() {
                    return upload(handle, RunnerAccess::replay(_graph)._device_shadows.get(id), tdesc->size_bytes,
                                  RunnerAccess::replay(_graph)._transfers);
                });
            }
        }
    } else if (node.kind == OpKind::DeviceToHost) {
        // D2H transfer. On unified memory the result is already in host-accessible memory.
        if constexpr (!gpu::has_unified_memory) {
            if (auto const *tdesc = node.op_data.get_if<TransferDescriptor>(); tdesc != nullptr && _device_work) {
                // Copies only when the device copy is the current one. A DeviceToHost node is
                // placed by TransferInsertion on the assumption that the GPU node before it ran
                // on the device; when that node instead fell back to the CPU, the host result is
                // the live one and the shadow still holds whatever was uploaded before the op.
                // Copying it back then overwrites the correct answer with stale data - typically
                // the zeros an output tensor was initialized to, which is what made whole graphs
                // come out zero.
                TensorId const      id     = tdesc->tensor_id;
                TensorHandle const *handle = _graph.find_tensor(id);
                _coherence.need_on_host(id, [&]() {
                    return download(handle, RunnerAccess::replay(_graph)._device_shadows.get(id), tdesc->size_bytes,
                                    RunnerAccess::replay(_graph)._transfers);
                });
            }
        }
    } else if (on_device) {
        // Device execution.
        // Every operand must be transferable before anything is placed on
        // the device. A tensor with no live host storage can neither be
        // uploaded nor receive a result, so a node touching one has to run
        // on the host instead.
        //
        // Checking up front matters: skipping just the upload would still
        // leave a shadow allocated, and resolve_device_ptr hands that
        // non-null buffer to the dispatcher, which then computes on
        // uninitialized device memory.
        bool operands_ready = true;
        if constexpr (!gpu::has_unified_memory) {
            for (auto const *ids : {&node.inputs, &node.outputs}) {
                for (TensorId const id : *ids) {
                    TensorHandle const *handle = _graph.find_tensor(id);
                    if (handle == nullptr ||
                        (_coherence.state(id) == detail::DeviceCoherence::State::HostOnly && live_host_ptr(*handle) == nullptr)) {
                        operands_ready = false;
                    }
                }
            }
            if (!operands_ready) {
                EINSUMS_LOG_DEBUG("Graph::execute: node {} ({}) kept on the host, an operand has no host storage to transfer", node.id,
                                  node.label);
            }

            if (operands_ready) {
                // Make every operand current on the device. Outputs too: an accumulating op
                // (nonzero C prefactor) reads its output operand before writing it.
                //
                // TransferInsertion normally emits an explicit HostToDevice ahead of this node,
                // and the operand is then current already. Uploading here as well keeps the
                // replay correct whichever passes ran: a GPU node reached without a preceding
                // H2D (a hand-set Target::GPU, or an H2D TransferElimination dropped) would
                // otherwise compute on uninitialized device memory.
                for (auto const *ids : {&node.inputs, &node.outputs}) {
                    for (TensorId const id : *ids) {
                        TensorHandle const *handle = _graph.find_tensor(id);
                        void               *shadow = RunnerAccess::replay(_graph)._device_shadows.get(id);
                        if (!_coherence.need_on_device(id, [&]() {
                                return upload(handle, shadow, handle->total_bytes(), RunnerAccess::replay(_graph)._transfers);
                            })) {
                            // No device buffer to compute in: the node runs on the host below.
                            operands_ready = false;
                        }
                    }
                }
            }
        }
        // The dispatcher reads every operand through its device shadow (resolve_device_ptr), never
        // through the tensor object, so the tensors keep their host data the whole time and a node
        // running beside this one on another thread sees nothing change.

        // Execute via GPU BLAS dispatch if possible, otherwise CPU fallback.
        bool gpu_dispatched = false;
        try {
            gpu_dispatched =
                operands_ready && try_gpu_blas_dispatch(node, _graph.tensors_map(), RunnerAccess::replay(_graph)._device_shadows);
            if (gpu_dispatched) {
                waggle::annotate("gpu_dispatch", "gemm");
            }
        } catch (std::exception const &e) {
            EINSUMS_LOG_WARN("GPU GEMM dispatch failed for node {} ({}): {}", node.id, node.label, e.what());
        }

        device_kernel = gpu_dispatched;
        if (gpu_dispatched) {
            if constexpr (!gpu::has_unified_memory) {
                for (TensorId const id : node.outputs) {
                    _coherence.wrote_on_device(id);
                }
            }
        } else {
            waggle::annotate("gpu_dispatch", "cpu_fallback");

            // Every operand must be current on the host, because a preceding
            // GPU node may have produced it on the device.
            if constexpr (!gpu::has_unified_memory) {
                for (auto const *ids : {&node.inputs, &node.outputs}) {
                    for (TensorId const id : *ids) {
                        host_current(id);
                    }
                }
            }

            // node.cpu_fallback and node.execute are the same callable -
            // GPUPlacement assigns cpu_fallback = execute - so the old
            // try/catch-then-retry structure re-ran an identical call and
            // could only fail twice. One call, one place to report.
            if (node.execute) {
                node.execute();
            }
            for (TensorId const id : node.outputs) {
                RunnerAccess::mark_host_resident(_graph, id);
                if constexpr (!gpu::has_unified_memory) {
                    _coherence.wrote_on_host(id);
                }
            }
        }
    } else {
        // CPU node. Whatever it reads or writes must be current on the host first, and whatever it
        // writes is then current there alone.
        //
        // TransferInsertion normally emits a DeviceToHost for this, but the replay cannot depend on
        // that: an accumulating einsum reads its output operand, and when the producing GPU node
        // ran on the device while the consumer fell back to the host, the host copy is stale.
        if (_device_work) {
            if (auto const io = _control_io.find(idx); io != _control_io.end()) {
                // A loop, conditional or setup node replays a body through the body's own runner and
                // shadows, so the parent's copies of what the body touches are brought to the host
                // first, and what it writes is current only there afterwards. Only those buffers:
                // another node may be writing anything else on the device at this moment.
                for (TensorId const id : io->second.first) {
                    host_current(id);
                }
                for (TensorId const id : io->second.second) {
                    host_current(id);
                }
            } else {
                for (auto const *ids : {&node.inputs, &node.outputs}) {
                    for (TensorId const id : *ids) {
                        host_current(id);
                    }
                }
            }
        }
        if (node.execute) {
            node.execute();
        } else if (node.async_start && node.async_finish) {
            // Async node (e.g., iallreduce): run both phases synchronously.
            // True overlap only happens with DataflowExecutor.
            node.async_start();
            node.async_finish();
        } else {
            EINSUMS_LOG_WARN("Node {} ({}) has no executor!", node.id, node.label);
            return;
        }
        if (_device_work) {
            auto const io = _control_io.find(idx);
            for (TensorId const id : io != _control_io.end() ? io->second.second : node.outputs) {
                _coherence.wrote_on_host(id);
            }
        }
    }

    auto t_end = std::chrono::steady_clock::now();

    double const ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    // This node's own slot: concurrent runs of other nodes write other slots.
    RunnerAccess::replay(_graph)._run_slots[idx] = {.ms = ms, .width = width, .on_device = device_kernel};
    // node_zone pops here (and on any early return / thrown node).
}

void NodeRunner::finish() {
    flush_device();

    // In node order, which is the order the samples have always been reported in, whatever order
    // a concurrent executor ran the nodes in.
    std::vector<NodeTimingSample> samples;
    samples.reserve(_graph.nodes().size());
    for (std::size_t i = 0; i < _graph.nodes().size(); ++i) {
        detail::NodeRunSlot const &slot = RunnerAccess::replay(_graph)._run_slots[i];
        if (slot.ms >= 0.0) {
            Node const &node = _graph.nodes()[i];
            samples.push_back({.id = node.id, .kind = node.kind, .duration_ms = slot.ms, .width = slot.width});
        }
    }
    RunnerAccess::record_node_timings(_graph, std::move(samples));
}

void NodeRunner::flush_device() {
    // Final D2H flush (discrete GPU only).
    // On unified memory, GPU wrote directly to host-accessible tensor data, no copy needed.
    if constexpr (!gpu::has_unified_memory) {
        // Copy back every tensor whose device copy is still the only current one at the end of
        // the replay. The coherence record is kept as the nodes run, so it answers this exactly.
        // An earlier version scanned the whole node list for "was any CPU node an output writer"
        // - order-blind, and wrong in both directions: a tensor written CPU-then-GPU was excluded
        // and its GPU result silently dropped.
        if (_device_work) {
            _coherence.flush([&](TensorId id) {
                TensorHandle const *handle = _graph.find_tensor(id);
                return download(handle, RunnerAccess::replay(_graph)._device_shadows.get(id), handle != nullptr ? handle->total_bytes() : 0,
                                RunnerAccess::replay(_graph)._transfers);
            });
        }
    }
}

EINSUMS_NAMESPACE_END(compute_graph)
