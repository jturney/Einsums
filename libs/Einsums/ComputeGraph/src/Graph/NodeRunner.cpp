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
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/GPU/Runtime.hpp>
#include <Einsums/Logging.hpp>
#include <Einsums/Profile/Profile.hpp>

#include <chrono>
#include <cstddef>
#include <exception>
#include <optional>
#include <unordered_set>
#include <utility>
#include <vector>

#include "GpuDispatch.hpp"

EINSUMS_NAMESPACE_BEGIN(compute_graph)

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
bool upload(TensorHandle const *handle, void *shadow, std::size_t bytes) {
    void *host = handle != nullptr ? live_host_ptr(*handle) : nullptr;
    if (shadow == nullptr || host == nullptr) {
        return false;
    }
    gpu::memcpy_host_to_device(shadow, host, bytes);
    return true;
}

/// Copy @p bytes of @p shadow down into @p handle's live host data.
/// @return Whether a copy happened; see upload.
bool download(TensorHandle const *handle, void const *shadow, std::size_t bytes) {
    void *host = handle != nullptr ? live_host_ptr(*handle) : nullptr;
    if (shadow == nullptr || host == nullptr) {
        return false;
    }
    gpu::memcpy_device_to_host(host, shadow, bytes);
    return true;
}

} // namespace

NodeRunner::NodeRunner(Graph &graph, bool recording) : _graph(graph), _recording(recording) {
    // assign() reuses the slots' storage, so a replay of an unchanged graph allocates nothing.
    _graph._replay._run_slots.assign(_graph.nodes().size(), detail::NodeRunSlot{});
    if constexpr (!gpu::has_unified_memory) {
        for (Node const &node : _graph.nodes()) {
            if (node.target == Target::GPU || node.kind == OpKind::HostToDevice || node.kind == OpKind::DeviceToHost) {
                prepare_device_work();
                break;
            }
        }
    }
}

void NodeRunner::prepare_device_work() {
    _device_work = true;

    // Every shadow is allocated here, before any node runs, so the shadow map is only read while
    // nodes run, from whichever threads run them.
    std::vector<TensorId> tracked;
    auto const            track = [&](TensorId id, std::size_t bytes) {
        tracked.push_back(id);
        _graph._replay._device_shadows.ensure(id, bytes);
    };
    for (Node const &node : _graph.nodes()) {
        if (node.kind == OpKind::HostToDevice || node.kind == OpKind::DeviceToHost) {
            if (auto const *tdesc = node.op_data.get_if<TransferDescriptor>()) {
                track(tdesc->tensor_id, tdesc->size_bytes);
            }
        } else if (node.target == Target::GPU) {
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
    _coherence.need_on_host(
        id, [&]() { return download(handle, _graph._replay._device_shadows.get(id), handle != nullptr ? handle->total_bytes() : 0); });
}

void NodeRunner::open_zone(std::optional<waggle::ScopedZone> &zone, std::size_t idx) const {
    if (!_recording) {
        return;
    }
    Node const &node = _graph.nodes()[idx];

    // A node whose cache entry does not match falls back to its bare label
    // (defensive: an undeclared mutation after the profile strings were built).
    static detail::ReplayState::NodeProfileStrings const kEmptyEntry{};
    detail::ReplayState::NodeProfileStrings const       &ps =
        (idx < _graph._replay._profile_strings.size() && _graph._replay._profile_strings[idx].node_id == node.id)
            ? _graph._replay._profile_strings[idx]
            : kEmptyEntry;

    if (ps.zone_id != 0) {
        zone.emplace(graph_node_site(), ps.zone_id);
    } else {
        zone.emplace(node.label);
    }

    // These two are static strings, interned once per process.
    waggle::annotate("op_kind", op_kind_name(node.kind));
    waggle::annotate("device", node.target == Target::GPU ? "GPU" : "CPU");

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
    open_zone(zone, idx);
    // The two halves are separate calls, possibly on two threads: the observer hears the node
    // begin here and end where finish_async completes.
    detail::notify_node(*_graph._replay._run_context, EventKind::NodeStart, _graph, node, 0.0);
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
    open_zone(zone, idx);
    auto const t_start = std::chrono::steady_clock::now();
    node.async_finish();
    if (_device_work) {
        for (TensorId const id : node.outputs) {
            _coherence.wrote_on_host(id);
        }
    }
    double const ms                   = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_start).count();
    _graph._replay._run_slots[idx].ms = ms;
    detail::notify_node(*_graph._replay._run_context, EventKind::NodeEnd, _graph, node, ms);
}

void NodeRunner::run(std::size_t idx, unsigned width) {
    Node const &node = _graph.nodes()[idx];

    std::optional<waggle::ScopedZone> node_zone;
    open_zone(node_zone, idx);

    // An attached observer hears the node begin here and end however this call ends.
    detail::NodeObservation const observed(*_graph._replay._run_context, _graph, node);
    auto                          t_start = std::chrono::steady_clock::now();

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
                _coherence.need_on_device(id, [&]() { return upload(handle, _graph._replay._device_shadows.get(id), tdesc->size_bytes); });
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
                _coherence.need_on_host(id, [&]() { return download(handle, _graph._replay._device_shadows.get(id), tdesc->size_bytes); });
            }
        }
    } else if (node.target == Target::GPU) {
        // GPU node execution.
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
                        void               *shadow = _graph._replay._device_shadows.get(id);
                        if (!_coherence.need_on_device(id, [&]() { return upload(handle, shadow, handle->total_bytes()); })) {
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
            gpu_dispatched = operands_ready && try_gpu_blas_dispatch(node, _graph.tensors_map(), _graph._replay._device_shadows);
            if (gpu_dispatched) {
                waggle::annotate("gpu_dispatch", "gemm");
            }
        } catch (std::exception const &e) {
            EINSUMS_LOG_WARN("GPU GEMM dispatch failed for node {} ({}): {}", node.id, node.label, e.what());
        }

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
                if (auto *handle = _graph.find_mutable_tensor(id)) {
                    handle->residency = Residency::Host;
                }
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
    _graph._replay._run_slots[idx] = {.ms = ms, .width = width};
    // node_zone pops here (and on any early return / thrown node).
}

void NodeRunner::finish() {
    flush_device();

    // In node order, which is the order the samples have always been reported in, whatever order
    // a concurrent executor ran the nodes in.
    std::vector<NodeTimingSample> samples;
    samples.reserve(_graph.nodes().size());
    for (std::size_t i = 0; i < _graph.nodes().size(); ++i) {
        detail::NodeRunSlot const &slot = _graph._replay._run_slots[i];
        if (slot.ms >= 0.0) {
            Node const &node = _graph.nodes()[i];
            samples.push_back({.id = node.id, .kind = node.kind, .duration_ms = slot.ms, .width = slot.width});
        }
    }
    _graph.record_node_timings(std::move(samples));
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
                return download(handle, _graph._replay._device_shadows.get(id), handle != nullptr ? handle->total_bytes() : 0);
            });
        }
    }
}

EINSUMS_NAMESPACE_END(compute_graph)
