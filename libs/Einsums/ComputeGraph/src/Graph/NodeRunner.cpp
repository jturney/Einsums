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

} // namespace

NodeRunner::NodeRunner(Graph &graph, bool recording) : _graph(graph), _recording(recording) {
    // assign() reuses the slots' storage, so a replay of an unchanged graph allocates nothing.
    _graph._run_slots.assign(_graph._nodes.size(), detail::NodeRunSlot{});
}

void NodeRunner::open_zone(std::optional<waggle::ScopedZone> &zone, std::size_t idx) const {
    if (!_recording) {
        return;
    }
    Node const &node = _graph._nodes[idx];

    // A node whose cache entry does not match falls back to its bare label
    // (defensive: an undeclared mutation after the profile strings were built).
    static Graph::NodeProfileStrings const kEmptyEntry{};
    Graph::NodeProfileStrings const       &ps = (idx < _graph._profile_strings.size() && _graph._profile_strings[idx].node_id == node.id)
                                                    ? _graph._profile_strings[idx]
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
    Node const &node = _graph._nodes[idx];
    return node.async_start && node.async_finish;
}

bool NodeRunner::serial_only() const {
    if constexpr (gpu::has_unified_memory) {
        return false;
    } else {
        for (Node const &node : _graph._nodes) {
            if (node.target == Target::GPU || node.kind == OpKind::HostToDevice || node.kind == OpKind::DeviceToHost) {
                return true;
            }
        }
        return false;
    }
}

void NodeRunner::start_async(std::size_t idx) {
    Node const                       &node = _graph._nodes[idx];
    std::optional<waggle::ScopedZone> zone;
    open_zone(zone, idx);
    // The two halves are separate calls, possibly on two threads: the observer hears the node
    // begin here and end where finish_async completes.
    detail::notify_node(*_graph._run_context, EventKind::NodeStart, _graph, node, 0.0);
    node.async_start();
}

void NodeRunner::finish_async(std::size_t idx) {
    Node const                       &node = _graph._nodes[idx];
    std::optional<waggle::ScopedZone> zone;
    open_zone(zone, idx);
    auto const t_start = std::chrono::steady_clock::now();
    node.async_finish();
    double const ms           = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_start).count();
    _graph._run_slots[idx].ms = ms;
    detail::notify_node(*_graph._run_context, EventKind::NodeEnd, _graph, node, ms);
}

void NodeRunner::run(std::size_t idx, unsigned width) {
    Node &node = _graph._nodes[idx];

    std::optional<waggle::ScopedZone> node_zone;
    open_zone(node_zone, idx);

    // An attached observer hears the node begin here and end however this call ends.
    detail::NodeObservation const observed(*_graph._run_context, _graph, node);
    auto                          t_start = std::chrono::steady_clock::now();

    if (node.kind == OpKind::HostToDevice) {
        // H2D transfer.
        auto const *tdesc = node.op_data.get_if<TransferDescriptor>();
        if (tdesc) {
            if constexpr (!gpu::has_unified_memory) {
                // Discrete GPU: copy host data → device shadow.
                auto &handle = _graph._tensors[tdesc->tensor_id];
                void *shadow = _graph._device_shadows.ensure(tdesc->tensor_id, tdesc->size_bytes);
                // A tensor whose storage has not been materialized yet has a
                // null data_ptr - graphs built inside a Setup closure reach
                // their transfer nodes before the allocation happens. Copying
                // from it is a null dereference, and there is nothing to copy:
                // the device buffer is exactly as defined as the host one.
                void *host = live_host_ptr(handle);
                if (shadow && host) {
                    gpu::memcpy_host_to_device(shadow, host, tdesc->size_bytes);
                    _device_valid.insert(tdesc->tensor_id);
                }
            }
            // Unified memory: no copy needed, GPU reads host memory directly.
        }
    } else if (node.kind == OpKind::DeviceToHost) {
        // D2H transfer.
        auto const *tdesc = node.op_data.get_if<TransferDescriptor>();
        if (tdesc) {
            if constexpr (!gpu::has_unified_memory) {
                // Discrete GPU: copy device shadow → host.
                auto       &handle = _graph._tensors[tdesc->tensor_id];
                void const *shadow = _graph._device_shadows.get(tdesc->tensor_id);
                // Only copy down when the device copy is the authoritative
                // one. A DeviceToHost node is placed by TransferInsertion on
                // the assumption that the GPU node before it ran on the
                // device; when that node instead fell back to the CPU, the
                // host result is the live one and the shadow still holds
                // whatever was uploaded before the op. Copying it back then
                // overwrites the correct answer with stale data - typically
                // the zeros an output tensor was initialized to, which is
                // what made whole graphs come out zero.
                void *host = live_host_ptr(handle);
                if (shadow && host && _device_valid.count(tdesc->tensor_id)) {
                    gpu::memcpy_device_to_host(host, shadow, tdesc->size_bytes);
                }
                _device_valid.erase(tdesc->tensor_id);
            }
            // Unified memory: no copy needed, result is already in host-accessible memory.
        }
    } else if (node.target == Target::GPU) {
        // GPU node execution.
        std::vector<std::pair<TensorId, void *>> saved_ptrs;

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
            auto check_ready = [&](TensorId tid) {
                auto const *handle = _graph.find_tensor(tid);
                if (handle == nullptr) {
                    operands_ready = false;
                    return;
                }
                if (!_device_valid.count(tid) && live_host_ptr(*handle) == nullptr) {
                    operands_ready = false;
                }
            };
            for (auto tid : node.inputs)
                check_ready(tid);
            for (auto tid : node.outputs)
                check_ready(tid);

            if (!operands_ready) {
                EINSUMS_LOG_DEBUG("Graph::execute: node {} ({}) kept on the host, an operand has no host storage to transfer", node.id,
                                  node.label);
            }
        }

        if constexpr (!gpu::has_unified_memory) {
            if (operands_ready) {
                // Discrete GPU: swap tensor data pointers to device shadows.
                std::unordered_set<TensorId> swapped;
                auto                         swap_to_shadow = [&](TensorId tid) {
                    if (swapped.count(tid))
                        return;
                    auto &handle = _graph._tensors[tid];
                    void *shadow = _graph._device_shadows.ensure(tid, handle.total_bytes());
                    if (!shadow)
                        return;

                    // Populate the shadow when the host copy is still the
                    // authoritative one. TransferInsertion normally emits an
                    // explicit HostToDevice node and marks the tensor
                    // Residency::Device, in which case this is skipped.
                    //
                    // Doing it here as well keeps the executor correct whichever
                    // passes ran: a GPU node reached without a preceding H2D (a
                    // hand-set Target::GPU, or an H2D TransferElimination dropped)
                    // would otherwise compute on uninitialized device memory.
                    if (!_device_valid.count(tid)) {
                        // Non-null by construction: checked above before any
                        // operand of this node was placed.
                        gpu::memcpy_host_to_device(shadow, live_host_ptr(handle), handle.total_bytes());
                    }
                    _device_valid.insert(tid);

                    if (handle.swap_data) {
                        void *old_ptr = handle.swap_data(shadow);
                        saved_ptrs.emplace_back(tid, old_ptr);
                        swapped.insert(tid);
                    }
                };
                // Outputs are copied up too: an accumulating op (nonzero C
                // prefactor) reads its output operand before writing it.
                for (auto tid : node.inputs)
                    swap_to_shadow(tid);
                for (auto tid : node.outputs)
                    swap_to_shadow(tid);
            }
        }
        // Unified memory: no swap needed, GPU reads tensor.data() directly.
        // MPS wrap_or_copy will create a zero-copy MTLBuffer wrapper.

        // Execute via GPU BLAS dispatch if possible, otherwise CPU fallback.
        bool gpu_dispatched = false;
        try {
            gpu_dispatched = operands_ready && try_gpu_blas_dispatch(node, _graph._tensors, _graph._device_shadows);
            if (gpu_dispatched) {
                waggle::annotate("gpu_dispatch", "gemm");
            }
        } catch (std::exception const &e) {
            EINSUMS_LOG_WARN("GPU GEMM dispatch failed for node {} ({}): {}", node.id, node.label, e.what());
        }

        if (!gpu_dispatched) {
            waggle::annotate("gpu_dispatch", "cpu_fallback");

            // The CPU lambda must run on HOST pointers: on a discrete device
            // the tensors currently point at cudaMalloc'd shadows, and a host
            // BLAS kernel on them segfaults.
            //
            // Inputs that were copied up must also come back down, because a
            // preceding GPU node in the same chain may have produced them on
            // the device and the host copy is then stale.
            if constexpr (!gpu::has_unified_memory) {
                for (auto const &[tid, old_ptr] : saved_ptrs) {
                    auto &handle = _graph._tensors[tid];
                    void *shadow = _graph._device_shadows.get(tid);
                    if (handle.swap_data) {
                        handle.swap_data(old_ptr);
                    }
                    if (void *host = live_host_ptr(handle); shadow && host && _device_valid.count(tid)) {
                        gpu::memcpy_device_to_host(host, shadow, handle.total_bytes());
                    }
                    _device_valid.erase(tid);
                }
                saved_ptrs.clear();
            }

            // node.cpu_fallback and node.execute are the same callable -
            // GPUPlacement assigns cpu_fallback = execute - so the old
            // try/catch-then-retry structure re-ran an identical call and
            // could only fail twice. One call, one place to report.
            if (node.execute) {
                node.execute();
            }
            for (auto tid : node.outputs) {
                if (auto *handle = _graph.find_tensor(tid)) {
                    handle->residency = Residency::Host;
                }
            }
        }

        // 3. Restore original host pointers (only needed on discrete GPU).
        // Empty when the fallback above already restored them.
        for (auto const &[tid, old_ptr] : saved_ptrs) {
            auto &handle = _graph._tensors[tid];
            if (handle.swap_data) {
                handle.swap_data(old_ptr);
            }
        }
    } else {
        // CPU node: execute normally. Anything it writes makes the host copy
        // authoritative again, so a later GPU node must re-upload it.
        if constexpr (!gpu::has_unified_memory) {
            // Bring down anything this node will read whose live value is
            // sitting in a device shadow.
            //
            // TransferInsertion normally emits a DeviceToHost for this, but
            // the executor cannot depend on that: an accumulating einsum
            // reads its output operand, and when the producing GPU node ran
            // on the device while the consumer fell back to the host, the
            // host copy is stale. Erasing _device_valid without copying -
            // which is what this did - silently drops the GPU result.
            auto sync_down = [&](TensorId tid) {
                if (!_device_valid.count(tid)) {
                    return;
                }
                if (auto const *handle = _graph.find_tensor(tid)) {
                    void const *shadow = _graph._device_shadows.get(tid);
                    if (void *host = live_host_ptr(*handle); shadow && host) {
                        gpu::memcpy_device_to_host(host, shadow, handle->total_bytes());
                    }
                }
                _device_valid.erase(tid);
            };

            if (node.kind == OpKind::Setup || node.kind == OpKind::Loop || node.kind == OpKind::Conditional) {
                // These carry a subgraph, and the nested replay writes host
                // tensors through its own executor and its own shadow map.
                // node.outputs does not describe what it touched, so no
                // device shadow this level holds can be assumed current
                // afterwards. Anything still needed on the device gets
                // re-uploaded by the next GPU node that wants it.
                // A nested replay can read anything, so everything the
                // device currently owns has to come back first.
                std::vector<TensorId> const owned(_device_valid.begin(), _device_valid.end());
                for (auto tid : owned)
                    sync_down(tid);
                _device_valid.clear();
            } else {
                for (auto tid : node.inputs)
                    sync_down(tid);
                for (auto tid : node.outputs)
                    sync_down(tid);
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
    }

    auto t_end = std::chrono::steady_clock::now();

    double const ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    // This node's own slot: concurrent runs of other nodes write other slots.
    _graph._run_slots[idx] = {.ms = ms, .width = width};
    // node_zone pops here (and on any early return / thrown node).
}

void NodeRunner::finish() {
    flush_device();

    // In node order, which is the order the samples have always been reported in, whatever order
    // a concurrent executor ran the nodes in.
    std::vector<NodeTimingSample> samples;
    samples.reserve(_graph._nodes.size());
    for (std::size_t i = 0; i < _graph._nodes.size(); ++i) {
        detail::NodeRunSlot const &slot = _graph._run_slots[i];
        if (slot.ms >= 0.0) {
            Node const &node = _graph._nodes[i];
            samples.push_back({.id = node.id, .kind = node.kind, .duration_ms = slot.ms, .width = slot.width});
        }
    }
    _graph.record_node_timings(std::move(samples));
}

void NodeRunner::flush_device() {
    // Final D2H flush (discrete GPU only).
    // On unified memory, GPU wrote directly to host-accessible tensor data, no copy needed.
    if constexpr (!gpu::has_unified_memory) {
        // Copy back every tensor whose device shadow is still the authoritative
        // copy at the end of the replay.
        //
        // _device_valid is maintained as the nodes run, so it answers this
        // exactly. The previous version scanned the whole node list for "was
        // any CPU node an output writer" - order-blind, and wrong in both
        // directions: a tensor written CPU-then-GPU was excluded and its GPU
        // result silently dropped.
        for (auto &tid : _device_valid) {
            auto const *handle = _graph.find_tensor(tid);
            if (handle == nullptr)
                continue;
            void const *shadow = _graph._device_shadows.get(tid);
            if (void *host = live_host_ptr(*handle); shadow && host) {
                gpu::memcpy_device_to_host(host, shadow, handle->total_bytes());
            }
        }
    }
}

EINSUMS_NAMESPACE_END(compute_graph)
