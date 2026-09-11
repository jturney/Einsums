//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file Execute.cpp
/// @brief Replay: running a captured graph, eagerly or through an executor.
///
/// @ref Graph::execute is the serial walk and the reference semantics; the
/// @ref Executor overload hands the same schedule to a parallel backend. Both
/// replay a schedule `Schedule.cpp` built, which is why neither computes an
/// order of its own.
///
/// Per node, the GPU is offered the work first through `GpuDispatch.hpp` and the
/// CPU closure runs whenever it declines. The optimization entry points sit here
/// too, because `optimize` is the other thing a caller does with a captured graph
/// and it is defined by what a later `execute` will see.

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

#include "GpuDispatch.hpp"

EINSUMS_NAMESPACE_BEGIN(compute_graph)

using namespace gpu_dispatch;

void Graph::execute() {
    // Storage-level aliasing must be resolved before anything reasons about
    // which buffer a node touches; cheap and idempotent after the first call.
    link_alias_storage();
    // An installed executor (set_executor) takes over the whole run. This is
    // how loop bodies get a parallel backend: the loop node replays its body
    // via this argument-less execute().
    if (_executor) {
        execute(*_executor);
        return;
    }

    // Wall clock of this replay, for the thread-plan trial: replays are the
    // only thing the trial's two sides can be compared on.
    auto const replay_t0 = std::chrono::steady_clock::now();

    // Rebuild when the order is unknown OR a pass vouched for the order via
    // mark_sorted() but left the position-keyed _deps stale (_deps_valid
    // false). topological_sort() takes the cheap rebuild_deps path in that
    // second case, keeping the pass-chosen order.
    if (!_sorted || !_deps_valid) {
        topological_sort();
    }

    // Validate slot pointers once per change to the slot table (cheap check).
    // This catches cross-pipeline tensor misuse before it segfaults. Nothing
    // can invalidate a pointer between two replays of an unchanged graph, so
    // the walk is skipped on the replays that iterative workloads spend their
    // time in; every slot create/rebind/redirect and every pass run clears the
    // flag.
    if (!_slots_validated) {
        for (auto const &[id, slot] : _slot_map) {
            if (!slot)
                continue;
            if (slot->ptr == nullptr || reinterpret_cast<uintptr_t>(slot->ptr) < 4096) {
                std::string tname = slot->name.empty() ? fmt::format("id={}", id) : slot->name;
                EINSUMS_THROW_EXCEPTION(std::runtime_error,
                                        "Graph '{}': tensor slot '{}' has invalid pointer (0x{:x}). "
                                        "This usually means the tensor was declared on a different pipeline/graph "
                                        "and wasn't properly shared via the workspace. "
                                        "Declare shared tensors on the Workspace, not on individual Pipelines.",
                                        _name, tname, reinterpret_cast<uintptr_t>(slot->ptr));
            }
        }
        _slots_validated = true;
    }

    if (!_executed) {
        auto validation = validate_tensors();
        if (!validation) {
            EINSUMS_THROW_EXCEPTION(std::runtime_error, "{}", validation.error().message);
        }
        register_graph(this);
    }

    clear_timing_report();
    _timing_samples.reserve(_nodes.size());
    _timing_report_valid = false; // samples are about to be written

    // Read once so the zone pushes and their matching pops agree even if
    // another thread flips recording mid-run: an unbalanced zone deepens the
    // profiler tree without bound.
    bool const recording = profile::Profiler::instance().enabled();

    // Two call sites shared by every graph and every node. The NAMES are per
    // graph and per node and come pre-interned from _profile_strings; the site
    // supplies the file/function/line, which are the same every time.
    static profile::ZoneSite const kGraphExecSite{"ComputeGraph::execute", __FILE__, __LINE__, __func__};
    static profile::ZoneSite const kGraphNodeSite{"ComputeGraph::node", __FILE__, __LINE__, __func__};

    // All fmt::format work for zones/annotations is precomputed, and so is the
    // string interning behind them; replays of an unchanged graph only pay the
    // profiler's event write. A run that records nothing builds none of it.
    if (recording && !_profile_strings_valid) {
        rebuild_profile_strings();
    }

    // RAII zones (see execute(Executor&)): a node that throws must still pop its
    // profiler zone, or the tree depth grows without bound across failed runs.
    std::optional<profile::ScopedZone> exec_zone;
    if (recording) {
        exec_zone.emplace(kGraphExecSite, _exec_zone_id, _exec_zone_name);
    }

    // A node whose cache entry does not match falls back to its bare label
    // (defensive: an undeclared mutation after the rebuild above).
    static NodeProfileStrings const kEmptyEntry{};

    // Tensors whose DEVICE shadow currently holds the authoritative data.
    //
    // Tracked at run time rather than read off TensorHandle::residency, which
    // is a static annotation left by TransferInsertion and says nothing about a
    // tensor produced by one GPU node and consumed by the next: no transfer is
    // inserted for those, so residency stays Host while the live value is on the
    // device. Copying the host side up on that basis would clobber the producer's
    // result.
    std::unordered_set<TensorId> device_valid;

    for (size_t idx = 0; idx < _nodes.size(); idx++) {
        Node &node = _nodes[idx];

        std::optional<profile::ScopedZone> node_zone;
        if (recording) {
            NodeProfileStrings const &ps =
                (idx < _profile_strings.size() && _profile_strings[idx].node_id == node.id) ? _profile_strings[idx] : kEmptyEntry;

            if (ps.zone_id != 0) {
                node_zone.emplace(kGraphNodeSite, ps.zone_id, ps.zone);
            } else {
                node_zone.emplace(node.label);
            }

            // These two are static strings, interned once per process.
            profile::annotate("op_kind", op_kind_name(node.kind));
            profile::annotate("device", node.target == Target::GPU ? "GPU" : "CPU");

            for (auto const &[key, value] : ps.texts) {
                profile::annotate_interned(key, value);
            }
            for (auto const &[key, value] : ps.numbers) {
                profile::annotate_interned(key, value);
            }
            for (auto const &[key, value] : ps.reals) {
                profile::annotate_interned(key, value);
            }
        }

        auto t_start = std::chrono::steady_clock::now();

        if (node.kind == OpKind::HostToDevice) {
            // H2D transfer.
            auto const *tdesc = std::get_if<TransferDescriptor>(&node.op_data);
            if (tdesc) {
                if constexpr (!gpu::has_unified_memory) {
                    // Discrete GPU: copy host data → device shadow.
                    auto &handle = _tensors[tdesc->tensor_id];
                    void *shadow = _device_shadows.ensure(tdesc->tensor_id, tdesc->size_bytes);
                    // A tensor whose storage has not been materialized yet has a
                    // null data_ptr - graphs built inside a Setup closure reach
                    // their transfer nodes before the allocation happens. Copying
                    // from it is a null dereference, and there is nothing to copy:
                    // the device buffer is exactly as defined as the host one.
                    void *host = live_host_ptr(handle);
                    if (shadow && host) {
                        gpu::memcpy_host_to_device(shadow, host, tdesc->size_bytes);
                        device_valid.insert(tdesc->tensor_id);
                    }
                }
                // Unified memory: no copy needed, GPU reads host memory directly.
            }
        } else if (node.kind == OpKind::DeviceToHost) {
            // D2H transfer.
            auto const *tdesc = std::get_if<TransferDescriptor>(&node.op_data);
            if (tdesc) {
                if constexpr (!gpu::has_unified_memory) {
                    // Discrete GPU: copy device shadow → host.
                    auto       &handle = _tensors[tdesc->tensor_id];
                    void const *shadow = _device_shadows.get(tdesc->tensor_id);
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
                    if (shadow && host && device_valid.count(tdesc->tensor_id)) {
                        gpu::memcpy_device_to_host(host, shadow, tdesc->size_bytes);
                    }
                    device_valid.erase(tdesc->tensor_id);
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
                    auto it = _tensors.find(tid);
                    if (it == _tensors.end()) {
                        operands_ready = false;
                        return;
                    }
                    if (!device_valid.count(tid) && live_host_ptr(it->second) == nullptr) {
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
                        auto &handle = _tensors[tid];
                        void *shadow = _device_shadows.ensure(tid, handle.total_bytes());
                        if (!shadow)
                            return;

                        // Populate the shadow when the host copy is still the
                        // authoritative one. TransferInsertion normally emits an
                        // explicit HostToDevice node and marks the tensor
                        // Residency::Device, in which case this is skipped.
                        //
                        // Doing it here as well is what keeps the executor correct
                        // independently of which passes ran. Previously this branch
                        // called ensure() and swapped the pointer but never copied,
                        // so a GPU node reached without a preceding H2D - a graph
                        // with Target::GPU set by hand, or one where
                        // TransferElimination dropped an H2D it judged redundant -
                        // computed on uninitialized device memory. Under unified
                        // memory the swap is a no-op, so the bug was invisible.
                        if (!device_valid.count(tid)) {
                            // Non-null by construction: checked above before any
                            // operand of this node was placed.
                            gpu::memcpy_host_to_device(shadow, live_host_ptr(handle), handle.total_bytes());
                        }
                        device_valid.insert(tid);

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
                gpu_dispatched = operands_ready && try_gpu_blas_dispatch(node, _tensors, _device_shadows);
                if (gpu_dispatched) {
                    profile::annotate("gpu_dispatch", "gemm");
                }
            } catch (std::exception const &e) {
                EINSUMS_LOG_WARN("GPU GEMM dispatch failed for node {} ({}): {}", node.id, node.label, e.what());
            }

            if (!gpu_dispatched) {
                profile::annotate("gpu_dispatch", "cpu_fallback");

                // The CPU lambda must run on HOST pointers. Restoring them first
                // is not a tidiness detail: on a discrete device the tensors are
                // currently pointing at cudaMalloc'd shadows, so calling a host
                // BLAS kernel on them is a segfault. This code used to run the
                // fallback "with pointers still swapped to shadows" - harmless
                // only because every backend that ever executed it had unified
                // memory, which made the swap a no-op.
                //
                // Inputs that were copied up must also come back down, because a
                // preceding GPU node in the same chain may have produced them on
                // the device and the host copy is then stale.
                if constexpr (!gpu::has_unified_memory) {
                    for (auto const &[tid, old_ptr] : saved_ptrs) {
                        auto &handle = _tensors[tid];
                        void *shadow = _device_shadows.get(tid);
                        if (handle.swap_data) {
                            handle.swap_data(old_ptr);
                        }
                        if (void *host = live_host_ptr(handle); shadow && host && device_valid.count(tid)) {
                            gpu::memcpy_device_to_host(host, shadow, handle.total_bytes());
                        }
                        device_valid.erase(tid);
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
                    auto it = _tensors.find(tid);
                    if (it != _tensors.end()) {
                        it->second.residency = Residency::Host;
                    }
                }
            }

            // 3. Restore original host pointers (only needed on discrete GPU).
            // Empty when the fallback above already restored them.
            for (auto const &[tid, old_ptr] : saved_ptrs) {
                auto &handle = _tensors[tid];
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
                // host copy is stale. Erasing device_valid without copying -
                // which is what this did - silently drops the GPU result.
                auto sync_down = [&](TensorId tid) {
                    if (!device_valid.count(tid)) {
                        return;
                    }
                    auto it = _tensors.find(tid);
                    if (it != _tensors.end()) {
                        void const *shadow = _device_shadows.get(tid);
                        if (void *host = live_host_ptr(it->second); shadow && host) {
                            gpu::memcpy_device_to_host(host, shadow, it->second.total_bytes());
                        }
                    }
                    device_valid.erase(tid);
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
                    std::vector<TensorId> const owned(device_valid.begin(), device_valid.end());
                    for (auto tid : owned)
                        sync_down(tid);
                    device_valid.clear();
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
                continue;
            }
        }

        auto t_end = std::chrono::steady_clock::now();

        double const ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        _timing_samples.push_back({.id = node.id, .kind = node.kind, .duration_ms = ms});
        // node_zone pops here (and on any early continue / thrown node).
    }

    // Final D2H flush (discrete GPU only).
    // On unified memory, GPU wrote directly to host-accessible tensor data, no copy needed.
    if constexpr (!gpu::has_unified_memory) {
        // Copy back every tensor whose device shadow is still the authoritative
        // copy at the end of the replay.
        //
        // device_valid is maintained as the nodes run, so it answers this
        // exactly. The previous version scanned the whole node list for "was
        // any CPU node an output writer" - order-blind, and wrong in both
        // directions: a tensor written CPU-then-GPU was excluded and its GPU
        // result silently dropped.
        for (auto &tid : device_valid) {
            auto const it = _tensors.find(tid);
            if (it == _tensors.end())
                continue;
            auto       &handle = it->second;
            void const *shadow = _device_shadows.get(tid);
            if (void *host = live_host_ptr(handle); shadow && host) {
                gpu::memcpy_device_to_host(host, shadow, handle.total_bytes());
            }
        }
    }

    // exec_zone pops here.
    _executed = true;

    // The safe point for the thread-plan trial (see plan_threads): the replay
    // has returned, so nothing is reading a width; every nested body replay it
    // started has returned with it; and this run's timings are complete and
    // recorded, which is the whole reason to wait for it.
    finish_replay_thread_plan(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - replay_t0).count());
}

void Graph::execute(Executor &executor) {
    auto const replay_t0 = std::chrono::steady_clock::now();
    // Rebuild when the order is unknown OR a pass vouched for the order via
    // mark_sorted() but left the position-keyed _deps stale (_deps_valid
    // false). Concurrent executors read _deps directly, so stale lists here
    // would drop storage-reuse and lifecycle edges a serial run never needed.
    if (!_sorted || !_deps_valid) {
        topological_sort();
    }

    if (!_executed) {
        auto validation = validate_tensors();
        if (!validation) {
            EINSUMS_THROW_EXCEPTION(std::runtime_error, "{}", validation.error().message);
        }
    }

    clear_timing_report();
    _timing_samples.reserve(_nodes.size());
    _timing_report_valid = false; // the executor is about to write samples

    {
        // RAII: pop the profiler zone even if the executor propagates an
        // exception. A bare push/pop here leaks an unclosed zone on every
        // failed execute, deepening the profiler tree without bound (which
        // makes the profiler→viewer serialization progressively slower).
        //
        // The name is built through the callable overload so that a run with
        // recording off pays neither the fmt::format nor executor.name()'s
        // returned std::string - this fired on every replay regardless of
        // profiler state.
        static profile::ZoneSite const site{"ComputeGraph::execute(executor)", __FILE__, __LINE__, __func__};
        profile::ScopedZone const _zone(site,
                                        [&]() { return fmt::format("ComputeGraph::execute({}, executor={})", _name, executor.name()); });
        executor.execute(*this);
    }
    _executed = true;

    // Same safe point as the argument-less execute(): the replay has returned
    // and its timings are complete.
    finish_replay_thread_plan(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - replay_t0).count());
}

UsageAnalysis const &Graph::usage() {
    if (_usage_version != _analysis_version || _usage.node_count() != _nodes.size()) {
        _usage         = UsageAnalysis::build(*this);
        _usage_version = _analysis_version;
    }
    return _usage;
}

bool Graph::apply(PassManager &pm) {
    // Storage-level aliasing must be resolved before anything reasons about
    // which buffer a node touches; cheap and idempotent after the first call.
    //
    // Recursively, and that is not a refinement. Fourteen passes opt into
    // ``recurse_into_subgraphs()`` and rewrite loop bodies, and every one of
    // them asks ``Graph::resolve_alias`` which buffer a node touches. Linking
    // the root alone left every body unlinked, so inside a loop that question
    // answered "this view aliases nothing": Reorder's hazard scan then missed
    // the view/parent edges entirely and was free to move a writer past a
    // reader of the same buffer. Silent, and invisible to a straight-line test.
    // for_each_subgraph visits one level, so this recurses: a loop nested in a
    // loop needs linking as much as the outer one does.
    auto link_tree = [](Graph &g, auto &&self) -> void {
        g.link_alias_storage();
        g.for_each_subgraph([&self](Graph &sub) { self(sub, self); });
    };
    link_tree(*this, link_tree);
    std::scoped_lock const lock(*_content_mutex);
    bool const             modified = pm.run(*this);
    if (modified) {
        _executed = false;
    }
    return modified;
}

bool Graph::optimize() {
    return optimize(OptLevel::O2);
}

bool Graph::optimize(OptLevel level) {
    size_t const before = _nodes.size();

    auto       pm       = PassManager::create_for(level);
    bool const modified = apply(pm);

    _last_optimize_report =
        fmt::format("optimize(O{}) on '{}': {} -> {} node(s)\n{}", static_cast<int>(level), _name, before, _nodes.size(), pm.explain());
    return modified;
}

EINSUMS_NAMESPACE_END(compute_graph)
