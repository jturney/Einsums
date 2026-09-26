//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/Detail/ScalarDispatch.hpp>
#include <Einsums/ComputeGraph/EinsumSpec.hpp>
#include <Einsums/ComputeGraph/ExecutorBuilder.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Passes/PassUtil.hpp>
#include <Einsums/ComputeGraph/Passes/ScratchPrivatization.hpp>
#include <Einsums/ComputeGraph/StringDispatch.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph::passes)

namespace {

/// One touch of a candidate tensor, in node order. Reads and RMW writes extend
/// the current generation; a pure full-tensor overwrite starts a new one.
struct Access {
    size_t node_idx;
    bool   starts_generation;
};

/// The dtypes detail::dispatch_scalar_type can dispatch (clone declaration and
/// the rebuilt executors both go through it).
bool dispatchable_dtype(packed_gemm::ScalarType dtype) {
    switch (dtype) {
    case packed_gemm::ScalarType::Float32:
    case packed_gemm::ScalarType::Float64:
    case packed_gemm::ScalarType::Complex64:
    case packed_gemm::ScalarType::Complex128:
        return true;
    default:
        return false;
    }
}

/// Can rebuild_node() below reproduce this node's computation with a renamed
/// operand? Rebuilding replaces the executor, so every operand must resolve
/// through the graph (impl_fn) and the descriptor must actually describe what
/// the baked lambda computes.
bool rebuildable(Graph &graph, Node const &nd) {
    auto const operands_resolve = [&]() {
        for (auto const tid : nd.inputs) {
            auto const *h = graph.find_tensor(tid);
            if (h == nullptr || !h->impl_fn || !dispatchable_dtype(h->dtype)) {
                return false;
            }
        }
        for (auto const tid : nd.outputs) {
            auto const *h = graph.find_tensor(tid);
            if (h == nullptr || !h->impl_fn || !dispatchable_dtype(h->dtype)) {
                return false;
            }
        }
        return true;
    };

    switch (nd.kind) {
    case OpKind::Einsum: {
        auto const *d = nd.op_data.get_if<EinsumDescriptor>();
        return d != nullptr && d->params != nullptr && d->indices != nullptr && nd.inputs.size() >= 2 && nd.inputs.size() <= 3 &&
               nd.outputs.size() == 1 && operands_resolve();
    }
    case OpKind::Permute: {
        // Exactly one input and one output: a SymmetrizedAccumulation-rewritten
        // permute accumulates into a second input and its descriptor no longer
        // matches its baked executor, so it must not be rebuilt from it.
        auto const *d = nd.op_data.get_if<PermuteDescriptor>();
        return d != nullptr && nd.inputs.size() == 1 && nd.outputs.size() == 1 && operands_resolve();
    }
    case OpKind::Axpby: {
        // The live params are what the captured executor reads; a descriptor
        // without them cannot be trusted to match the lambda.
        auto const *d = nd.op_data.get_if<AxpbyDescriptor>();
        return d != nullptr && d->params != nullptr && (nd.inputs.size() == 1 || nd.inputs.size() == 2) && nd.outputs.size() == 1 &&
               operands_resolve();
    }
    default:
        return false;
    }
}

/// Replace @p old_id with @p new_id in @p nd's operand lists and rebuild its
/// executor so the computation follows. Caller guarantees rebuildable().
void rebuild_node(Graph &graph, Node &nd, TensorId old_id, TensorId new_id) {
    auto const sub = [&](TensorId tid) { return tid == old_id ? new_id : tid; };

    if (nd.kind == OpKind::Einsum) {
        auto const    *d = nd.op_data.get_if<EinsumDescriptor>();
        TensorId const a = sub(nd.inputs[0]);
        TensorId const b = sub(nd.inputs[1]);
        TensorId const c = sub(nd.outputs[0]);
        // The live indices/params are what the old executor read each call, so
        // the rebuild starts from them, not from the capture-time snapshots.
        Node rebuilt = graph.make_einsum_node(a, b, c, d->indices->spec, d->params->c_pf, d->params->ab_pf, d->params->conj_a,
                                              d->params->conj_b, nd.label);
        rebuilt.id   = nd.id; // keep NodeId-keyed state (profiler payloads) attached
        nd           = std::move(rebuilt);
        return;
    }

    // Permute and Axpby: the descriptor already says everything the executor does, its live
    // params included, so the node keeps it and only the operand ids change. The executor comes
    // from build_executor, as for a captured node. (A hand-built permute executor used to drop the
    // permutation operators and read the snapshot scalars.) Only the ids change in the operand
    // lists, so an accumulating axpby keeps listing Y as an input.
    for (auto &tid : nd.inputs) {
        tid = sub(tid);
    }
    for (auto &tid : nd.outputs) {
        tid = sub(tid);
    }
    auto const &out = graph.tensor(nd.outputs.front());
    nd.execute      = build_executor(nd.kind, out.dtype, out.rank, nd.op_data, graph, nd.inputs, nd.outputs);
}

/// Declare one clone of @p handle on @p graph: a graph-owned deferred
/// intermediate with the same dims and dtype. Returns 0 on failure.
TensorId declare_clone(Graph &graph, TensorHandle const &handle, std::string name) {
    void const *ptr = nullptr;
    detail::dispatch_scalar_type(handle.dtype, [&]<typename T>(T /*tag*/) {
        ptr = &graph.declare_zero_runtime_tensor<T>(std::move(name), handle.dims, /*intermediate=*/true);
    });
    if (ptr == nullptr) {
        return 0;
    }
    return graph.find_tensor_id_by_ptr(ptr);
}

} // namespace

void ScratchPrivatization::reset_stats() {
    _num_tensors_privatized = 0;
    _num_copies_created     = 0;
    _num_nodes_rebuilt      = 0;
}

bool ScratchPrivatization::run(Graph &graph) {
    PassCounter const nodes_rebuilt{_num_nodes_rebuilt};
    run_recursive(graph);
    return nodes_rebuilt.moved();
}

void ScratchPrivatization::run_recursive(Graph &graph) {
    for (auto &node : graph.nodes()) {
        if (auto *loop = node.op_data.get_if<LoopDescriptor>(); loop != nullptr && loop->body) {
            run_recursive(*loop->body);
        } else if (auto *cond = node.op_data.get_if<ConditionalDescriptor>(); cond != nullptr) {
            if (cond->then_branch) {
                run_recursive(*cond->then_branch);
            }
            if (cond->else_branch) {
                run_recursive(*cond->else_branch);
            }
        }
    }
    privatize_one_graph(graph);
}

void ScratchPrivatization::privatize_one_graph(Graph &graph) {
    // Under the built-in sequential replay the clones cost cache locality and
    // buy no width, so only graphs that will actually run on a parallel
    // executor are rewritten (Graph::set_executor, installed before apply()).
    auto &nodes = graph.nodes();
    if (nodes.empty()) {
        return;
    }
    if (_require_executor && graph.executor() == nullptr) {
        note_skip("no parallel executor is installed, and under sequential replay a clone buys no width",
                  fmt::format("graph '{}'", graph.name()));
        return;
    }

    // ── Scan: per-tensor access sequence + local disqualifiers ──────────────
    std::unordered_map<TensorId, std::vector<Access>> accesses;
    // Why each disqualified tensor is left alone: the first reason found, which is what the skip
    // tally reports if the tensor turns out to be reused scratch.
    std::unordered_map<TensorId, char const *> disqualified;

    for (size_t i = 0; i < nodes.size(); ++i) {
        Node const &nd = nodes[i];

        // A nested body/branch touching the tensor makes local generation
        // analysis blind (its accesses interleave with ours only at the
        // control-flow node's granularity); leave such tensors alone.
        if (is_control_flow(nd.kind)) {
            auto [eff_in, eff_out] = graph.effective_io(nd);
            for (auto const raw : eff_in) {
                disqualified.emplace(graph.buffer_of(raw), "a control-flow node's body touches it");
            }
            for (auto const raw : eff_out) {
                disqualified.emplace(graph.buffer_of(raw), "a control-flow node's body touches it");
            }
            continue;
        }

        bool const lifecycle = is_lifecycle(nd.kind);
        for (auto const raw : nd.inputs) {
            TensorId const tid = graph.buffer_of(raw);
            if (raw != tid || lifecycle) {
                // A view access is a partial touch; a lifecycle node ties the
                // buffer's storage to this graph's schedule. Both make
                // renaming unsafe to reason about locally.
                disqualified.emplace(tid, lifecycle ? "an allocation or free node touches it" : "it is accessed through a view");
                continue;
            }
            accesses[tid].push_back({.node_idx = i, .starts_generation = false});
        }
        for (auto const raw : nd.outputs) {
            TensorId const tid = graph.buffer_of(raw);
            if (raw != tid || lifecycle) {
                disqualified.emplace(tid, lifecycle ? "an allocation or free node touches it" : "it is accessed through a view");
                continue;
            }
            bool const reads_self = std::find(nd.inputs.begin(), nd.inputs.end(), raw) != nd.inputs.end() || reads_destination(nd);
            accesses[tid].push_back({.node_idx = i, .starts_generation = !reads_self && pure_overwrite(nd)});
        }
    }

    // Deterministic candidate order (unordered_map iteration is not).
    std::vector<TensorId> candidates;
    candidates.reserve(accesses.size());
    for (auto const &[tid, accs] : accesses) {
        candidates.push_back(tid);
    }
    std::sort(candidates.begin(), candidates.end());

    size_t const cap = _max_copies != 0 ? _max_copies : std::max<size_t>(2, static_cast<size_t>(std::thread::hardware_concurrency()));

    bool changed = false;
    for (auto const tid : candidates) {
        auto const &accs = accesses[tid];
        // Only a tensor overwritten whole more than once is reused scratch. Anything else is an
        // input, a result or single-use scratch: declining it is not news, and it could not be
        // split anyway, so it is passed over without a word in the skip tally.
        auto const overwrites = std::ranges::count_if(accs, [](Access const &a) { return a.starts_generation; });
        if (overwrites < 2) {
            continue;
        }

        auto const *handle = graph.find_tensor(tid);
        auto const  skip   = [&](std::string_view why) {
            note_skip(why, fmt::format("tensor '{}'", handle != nullptr ? handle->name : fmt::format("id {}", tid)));
        };
        if (auto const d = disqualified.find(tid); d != disqualified.end()) {
            skip(d->second);
            continue;
        }
        if (handle == nullptr || !handle->impl_fn || handle->dims.empty() || !dispatchable_dtype(handle->dtype)) {
            skip("its storage or element type cannot be cloned");
            continue;
        }
        if (handle->aliases != 0) {
            skip("other tensors alias it");
            continue;
        }

        // A read before the first pure overwrite consumes a value carried in
        // from outside this graph (e.g. the previous loop iteration); the
        // tensor is not scratch here.
        if (!accs.front().starts_generation) {
            skip("it is read before its first whole overwrite, so it carries a value in");
            continue;
        }

        // Split into generations, deduplicating a node that touches the tensor
        // as both input and output within one generation.
        std::vector<std::vector<size_t>> gens;
        for (auto const &a : accs) {
            if (a.starts_generation) {
                gens.emplace_back();
            }
            if (gens.back().empty() || gens.back().back() != a.node_idx) {
                gens.back().push_back(a.node_idx);
            }
        }
        if (gens.size() < 2) {
            continue;
        }

        // Only the interior generations are renamed; the last stays on the
        // original tensor so every value observable after execution is
        // unchanged. All their nodes must be rebuildable.
        size_t const interior       = gens.size() - 1;
        bool         ok             = true;
        Node const  *not_understood = nullptr;
        for (size_t j = 0; j < interior && ok; ++j) {
            for (size_t const idx : gens[j]) {
                if (!understands(graph, nodes[idx])) {
                    not_understood = &nodes[idx];
                    ok             = false;
                    break;
                }
                if (!rebuildable(graph, nodes[idx])) {
                    ok = false;
                    break;
                }
            }
        }
        if (not_understood != nullptr) {
            note_skip("the node carries a feature this pass does not understand",
                      fmt::format("node '{}': {}", not_understood->label, describe_features(features_of(graph, *not_understood))));
            continue;
        }
        if (!ok) {
            skip("a node in one of its interior generations cannot be rebuilt onto a clone");
            continue;
        }

        size_t const num_clones = std::min(interior, cap);
        // Copy before declaring: registering a clone rehashes the tensor map
        // and would dangle the handle pointer.
        std::string const     base_name = handle->name;
        TensorHandle const    proto     = *handle;
        std::vector<TensorId> clones;
        clones.reserve(num_clones);
        for (size_t k = 0; k < num_clones; ++k) {
            TensorId const clone = declare_clone(graph, proto, fmt::format("_sp_{}_{}", base_name, k));
            if (clone == 0) {
                break;
            }
            clones.push_back(clone);
        }
        if (clones.size() != num_clones) {
            skip("declaring a clone failed"); // leave the tensor untouched
            continue;
        }

        for (size_t j = 0; j < interior; ++j) {
            TensorId const target = clones[j % num_clones];
            for (size_t const idx : gens[j]) {
                rebuild_node(graph, nodes[idx], tid, target);
                ++_num_nodes_rebuilt;
            }
        }
        ++_num_tensors_privatized;
        _num_copies_created += num_clones;
        changed = true;

        if (_verbosity >= 2) {
            report(2,
                   fmt::format("'{}': {} generations onto {} clone(s), {} node(s) rebuilt", base_name, gens.size(), num_clones, interior));
        }
    }

    if (changed) {
        // Node order is unchanged (renaming only removes dependences), so the
        // existing order stays topologically valid; deps and cached analyses
        // are stale.
        graph.mark_sorted();
    }
}

std::vector<std::string> ScratchPrivatization::explain() const {
    if (num_tensors_privatized() == 0) {
        return {};
    }
    return {fmt::format("ScratchPrivatization: {} scratch tensor(s) split onto {} clone(s), {} node(s) rebuilt", num_tensors_privatized(),
                        num_copies_created(), num_nodes_rebuilt())};
}

EINSUMS_NAMESPACE_END(compute_graph::passes)
