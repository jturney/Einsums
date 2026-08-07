//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/Detail/ScalarDispatch.hpp>
#include <Einsums/ComputeGraph/EinsumSpec.hpp>
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
#include <thread>
#include <unordered_map>
#include <unordered_set>
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
        auto const *d = std::get_if<EinsumDescriptor>(&nd.op_data);
        return d != nullptr && d->params != nullptr && d->indices != nullptr && nd.inputs.size() >= 2 && nd.inputs.size() <= 3 &&
               nd.outputs.size() == 1 && operands_resolve();
    }
    case OpKind::Permute: {
        // Exactly one input and one output: a SymmetrizedAccumulation-rewritten
        // permute accumulates into a second input and its descriptor no longer
        // matches its baked executor, so it must not be rebuilt from it.
        auto const *d = std::get_if<PermuteDescriptor>(&nd.op_data);
        return d != nullptr && nd.inputs.size() == 1 && nd.outputs.size() == 1 && operands_resolve();
    }
    case OpKind::Axpby: {
        // The live params are what the captured executor reads; a descriptor
        // without them cannot be trusted to match the lambda.
        auto const *d = std::get_if<AxpbyDescriptor>(&nd.op_data);
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
        auto const    *d = std::get_if<EinsumDescriptor>(&nd.op_data);
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

    if (nd.kind == OpKind::Permute) {
        auto const    *d = std::get_if<PermuteDescriptor>(&nd.op_data);
        TensorId const a = sub(nd.inputs[0]);
        TensorId const c = sub(nd.outputs[0]);

        ParsedPermuteSpec pspec;
        pspec.c_indices = d->c_indices;
        pspec.a_indices = d->a_indices;
        pspec.raw       = fmt::format("{} <- {}", fmt::join(d->c_indices, ","), fmt::join(d->a_indices, ","));

        // PrefactorScalar, not the raw complex<double>: `as<T>` narrows to the
        // element type exactly and throws rather than silently dropping a
        // non-zero imaginary part into a real permute.
        PrefactorScalar const alpha{d->alpha};
        PrefactorScalar const beta{d->beta};
        auto const            dtype = graph.tensor(c).dtype;

        Graph *g   = &graph;
        nd.execute = [g, a, c, pspec = std::move(pspec), alpha, beta, dtype]() {
            detail::dispatch_scalar_type(dtype, [&]<typename T>(T /*tag*/) {
                using Impl = ::einsums::detail::TensorImpl<T>;
                RuntimeTensorView<T> const A{*static_cast<Impl *>(g->tensor(a).impl_fn())};
                RuntimeTensorView<T>       C{*static_cast<Impl *>(g->tensor(c).impl_fn())};
                dispatch::string_permute(pspec, as<T>(beta), &C, as<T>(alpha), A);
            });
        };
        nd.inputs  = {a};
        nd.outputs = {c};
        return;
    }

    // Axpby: Y = alpha*X + beta*Y. Keep the SHARED params object so later
    // scalar rewrites (ScaleAbsorption-style) still reach this executor.
    auto const    *d      = std::get_if<AxpbyDescriptor>(&nd.op_data);
    TensorId const x      = sub(nd.inputs[0]);
    TensorId const y      = sub(nd.outputs[0]);
    auto           params = d->params;
    auto const     dtype  = graph.tensor(y).dtype;

    Graph *g   = &graph;
    nd.execute = [g, x, y, params, dtype]() {
        detail::dispatch_scalar_type(dtype, [&]<typename T>(T /*tag*/) {
            using Impl = ::einsums::detail::TensorImpl<T>;
            RuntimeTensorView<T> const X{*static_cast<Impl *>(g->tensor(x).impl_fn())};
            RuntimeTensorView<T>       Y{*static_cast<Impl *>(g->tensor(y).impl_fn())};
            linear_algebra::axpby(as<T>(params->alpha), X, as<T>(params->beta), &Y);
        });
    };
    // Preserve the RMW convention of the original lists (beta != 0 lists Y as
    // an input too); only the ids change.
    for (auto &tid : nd.inputs) {
        tid = sub(tid);
    }
    nd.outputs = {y};
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
    for (auto const &[tid, h] : graph.tensors_map()) {
        if (h.tensor_ptr == ptr) {
            return tid;
        }
    }
    return 0;
}

} // namespace

void ScratchPrivatization::reset_stats() {
    _num_tensors_privatized = 0;
    _num_copies_created     = 0;
    _num_nodes_rebuilt      = 0;
}

bool ScratchPrivatization::run(Graph &graph) {
    // Per-apply counters: compare against entry values, not zero (the recursive
    // driver calls run() once and reset_stats() once per apply).
    size_t const rebuilt_at_entry = _num_nodes_rebuilt;
    run_recursive(graph);
    return _num_nodes_rebuilt > rebuilt_at_entry;
}

void ScratchPrivatization::run_recursive(Graph &graph) {
    for (auto &node : graph.nodes()) {
        if (auto *loop = std::get_if<LoopDescriptor>(&node.op_data); loop != nullptr && loop->body) {
            run_recursive(*loop->body);
        } else if (auto *cond = std::get_if<ConditionalDescriptor>(&node.op_data); cond != nullptr) {
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
    if (_require_executor && graph.executor() == nullptr) {
        return;
    }
    auto &nodes = graph.nodes();
    if (nodes.empty()) {
        return;
    }

    // ── Scan: per-tensor access sequence + local disqualifiers ──────────────
    std::unordered_map<TensorId, std::vector<Access>> accesses;
    std::unordered_set<TensorId>                      disqualified;

    for (size_t i = 0; i < nodes.size(); ++i) {
        Node const &nd = nodes[i];

        // A nested body/branch touching the tensor makes local generation
        // analysis blind (its accesses interleave with ours only at the
        // control-flow node's granularity); leave such tensors alone.
        if (is_control_flow(nd.kind)) {
            auto [eff_in, eff_out] = graph.effective_io(nd);
            for (auto const raw : eff_in) {
                disqualified.insert(graph.resolve_alias(raw));
            }
            for (auto const raw : eff_out) {
                disqualified.insert(graph.resolve_alias(raw));
            }
            continue;
        }

        bool const lifecycle = is_lifecycle(nd.kind);
        for (auto const raw : nd.inputs) {
            TensorId const tid = graph.resolve_alias(raw);
            if (raw != tid || lifecycle) {
                // A view access is a partial touch; a lifecycle node ties the
                // buffer's storage to this graph's schedule. Both make
                // renaming unsafe to reason about locally.
                disqualified.insert(tid);
                continue;
            }
            accesses[tid].push_back({.node_idx = i, .starts_generation = false});
        }
        for (auto const raw : nd.outputs) {
            TensorId const tid = graph.resolve_alias(raw);
            if (raw != tid || lifecycle) {
                disqualified.insert(tid);
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
        if (disqualified.contains(tid)) {
            continue;
        }
        auto const *handle = graph.find_tensor(tid);
        if (handle == nullptr || handle->aliases != 0 || !handle->impl_fn || handle->dims.empty() || !dispatchable_dtype(handle->dtype)) {
            continue;
        }

        auto const &accs = accesses[tid];
        // A read before the first pure overwrite consumes a value carried in
        // from outside this graph (e.g. the previous loop iteration); the
        // tensor is not scratch here.
        if (accs.empty() || !accs.front().starts_generation) {
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
        size_t const interior = gens.size() - 1;
        bool         ok       = true;
        for (size_t j = 0; j < interior && ok; ++j) {
            for (size_t const idx : gens[j]) {
                if (!rebuildable(graph, nodes[idx])) {
                    ok = false;
                    break;
                }
            }
        }
        if (!ok) {
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
            continue; // clone declaration failed; leave the tensor untouched
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
