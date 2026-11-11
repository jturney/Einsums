//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/Detail/ScalarDispatch.hpp>
#include <Einsums/ComputeGraph/Detail/ScalarTypes.hpp>
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

/// Can rebuild_node() below reproduce this node's computation with a renamed
/// operand? Rebuilding replaces the executor, so every operand must resolve
/// through the graph (impl_fn) and the descriptor must actually describe what
/// the baked lambda computes.
bool rebuildable(Graph &graph, Node const &nd) {
    auto const operands_resolve = [&]() {
        for (auto const tid : nd.inputs) {
            auto const *h = graph.find_tensor(tid);
            if (h == nullptr || !h->impl_fn || !compute_graph::detail::is_dispatchable(h->dtype)) {
                return false;
            }
        }
        for (auto const tid : nd.outputs) {
            auto const *h = graph.find_tensor(tid);
            if (h == nullptr || !h->impl_fn || !compute_graph::detail::is_dispatchable(h->dtype)) {
                return false;
            }
        }
        return true;
    };

    switch (nd.kind) {
    case OpKind::Einsum: {
        auto const *d = nd.op_data.get_if<EinsumDescriptor>();
        return d != nullptr && nd.inputs.size() >= 2 && nd.inputs.size() <= 3 && nd.outputs.size() == 1 && operands_resolve();
    }
    case OpKind::Permute: {
        // The source, then the destination when the permute accumulates into it.
        auto const *d = nd.op_data.get_if<PermuteDescriptor>();
        return d != nullptr && (nd.inputs.size() == 1 || nd.inputs.size() == 2) && nd.outputs.size() == 1 && operands_resolve();
    }
    case OpKind::Axpby: {
        auto const *d = nd.op_data.get_if<AxpbyDescriptor>();
        return d != nullptr && (nd.inputs.size() == 1 || nd.inputs.size() == 2) && nd.outputs.size() == 1 && operands_resolve();
    }
    default:
        return false;
    }
}

/// Replace @p old_id with @p new_id in the operand lists of node @p index; the edit rebuilds its
/// executor so the computation follows. Caller guarantees rebuildable().
void rebuild_node(Graph &graph, std::size_t index, TensorId old_id, TensorId new_id) {
    auto const sub = [&](TensorId tid) { return tid == old_id ? new_id : tid; };

    Node const &nd = graph.nodes()[index];
    if (nd.kind == OpKind::Einsum) {
        // The descriptor is re-derived for the new operands: its letter spaces and GEMM hint name
        // them.
        auto const *d = nd.op_data.get_if<EinsumDescriptor>();
        Node rebuilt  = graph.make_einsum_node(sub(nd.inputs[0]), sub(nd.inputs[1]), sub(nd.outputs[0]), detail::einsum_indices_of(*d).spec,
                                               d->c_prefactor, d->ab_prefactor, d->conj_a, d->conj_b, nd.label);
        graph.edit(index, [&](NodeEdit &e) {
            e->inputs  = std::move(rebuilt.inputs);
            e->outputs = std::move(rebuilt.outputs);
            e->op_data = std::move(rebuilt.op_data);
        });
        return;
    }

    // Permute and Axpby: the descriptor already says everything the executor does, its live
    // params included, so the node keeps it and only the operand ids change. Only the ids change
    // in the operand lists, so an accumulating axpby or permute keeps listing its destination as
    // an input.
    graph.edit(index, [&](NodeEdit &e) {
        for (auto &tid : e->inputs) {
            tid = sub(tid);
        }
        for (auto &tid : e->outputs) {
            tid = sub(tid);
        }
    });
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

bool ScratchPrivatization::transform(Graph &graph) {
    PassCounter const nodes_rebuilt{_stats.num_nodes_rebuilt};
    run_recursive(graph);
    return nodes_rebuilt.moved();
}

void ScratchPrivatization::run_recursive(Graph &graph) {
    graph.for_each_subgraph([this](Graph &child) { run_recursive(child); }, /*include_setup=*/false);
    privatize_one_graph(graph);
}

void ScratchPrivatization::privatize_one_graph(Graph &graph) {
    // Under the built-in sequential replay the clones cost cache locality and
    // buy no width, so only graphs that will actually run on a parallel
    // executor are rewritten (Graph::set_executor, installed before apply()).
    auto const &nodes = graph.nodes();
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
        if (handle == nullptr || !handle->impl_fn || handle->dims.empty() || !compute_graph::detail::is_dispatchable(handle->dtype)) {
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
            decline_unknown_features(graph, *not_understood);
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
                rebuild_node(graph, idx, tid, target);
                ++_stats.num_nodes_rebuilt;
            }
        }
        ++_stats.num_tensors_privatized;
        _stats.num_copies_created += num_clones;
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
