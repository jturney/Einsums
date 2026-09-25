//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file Operations.cpp
/// @brief Making nodes and the closures that run them, and checking the operands.
///
/// The factories a pass calls when it rewrites a graph: create one tensor,
/// build one node, or build the `std::function` a node executes. They are here
/// rather than in `Execute.cpp` because they run at CAPTURE and rewrite time,
/// not at replay, and what they mostly do is type erasure, turning a dtype and a
/// rank into a concrete call.
///
/// @ref Graph::validate_tensors and @ref Graph::validate_shapes_at_capture end
/// the file because they answer the same question from the other side: given the
/// nodes these factories made, do the operands agree.

#include <Einsums/CXX23/Expected.hpp>
#include <Einsums/ComputeGraph/CaptureContext.hpp>
#include <Einsums/ComputeGraph/Detail/MixedPrecision.hpp>
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

void Graph::update_prefactors(NodeId node_id, PrefactorScalar c_pf, PrefactorScalar ab_pf) {
    for (auto &node : _nodes) {
        if (node.id != node_id) {
            continue;
        }
        auto *desc = node.op_data.get_if<EinsumDescriptor>();
        if (desc == nullptr) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                    "Graph '{}': node {} ({}) is not an einsum; update_prefactors only applies to einsum nodes", _name,
                                    node_id, node.kind);
        }
        // Keep both prefactor sources in sync: the descriptor snapshot
        // (read by GPU dispatch and analysis passes) and the shared
        // EinsumParams the CPU executor lambda reads live. The descriptor
        // owns its params handle, so this stays correct when passes
        // reorder or remove nodes.
        desc->c_prefactor  = c_pf;
        desc->ab_prefactor = ab_pf;
        if (desc->params) {
            desc->params->c_pf  = c_pf;
            desc->params->ab_pf = ab_pf;
        }
        // Prefactors appear in the cached profiler annotations.
        _profile_strings_valid = false;
        return;
    }
    EINSUMS_THROW_EXCEPTION(std::out_of_range, "Graph '{}': no node with id {}", _name, node_id);
}

expected<std::pair<TensorId, void *>, GraphError> Graph::create_tensor_dynamic(std::string name, packed_gemm::ScalarType dtype,
                                                                               std::vector<size_t> const &dims) {
    if (dims.empty()) {
        return unexpected(GraphError::type_error("create_tensor_dynamic: dims must not be empty"));
    }

    // Typed-tensor create-by-rank dispatch. Caps at rank 8 because the
    // typed Tensor<T, K> family requires a compile-time switch case per
    // rank; passes that consume the void* result static_cast it back to
    // Tensor<T, K> (e.g. DistributiveFactoring's slot-redirect trick),
    // so we can't transparently substitute RuntimeTensor here. Callers
    // that need higher ranks or want a single runtime-rank surface should
    // use Graph::create_runtime_tensor / create_zero_runtime_tensor
    // directly.
    auto make = [&]<typename T>(T /*tag*/) -> expected<std::pair<TensorId, void *>, GraphError> {
        switch (dims.size()) {
        case 1: {
            auto &t = create_zero_tensor<T, 1>(std::move(name), dims[0]);
            return std::pair{find_tensor_id_by_ptr(&t), static_cast<void *>(&t)};
        }
        case 2: {
            auto &t = create_zero_tensor<T, 2>(std::move(name), dims[0], dims[1]);
            return std::pair{find_tensor_id_by_ptr(&t), static_cast<void *>(&t)};
        }
        case 3: {
            auto &t = create_zero_tensor<T, 3>(std::move(name), dims[0], dims[1], dims[2]);
            return std::pair{find_tensor_id_by_ptr(&t), static_cast<void *>(&t)};
        }
        case 4: {
            auto &t = create_zero_tensor<T, 4>(std::move(name), dims[0], dims[1], dims[2], dims[3]);
            return std::pair{find_tensor_id_by_ptr(&t), static_cast<void *>(&t)};
        }
        case 5: {
            auto &t = create_zero_tensor<T, 5>(std::move(name), dims[0], dims[1], dims[2], dims[3], dims[4]);
            return std::pair{find_tensor_id_by_ptr(&t), static_cast<void *>(&t)};
        }
        case 6: {
            auto &t = create_zero_tensor<T, 6>(std::move(name), dims[0], dims[1], dims[2], dims[3], dims[4], dims[5]);
            return std::pair{find_tensor_id_by_ptr(&t), static_cast<void *>(&t)};
        }
        case 7: {
            auto &t = create_zero_tensor<T, 7>(std::move(name), dims[0], dims[1], dims[2], dims[3], dims[4], dims[5], dims[6]);
            return std::pair{find_tensor_id_by_ptr(&t), static_cast<void *>(&t)};
        }
        case 8: {
            auto &t = create_zero_tensor<T, 8>(std::move(name), dims[0], dims[1], dims[2], dims[3], dims[4], dims[5], dims[6], dims[7]);
            return std::pair{find_tensor_id_by_ptr(&t), static_cast<void *>(&t)};
        }
        default:
            return unexpected(GraphError::type_error(
                fmt::format("create_tensor_dynamic: unsupported rank {}; use create_runtime_tensor for higher ranks", dims.size())));
        }
    };

    switch (dtype) {
    case packed_gemm::ScalarType::Float32:
        return make(float{});
    case packed_gemm::ScalarType::Float64:
        return make(double{});
    case packed_gemm::ScalarType::Complex64:
        return make(std::complex<float>{});
    case packed_gemm::ScalarType::Complex128:
        return make(std::complex<double>{});
    default:
        return unexpected(GraphError::type_error("create_tensor_dynamic: unknown ScalarType"));
    }
}

// ── Runtime dispatch helpers for type-erased operations ────────────────────

namespace {

/// Dispatch a binary operation on two tensors with matching dtype and rank.
/// The Fn receives typed pointers: fn(Tensor<T,Rank>*, Tensor<T,Rank>*)
/// Dispatch a binary operation on two tensors.
///
/// Both operands are reached through @ref TensorHandle::live_ptr rather than
/// ``tensor_ptr``. These helpers back the ``make_*_executor`` family, which
/// resolves its operands by id at REPLAY, and by then the caller's wrapper may
/// legally be gone: capture's whole contract is that an operand's wrapper may
/// be destroyed before ``execute()``. Reading the identity pointer there is a
/// use-after-free, and it was one, silently, for every pass-built axpy.
template <typename Fn>
void dispatch_binary(TensorHandle const &a, TensorHandle const &b, Fn &&fn) {
    if (a.dtype != b.dtype || a.rank != b.rank) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "dispatch_binary: dtype or rank mismatch");
    }
    // A runtime tensor's storage layout differs from Tensor<T, Rank>; casting one
    // handle's pointer with the other's shape is type confusion. Both operands
    // must be the same kind (callers gate on is_runtime, so this only guards misuse).
    if (a.is_runtime != b.is_runtime) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "dispatch_binary: cannot mix runtime and compile-time tensors");
    }

    auto go = [&]<typename T>(T /*tag*/) {
        // GeneralRuntimeTensor<T> carries its rank dynamically, so one cast covers
        // every rank; branch on the handle kind before the compile-time rank switch.
        if (a.is_runtime) {
            using RT = GeneralRuntimeTensor<T, std::allocator<T>>;
            fn(static_cast<RT *>(a.live_ptr()), static_cast<RT *>(b.live_ptr()));
            return;
        }
        detail::dispatch_by_rank(a.rank, [&](auto rank_tag) {
            constexpr std::size_t K = decltype(rank_tag)::value;
            fn(static_cast<Tensor<T, K> *>(a.live_ptr()), static_cast<Tensor<T, K> *>(b.live_ptr()));
        });
    };

    detail::dispatch_scalar_type(a.dtype, go);
}

/// Dispatch a unary operation on one tensor.
template <typename Fn>
void dispatch_unary(TensorHandle const &a, Fn &&fn) {
    auto go = [&]<typename T>(T /*tag*/) {
        // See dispatch_binary: a runtime handle casts to GeneralRuntimeTensor<T>
        // (rank carried dynamically) rather than the compile-time Tensor<T, Rank>.
        if (a.is_runtime) {
            using RT = GeneralRuntimeTensor<T, std::allocator<T>>;
            fn(static_cast<RT *>(a.live_ptr()));
            return;
        }
        detail::dispatch_by_rank(a.rank, [&](auto rank_tag) {
            constexpr std::size_t K = decltype(rank_tag)::value;
            fn(static_cast<Tensor<T, K> *>(a.live_ptr()));
        });
    };

    detail::dispatch_scalar_type(a.dtype, go);
}

} // namespace

std::function<void()> Graph::make_axpy_executor(double alpha, TensorId src_id, TensorId dst_id) {
    return [this, alpha, src_id, dst_id]() {
        auto const &src = tensor(src_id);
        auto       &dst = tensor(dst_id);
        dispatch_binary(src, dst, [alpha](auto *s, auto *d) {
            using T = typename std::remove_pointer_t<decltype(s)>::ValueType;
            linear_algebra::axpy(static_cast<T>(alpha), *s, d);
        });
    };
}

std::function<void()> Graph::make_axpby_executor(std::shared_ptr<AxpbyParams> params, TensorId src_id, TensorId dst_id) {
    // Reads the scalars from the shared params on every replay, so a node built
    // with this executor can carry a real AxpbyDescriptor: passes rewrite the
    // params and the replay honors them. The beta == 1 case keeps the BLAS axpy
    // fast path - the common one, since accumulation is what pass-built nodes of
    // this shape are for.
    return [this, params = std::move(params), src_id, dst_id]() {
        auto const &src = tensor(src_id);
        auto       &dst = tensor(dst_id);
        dispatch_binary(src, dst, [&params](auto *s, auto *d) {
            using T          = typename std::remove_pointer_t<decltype(s)>::ValueType;
            auto const alpha = as<T>(params->alpha);
            auto const beta  = as<T>(params->beta);
            if (beta == T{1}) {
                linear_algebra::axpy(alpha, *s, d);
            } else {
                linear_algebra::axpby(alpha, *s, beta, d);
            }
        });
    };
}

std::function<void()> Graph::make_copy_executor(TensorId src_id, TensorId dst_id) {
    return [this, src_id, dst_id]() {
        auto const &src = tensor(src_id);
        auto       &dst = tensor(dst_id);
        dispatch_binary(src, dst, [](auto *s, auto *d) {
            // Element-by-element copy (works for any rank)
            size_t const n  = s->size();
            auto        *sp = s->data();
            auto        *dp = d->data();
            std::memcpy(dp, sp, n * sizeof(*sp));
        });
    };
}

expected<std::pair<TensorId, void *>, GraphError> Graph::create_zero_runtime_tensor_dynamic(std::string name, packed_gemm::ScalarType dtype,
                                                                                            std::vector<size_t> const &dims) {
    if (dims.empty()) {
        return unexpected(GraphError::type_error("create_zero_runtime_tensor_dynamic: dims must not be empty"));
    }

    auto make = [&]<typename T>(T /*tag*/) -> std::pair<TensorId, void *> {
        auto &t = create_zero_runtime_tensor<T, std::allocator<T>>(std::move(name), dims, /*intermediate=*/true);
        return {find_tensor_id_by_ptr(&t), static_cast<void *>(&t)};
    };

    switch (dtype) {
    case packed_gemm::ScalarType::Float32:
        return make(float{});
    case packed_gemm::ScalarType::Float64:
        return make(double{});
    case packed_gemm::ScalarType::Complex64:
        return make(std::complex<float>{});
    case packed_gemm::ScalarType::Complex128:
        return make(std::complex<double>{});
    default:
        return unexpected(GraphError::type_error("create_zero_runtime_tensor_dynamic: unknown ScalarType"));
    }
}

std::function<void()> Graph::make_gemm_executor(TensorId a_id, TensorId b_id, TensorId c_id, double alpha, double beta) {
    std::array<TensorId, 2> const inputs{a_id, b_id};

    OpData const op_data(GemmDescriptor{.alpha = PrefactorScalar{alpha}, .beta = PrefactorScalar{beta}, .trans_a = 'n', .trans_b = 'n'});

    return build_executor(OpKind::Gemm, tensor(a_id).dtype, 2, op_data, *this, inputs, std::span<TensorId const>{&c_id, 1});
}

Node Graph::make_einsum_node(TensorId a_id, TensorId b_id, TensorId c_id, ParsedEinsumSpec const &spec, PrefactorScalar c_pf,
                             PrefactorScalar ab_pf, bool conj_a, bool conj_b, std::string label) {
    auto const &a_h = tensor(a_id);
    auto const &b_h = tensor(b_id);
    auto const &c_h = tensor(c_id);

    // Every operand must expose a rank-erased TensorImpl. That covers runtime
    // tensors AND statically typed Tensor<T, Rank>: the impl carries data, dims and
    // strides as runtime values, so one dtype dispatch serves every rank and no
    // static-rank cast is needed -- which is what used to restrict this to runtime
    // tensors. Only tile-wise sparse tensors lack an impl; they have no single
    // buffer to contract over, so a pass must not route them here.
    if (!a_h.impl_fn || !b_h.impl_fn || !c_h.impl_fn) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                "Graph::make_einsum_node: every operand needs a rank-erased impl (a={} b={} c={}); tile-wise sparse "
                                "tensors have none and cannot be contracted through this path",
                                static_cast<bool>(a_h.impl_fn), static_cast<bool>(b_h.impl_fn), static_cast<bool>(c_h.impl_fn));
    }
    if (a_h.dtype != c_h.dtype || b_h.dtype != c_h.dtype) {
        // Operands of different element types contract through the mixed-precision generic loop,
        // whose executor reads each operand's type from its accessor. What cannot store its result
        // is rejected here, where the node is built, rather than when it first runs: a complex
        // contraction or a complex prefactor into a real output, and the permutation operators the
        // mixed loop does not implement.
        detail::check_mixed_einsum(c_h.dtype, a_h.dtype, b_h.dtype, !is_real_valued(c_pf) || !is_real_valued(ab_pf),
                                   "Graph::make_einsum_node");
        if (!spec.operators.empty()) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                    "Graph::make_einsum_node: '{}' has permutation operators, which are not supported when the "
                                    "operands' element types differ",
                                    spec.raw);
        }
    }
    auto const dtype = c_h.dtype;

    // Live state, shared with the executor. Passes mutate these; the executor
    // dereferences them on every call, so a rewrite lands on the next execute.
    auto params    = std::make_shared<EinsumParams>();
    params->c_pf   = c_pf;
    params->ab_pf  = ab_pf;
    params->conj_a = conj_a;
    params->conj_b = conj_b;

    auto indices          = std::make_shared<EinsumIndices>();
    indices->spec         = spec;
    indices->link_indices = spec.link_indices();

    auto desc    = detail::build_einsum_descriptor(spec, c_pf, ab_pf, conj_a, conj_b);
    desc.params  = params;
    desc.indices = indices;

    // Same derivation the capture path uses, from the operand handles this node was handed. A
    // pass that rebuilds a node must not drop its letter map, and re-deriving it (rather than
    // copying the old node's) is what keeps the map right when the pass changed the operands.
    desc.letter_spaces =
        detail::bind_einsum_spaces(*this, a_id, b_id, c_id, spec.a_indices, spec.b_indices, spec.c_indices, "Graph::make_einsum_node");

    // BLAS batching hint. One derivation, shared with the capture path in
    // Operations.hpp: the gate, the m/n/k arithmetic and the roles clause used
    // to be duplicated here, and two copies of a rule whose failures are
    // invisible until GEMMBatching forms a batch is one copy too many.
    //
    // Building this at pass time is no weaker than building it at capture. The
    // dims come from the same place either way, and GEMMBatching consumes the
    // hint BEFORE DistributionPlanning and Materialization run, so a captured
    // hint on graph-owned scratch is derived from the same shell geometry this
    // is. (A deferred shell carries valid dims and strides; only data() is null.)
    desc.gemm_hint = derive_gemm_hint(dtype, desc.spec, *this, a_id, b_id, c_id);

    Node node;
    node.id    = reserve_node_id();
    node.kind  = OpKind::Einsum;
    node.label = label.empty() ? fmt::format("einsum({} <- {} ; {})", fmt::join(spec.c_indices, ","), fmt::join(spec.a_indices, ","),
                                             fmt::join(spec.b_indices, ","))
                               : std::move(label);
    // RMW convention: a nonzero output prefactor means the node READS its output,
    // so the output must appear as an input too or the schedulers and the liveness
    // passes cannot see the accumulation ordering (bug-1009).
    node.inputs  = is_zero(c_pf) ? std::vector<TensorId>{a_id, b_id} : std::vector<TensorId>{a_id, b_id, c_id};
    node.outputs = {c_id};

    // This node's packed-GEMM memo (see packed_gemm::ContractionSite). One per
    // node, so per-tile nodes from a tiled expansion never share one and a
    // parallel executor needs no synchronization around it. Dtype-agnostic:
    // the key records the scalar type, so a rebind to another dtype misses.
    // Set before the executor is built, because the builder adopts it from the
    // descriptor: that is what lets a plan-time pass pin this node's kernel
    // route where the dispatch will read it.
    desc.site = std::make_shared<packed_gemm::ContractionSite>();

    // One lowering, shared with capture and with a future loader: the executor
    // is derived from (kind, dtype, rank, descriptor, operand ids) and nothing
    // else (design part 3.2). It resolves operands through the graph's slots,
    // so rebind() and redirect_slot() are honored, and reads the descriptor's
    // live params and indices, so a pass that rewrites a prefactor or an index
    // list takes effect on the next execute rather than being silently ignored
    // (the desync class of bug-1002).
    node.op_data = std::move(desc);
    node.execute = build_executor(OpKind::Einsum, dtype, c_h.rank, node.op_data, *this, std::span<TensorId const>{node.inputs},
                                  std::span<TensorId const>{node.outputs});
    return node;
}

Node Graph::make_axpby_node(TensorId x, TensorId y, PrefactorScalar alpha, PrefactorScalar beta, std::string label) {
    // Live scalars shared with the executor, the same contract a captured axpby
    // has: the descriptor is what downstream passes read AND what the executor
    // uses, so a fold into alpha reaches the replay. A descriptor the executor
    // ignored would be worse than none.
    auto params   = std::make_shared<AxpbyParams>();
    params->alpha = alpha;
    params->beta  = beta;

    AxpbyDescriptor desc;
    desc.alpha  = params->alpha;
    desc.beta   = params->beta;
    desc.params = params;

    Node node;
    node.id      = reserve_node_id();
    node.kind    = OpKind::Axpby;
    node.label   = std::move(label);
    node.inputs  = {x, y};
    node.outputs = {y};
    node.op_data = std::move(desc);
    node.execute = make_axpby_executor(std::move(params), x, y);
    return node;
}

Node Graph::make_permute_node(TensorId a_id, TensorId c_id, ParsedPermuteSpec const &spec, PrefactorScalar alpha, PrefactorScalar beta,
                              std::string label) {
    auto const &a_h = tensor(a_id);
    auto const &c_h = tensor(c_id);

    // The rank-erased impl gate make_einsum_node states, for the same reason:
    // only tile-wise sparse tensors lack one, and they have no single buffer to
    // permute.
    if (!a_h.impl_fn || !c_h.impl_fn) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                "Graph::make_permute_node: both operands need a rank-erased impl (a={} c={}); tile-wise sparse "
                                "tensors have none",
                                static_cast<bool>(a_h.impl_fn), static_cast<bool>(c_h.impl_fn));
    }
    if (a_h.dtype != c_h.dtype) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "Graph::make_permute_node: operand dtypes disagree; a permute does not convert");
    }
    auto const dtype = c_h.dtype;

    // Live scalars shared with the executor, the contract make_axpby_node states:
    // the descriptor is both what passes read and what the replay uses, so a
    // later fold into alpha reaches execute() rather than being ignored.
    auto params   = std::make_shared<ElementwiseParams>();
    params->alpha = alpha;
    params->beta  = beta;

    PermuteDescriptor desc;
    desc.alpha     = as<std::complex<double>>(alpha);
    desc.beta      = as<std::complex<double>>(beta);
    desc.c_indices = spec.c_indices;
    desc.a_indices = spec.a_indices;
    desc.operators = spec.operators;
    desc.params    = params;

    Node node;
    node.id   = reserve_node_id();
    node.kind = OpKind::Permute;
    node.label =
        label.empty() ? fmt::format("permute({} <- {})", fmt::join(spec.c_indices, ","), fmt::join(spec.a_indices, ",")) : std::move(label);
    // One input even when beta is nonzero. See the note on the declaration: this
    // matches what capturing a cg::permute produces, and four passes plus
    // build_executor gate on a permute having exactly one input.
    node.inputs  = {a_id};
    node.outputs = {c_id};
    node.op_data = std::move(desc);
    node.execute = build_executor(OpKind::Permute, dtype, c_h.rank, node.op_data, *this, std::span<TensorId const>{node.inputs},
                                  std::span<TensorId const>{node.outputs});
    return node;
}

std::function<void()> Graph::make_zero_executor(TensorId tensor_id) {
    return [this, tensor_id]() {
        auto &h = tensor(tensor_id);
        dispatch_unary(h, [](auto *t) { t->zero(); });
    };
}

expected<void, GraphError> Graph::validate_tensors() const {
    // Lazily built: TensorIds and tensor_ptrs that some Materialize node in
    // this graph will bring to life during execution. (tensor_ptr matters for
    // the hoist/dedup case where the node carries a different TensorId than
    // the handle being checked but targets the same underlying buffer.)
    bool                             materialize_targets_built = false;
    std::unordered_set<TensorId>     materialize_tids;
    std::unordered_set<void const *> materialize_ptrs;
    std::unordered_set<TensorId>     referenced_tids;

    for (auto const &[id, handle] : _tensors) {
        if (handle.alloc_state == AllocState::Deferred) {
            // The snapshot says "deferred", but the user may have called
            // tensor.materialize() directly since registration - ask the
            // tensor itself when possible.
            if (handle.is_materialized_fn && handle.is_materialized_fn())
                continue;
            if (!materialize_targets_built) {
                materialize_targets_built = true;
                for (auto const &node : _nodes) {
                    for (auto in : node.inputs)
                        referenced_tids.insert(in);
                    for (auto out : node.outputs)
                        referenced_tids.insert(out);
                    if (node.kind != OpKind::Materialize)
                        continue;
                    for (auto out : node.outputs) {
                        materialize_tids.insert(out);
                        if (auto it = _tensors.find(out); it != _tensors.end() && it->second.tensor_ptr != nullptr) {
                            materialize_ptrs.insert(it->second.tensor_ptr);
                        }
                    }
                }
                // A Materialize can also live inside a sub-graph, and one that does still
                // brings the buffer to life before anything here reads it. This used never
                // to happen, because Materialization HOISTS a body tensor's lifecycle into
                // the parent; a setup body is the exception, since its lifecycle has to be
                // skipped on the replays that skip the fitting. Matched by tensor_ptr, which
                // is the identity two graphs share; ids are per-graph and would not.
                // NOLINTNEXTLINE(misc-no-recursion): sub-graphs nest, so the walk over them does too.
                std::function<void(Graph const &)> collect_sub = [&](Graph const &sub) {
                    for (auto const &node : sub._nodes) {
                        if (node.kind != OpKind::Materialize) {
                            continue;
                        }
                        for (auto out : node.outputs) {
                            if (auto it = sub._tensors.find(out); it != sub._tensors.end() && it->second.tensor_ptr != nullptr) {
                                materialize_ptrs.insert(it->second.tensor_ptr);
                            }
                        }
                    }
                    sub.for_each_subgraph(collect_sub);
                };
                for_each_subgraph(collect_sub);
            }
            // A deferred handle no node references cannot corrupt execution.
            // These exist by design: effective_io registers orphan parent
            // handles for buffers living only inside Loop/Conditional bodies
            // (the body's own handle gets the hoisted Materialize; the parent
            // orphan is just an id anchor for dependency edges).
            if (!referenced_tids.contains(id))
                continue;
            if (materialize_tids.contains(id))
                continue;
            if (handle.tensor_ptr != nullptr && materialize_ptrs.contains(handle.tensor_ptr))
                continue;
            return unexpected(GraphError::validation(
                fmt::format("Graph '{}': tensor '{}' (id={}) is still deferred - it was declared (declare_tensor / "
                            "declare_runtime_tensor) but never given backing storage, and no Materialize node exists for it. "
                            "Run the Materialization pass (graph.optimize() or PassManager) or Workspace::materialize_all() "
                            "before execute().",
                            _name, handle.name, id)));
        }
        if (handle.validator && !handle.validator()) {
            return unexpected(
                GraphError::validation(fmt::format("Graph '{}': tensor '{}' (id={}) appears to have been destroyed. "
                                                   "Ensure all tensors outlive the graph, or use graph.create_tensor() for intermediates.",
                                                   _name, handle.name, id)));
        }
    }

    return {};
}

void Graph::validate_shapes_at_capture() const {
    // Verify tensor ranks match index counts for Einsum nodes
    for (auto const &node : _nodes) {
        if (node.kind != OpKind::Einsum)
            continue;

        auto *desc = node.op_data.get_if<EinsumDescriptor>();
        if (!desc)
            continue;

        // Check each input tensor's rank matches its index count
        for (size_t inp = 0; inp < node.inputs.size() && inp < 2; inp++) {
            auto it = _tensors.find(node.inputs[inp]);
            if (it == _tensors.end())
                continue;

            auto const &handle        = it->second;
            size_t      expected_rank = (inp == 0) ? desc->spec.a_indices.size() : desc->spec.b_indices.size();

            if (handle.rank != 0 && handle.rank != expected_rank) {
                EINSUMS_THROW_EXCEPTION(std::runtime_error,
                                        "Graph '{}': shape mismatch in node '{}': "
                                        "input tensor '{}' has rank {} but {} indices specified",
                                        _name, node.label, handle.name, handle.rank, expected_rank);
            }
        }

        // Check output tensor rank matches C index count
        if (!node.outputs.empty()) {
            auto it = _tensors.find(node.outputs[0]);
            if (it != _tensors.end()) {
                auto const &handle        = it->second;
                size_t      expected_rank = desc->spec.c_indices.size();
                // Scalar output ("<- ij ; ij") carries no indices, and the
                // dispatch writes the result through C->data()[0]. The sink
                // for that is a one-element rank-1 tensor, which is the
                // convention every scalar-writing op here uses, so a rank
                // ABOVE the index count is allowed when the whole output holds
                // one element. Without this the contraction ran eagerly but
                // could not be captured.
                bool const scalar_sink = expected_rank == 0 && handle.total_elems() == 1;
                if (handle.rank != 0 && handle.rank != expected_rank && !scalar_sink) {
                    EINSUMS_THROW_EXCEPTION(std::runtime_error,
                                            "Graph '{}': shape mismatch in node '{}': "
                                            "output tensor '{}' has rank {} but {} indices specified",
                                            _name, node.label, handle.name, handle.rank, expected_rank);
                }
            }
        }
    }
}

EINSUMS_NAMESPACE_END(compute_graph)
