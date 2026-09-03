//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

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

namespace {

/// Try to dispatch a GPU node via gpu::blas (GEMM or GEMV).
/// Returns true if dispatched, false if not applicable (caller should use CPU fallback).
bool try_gpu_blas_dispatch(Node const &node, std::unordered_map<TensorId, TensorHandle> const &tensors, DeviceShadowMap &shadows);

/// Resolve the device-visible pointer for one operand. On unified memory the GPU
/// reads the host tensor's data directly; on a discrete device the data has been
/// copied into a shadow buffer keyed by tensor id. Shared by every try_gpu_*
/// helper below, which otherwise repeated this if-constexpr branch per operand.
[[nodiscard]] inline void *resolve_device_ptr([[maybe_unused]] TensorHandle const &h, [[maybe_unused]] TensorId id,
                                              [[maybe_unused]] DeviceShadowMap &shadows) {
    if constexpr (gpu::has_unified_memory) {
        return h.data_ptr;
    } else {
        return shadows.get(id);
    }
}

/// Try GEMM dispatch: 2 target indices + 1 link index, rank-2 tensors.
bool try_gpu_gemm(EinsumDescriptor const &desc, Node const &node, std::unordered_map<TensorId, TensorHandle> const &tensors,
                  DeviceShadowMap &shadows) {
    // Must be a standard GEMM pattern: 2 target indices, 1 link index.
    if (desc.spec.target_indices.size() != 2 || desc.spec.link_indices.size() != 1)
        return false;

    // Need exactly 1 output (C).
    if (node.outputs.size() != 1)
        return false;

    // C is the output. A and B are the non-C inputs.
    TensorId const c_id = node.outputs[0];
    TensorId       a_id = 0, b_id = 0;
    bool           found_a = false;

    for (auto tid : node.inputs) {
        if (tid == c_id)
            continue; // Skip C if it's also in inputs.
        if (!found_a) {
            a_id    = tid;
            found_a = true;
        } else {
            b_id = tid;
        }
    }

    if (!found_a || b_id == 0)
        return false;

    auto a_it = tensors.find(a_id);
    auto b_it = tensors.find(b_id);
    auto c_it = tensors.find(c_id);
    if (a_it == tensors.end() || b_it == tensors.end() || c_it == tensors.end())
        return false;

    auto const &ha = a_it->second;
    auto const &hb = b_it->second;
    auto const &hc = c_it->second;

    // Must all be rank-2 matrices.
    if (ha.rank != 2 || hb.rank != 2 || hc.rank != 2)
        return false;

    // Must all be the same dtype.
    if (ha.dtype != hb.dtype || ha.dtype != hc.dtype)
        return false;

    void *ptr_a = resolve_device_ptr(ha, a_id, shadows);
    void *ptr_b = resolve_device_ptr(hb, b_id, shadows);
    void *ptr_c = resolve_device_ptr(hc, c_id, shadows);
    if (!ptr_a || !ptr_b || !ptr_c)
        return false;

    // Determine M, N, K and transpose flags from the contraction pattern.
    // Standard einsum: C[i,j] = A[?,?] * B[?,?] where ? matches indices.
    // Column-major GEMM: C(M×N) = A(M×K) * B(K×N), with lda=M, ldb=K, ldc=M.
    //
    // From the spec:
    //   c_indices[0] = row index of C, c_indices[1] = col index of C
    //   We need to figure out if A or B is transposed based on where the indices appear.

    auto const &ci = desc.spec.c_indices;    // e.g., ["i", "j"]
    auto const &ai = desc.spec.a_indices;    // e.g., ["i", "k"]
    auto const &bi = desc.spec.b_indices;    // e.g., ["k", "j"]
    auto const &li = desc.spec.link_indices; // e.g., ["k"]

    if (ci.size() != 2 || ai.size() != 2 || bi.size() != 2)
        return false;

    // C is column-major: C[row, col] with dims[0]=rows, dims[1]=cols.
    // M = C rows, N = C cols.
    auto    M = static_cast<int64_t>(hc.dims[0]);
    auto    N = static_cast<int64_t>(hc.dims[1]);
    int64_t K = 0;

    // Find K from the link index dimension.
    std::string const &link = li[0];
    // K is the dimension of the link index in A (or B).
    for (size_t d = 0; d < 2; d++) {
        if (ai[d] == link) {
            K = static_cast<int64_t>(ha.dims[d]);
            break;
        }
    }
    if (K == 0)
        return false;

    // Determine transpose for A:
    // Column-major A: A[row, col]. If A's indices match [row_of_C, link] → no transpose.
    // If A's indices match [link, row_of_C] → transpose.
    char transa = 'n';
    if (ai[0] == link && ai[1] == ci[0]) {
        transa = 't'; // A is K×M stored, need M×K → transpose
    } else if (ai[0] == ci[0] && ai[1] == link) {
        transa = 'n'; // A is M×K stored → no transpose
    } else {
        return false; // Unrecognized pattern
    }

    // Determine transpose for B:
    char transb = 'n';
    if (bi[0] == ci[1] && bi[1] == link) {
        transb = 't'; // B is N×K stored, need K×N → transpose
    } else if (bi[0] == link && bi[1] == ci[1]) {
        transb = 'n'; // B is K×N stored → no transpose
    } else {
        return false;
    }

    auto lda = static_cast<int64_t>(ha.dims[0]); // leading dimension = rows in column-major
    auto ldb = static_cast<int64_t>(hb.dims[0]);
    auto ldc = static_cast<int64_t>(hc.dims[0]);

    // Dispatch based on dtype.
    if (ha.dtype == packed_gemm::ScalarType::Float32) {
        auto alpha = as<float>(desc.ab_prefactor);
        auto beta  = as<float>(desc.c_prefactor);
        gpu::blas::gemm<float>(transa, transb, M, N, K, alpha, static_cast<float const *>(ptr_a), lda, static_cast<float const *>(ptr_b),
                               ldb, beta, static_cast<float *>(ptr_c), ldc);
        return true;
    } else if (ha.dtype == packed_gemm::ScalarType::Float64) {
        auto alpha = as<double>(desc.ab_prefactor);
        auto beta  = as<double>(desc.c_prefactor);
        gpu::blas::gemm<double>(transa, transb, M, N, K, alpha, static_cast<double const *>(ptr_a), lda, static_cast<double const *>(ptr_b),
                                ldb, beta, static_cast<double *>(ptr_c), ldc);
        return true;
    }

    return false; // Complex or unsupported dtype
}

/// Try GEMV dispatch: 1 target index + 1 link index, rank-1 output (vector), rank-2 input (matrix).
bool try_gpu_gemv(EinsumDescriptor const &desc, Node const &node, std::unordered_map<TensorId, TensorHandle> const &tensors,
                  DeviceShadowMap &shadows) {
    // Must be: y[i] = A[i,k] * x[k] or y[i] = A[k,i] * x[k] (with transpose)
    if (desc.spec.target_indices.size() != 1 || desc.spec.link_indices.size() != 1)
        return false;

    if (node.outputs.size() != 1)
        return false;

    TensorId const y_id = node.outputs[0];

    // Find A (rank-2) and x (rank-1) among inputs.
    TensorId   a_id = 0, x_id = 0;
    bool const found = false;

    for (auto tid : node.inputs) {
        if (tid == y_id)
            continue;
        auto it = tensors.find(tid);
        if (it == tensors.end())
            continue;
        if (it->second.rank == 2 && a_id == 0) {
            a_id = tid;
        } else if (it->second.rank == 1 && x_id == 0) {
            x_id = tid;
        }
    }

    if (a_id == 0 || x_id == 0)
        return false;

    auto a_it = tensors.find(a_id);
    auto x_it = tensors.find(x_id);
    auto y_it = tensors.find(y_id);

    auto const &ha = a_it->second;
    auto const &hx = x_it->second;
    auto const &hy = y_it->second;

    if (hy.rank != 1 || ha.rank != 2 || hx.rank != 1)
        return false;
    if (ha.dtype != hx.dtype || ha.dtype != hy.dtype)
        return false;

    void *ptr_a = resolve_device_ptr(ha, a_id, shadows);
    void *ptr_x = resolve_device_ptr(hx, x_id, shadows);
    void *ptr_y = resolve_device_ptr(hy, y_id, shadows);

    if (!ptr_a || !ptr_x || !ptr_y)
        return false;

    auto const &ai = desc.spec.a_indices;
    auto const &bi = desc.spec.b_indices; // "b" is actually x for GEMV
    auto const &ci = desc.spec.c_indices; // "c" is actually y for GEMV
    auto const &li = desc.spec.link_indices;

    if (ci.size() != 1 || li.size() != 1)
        return false;

    // A is M×N column-major (dims[0]=M=rows, dims[1]=N=cols).
    auto          M   = static_cast<int64_t>(ha.dims[0]);
    auto          N   = static_cast<int64_t>(ha.dims[1]);
    int64_t const lda = M;

    // Determine transpose: does the target index appear as A's row or column?
    char trans = 'n';
    if (ai.size() == 2) {
        if (ai[0] == ci[0] && ai[1] == li[0]) {
            trans = 'n'; // A[target, link] → no transpose, y has M elements
        } else if (ai[0] == li[0] && ai[1] == ci[0]) {
            trans = 't'; // A[link, target] → transpose, y has N elements
        } else {
            return false;
        }
    } else {
        return false;
    }

    if (ha.dtype == packed_gemm::ScalarType::Float32) {
        auto alpha = as<float>(desc.ab_prefactor);
        auto beta  = as<float>(desc.c_prefactor);
        gpu::blas::gemv<float>(trans, M, N, alpha, static_cast<float const *>(ptr_a), lda, static_cast<float const *>(ptr_x), 1, beta,
                               static_cast<float *>(ptr_y), 1);
        return true;
    } else if (ha.dtype == packed_gemm::ScalarType::Float64) {
        auto alpha = as<double>(desc.ab_prefactor);
        auto beta  = as<double>(desc.c_prefactor);
        gpu::blas::gemv<double>(trans, M, N, alpha, static_cast<double const *>(ptr_a), lda, static_cast<double const *>(ptr_x), 1, beta,
                                static_cast<double *>(ptr_y), 1);
        return true;
    }

    return false;
}

/// Try Scale dispatch: x = alpha * x
bool try_gpu_scale(Node const &node, std::unordered_map<TensorId, TensorHandle> const &tensors, DeviceShadowMap &shadows) {
    if (node.kind != OpKind::Scale)
        return false;

    auto const *desc = std::get_if<ScaleDescriptor>(&node.op_data);
    if (!desc)
        return false;

    if (node.outputs.size() != 1)
        return false;

    TensorId const tid = node.outputs[0];
    auto           it  = tensors.find(tid);
    if (it == tensors.end())
        return false;

    auto const &handle = it->second;

    void *ptr = resolve_device_ptr(handle, tid, shadows);
    if (!ptr)
        return false;

    auto n = static_cast<int64_t>(handle.total_bytes() / handle.element_size);

    // Only the real device kernels are wired up here, so a complex factor
    // declines rather than being projected onto its real part: this path used
    // to truncate silently, which is a wrong answer and not a slow one.
    if (!is_real_valued(desc->factor)) {
        return false;
    }
    if (handle.dtype == packed_gemm::ScalarType::Float32) {
        gpu::blas::scal<float>(n, as_real<float>(desc->factor), static_cast<float *>(ptr), 1);
        return true;
    } else if (handle.dtype == packed_gemm::ScalarType::Float64) {
        gpu::blas::scal<double>(n, as_real<double>(desc->factor), static_cast<double *>(ptr), 1);
        return true;
    }
    return false;
}

/// Try Axpy/Axpby dispatch: y = alpha * x + beta * y
///
/// The scalars come from the AxpbyDescriptor. They used to be read from a
/// ScaleDescriptor, which an axpy/axpby node never carries, so `alpha` silently
/// defaulted to 1.0 and every prefactor was dropped on the device path - a
/// wrong answer, not a slow one. A node whose scalars cannot be read now
/// declines here and runs its own (correct) CPU executor instead of guessing.
bool try_gpu_axpy(Node const &node, std::unordered_map<TensorId, TensorHandle> const &tensors, DeviceShadowMap &shadows) {
    if (node.kind != OpKind::Axpby)
        return false;

    if (node.inputs.size() < 1 || node.outputs.size() != 1)
        return false;

    // Axpy: inputs = [x], outputs = [y] (y is also implicitly read)
    TensorId const x_id = node.inputs[0];
    TensorId const y_id = node.outputs[0];

    auto x_it = tensors.find(x_id);
    auto y_it = tensors.find(y_id);
    if (x_it == tensors.end() || y_it == tensors.end())
        return false;

    auto const &hx = x_it->second;
    auto const &hy = y_it->second;

    if (hx.dtype != hy.dtype)
        return false;

    void *ptr_x = resolve_device_ptr(hx, x_id, shadows);
    void *ptr_y = resolve_device_ptr(hy, y_id, shadows);
    if (!ptr_x || !ptr_y)
        return false;

    auto n = static_cast<int64_t>(hy.total_bytes() / hy.element_size);

    // Read the live scalars. Prefer the shared params over the descriptor
    // snapshot: a pass that folded a scale into this node wrote them there, and
    // that is what the CPU executor would use.
    auto const *desc = std::get_if<AxpbyDescriptor>(&node.op_data);
    if (desc == nullptr) {
        return false; // pass-built node with no readable scalars: let the CPU executor run
    }
    PrefactorScalar const &alpha_pf = live_alpha(*desc);
    PrefactorScalar const &beta_pf  = live_beta(*desc);

    // Only real scalars are representable by the real gpu::blas entry points
    // reached below; a complex prefactor on a real tensor has nowhere to go.
    if (!is_real_valued(alpha_pf) || !is_real_valued(beta_pf)) {
        return false;
    }
    double const alpha = as<double>(alpha_pf);
    double const beta  = as<double>(beta_pf);

    // y = alpha*x + beta*y. beta == 1 is the axpy fast path; otherwise scale the
    // destination first. beta == 0 is handled by the same scal (y := 0) and then
    // the axpy, which is exact.
    auto apply = [&]<typename T>(T /*tag*/) {
        if (beta != 1.0) {
            gpu::blas::scal<T>(n, static_cast<T>(beta), static_cast<T *>(ptr_y), 1);
        }
        gpu::blas::axpy<T>(n, static_cast<T>(alpha), static_cast<T const *>(ptr_x), 1, static_cast<T *>(ptr_y), 1);
    };

    if (hx.dtype == packed_gemm::ScalarType::Float32) {
        apply(float{});
        return true;
    } else if (hx.dtype == packed_gemm::ScalarType::Float64) {
        apply(double{});
        return true;
    }
    return false;
}

/// Try strided-batched GEMM dispatch for OpKind::BatchedGemm nodes.
/// Only the strided mode is handled here, that's what the 3D-batch
/// capture path produces. The pointer-array mode (output of the
/// GEMMBatching pass over N independent 2D einsums) is CPU-only
/// today; extending it to GPU would require either copying each 2D
/// tensor onto a contiguous device buffer first or adding a
/// pointer-array batched GPU wrapper.
bool try_gpu_batched_gemm(BatchedGemmDescriptor const &desc, Node const &node, std::unordered_map<TensorId, TensorHandle> const &tensors,
                          DeviceShadowMap &shadows) {
    if (!desc.strided)
        return false;
    if (node.inputs.size() < 2 || node.outputs.empty())
        return false;

    TensorId const a_id = node.inputs[0];
    TensorId const b_id = node.inputs[1];
    TensorId const c_id = node.outputs[0];

    auto a_it = tensors.find(a_id);
    auto b_it = tensors.find(b_id);
    auto c_it = tensors.find(c_id);
    if (a_it == tensors.end() || b_it == tensors.end() || c_it == tensors.end())
        return false;

    void *ptr_a = resolve_device_ptr(a_it->second, a_id, shadows);
    void *ptr_b = resolve_device_ptr(b_it->second, b_id, shadows);
    void *ptr_c = resolve_device_ptr(c_it->second, c_id, shadows);
    if (!ptr_a || !ptr_b || !ptr_c)
        return false;

    switch (desc.scalar) {
    case BlasScalar::Float: {
        gpu::blas::gemm_strided_batched<float>(
            desc.trans_a, desc.trans_b, desc.m, desc.n, desc.k, static_cast<float>(desc.alpha.real()), static_cast<float const *>(ptr_a),
            desc.lda, desc.batch_stride_a, static_cast<float const *>(ptr_b), desc.ldb, desc.batch_stride_b,
            static_cast<float>(desc.beta.real()), static_cast<float *>(ptr_c), desc.ldc, desc.batch_stride_c, desc.batch_count);
        return true;
    }
    case BlasScalar::Double: {
        gpu::blas::gemm_strided_batched<double>(desc.trans_a, desc.trans_b, desc.m, desc.n, desc.k, desc.alpha.real(),
                                                static_cast<double const *>(ptr_a), desc.lda, desc.batch_stride_a,
                                                static_cast<double const *>(ptr_b), desc.ldb, desc.batch_stride_b, desc.beta.real(),
                                                static_cast<double *>(ptr_c), desc.ldc, desc.batch_stride_c, desc.batch_count);
        return true;
    }
    case BlasScalar::ComplexFloat: {
        std::complex<float> const alpha{static_cast<float>(desc.alpha.real()), static_cast<float>(desc.alpha.imag())};
        std::complex<float> const beta{static_cast<float>(desc.beta.real()), static_cast<float>(desc.beta.imag())};
        gpu::blas::gemm_strided_batched<std::complex<float>>(
            desc.trans_a, desc.trans_b, desc.m, desc.n, desc.k, alpha, static_cast<std::complex<float> const *>(ptr_a), desc.lda,
            desc.batch_stride_a, static_cast<std::complex<float> const *>(ptr_b), desc.ldb, desc.batch_stride_b, beta,
            static_cast<std::complex<float> *>(ptr_c), desc.ldc, desc.batch_stride_c, desc.batch_count);
        return true;
    }
    case BlasScalar::ComplexDouble: {
        std::complex<double> const alpha = desc.alpha;
        std::complex<double> const beta  = desc.beta;
        gpu::blas::gemm_strided_batched<std::complex<double>>(
            desc.trans_a, desc.trans_b, desc.m, desc.n, desc.k, alpha, static_cast<std::complex<double> const *>(ptr_a), desc.lda,
            desc.batch_stride_a, static_cast<std::complex<double> const *>(ptr_b), desc.ldb, desc.batch_stride_b, beta,
            static_cast<std::complex<double> *>(ptr_c), desc.ldc, desc.batch_stride_c, desc.batch_count);
        return true;
    }
    }
    return false;
}

/// Top-level GPU BLAS dispatcher: tries GEMM, GEMV, Scale, Axpy, BatchedGemm.
bool try_gpu_blas_dispatch(Node const &node, std::unordered_map<TensorId, TensorHandle> const &tensors, DeviceShadowMap &shadows) {
    // Einsum operations: try GEMM, then GEMV.
    if (auto const *desc = std::get_if<EinsumDescriptor>(&node.op_data)) {
        if (try_gpu_gemm(*desc, node, tensors, shadows))
            return true;
        if (try_gpu_gemv(*desc, node, tensors, shadows))
            return true;
    }

    // Strided-batched GEMM (3D batch-contiguous einsums captured as BatchedGemm).
    if (auto const *desc = std::get_if<BatchedGemmDescriptor>(&node.op_data)) {
        if (try_gpu_batched_gemm(*desc, node, tensors, shadows))
            return true;
    }

    // BLAS Level 1 operations.
    if (try_gpu_scale(node, tensors, shadows))
        return true;
    if (try_gpu_axpy(node, tensors, shadows))
        return true;

    return false;
}

} // namespace

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
    _indices_store         = std::move(other._indices_store);
    _device_shadows        = std::move(other._device_shadows);
    _executor              = std::move(other._executor);
    _aliases_linked        = other._aliases_linked;
    _slots_validated       = other._slots_validated;
    _timing_samples        = std::move(other._timing_samples);
    _timing_report         = std::move(other._timing_report);
    _timing_report_valid   = other._timing_report_valid;
    _params_store          = std::move(other._params_store);
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

namespace {

/// Half-open byte span of a handle's storage, or false when it has none that
/// can be reasoned about (deferred allocation, tiled layout, zero extent).
///
/// A tiled handle has no single buffer, so it is refused here rather than in the
/// shared span helper, which knows only about one strided allocation.
bool handle_byte_span(TensorHandle const &h, char const *&lo, char const *&hi) {
    if (h.is_tiled) {
        return false;
    }
    return detail::strided_byte_span(h.data_ptr, h.dims, h.strides, h.element_size, lo, hi);
}

/// How well a handle's strides describe a lattice, which is what decides
/// whether an offset may be decoded into per-axis indices at all.
enum class LayoutFit {
    None,   ///< Overlapping or degenerate; nothing may be decoded from an offset.
    Nested, ///< Injective: each traversed axis starts at or past the end of the one below it.
    Packed  ///< Nested with no gaps, so every offset under the total is an index.
};

/// Classify @p h's layout and hand back its traversed axes, largest stride
/// first, which is the order both derivations peel an offset in.
///
/// Nesting is the property that makes an offset decode to ONE index: sort the
/// axes holding more than one element by stride, and require each one to start
/// at or past the end of everything below it. Without it a layout can be
/// non-injective - dims (4, 4) with strides (1, 2) reaches offset 4 as both
/// (0, 2) and (2, 1) - and two boxes that do not intersect in the axis space
/// can still name the same memory, which is the one way a box can be too
/// narrow. Packing additionally forbids gaps, which is what makes every offset
/// in a range decodable and so is what the span derivation below needs.
///
/// Axes holding one element are left out of @p order: their only index is 0,
/// and a dense layout is free to give them any stride at all.
///
/// @param[in]  h     Handle to classify.
/// @param[out] order Its traversed axes, largest stride first.
/// @param[out] total Number of elements, meaningful for a Packed layout.
LayoutFit layout_axis_order(TensorHandle const &h, std::vector<size_t> &order, size_t &total) {
    size_t const rank = h.dims.size();
    if (rank == 0 || h.strides.size() != rank || h.is_tiled) {
        return LayoutFit::None;
    }
    order.clear();
    total = 1;
    for (size_t d = 0; d < rank; ++d) {
        if (h.dims[d] == 0) {
            return LayoutFit::None;
        }
        total *= h.dims[d];
        if (h.dims[d] > 1) {
            order.push_back(d);
        }
    }
    std::ranges::sort(order, [&](size_t a, size_t b) { return h.strides[a] < h.strides[b]; });
    size_t reach  = 1; // one past the last offset the axes so far can reach
    bool   packed = true;
    for (size_t const d : order) {
        if (h.strides[d] < reach) {
            return LayoutFit::None;
        }
        packed = packed && h.strides[d] == reach;
        reach  = h.strides[d] * h.dims[d];
    }
    std::ranges::reverse(order); // the digit peel runs most significant first
    return packed ? LayoutFit::Packed : LayoutFit::Nested;
}

/// Recover @p child's region in @p parent's axis space by matching each child
/// axis to the parent axis that carries the same stride. Exact when it answers,
/// and it answers for every view whose axes are parent axes: a sub-block, a
/// dropped axis (which is a single-index interval in the parent), a transposed
/// view, and any mix of the three.
///
/// Returns false whenever the layout is not provably a sub-box, which leaves the
/// caller to try the looser derivation below or, failing that, to treat the
/// access as whole-tensor. Being wrong here would UNDER-serialize, so every
/// ambiguity declines rather than guesses.
bool derive_matched_alias_box(TensorHandle const &parent, TensorHandle const &child,
                              std::vector<std::pair<std::int64_t, std::int64_t>> &box) {
    size_t const rank = parent.dims.size();
    if (rank == 0 || parent.element_size == 0 || child.element_size != parent.element_size) {
        return false;
    }
    if (parent.strides.size() != rank || child.strides.size() != child.dims.size() || parent.is_tiled || child.is_tiled) {
        return false;
    }
    // A layout that is not nested cannot be decoded: the offset would name more
    // than one index, and the axis matching would be ambiguous with it. Equal
    // strides on two traversed axes are the common way in and are rejected
    // here, along with the overlapping strides that are the subtle way in.
    std::vector<size_t> order;
    size_t              total = 0;
    if (layout_axis_order(parent, order, total) == LayoutFit::None) {
        return false;
    }

    auto const *p = static_cast<char const *>(parent.data_ptr);
    auto const *c = static_cast<char const *>(child.data_ptr);
    if (c < p) {
        return false;
    }
    size_t off = static_cast<size_t>(c - p);
    if (off % parent.element_size != 0) {
        return false;
    }
    off /= parent.element_size;

    // Peel the offset apart largest stride first, which is unique for a nested
    // layout. An offset that is not on the lattice cannot survive it: it leaves
    // a non-zero remainder, or a digit past its axis's extent that the range
    // check at the end rejects.
    std::vector<std::int64_t> start(rank, 0);
    for (size_t const d : order) {
        start[d] = static_cast<std::int64_t>(off / parent.strides[d]);
        off %= parent.strides[d];
    }
    if (off != 0) {
        return false; // offset does not land on a parent index
    }

    // A child axis that reuses a parent stride keeps that axis; the rest are
    // pinned to a single index by the offset.
    std::vector<std::int64_t> extent(rank, 1);
    std::vector<bool>         matched(child.dims.size(), false);
    for (size_t d = 0; d < rank; ++d) {
        for (size_t e = 0; e < child.dims.size(); ++e) {
            if (!matched[e] && child.strides[e] == parent.strides[d] && child.dims[e] > 1) {
                extent[d]  = static_cast<std::int64_t>(child.dims[e]);
                matched[e] = true;
                break;
            }
        }
    }
    for (size_t e = 0; e < child.dims.size(); ++e) {
        if (!matched[e] && child.dims[e] > 1) {
            return false; // a traversed child axis with no parent counterpart
        }
    }

    box.clear();
    box.reserve(rank);
    for (size_t d = 0; d < rank; ++d) {
        std::int64_t const hi = start[d] + extent[d];
        if (start[d] < 0 || hi > static_cast<std::int64_t>(parent.dims[d])) {
            return false;
        }
        box.emplace_back(start[d], hi);
    }
    return true;
}

/// Recover @p child's region in @p parent's axis space from the contiguous
/// OFFSET RANGE it spans, for the children the axis matching above cannot
/// describe: a reshaped window carries axes that are products of parent axes
/// rather than parent axes, so there is no correspondence to recover.
///
/// The shape comes from the DLPNO port, whose rank-3 dressed factors are
/// reshaped windows of a flat scratch pool. It buys that port no schedule
/// today, and the reason is worth recording: its hand-outs are PREFIXES of
/// their buffer, so two of them overlap however precisely they are described,
/// and the parallelism there comes from the pool being several buffers rather
/// than from any box. What the bound buys is that a pool carved into DISJOINT
/// windows is separable at all, which the axis match cannot do at any width.
///
/// **Soundness.** Strides are unsigned, so every element the child addresses
/// lies at an offset in `[base, base + sum (dim - 1) * stride]`, whatever its
/// axis order and however its own axes overlap each other. A packed parent
/// decodes each offset in that range to exactly one index, so the smallest box
/// containing the decoded ends contains every element the child can touch. That
/// box is looser than the matched one - a range of offsets is not a box, so
/// once the two ends differ in some axis every lower axis has to open to its
/// full extent - and loose is the safe direction: a box that is too WIDE costs
/// hazard edges, only a box that is too NARROW loses one.
bool derive_span_alias_box(TensorHandle const &parent, TensorHandle const &child, std::vector<std::pair<std::int64_t, std::int64_t>> &box) {
    if (child.is_tiled || parent.element_size == 0 || child.element_size != parent.element_size) {
        return false;
    }
    if (child.strides.size() != child.dims.size() || parent.data_ptr == nullptr || child.data_ptr == nullptr) {
        return false;
    }

    // Packed, not merely nested: a gapped parent has offsets between its
    // elements, and the bound below walks a RANGE of offsets rather than the
    // lattice points the matched derivation sticks to.
    std::vector<size_t> order;
    size_t              total = 0;
    if (layout_axis_order(parent, order, total) != LayoutFit::Packed) {
        return false;
    }

    auto const *p = static_cast<char const *>(parent.data_ptr);
    auto const *c = static_cast<char const *>(child.data_ptr);
    if (c < p) {
        return false;
    }
    size_t base = static_cast<size_t>(c - p);
    if (base % parent.element_size != 0) {
        return false;
    }
    base /= parent.element_size;

    size_t reach = 0; // offset of the last element the child can address
    for (size_t e = 0; e < child.dims.size(); ++e) {
        if (child.dims[e] == 0) {
            return false; // an empty access has no region to bound
        }
        reach += (child.dims[e] - 1) * child.strides[e];
    }
    if (base >= total || reach > total - 1 - base) {
        return false; // not contained in the parent, so not describable in its axes
    }

    box.clear();
    box.reserve(parent.dims.size());
    for (size_t const d : parent.dims) {
        box.emplace_back(0, static_cast<std::int64_t>(d));
    }
    // Peel both ends most significant first. While the digits agree the box is
    // that single index; the first axis where they differ takes the span
    // between them and every axis below it keeps the full extent it started
    // with, because the offsets between the two ends run through all of them.
    size_t lo = base;
    size_t hi = base + reach;
    for (size_t const d : order) {
        size_t const stride = parent.strides[d];
        size_t const lo_d   = lo / stride;
        size_t const hi_d   = hi / stride;
        lo %= stride;
        hi %= stride;
        if (hi_d >= parent.dims[d]) {
            return false; // cannot happen for a contained child of a packed parent
        }
        box[d] = {static_cast<std::int64_t>(lo_d), static_cast<std::int64_t>(hi_d) + 1};
        if (lo_d != hi_d) {
            break;
        }
    }
    return true;
}

/// @p child's region in @p parent's axis space: the exact per-axis match where
/// the layouts allow it, the looser offset-span bound where they do not, and no
/// box at all (a conservatively whole-tensor access) where neither is provable.
///
/// A box covering the whole parent is reported as NO box, which is the same
/// statement - both overlap every access and cover every access - in the form
/// the hazard scan reasons about better. A whole-tensor write there retires
/// every writer before it, because it dominates them; a full-cover BOX is not
/// recognized as dominating anything, so the writer list grows without bound
/// and each later access emits an edge against all of it. The two spellings
/// schedule identically and the quadratic one is worth avoiding: adding the
/// span bound above WITHOUT this normalization put 103,000 extra edges on the
/// DLPNO merged iteration, every one of them from a reshaped view that covers
/// its whole parent.
bool derive_alias_box(TensorHandle const &parent, TensorHandle const &child, std::vector<std::pair<std::int64_t, std::int64_t>> &box) {
    if (!derive_matched_alias_box(parent, child, box) && !derive_span_alias_box(parent, child, box)) {
        return false;
    }
    bool whole = box.size() == parent.dims.size();
    for (size_t d = 0; whole && d < box.size(); ++d) {
        whole = box[d].first == 0 && box[d].second == static_cast<std::int64_t>(parent.dims[d]);
    }
    if (whole) {
        box.clear();
        return false;
    }
    return true;
}

/// Per-axis half-open interval list, the representation both derivations speak.
using AliasBox = std::vector<std::pair<std::int64_t, std::int64_t>>;

/// True when @p box is every element of a parent with extents @p dims, which is
/// the shape `derive_alias_box` normalizes away. See its comment for why: a
/// full-cover BOX is not recognized as dominating anything, so the writer list
/// grows without bound, while the same statement spelled as NO box retires
/// every writer it covers.
bool whole_cover(AliasBox const &box, std::vector<size_t> const &dims) {
    if (box.size() != dims.size()) {
        return false;
    }
    for (size_t d = 0; d < box.size(); ++d) {
        if (box[d].first != 0 || !std::cmp_equal(box[d].second, dims[d])) {
            return false;
        }
    }
    return true;
}

/// What a structural walk knows about one tensor's relation to its alias root.
///
/// @ref box and @ref axis_map are only meaningful under their respective flags;
/// @ref root is always meaningful, because a walk that cannot describe a region
/// still knows which buffer the tensor is part of, and that is the half whose
/// absence races.
struct StructuralAlias {
    /// The tensor at the end of the alias chain. Equal to the tensor itself when
    /// it owns its storage.
    TensorId root{0};
    /// The region the tensor covers, in @ref root's axis space. Valid only when
    /// @ref box_known.
    AliasBox box;
    /// This tensor's axis @c r maps onto @ref root's axis ``axis_map[r]``. One
    /// entry per axis of THIS tensor, so a Drop axis contributes none. Valid only
    /// when @ref map_known, and needed by a child that composes through it.
    std::vector<size_t> axis_map;
    bool                box_known{false}; ///< Whether @ref box describes the region.
    bool                map_known{false}; ///< Whether @ref axis_map describes the axis correspondence.
};

/// Alias discovery with NO addresses: the relation `(root, region)` derived from
/// ``View`` nodes, their @ref ViewDescriptor axes, and the alias links already on
/// the handles, with no data pointer consulted anywhere.
///
/// This is the counterpart of the pointer-derived containment search in
/// @ref Graph::link_alias_storage and it is what a loaded graph has instead of
/// one, because a graph read from a file has allocated nothing and the tensors
/// bound to it afterwards are not the tensors that were captured. It is also
/// STRICTLY more general than the View-node scan the hazard pass used to run
/// inline, which refused a permuted view and a view of a view; both compose here.
///
/// **Composition.** A view's descriptor gives, per parent axis, the interval that
/// axis is restricted to (a whole axis, a constant Range, or the single index a
/// Drop pins), and its permutation says which parent axis each RESULT axis reads.
/// Composing a chain is therefore: take the parent's own region in the root's
/// axis space, and place the child's per-parent-axis intervals inside it by
/// offsetting each one by where the parent's corresponding axis starts. The axis
/// map is what makes that placement possible past one hop, and it is why a view
/// of a permuted view is describable at all.
///
/// **Where it declines.** A non-constant bound (a Param or a Callback, whose
/// value is not known until execute) yields no box, which reads as the whole
/// parent - the same conservative answer the pointer path gives for a region it
/// cannot prove. So does a root whose own strides are not injective: such a
/// layout reaches one element through more than one index tuple, so two boxes
/// that share no index can still name the same memory, and trusting the axis
/// space there would DROP a real hazard edge. That fence is checked against the
/// root's registration-time strides when it has them; a loaded graph carries
/// none yet, and its axis space is trusted because nothing else can be.
class StructuralAliasResolver {
  public:
    /// Index the graph's ``View`` nodes. One pass; the walk itself is memoized.
    explicit StructuralAliasResolver(Graph const &graph) : _graph(&graph) {
        for (auto const &nd : graph.nodes()) {
            if (nd.kind != OpKind::View || nd.outputs.size() != 1) {
                continue;
            }
            auto const *vd = std::get_if<ViewDescriptor>(&nd.op_data);
            if (vd == nullptr) {
                continue;
            }
            // First View node writing a tensor wins. A second one would be a
            // re-description of the same slice; taking the first keeps the
            // derivation independent of node order.
            _views.emplace(nd.outputs[0], vd);
        }
    }

    /// Whether @p id is the output of a ``View`` node, i.e. whether this
    /// derivation has anything to say about it that the handle does not.
    [[nodiscard]] bool is_view(TensorId id) const { return _views.contains(id); }

    /// @return The relation, memoized.
    ///
    /// Walks UP the view chain collecting what this answer depends on, then
    /// composes back DOWN. Iterative rather than recursive on purpose: a chain
    /// is bounded only by the tensor count (DLPNO-MP2 registers ~13k), and a
    /// stack frame per hop is the wrong thing to bound it with. It also makes
    /// the cycle case a plain visited-set test rather than a depth guard - a
    /// cycle is not constructible today, and if one ever is, the tensor is
    /// reported conservatively instead of looped on.
    StructuralAlias const &resolve(TensorId id) {
        if (auto const it = _memo.find(id); it != _memo.end()) {
            return it->second;
        }

        std::vector<TensorId>        chain;
        std::unordered_set<TensorId> on_chain;
        for (TensorId current = id;;) {
            if (_memo.contains(current)) {
                break; // an answer already exists to compose against
            }
            auto const vit = _views.find(current);
            if (vit == _views.end()) {
                _memo.emplace(current, self(current)); // chain ends at a tensor no View node describes
                break;
            }
            if (!on_chain.insert(current).second) {
                _memo.emplace(current, StructuralAlias{.root = current}); // cycle: conservative
                break;
            }
            chain.push_back(current);
            TensorId const parent = vit->second->parent_id;
            if (parent == current || _graph->find_tensor(parent) == nullptr) {
                chain.pop_back();
                _memo.emplace(current, self(current)); // a View node naming nothing usable
                break;
            }
            current = parent;
        }

        // Compose downward: every entry's parent is already answered.
        for (TensorId const tid : std::views::reverse(chain)) {
            _memo.emplace(tid, compose(tid));
        }
        return _memo.at(id);
    }

  private:
    /// The answer for a tensor that no ``View`` node describes: it owns its
    /// storage, or its handle already names an alias parent whose axis space
    /// this derivation cannot recover (a pointer-linked slice, a manifest
    /// declaration). Either way the ROOT is known and the region is not.
    [[nodiscard]] StructuralAlias self(TensorId id) const {
        StructuralAlias res;
        res.root                   = id;
        TensorHandle const *handle = _graph->find_tensor(id);
        if (handle == nullptr) {
            return res;
        }
        if (handle->aliases != 0) {
            // resolve_alias throws only on a cycle, which every writer of
            // ``aliases`` makes unconstructible; a conservative catch here would
            // hide exactly the corruption it is there to report.
            res.root = _graph->resolve_alias(handle->aliases);
            return res; // region unknown: a box on the handle lives in a space this walk did not build
        }
        if (handle->is_tiled) {
            return res; // no single axis space to place a region in
        }
        // The injectivity fence. Only checked when the root carries the strides
        // to check it with; see the class comment.
        if (handle->strides.size() == handle->dims.size() && !handle->strides.empty()) {
            std::vector<size_t> order;
            size_t              total = 0;
            if (layout_axis_order(*handle, order, total) == LayoutFit::None) {
                return res;
            }
        }
        res.box.reserve(handle->dims.size());
        res.axis_map.reserve(handle->dims.size());
        for (size_t d = 0; d < handle->dims.size(); ++d) {
            res.box.emplace_back(0, static_cast<std::int64_t>(handle->dims[d]));
            res.axis_map.push_back(d);
        }
        res.box_known = true;
        res.map_known = true;
        return res;
    }

    /// Place @p id's own slice inside the region its parent already occupies.
    /// Only called by @ref resolve, and only once the parent is answered.
    [[nodiscard]] StructuralAlias compose(TensorId id) const {
        ViewDescriptor const  *vd     = _views.at(id);
        TensorHandle const    *ph     = _graph->find_tensor(vd->parent_id);
        StructuralAlias const &parent = _memo.at(vd->parent_id);
        StructuralAlias        res;
        res.root = parent.root;
        if (ph == nullptr) {
            return res;
        }

        size_t const prank = ph->dims.size();
        if (!parent.box_known || !parent.map_known || parent.axis_map.size() != prank || vd->axes.size() != prank) {
            return res; // the parent's own region is not placeable, so neither is this one
        }

        // The child's region in the PARENT's axis space, plus which parent axis
        // each of the child's own axes reads.
        AliasBox            local(prank, {0, 0});
        std::vector<size_t> local_map;
        std::vector<bool>   touched(prank, false);
        local_map.reserve(prank);
        for (size_t i = 0; i < prank; ++i) {
            // Result axis i slices parent axis p. Empty permutation is identity;
            // a permutation is a bijection of [0, prank), so every parent axis is
            // named exactly once and the box has no hole.
            size_t const p = vd->permutation.empty() ? i : vd->permutation[i];
            if (p >= prank || touched[p]) {
                return res;
            }
            touched[p]          = true;
            ViewAxis const &ax  = vd->axes[i];
            auto const      dim = static_cast<std::int64_t>(ph->dims[p]);
            switch (ax.kind) {
            case ViewAxis::Kind::Full:
                local[p] = {0, dim};
                local_map.push_back(p);
                break;
            case ViewAxis::Kind::Range:
                if (!ax.lo.is_const() || !ax.hi.is_const()) {
                    return res; // a runtime bound conflicts as the whole parent
                }
                local[p] = {ax.lo.const_value(), ax.hi.const_value()};
                local_map.push_back(p);
                break;
            case ViewAxis::Kind::Drop:
                if (!ax.lo.is_const()) {
                    return res;
                }
                local[p] = {ax.lo.const_value(), ax.lo.const_value() + 1};
                break; // a dropped axis contributes its collapsed index and no result axis
            }
            if (local[p].first < 0 || local[p].first > local[p].second || local[p].second > dim) {
                return res; // out of the parent, so not describable in its axes
            }
        }

        // Place the local intervals inside the parent's own region: parent axis p
        // is root axis q, and its index 0 sits at the root index the parent's box
        // starts at.
        res.box = parent.box;
        for (size_t p = 0; p < prank; ++p) {
            size_t const q = parent.axis_map[p];
            if (q >= res.box.size()) {
                return {.root = parent.root};
            }
            std::int64_t const base = parent.box[q].first;
            res.box[q]              = {base + local[p].first, base + local[p].second};
            if (res.box[q].second > parent.box[q].second) {
                return {.root = parent.root};
            }
        }
        res.axis_map.reserve(local_map.size());
        for (size_t const p : local_map) {
            res.axis_map.push_back(parent.axis_map[p]);
        }
        res.box_known = true;
        res.map_known = true;
        return res;
    }

    Graph const                                         *_graph;
    std::unordered_map<TensorId, ViewDescriptor const *> _views;
    std::unordered_map<TensorId, StructuralAlias>        _memo;
};

} // namespace

void Graph::declare_alias(TensorId child, TensorId parent) {
    if (child == 0 || parent == 0 || child == parent) {
        return;
    }
    auto const child_it = _tensors.find(child);
    if (child_it == _tensors.end() || !_tensors.contains(parent)) {
        return;
    }
    // A cycle would turn every later resolve_alias into a throw, and the message
    // there is about a corrupt link rather than about the declaration that made
    // it. Refuse here, where the two names are still in hand: a saved graph
    // claiming A is part of B and B part of A describes nothing.
    for (TensorId walk = parent, hops = 0; walk != 0 && hops <= _tensors.size(); ++hops) {
        if (walk == child) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                    "Graph '{}': declare_alias({}, {}) would make the two tensors each other's alias parent; an "
                                    "alias declaration has to name a containing buffer, and containment is not symmetric",
                                    _name, child, parent);
        }
        auto const it = _tensors.find(walk);
        if (it == _tensors.end()) {
            break;
        }
        walk = it->second.aliases;
    }
    _declared_aliases.insert_or_assign(child, parent);
    child_it->second.aliases = parent;
    // A declaration says WHICH buffer, never WHICH REGION - the manifest schema
    // carries no box - so the declared alias conflicts as the whole parent.
    child_it->second.alias_box.clear();
    // The hazard relation just changed, so anything derived from it is stale.
    _deps_valid = false;
}

void Graph::clear_alias_links() noexcept {
    for (auto &[id, handle] : _tensors) {
        handle.aliases = 0;
        handle.alias_box.clear();
    }
    _aliases_linked = false;
    _deps_valid     = false;
}

void Graph::apply_declared_aliases() {
    for (auto const &[child, parent] : _declared_aliases) {
        auto const it = _tensors.find(child);
        if (it == _tensors.end() || child == parent || !_tensors.contains(parent)) {
            continue;
        }
        it->second.aliases = parent;
        it->second.alias_box.clear();
    }
}

void Graph::link_alias_structural() {
    // Declarations first, and they are what the pointer path cannot see: two
    // caller-supplied tensors that happen to share storage have no View node
    // recording it, and in a loaded graph they have no addresses to compare
    // either. Applying them first also means a view OF a declared alias
    // composes down onto the declared root below.
    apply_declared_aliases();

    StructuralAliasResolver resolver(*this);

    // Derive first, write second. The walk reads ``aliases`` for the handles no
    // View node describes, so rewriting a handle mid-walk would let a later
    // tensor resolve against a half-updated relation.
    struct Update {
        TensorId tid;
        TensorId root;
        AliasBox box;
    };
    std::vector<Update> updates;
    for (auto const &[id, handle] : _tensors) {
        if (!resolver.is_view(id)) {
            continue;
        }
        StructuralAlias const &res = resolver.resolve(id);
        if (res.root == 0 || res.root == id) {
            continue;
        }
        AliasBox box;
        if (res.box_known) {
            TensorHandle const *root = find_tensor(res.root);
            // A box covering the whole root is reported as NO box, exactly as
            // the pointer path's derive_alias_box does and for the same
            // dominance reason. Keeping the two normalizations identical is
            // what makes the derivations comparable at all.
            if (root != nullptr && !whole_cover(res.box, root->dims)) {
                box = res.box;
            }
        }
        updates.push_back({.tid = id, .root = res.root, .box = std::move(box)});
    }
    for (auto &update : updates) {
        auto const it = _tensors.find(update.tid);
        if (it == _tensors.end()) {
            continue;
        }
        it->second.aliases   = update.root;
        it->second.alias_box = std::move(update.box);
    }

    // Deliberately NOT marking the pointer derivation as done. Structural adds
    // what the ``View`` nodes and the declarations say and claims nothing about
    // the relations only an address can reveal, so a graph that HAS addresses
    // still gets its containment search - which is also what makes "run one, then
    // the other, and nothing moves" an invariant every scheduling test enforces
    // for free rather than a property one test remembers to check.
    // Graph::link_alias_storage sets the flag itself before delegating here, so
    // the loaded-graph path does not re-enter.

    // A Loop body and a Conditional branch are separate graphs with their own
    // handles, and fourteen passes rewrite them; a body left unlinked answers
    // "this view aliases nothing" and its hazard edges vanish. The pointer path
    // is recursed by Graph::apply for that reason, and this recurses itself so
    // a direct call (the loader's, and the tests') gets the same tree.
    for_each_subgraph([](Graph &sub) { sub.link_alias_structural(); });
}

void Graph::link_alias_storage() {
    if (_aliases_linked) {
        return;
    }
    _aliases_linked = true;

    // A declaration is authoritative and an address coincidence is not, so
    // declarations go on first and the containment search below leaves them
    // alone (it skips any handle that already names a parent).
    apply_declared_aliases();

    // The loaded-graph state: structure was read from a file, nothing is
    // allocated, and no handle has an address to compare. Linking nothing here
    // is not a safe default - it is the exact shape of the full-cover alias bug,
    // a silently incomplete relation that surfaces as a race - so the derivation
    // switches to the structural one, which needs no addresses at all.
    //
    // MIXED graphs are the ordinary case, not an error: a deferred shell has no
    // address until it is materialized, and TensorHandle::data_ptr is a
    // registration-time snapshot nothing refreshes. Refusing to mix would refuse
    // most real graphs. The rule is therefore the honest one - the pointer
    // derivation runs whenever ANY handle carries an address, unchanged from
    // before, and it simply cannot see a null-address handle; the structural
    // derivation takes the whole graph only when NO handle carries one. What
    // spans the two modes is the declaration, which is applied above in both and
    // is the only way an alias between two address-less operands is expressible.
    if (!_tensors.empty() && std::ranges::none_of(_tensors, [](auto const &kv) { return kv.second.data_ptr != nullptr; })) {
        link_alias_structural();
        return;
    }

    struct Entry {
        char const *lo;
        char const *hi;
        TensorId    id;
    };
    std::vector<Entry> spans;
    spans.reserve(_tensors.size());
    for (auto const &[id, h] : _tensors) {
        char const *lo = nullptr;
        char const *hi = nullptr;
        if (handle_byte_span(h, lo, hi)) {
            spans.push_back({.lo = lo, .hi = hi, .id = id});
        }
    }
    if (spans.size() < 2) {
        return;
    }
    // The backward walk below only visits spans sorted at or before its own,
    // so a span must sort after every span that can own it: wider spans first
    // among equal starts, and among IDENTICAL spans the designated owner
    // (lower id, see below) first.
    std::ranges::sort(spans, [](Entry const &a, Entry const &b) {
        if (a.lo != b.lo) {
            return a.lo < b.lo;
        }
        if (a.hi != b.hi) {
            return a.hi > b.hi;
        }
        return a.id < b.id;
    });

    // Running max of hi over the prefix. Only a span starting at or before this
    // one can contain it, so once the best hi in that prefix falls short of our
    // end there is nothing left to find and the backward walk stops. Without it
    // the walk is the O(n^2) scan this replaces.
    std::vector<char const *> prefix_max_hi(spans.size());
    char const               *running = nullptr;
    for (size_t i = 0; i < spans.size(); ++i) {
        running          = (running == nullptr || spans[i].hi > running) ? spans[i].hi : running;
        prefix_max_hi[i] = running;
    }

    // Ownership is the strict order (more elements, then lower id): the id
    // tie-break keeps the relation acyclic when two handles cover the SAME
    // bytes, which is exactly what a view that spans its whole parent does.
    // Requiring strictly fewer elements instead left such a view linked to
    // nothing, so the hazard scan saw two unrelated tensors on one buffer and
    // emitted no edge between their accesses - DLPNO's singleton shape
    // classes hit this (`_W_pair` covers `_W` when a class has one pair).
    for (size_t i = 0; i < spans.size(); ++i) {
        auto self = _tensors.find(spans[i].id);
        if (self == _tensors.end() || self->second.aliases != 0) {
            continue;
        }
        // Last span starting at or before this one.
        size_t j = i;
        while (j > 0 && spans[j].lo > spans[i].lo) {
            --j;
        }
        for (;; --j) {
            if (prefix_max_hi[j] < spans[i].hi) {
                break; // nothing at or before j reaches far enough
            }
            if (spans[j].id != spans[i].id && spans[j].lo <= spans[i].lo && spans[i].hi <= spans[j].hi) {
                auto const owner = _tensors.find(spans[j].id);
                if (owner != _tensors.end() &&
                    (owner->second.total_elems() > self->second.total_elems() ||
                     (owner->second.total_elems() == self->second.total_elems() && owner->first < self->first))) {
                    // Link to the owner's ROOT, not to the owner. Containment
                    // is transitive, so both are correct answers to "who owns
                    // this", but only one of them keeps the chain short - and
                    // resolve_alias walks the chain on every hazard-scan
                    // lookup and gives up after a bounded number of hops.
                    //
                    // Without this, N handles covering the SAME bytes (the
                    // tie-break below the containment test links each to the
                    // one before it) form a chain of depth N, and past the hop
                    // limit resolve_alias returns a different mid-chain id for
                    // each of them. The hazard scan then keys their accesses
                    // under different owners and emits no edge between any of
                    // them, which under a threading executor is a silent data
                    // race - DLPNO-(T0) hit exactly this with one scratch
                    // buffer shared by 40 triplets, and the schedule's widest
                    // level came out at exactly (triplets - hop limit).
                    //
                    // Owners sort before their aliases and the outer loop runs
                    // in sorted order, so the owner's own link is already final
                    // here and one resolve gives the true root.
                    TensorId const root_id = resolve_alias(owner->first);
                    auto const     root    = _tensors.find(root_id);
                    if (root == _tensors.end()) {
                        break;
                    }
                    self->second.aliases = root_id;
                    // The box has to live in the axis space of whatever
                    // ``aliases`` names, which is now the root rather than the
                    // immediate container.
                    if (!derive_alias_box(root->second, self->second, self->second.alias_box)) {
                        self->second.alias_box.clear(); // unknown box reads as the whole parent
                    }
                    break;
                }
            }
            if (j == 0) {
                break;
            }
        }
    }

    // Second pass: PARTIAL overlap, which containment cannot express.
    //
    // Two handles that overlap without either containing the other - sliding
    // windows over a buffer no node names, so the common parent is never
    // registered - both come out of the loop above unlinked. The hazard scan
    // then sees two unrelated tensors sharing bytes and orders nothing between
    // their accesses, which under a threading executor is a silent data race.
    //
    // The fix is to give each run of mutually overlapping spans ONE root, so
    // the scan keys their accesses together. It cannot be exact: `aliases`
    // names a container, and a run like this has none, so the members' regions
    // are no longer expressible in the root's axis space. They lose their boxes
    // and conflict conservatively - which is the honest answer for a relation
    // this model does not describe, and still far better than no edge at all.
    //
    // A run whose members ALREADY share one root is left completely alone. That
    // is the common case and the important one: a registered parent and its
    // slices form a single run, they already resolve to the parent, and
    // relinking anything there would strip the slices of the boxes that keep
    // provably disjoint ones running in parallel.
    {
        size_t run_begin = 0;
        while (run_begin < spans.size()) {
            size_t      run_end = run_begin;
            char const *reach   = spans[run_begin].hi;
            while (run_end + 1 < spans.size() && spans[run_end + 1].lo < reach) {
                ++run_end;
                reach = std::max(reach, spans[run_end].hi);
            }

            // The canonical root is the widest span's, not the lowest id's: in
            // a run that mixes a real container with a partial overlapper, the
            // container is the one whose axis space the other members' boxes
            // are already written in, so choosing it keeps those boxes valid.
            // The id breaks ties so the choice does not depend on map order.
            TensorId canonical  = 0;
            size_t   widest     = 0;
            TensorId first_root = 0;
            bool     mixed      = false;
            for (size_t k = run_begin; k <= run_end; ++k) {
                TensorId const root = resolve_alias(spans[k].id);
                if (first_root == 0) {
                    first_root = root;
                } else if (root != first_root) {
                    mixed = true;
                }
                auto const it = _tensors.find(root);
                if (it == _tensors.end()) {
                    continue;
                }
                size_t const extent = it->second.total_elems();
                if (canonical == 0 || extent > widest || (extent == widest && root < canonical)) {
                    canonical = root;
                    widest    = extent;
                }
            }
            if (mixed && canonical != 0) {
                for (size_t k = run_begin; k <= run_end; ++k) {
                    TensorId const root = resolve_alias(spans[k].id);
                    if (root == canonical) {
                        continue;
                    }
                    auto it = _tensors.find(root);
                    if (it == _tensors.end()) {
                        continue;
                    }
                    // Only ever a root, and only ever onto a DIFFERENT root of
                    // the same run, so the relation stays acyclic and every
                    // member of the run resolves to `canonical` from here.
                    it->second.aliases = canonical;
                    it->second.alias_box.clear();
                }
            }
            run_begin = run_end + 1;
        }
    }
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

void Graph::collect_subtree_referenced_ptrs(std::unordered_set<void const *> &out) const {
    // Insert the tensor pointers referenced (read or written) by one graph's
    // own nodes. Resolves each TensorId through that graph's own map.
    auto collect_own = [](Graph const &g, std::unordered_set<void const *> &acc) {
        auto add = [&](TensorId tid) {
            auto it = g._tensors.find(tid);
            if (it != g._tensors.end() && it->second.tensor_ptr != nullptr) {
                acc.insert(it->second.tensor_ptr);
            }
        };
        for (auto const &node : g._nodes) {
            for (auto tid : node.inputs) {
                add(tid);
            }
            for (auto tid : node.outputs) {
                add(tid);
            }
        }
    };

    for_each_subgraph([&](Graph const &sub) {
        collect_own(sub, out);                    // sub's own references
        sub.collect_subtree_referenced_ptrs(out); // and sub's descendants
    });
}

std::pair<std::vector<TensorId>, std::vector<TensorId>> Graph::effective_io(Node const &node) {
    std::vector<TensorId> ins  = node.inputs;
    std::vector<TensorId> outs = node.outputs;

    if (!is_control_flow(node.kind)) {
        return {ins, outs};
    }

    // Walk the node's subtree (body / branches, recursively) and collect the
    // buffer pointers it reads and writes, keeping one representative handle per
    // buffer. Each sub-graph resolves its own TensorIds, so we key on the stable
    // tensor_ptr; the handle lets us register the buffer in the parent below if
    // it isn't already known there.
    std::set<void const *>                         read_ptrs;
    std::set<void const *>                         write_ptrs;
    std::unordered_map<void const *, TensorHandle> rep_handle;
    std::function<void(Graph const &)>             collect = [&](Graph const &sub) {
        for (auto const &nd : sub._nodes) {
            if (is_lifecycle(nd.kind)) {
                continue;
            }
            auto note = [&](TensorId tid, std::set<void const *> &dst) {
                // Resolve view aliases to the owning buffer so a read/write
                // through a view inside the subtree is attributed to the parent
                // tensor: otherwise an op outside the control-flow node that
                // touches the owner sees no dependency and can be misordered.
                auto it = sub._tensors.find(sub.resolve_alias(tid));
                if (it != sub._tensors.end() && it->second.tensor_ptr != nullptr) {
                    dst.insert(it->second.tensor_ptr);
                    rep_handle.emplace(it->second.tensor_ptr, it->second);
                }
            };
            for (auto tid : nd.inputs) {
                note(tid, read_ptrs);
            }
            for (auto tid : nd.outputs) {
                note(tid, write_ptrs);
            }
        }
        sub.for_each_subgraph(collect);
    };

    for_each_child_graph(node, collect);

    // Map subtree buffer pointers back to this graph's TensorIds. A buffer used
    // only inside sub-graphs has no parent TensorId yet; register one (a stable
    // shared id) so two control-flow nodes touching the same buffer resolve to
    // the same id and a dependency edge forms between them.
    //
    // The lookup is a scan of the tensor TABLE rather than find_or_register_tensor_ptr,
    // which asks the pointer INDEX the same question. The two disagree after a rebind:
    // rebind_impl repoints TensorHandle::tensor_ptr at the caller's new tensor and leaves
    // the index naming the old address, so the index answers "not registered" for a buffer
    // the table holds. Registering a second id for it then gives the graph two interface
    // tensors of one name and its next manifest() refuses the graph.
    std::unordered_map<void const *, TensorId> ptr_to_tid;
    for (auto const &[tid, handle] : _tensors) {
        if (handle.tensor_ptr != nullptr) {
            ptr_to_tid.emplace(handle.tensor_ptr, tid);
        }
    }
    auto resolve = [&](void const *ptr) -> TensorId {
        auto it = ptr_to_tid.find(ptr);
        if (it != ptr_to_tid.end()) {
            return it->second;
        }
        TensorId const tid = register_tensor(rep_handle.at(ptr));
        ptr_to_tid.emplace(ptr, tid);
        return tid;
    };

    // Collect the mapped TensorIds into ordered sets so the appended order is
    // deterministic (the dependency edge set is order-independent, but a stable
    // order keeps builds and any downstream iteration reproducible).
    std::set<TensorId> add_in;
    std::set<TensorId> add_out;
    for (auto const *p : read_ptrs) {
        add_in.insert(resolve(p));
    }
    for (auto const *p : write_ptrs) {
        add_out.insert(resolve(p));
    }

    std::unordered_set<TensorId> have_in(ins.begin(), ins.end());
    std::unordered_set<TensorId> have_out(outs.begin(), outs.end());
    for (TensorId const tid : add_in) {
        if (have_in.insert(tid).second) {
            ins.push_back(tid);
        }
    }
    for (TensorId const tid : add_out) {
        if (have_out.insert(tid).second) {
            outs.push_back(tid);
        }
    }

    return {ins, outs};
}

std::pair<std::span<TensorId const>, std::span<TensorId const>> Graph::effective_io_cached(Node const &node, EffectiveIoCache &cache) {
    if (!is_control_flow(node.kind)) {
        // Ordinary nodes: their own I/O lists ARE the effective lists; hand
        // out views instead of heap-copying two vectors per node per scan.
        return {node.inputs, node.outputs};
    }
    // Control-flow nodes: the subtree walk is expensive, memoize it for the
    // duration of one sort (both hazard scans). The cache must NOT outlive
    // the call: passes like LoopInvariantHoisting move nodes across loop-body
    // boundaries, which changes the subtree I/O between sorts.
    auto it = cache.find(node.id);
    if (it == cache.end()) {
        it = cache.emplace(node.id, effective_io(node)).first;
    }
    return {it->second.first, it->second.second};
}

template <typename F>
void Graph::for_each_hazard_edge(EffectiveIoCache &cache, F &&emit) {
    // Storage-level aliasing must be resolved before anything reasons about
    // which buffer a node touches; cheap and idempotent after the first call.
    link_alias_storage();
    // Owner-resolved (resolve_alias), subtree-aware (effective_io_cached)
    // RAW/WAW/WAR scan. Every emitted edge points from an earlier to a later
    // position, so program order remains a valid topological order.
    //
    // Accesses through views with STATICALLY DISJOINT extents do not conflict:
    // per-slice writes like the CCSD ladder's ``r2[i,j] += ...`` touch
    // provably different elements of one parent, and serializing them (the old
    // owner-only scan) chained every slice of a tensor behind every other,
    // leaving parallel executors no width. Each describable view gets a
    // per-ROOT-axis interval box; two accesses conflict only when their boxes
    // may overlap. Anything unprovable - a runtime bound, a non-injective
    // parent layout, a whole-tensor access - keeps a null box, which overlaps
    // everything (the previous behavior). Element-disjoint writes commute
    // bitwise, so relaxing the order cannot change results.
    //
    // The box comes from StructuralAliasResolver, the SAME derivation
    // link_alias_structural writes onto the handles, and deliberately so: two
    // derivations of one alias relation disagreeing is the shape of both the
    // full-cover bug and the 32-hop cap. Sharing it also widened what is
    // describable here, since the walk composes chains - a permuted view and a
    // view of a view were both refused outright by the scan this replaces.
    using Box = std::vector<std::pair<std::int64_t, std::int64_t>>; // per root axis: [lo, hi)

    std::unordered_map<TensorId, Box>      view_box;    // view tid -> box in root axis space
    std::unordered_map<TensorId, TensorId> view_parent; // view tid -> alias ROOT tid
    StructuralAliasResolver                resolver(*this);
    for (auto const &nd : _nodes) {
        if (nd.kind != OpKind::View || nd.outputs.size() != 1 || !std::holds_alternative<ViewDescriptor>(nd.op_data)) {
            continue;
        }
        TensorId const         vid = nd.outputs[0];
        StructuralAlias const &res = resolver.resolve(vid);
        if (!res.box_known || res.root == vid || view_box.contains(vid)) {
            continue;
        }
        TensorHandle const *root = find_tensor(res.root);
        // A box covering the whole root is the same statement as NO box and
        // schedules better: a whole-tensor write dominates and retires the
        // writers before it, while a full-cover box is not recognized as
        // dominating anything and every later access takes an edge against all
        // of them.
        if (root == nullptr || whole_cover(res.box, root->dims)) {
            continue;
        }
        view_parent.emplace(vid, res.root);
        view_box.emplace(vid, res.box);
    }

    // Views that reached the graph without a View node (sliced outside a
    // capture, linked by storage containment at registration; or declared by a
    // manifest and linked at bind) carry their box on the handle instead.
    // Without this they would still be ordered against the parent correctly, but
    // as whole-tensor accesses, chaining every slice behind every other and
    // costing the parallel executors their width.
    for (auto const &[tid, h] : _tensors) {
        if (h.aliases == 0 || h.alias_box.empty() || view_box.contains(tid)) {
            continue;
        }
        if (resolve_alias(h.aliases) != h.aliases) {
            continue; // box lives in the immediate parent's axis space
        }
        view_parent.emplace(tid, h.aliases);
        view_box.emplace(tid, Box(h.alias_box.begin(), h.alias_box.end()));
    }

    auto const may_overlap = [](Box const *a, Box const *b) {
        if (a == nullptr || b == nullptr || a->size() != b->size()) {
            return true; // unprovable -> conservative
        }
        for (size_t d = 0; d < a->size(); ++d) {
            if (std::max((*a)[d].first, (*b)[d].first) >= std::min((*a)[d].second, (*b)[d].second)) {
                return false; // some axis with empty intersection -> disjoint
            }
        }
        return true;
    };

    // a fully inside b. A retired reader may only be dropped when the write
    // COVERS it: an overlapped-but-uncovered reader still needs WAR edges
    // against later writers that touch its uncovered part.
    auto const covered_by = [](Box const *a, Box const *b) {
        if (b == nullptr) {
            return true; // whole-tensor write covers everything
        }
        if (a == nullptr || a->size() != b->size()) {
            return false;
        }
        for (size_t d = 0; d < a->size(); ++d) {
            if ((*a)[d].first < (*b)[d].first || (*a)[d].second > (*b)[d].second) {
                return false;
            }
        }
        return true;
    };

    // Box of an access through @p raw against owner @p tid; null = whole tensor.
    // A View node's own read/write of its parent is the METADATA rebind of the
    // slice it describes, so it carries that slice's box - a whole-tensor read
    // here would re-serialize every consumer of every other slice through it.
    auto const box_of = [&](Node const &nd, TensorId raw, TensorId tid) -> Box const * {
        if (auto it = view_box.find(raw); it != view_box.end() && resolve_alias(view_parent.at(raw)) == tid) {
            return &it->second;
        }
        if (nd.kind == OpKind::View && nd.outputs.size() == 1) {
            // ``view_parent`` names the alias ROOT, which for a view of a view
            // is not @p raw (the immediate parent). Comparing against the
            // resolved owner is what keeps a chained view's own metadata
            // rebind boxed rather than widening to the whole buffer.
            if (auto it = view_box.find(nd.outputs[0]); it != view_box.end() && view_parent.at(nd.outputs[0]) == tid) {
                return &it->second;
            }
        }
        return nullptr;
    };

    struct Access {
        size_t     pos;
        Box const *box;
    };
    std::unordered_map<TensorId, std::vector<Access>> writers;
    std::unordered_map<TensorId, std::vector<Access>> readers;

    // The same three hazards over the PARAMETER table. A WriteParam's effect
    // and a parameter-bound View's dependence on it never touch a tensor, so
    // the owner-resolved scan above emits no edge between them and the two are
    // free to be reordered - which silently freezes the slice at whatever the
    // table happened to hold. Keyed by parameter name; see param_writes /
    // param_reads in Node.hpp.
    std::unordered_map<std::string, std::vector<size_t>> param_writers;
    std::unordered_map<std::string, std::vector<size_t>> param_readers;

    size_t const n = _nodes.size();
    for (size_t i = 0; i < n; i++) {
        auto [eff_in, eff_out] = effective_io_cached(_nodes[i], cache);
        for (auto raw : eff_in) {
            TensorId const tid = resolve_alias(raw);
            Box const     *box = box_of(_nodes[i], raw, tid);
            for (auto const &w : writers[tid]) {
                if (w.pos != i && may_overlap(w.box, box)) {
                    emit(w.pos, i); // RAW: writer -> reader
                }
            }
            readers[tid].push_back({.pos = i, .box = box});
        }
        for (auto raw : eff_out) {
            TensorId const tid = resolve_alias(raw);
            Box const     *box = box_of(_nodes[i], raw, tid);
            auto          &wl  = writers[tid];
            for (auto const &w : wl) {
                if (w.pos != i && may_overlap(w.box, box)) {
                    emit(w.pos, i); // WAW: prior writer -> this writer
                }
            }
            auto &rl = readers[tid];
            for (auto const &r : rl) {
                if (r.pos != i && may_overlap(r.box, box)) {
                    emit(r.pos, i); // WAR: prior reader -> this writer
                }
            }
            // Ordered readers are consumed; anything not fully covered must
            // stay visible to future writers of its uncovered part.
            std::erase_if(rl, [&](Access const &r) { return covered_by(r.box, box); });
            // A COVERED prior writer is consumed by the same argument, and for
            // the same reason it has to be covered rather than merely
            // overlapped. Anything later that reaches the covered writer
            // reaches this one too, so it takes an edge from this one, which
            // already carries an edge from the covered writer: the order
            // survives as a path. What that saves is quadratic - a buffer
            // written n times in a row used to keep all n writers and emit an
            // edge from every one of them to every later access - and it used
            // to be spelled as a whole-tensor special case, which stopped
            // applying the moment the write carried a box.
            std::erase_if(wl, [&](Access const &w) { return covered_by(w.box, box); });
            wl.push_back({.pos = i, .box = box});
        }

        for (auto const &pname : param_reads(_nodes[i])) {
            for (size_t const w : param_writers[pname]) {
                if (w != i) {
                    emit(w, i); // RAW: parameter write -> slice that resolves it
                }
            }
            param_readers[pname].push_back(i);
        }
        for (auto const &pname : param_writes(_nodes[i])) {
            for (size_t const w : param_writers[pname]) {
                if (w != i) {
                    emit(w, i); // WAW: the later write must win
                }
            }
            for (size_t const r : param_readers[pname]) {
                if (r != i) {
                    emit(r, i); // WAR: readers of the old value must go first
                }
            }
            param_readers[pname].clear();
            param_writers[pname].clear();
            param_writers[pname].push_back(i);
        }
    }
}

void Graph::rebuild_deps(EffectiveIoCache &cache) {
    // Position-keyed dependency lists for the current node order.
    size_t const n = _nodes.size();
    _deps.successors.assign(n, {});
    _deps.predecessors.assign(n, {});

    for_each_hazard_edge(cache, [&](size_t producer, size_t consumer) {
        _deps.successors[producer].push_back(consumer);
        _deps.predecessors[consumer].push_back(producer);
    });

    rebuild_levels();
}

void Graph::verify_level_independence() const {
    // Byte span per registered tensor. A tensor with no span that can be
    // reasoned about (deferred allocation, tiled layout) is skipped entirely
    // rather than guessed at; the hazard scan skips it too, so a conflict
    // through one is out of scope for both.
    struct Span {
        char const *lo;
        char const *hi;
    };
    std::unordered_map<TensorId, Span> span;
    for (auto const &[id, h] : _tensors) {
        char const *lo = nullptr;
        char const *hi = nullptr;
        if (handle_byte_span(h, lo, hi)) {
            span.emplace(id, Span{.lo = lo, .hi = hi});
        }
    }
    if (span.size() < 2) {
        return;
    }

    // Group tensors whose byte ranges overlap, by merging sorted intervals.
    // This is the independence that matters: it never consults `aliases`, so a
    // defect in the alias links cannot hide a conflict from this check.
    std::vector<std::pair<TensorId, Span>> ordered(span.begin(), span.end());
    std::ranges::sort(ordered, [](auto const &a, auto const &b) {
        return a.second.lo != b.second.lo ? a.second.lo < b.second.lo : a.second.hi > b.second.hi;
    });
    std::unordered_map<TensorId, size_t> group_of;
    std::vector<char const *>            group_end;
    for (auto const &[id, s] : ordered) {
        if (!group_end.empty() && s.lo < group_end.back()) {
            group_of[id]     = group_end.size() - 1;
            group_end.back() = std::max(group_end.back(), s.hi);
        } else {
            group_of[id] = group_end.size();
            group_end.push_back(s.hi);
        }
    }

    // Per group, the widest member, which is the only candidate for an axis
    // space every other member's region can be expressed in. A group with no
    // single container keeps a null root and every access in it is treated as
    // whole-group, which is conservative.
    std::vector<TensorId> root(group_end.size(), 0);
    for (auto const &[id, s] : ordered) {
        size_t const g = group_of[id];
        if (root[g] == 0) {
            root[g] = id; // sorted: first member of a group starts earliest and spans furthest
        } else {
            auto const &r = span.at(root[g]);
            if (s.lo < r.lo || s.hi > r.hi) {
                root[g] = 0; // no single container; fall back to conservative
                // Keep the group; a zero root simply disables the box test.
            }
        }
    }

    struct Access {
        size_t   pos;
        TensorId tid;
        bool     is_write;
    };
    using Box = std::vector<std::pair<std::int64_t, std::int64_t>>;

    auto const may_overlap = [](Box const *a, Box const *b) {
        if (a == nullptr || b == nullptr || a->size() != b->size()) {
            return true;
        }
        for (size_t d = 0; d < a->size(); ++d) {
            if (std::max((*a)[d].first, (*b)[d].first) >= std::min((*a)[d].second, (*b)[d].second)) {
                return false;
            }
        }
        return true;
    };

    // A region's box in its group root's axis space, derived from the two
    // handles alone. Memoized: a tensor read by many nodes derives once.
    std::unordered_map<TensorId, Box> box_cache;
    std::unordered_set<TensorId>      box_absent;
    auto const                        box_for = [&](TensorId tid, size_t g) -> Box const                        *{
        if (root[g] == 0 || box_absent.contains(tid)) {
            return nullptr;
        }
        if (auto it = box_cache.find(tid); it != box_cache.end()) {
            return &it->second;
        }
        auto const self  = _tensors.find(tid);
        auto const owner = _tensors.find(root[g]);
        if (self == _tensors.end() || owner == _tensors.end()) {
            box_absent.insert(tid);
            return nullptr;
        }
        Box derived;
        if (tid == root[g]) {
            derived.reserve(owner->second.dims.size());
            for (size_t const d : owner->second.dims) {
                derived.emplace_back(0, static_cast<std::int64_t>(d));
            }
        } else if (!derive_alias_box(owner->second, self->second, derived)) {
            box_absent.insert(tid);
            return nullptr;
        }
        return &box_cache.emplace(tid, std::move(derived)).first->second;
    };

    EffectiveIoCache cache;
    for (size_t level_index = 0; level_index < _deps.levels.size(); ++level_index) {
        auto const &level = _deps.levels[level_index];
        if (level.size() < 2) {
            continue;
        }
        // Only nodes sharing a storage group can conflict, so the pairwise
        // test runs per group rather than over the level.
        std::unordered_map<size_t, std::vector<Access>> touched;
        // A View node's effect is the handle it binds, so its hazards are
        // against the OTHER nodes that name that handle rather than against
        // the storage: bound = who binds a handle, named = who else names it.
        std::unordered_map<TensorId, std::vector<size_t>> bound;
        std::unordered_map<TensorId, std::vector<size_t>> named;
        for (size_t const pos : level) {
            auto const &node       = _nodes[pos];
            auto [eff_in, eff_out] = const_cast<Graph *>(this)->effective_io_cached(node, cache);
            // A View node writes its slice handle's dims, strides and data
            // pointer, and no element of the parent: the executor re-emplaces
            // the handle from the parent's own pointer and strides. So it is
            // NOT a writer of the storage it spans, and two of them are
            // independent however their slices overlap - which is what every
            // graph that records more than one view of one buffer depends on.
            //
            // What it does read is the parent's pointer and extents, so it
            // stays a READER of the region it describes: a node that moves the
            // parent's data or its allocation out from under it on the same
            // level is still a conflict. The read is attributed to the SLICE
            // rather than to the parent it names as an input, because the
            // parent's whole extent would conflict with every disjoint slice
            // written on the level.
            if (node.kind == OpKind::View && node.outputs.size() == 1) {
                TensorId const slice = node.outputs[0];
                if (auto it = group_of.find(slice); it != group_of.end()) {
                    touched[it->second].push_back({.pos = pos, .tid = slice, .is_write = false});
                }
                bound[slice].push_back(pos);
                for (auto const tid : eff_in) {
                    named[tid].push_back(pos); // a view of a view reads the parent handle
                }
                continue;
            }
            for (auto const tid : eff_in) {
                named[tid].push_back(pos);
                if (auto it = group_of.find(tid); it != group_of.end()) {
                    touched[it->second].push_back({.pos = pos, .tid = tid, .is_write = false});
                }
            }
            for (auto const tid : eff_out) {
                named[tid].push_back(pos);
                if (auto it = group_of.find(tid); it != group_of.end()) {
                    touched[it->second].push_back({.pos = pos, .tid = tid, .is_write = true});
                }
            }
        }

        // The hazard a metadata write still carries. Everything that reads or
        // writes a slice reads the handle the View node binds, so it may not
        // run alongside it; and two View nodes binding the SAME handle are a
        // write-write on that handle.
        for (auto const &[tid, binders] : bound) {
            size_t const self  = binders.front();
            size_t       other = self;
            if (binders.size() > 1) {
                other = binders[1];
            } else if (auto const it = named.find(tid); it != named.end()) {
                for (size_t const pos : it->second) {
                    if (pos != self) {
                        other = pos;
                        break;
                    }
                }
            }
            if (other == self) {
                continue;
            }
            EINSUMS_THROW_EXCEPTION(std::runtime_error,
                                    "Graph '{}': nodes {} ({}) and {} ({}) share execution level {} but one binds tensor {}'s view "
                                    "metadata while the other uses it. A level-scheduling executor launches them together, so this is "
                                    "a data race on the handle; the hazard scan should have ordered them.",
                                    _name, self, _nodes[self].label, other, _nodes[other].label, level_index, tid);
        }

        for (auto const &[g, accesses] : touched) {
            for (size_t a = 0; a < accesses.size(); ++a) {
                for (size_t b = a + 1; b < accesses.size(); ++b) {
                    if (accesses[a].pos == accesses[b].pos || (!accesses[a].is_write && !accesses[b].is_write)) {
                        continue;
                    }
                    if (!may_overlap(box_for(accesses[a].tid, g), box_for(accesses[b].tid, g))) {
                        continue;
                    }
                    EINSUMS_THROW_EXCEPTION(std::runtime_error,
                                            "Graph '{}': nodes {} ({}) and {} ({}) share execution level {} but both touch overlapping "
                                            "storage (tensors {} and {}, at least one written). A level-scheduling executor launches them "
                                            "together, so this is a data race; the hazard scan should have ordered them.",
                                            _name, accesses[a].pos, _nodes[accesses[a].pos].label, accesses[b].pos,
                                            _nodes[accesses[b].pos].label, level_index, accesses[a].tid, accesses[b].tid);
                }
            }
        }
    }
}

void Graph::rebuild_levels() {
    // Level partition for level-scheduling executors. Edges always point
    // from earlier to later positions (the hazard scan links prior
    // writers/readers to the current node), so one forward pass suffices.
    size_t const        n = _deps.predecessors.size();
    std::vector<size_t> level(n, 0);
    size_t              max_level = 0;
    for (size_t i = 0; i < n; i++) {
        for (size_t const pred : _deps.predecessors[i]) {
            level[i] = std::max(level[i], level[pred] + 1);
        }
        max_level = std::max(max_level, level[i]);
    }
    _deps.levels.assign(max_level + 1, {});
    for (size_t i = 0; i < n; i++) {
        _deps.levels[level[i]].push_back(i);
    }
}

bool Graph::run_thread_planner(unsigned threads) {
    passes::ThreadPlanning planner(threads);
    // This path builds its pass directly rather than through a PassManager, so
    // nothing else would ever give the planner a verbosity and every report it
    // makes would be unreachable from a normal run. A thread plan is the one
    // result here that cannot be inferred from the outside: whether it widened
    // anything, and if not which gate declined, is otherwise invisible.
    planner.set_verbosity(static_cast<int>(config::get(option::PassVerbosity)));
    planner.run(*this);
    _planned_thread_count = static_cast<std::uint16_t>(threads);
    return planner.num_widened() > 0;
}

bool Graph::plan_threads(bool freeze) {
    auto &budget = task_pool::WidthBudget::get_singleton();
    // The budget is what admission rations against, so it is also what a plan
    // has to be made for; asking the hardware directly could disagree with it.
    budget.sync_machine_width();
    unsigned const threads = std::max(1U, budget.total());

    // Timings are cleared at the start of every execute() and written during
    // it, so a non-empty set means this graph has completed at least one
    // replay and the planner has measurements instead of a model.
    bool have_timings = false;
    {
        std::scoped_lock const lock(*_content_mutex);
        have_timings = !_timing_samples.empty();
    }

    bool const widened = run_thread_planner(threads);

    // Cold plans are model plans, and the model is the part of this that is
    // guessing. What the first real timings buy is a TRIAL, not a decree: a
    // re-planned candidate has to beat the cold plan on the wall clock before
    // it may replace it (see finish_replay_thread_plan for why estimates
    // cannot referee that contest).
    _plan_trial = (!freeze && !have_timings) ? ThreadPlanTrial::Armed : ThreadPlanTrial::None;
    _plan_incumbent.clear();
    _plan_candidate.clear();
    return widened;
}

namespace {

using PlanSnapshot = std::vector<std::pair<std::uint16_t, std::int64_t>>;

/// Record every node's planned width and admission priority, container bodies
/// included, in one deterministic walk order shared with apply_thread_plan.
///
/// Setup bodies are deliberately left out of both walks: the widths these two
/// snapshot and restore are the ones ThreadPlanning::plan_graph writes, and that
/// planner descends into loop bodies and conditional branches only.
void collect_thread_plan(Graph &graph, PlanSnapshot &out) {
    for (auto &node : graph.nodes()) {
        out.emplace_back(node.thread_width, node.admission_priority);
        for_each_child_graph(
            node, [&out](Graph &sub) { collect_thread_plan(sub, out); }, /*include_setup=*/false);
    }
}

void apply_thread_plan(Graph &graph, PlanSnapshot const &plan, size_t &pos) {
    for (auto &node : graph.nodes()) {
        if (pos >= plan.size()) {
            return; // structure changed under the trial; leave the rest alone
        }
        node.thread_width       = plan[pos].first;
        node.admission_priority = plan[pos].second;
        pos++;
        for_each_child_graph(
            node, [&](Graph &sub) { apply_thread_plan(sub, plan, pos); }, /*include_setup=*/false);
    }
}

[[nodiscard]] bool same_widths(PlanSnapshot const &a, PlanSnapshot const &b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); i++) {
        if (a[i].first != b[i].first) {
            return false;
        }
    }
    return true;
}

/// How much faster the candidate's replay must be before it replaces the cold
/// plan. Consecutive replays of one graph differ by a few percent on a machine
/// doing anything else, and the trial reads exactly one replay of each side,
/// so a candidate inside this band is indistinguishable from noise - and the
/// incumbent is the plan a calibrated model chose on purpose.
constexpr double kThreadPlanTrialMargin = 0.05;

} // namespace

void Graph::finish_replay_thread_plan(double replay_ms) {
    switch (_plan_trial) {
    case ThreadPlanTrial::None:
        return;

    case ThreadPlanTrial::Armed: {
        // Disarmed BEFORE planning, so a planner that throws does not leave the
        // graph re-planning at the end of every replay from here on.
        _plan_trial = ThreadPlanTrial::None;

        _plan_incumbent.clear();
        collect_thread_plan(*this, _plan_incumbent);

        auto &budget = task_pool::WidthBudget::get_singleton();
        budget.sync_machine_width();
        run_thread_planner(std::max(1U, budget.total()));

        _plan_candidate.clear();
        collect_thread_plan(*this, _plan_candidate);
        if (same_widths(_plan_candidate, _plan_incumbent)) {
            // The timings agree with the model about the widths, so there is
            // nothing to referee; the re-plan's measured admission priorities
            // are kept and the plan is final.
            _plan_incumbent.clear();
            _plan_candidate.clear();
            return;
        }
        // The candidate's widths are live now; the next replay times them.
        _plan_trial = ThreadPlanTrial::Candidate;
        return;
    }

    case ThreadPlanTrial::Candidate: {
        _plan_candidate_ms = replay_ms;
        size_t pos         = 0;
        apply_thread_plan(*this, _plan_incumbent, pos);
        _plan_trial = ThreadPlanTrial::Incumbent;
        return;
    }

    case ThreadPlanTrial::Incumbent: {
        _plan_trial          = ThreadPlanTrial::None;
        bool const candidate = _plan_candidate_ms < (1.0 - kThreadPlanTrialMargin) * replay_ms;
        if (candidate) {
            size_t pos = 0;
            apply_thread_plan(*this, _plan_candidate, pos);
        }
        if (config::get(option::PassVerbosity) >= 1) {
            fmt::print(stderr, "[ThreadPlanning] trial on '{}': candidate replay {:.1f} ms vs incumbent replay {:.1f} ms; keeping the {}\n",
                       _name, _plan_candidate_ms, replay_ms, candidate ? "re-planned widths" : "cold plan");
        }
        _plan_incumbent.clear();
        _plan_incumbent.shrink_to_fit();
        _plan_candidate.clear();
        _plan_candidate.shrink_to_fit();
        return;
    }
    }
}

size_t Graph::schedule_edge_count() {
    topological_sort();
    size_t edges = 0;
    for (auto const &succ : _deps.successors) {
        edges += succ.size();
    }
    return edges;
}

std::vector<size_t> Graph::schedule_level_sizes() {
    topological_sort();
    std::vector<size_t> sizes;
    sizes.reserve(_deps.levels.size());
    for (auto const &level : _deps.levels) {
        sizes.push_back(level.size());
    }
    return sizes;
}

void Graph::topological_sort() {
    std::scoped_lock const lock(*_content_mutex);
    // Defense in depth: a pass that mutates the node list without declaring
    // it (mark_sorted / add_node) leaves stale flags. A count mismatch is the
    // detectable symptom; downgrade to a full re-sort instead of letting a
    // consumer index _deps out of range.
    if (_deps.successors.size() != _nodes.size()) {
        _deps_valid = false;
    }

    if (_sorted && _deps_valid) {
        // Node order and dependency lists both current; the ~20-pass default
        // pipeline hits this on every pass that follows a non-mutating one.
        return;
    }

    if (_nodes.empty()) {
        _deps.successors.clear();
        _deps.predecessors.clear();
        _deps.levels.clear();
        _sorted     = true;
        _deps_valid = true;
        return;
    }

    if (_sorted) {
        // A pass rebuilt or filtered the node list and vouched for the order
        // via mark_sorted(); only the position-keyed _deps are stale. This
        // also means a pass-chosen order (e.g. Reorder's memory-aware
        // schedule) survives instead of being re-derived by a fresh Kahn.
        EffectiveIoCache cache;
        rebuild_deps(cache);
        _deps_valid = true;
        return;
    }

    // Build adjacency from data dependencies:
    // If node A writes tensor T and node B reads tensor T (and B comes after A),
    // then A → B (A must execute before B).

    size_t const n = _nodes.size();

    // Track dependencies: read-after-write, write-after-write, write-after-read.
    // Keyed by *owner* TensorId (resolve_alias), so reads/writes through a view
    // register against the parent tensor. Without this, the scheduler would
    // treat ``GEMM(C_occ, …)`` and ``Syev(C, …)`` as independent, they're not,
    // since C_occ aliases C.
    std::vector<std::vector<size_t>> adj(n);
    std::vector<size_t>              in_degree(n, 0);

    // eff_cache memoizes effective I/O across this scan and the rebuild_deps
    // call further below; keyed by NodeId so it survives the move of nodes into
    // their sorted positions.
    EffectiveIoCache eff_cache;

    // The position-keyed dependency lists come out of THIS scan rather than a
    // second one. They are the same edges, and the scan is what this function
    // costs: it walks every node's effective I/O and intersects view boxes
    // pairwise, which on a DLPNO-MP2 iteration body (nodes carrying ~1000
    // operands each) is 28 ms of a 83 ms graph build - paid twice. They are
    // only valid if the sort leaves the nodes where they are, which is why the
    // Kahn loop below reports whether anything moved.
    _deps.successors.assign(n, {});
    _deps.predecessors.assign(n, {});

    for_each_hazard_edge(eff_cache, [&](size_t producer, size_t consumer) {
        adj[producer].push_back(consumer);
        in_degree[consumer]++;
        _deps.successors[producer].push_back(consumer);
        _deps.predecessors[consumer].push_back(producer);
    });

    // Kahn's algorithm, taking the smallest ready POSITION rather than FIFO.
    // Hazard edges always point from an earlier to a later position, so
    // program order is itself a valid topological order and this reproduces
    // it exactly. A FIFO queue does not: a zero-in-degree node late in
    // program order pops ahead of an earlier node that waits on any edge, so
    // an edge the hazard scan missed became a REORDER that broke even serial
    // replay, instead of staying harmless there.
    std::priority_queue<size_t, std::vector<size_t>, std::greater<>> ready;
    for (size_t i = 0; i < n; i++) {
        if (in_degree[i] == 0) {
            ready.push(i);
        }
    }

    std::vector<Node> sorted;
    sorted.reserve(n);

    bool reordered = false;
    while (!ready.empty()) {
        size_t const idx = ready.top();
        ready.pop();
        if (idx != sorted.size()) {
            reordered = true;
        }
        sorted.push_back(std::move(_nodes[idx]));

        for (size_t const succ : adj[idx]) {
            if (--in_degree[succ] == 0) {
                ready.push(succ);
            }
        }
    }

    if (sorted.size() != n) {
        EINSUMS_THROW_EXCEPTION(std::runtime_error, "Graph '{}': topological sort failed — cycle detected", _name);
    }

    _nodes = std::move(sorted);

    if (reordered) {
        // Positions moved, so the lists built during the scan are keyed to the
        // wrong slots and there is nothing to do but scan again.
        rebuild_deps(eff_cache);
    } else {
        // Every node stayed put, so the lists are already right and only the
        // level partition is missing. This is the normal outcome rather than a
        // lucky one: every hazard edge points from an earlier position to a
        // later one, so program order is itself a topological order, and the
        // priority queue above takes the smallest ready position, which
        // reproduces it. The `reordered` flag is what keeps that an observation
        // instead of an assumption - if the invariant ever breaks, this falls
        // back to the full rebuild rather than to stale lists.
        rebuild_levels();
    }

    _sorted     = true;
    _deps_valid = true;
    // Positions changed but the node COUNT did not, so cached position-keyed
    // analyses (UsageAnalysis) must be invalidated explicitly - the count
    // defense in usage() cannot see a same-size reorder.
    _analysis_version++;
}

std::tuple<Graph &, Graph &> Graph::add_conditional(std::string label, std::function<bool()> predicate) {
    // An EMPTY function is wrapped rather than turned into a literal, so a
    // caller who passed one still gets the std::bad_function_call it has always
    // got instead of silently taking a branch.
    return add_conditional(std::move(label), PredExpr::callback(std::move(predicate)));
}

std::tuple<Graph &, Graph &> Graph::add_conditional(std::string label, PredExpr predicate) {
    auto then_graph = std::make_shared<Graph>(label + "/then");
    auto else_graph = std::make_shared<Graph>(label + "/else");

    ConditionalDescriptor desc;
    desc.predicate   = std::move(predicate);
    desc.then_branch = then_graph;
    desc.else_branch = else_graph;

    // Dtype and rank are meaningless for a control-flow node, which has no
    // tensor destination; the builder neither dispatches on nor validates them.
    OpData op_data(std::move(desc));
    auto   executor = build_executor(OpKind::Conditional, packed_gemm::ScalarType::Unknown, 0, op_data, *this, {}, {});

    Node node;
    node.kind    = OpKind::Conditional;
    node.label   = std::move(label);
    node.execute = std::move(executor);
    node.op_data = std::move(op_data);

    add_node(std::move(node));

    return {*then_graph, *else_graph};
}

std::tuple<Graph &, Graph &> Graph::add_conditional_flag(std::string label, GateFlags const &flags, size_t index) {
    // The buffer, not the handle: the node has to keep reading the same array after the caller's
    // GateFlags goes out of scope, and a shared_ptr copy is what makes that true. This used to be
    // a lambda closing over that buffer; PredExpr::FlagTest is the same load expressed as data,
    // so the node is now saveable as well as GIL-free.
    return add_conditional(std::move(label), PredExpr::flag(flags, index));
}

Graph &Graph::add_loop(std::string label, size_t max_iterations, std::function<bool(size_t)> condition) {
    // An absent condition has always meant "run to max_iterations", and a
    // default PredExpr is an unconditional true, which says exactly that.
    return add_loop(std::move(label), max_iterations, condition ? PredExpr::callback(std::move(condition)) : PredExpr{});
}

Graph &Graph::add_loop(std::string label, size_t max_iterations, PredExpr condition) {
    auto body_graph = std::make_shared<Graph>(label + "/body");

    LoopDescriptor desc;
    desc.body           = body_graph;
    desc.max_iterations = max_iterations;
    desc.condition      = std::move(condition);
    // Shared with the executor, so the iteration count the replay writes is
    // observable on the node afterwards. See LoopDescriptor::last_iteration_count.
    desc.state = std::make_shared<LoopState>();

    OpData op_data(std::move(desc));
    auto   executor = build_executor(OpKind::Loop, packed_gemm::ScalarType::Unknown, 0, op_data, *this, {}, {});

    Node node;
    node.kind    = OpKind::Loop;
    node.label   = std::move(label);
    node.execute = std::move(executor);
    node.op_data = std::move(op_data);

    add_node(std::move(node));

    return *body_graph;
}

void Graph::add_loop(std::string label, size_t max_iterations, std::function<bool(size_t)> condition, std::function<void()> body_fn) {
    auto              &body = add_loop(std::move(label), max_iterations, std::move(condition));
    CaptureGuard const g(body);
    body_fn();
}

// ── Setup subgraphs ─────────────────────────────────────────────────────────

Graph &Graph::add_setup(std::string label) {
    return add_setup_at(std::move(label), _nodes.size());
}

Graph &Graph::add_setup_at(std::string label, std::size_t position) {
    auto body_graph = std::make_shared<Graph>(label + "/setup");

    // One parameter table with the parent, not a fresh one. A setup body is a PHASE of this
    // graph rather than a separate scope: a value it writes out with cg::write_param exists to
    // be read by the caller or by a later node here, and a body holding its own table would
    // write it somewhere nobody looks. That is how the fitting diagnostic went missing the
    // first time it was asked for.
    body_graph->set_params_ptr(_params);

    SetupDescriptor desc;
    desc.body = body_graph;
    // Shared with the executor, so the "already computed" answer a replay writes is the
    // one a later bind clears. See LoopDescriptor::state for why this is not a plain field.
    desc.state = std::make_shared<SetupState>();
    // A setup node added after a key was declared still belongs to the problem the caller
    // named; the key is graph state and the node is where it has to be readable from.
    desc.state->pending_key = _setup_key;

    OpData op_data(std::move(desc));
    auto   executor = build_executor(OpKind::Setup, packed_gemm::ScalarType::Unknown, 0, op_data, *this, {}, {});

    Node node;
    node.kind    = OpKind::Setup;
    node.label   = std::move(label);
    node.execute = std::move(executor);
    node.op_data = std::move(op_data);
    node.id      = reserve_node_id();

    // Through insert_node_groups rather than add_node, because that is the splice that
    // already knows how to keep positions valid; appending and then moving would be a second
    // way of doing it and one more thing to keep in step.
    std::vector<std::pair<std::size_t, std::vector<Node>>> group;
    group.emplace_back(std::min(position, _nodes.size()), std::vector<Node>{std::move(node)});
    insert_node_groups(std::move(group));

    return *body_graph;
}

void Graph::add_setup(std::string label, std::function<void()> body_fn) {
    auto              &body = add_setup(std::move(label));
    CaptureGuard const g(body);
    body_fn();
}

bool Graph::has_setup() const noexcept {
    return std::any_of(_nodes.begin(), _nodes.end(), [](Node const &node) { return node.kind == OpKind::Setup; });
}

void Graph::run_setup(bool force) {
    for (auto &node : _nodes) {
        auto *desc = std::get_if<SetupDescriptor>(&node.op_data);
        if (desc == nullptr || !desc->body) {
            continue;
        }
        if (force && desc->state != nullptr) {
            // Clear both, not just the flag: a stale key would otherwise let the very next
            // guard skip the body that this call exists to force.
            desc->state->computed = false;
            desc->state->computed_key.clear();
        }
        // Through the node's own executor rather than by calling body->execute() here, so
        // there is one place that decides whether a setup body runs. A second copy of the
        // two guards is a second thing to keep in step with the first.
        node.execute();
    }
}

void Graph::invalidate_setup() {
    for (auto &node : _nodes) {
        auto *desc = std::get_if<SetupDescriptor>(&node.op_data);
        if (desc == nullptr || desc->state == nullptr) {
            continue;
        }
        desc->state->computed = false;
    }
}

// ── The accuracy contract ───────────────────────────────────────────────────

namespace {

/// Whether @p record's bound applies to @p output.
///
/// A record naming no outputs applies to all of them, which is the honest answer for a pass
/// that rewrote something feeding every result; and an EMPTY question means "the graph-wide
/// worst case", which every record answers.
bool record_covers(ApproximationRecord const &record, std::string_view output) {
    if (output.empty() || record.outputs.empty()) {
        return true;
    }
    return std::find(record.outputs.begin(), record.outputs.end(), output) != record.outputs.end();
}

/// The outputs @p candidate could collide with an existing record over.
bool records_overlap(ApproximationRecord const &a, ApproximationRecord const &b) {
    if (a.outputs.empty() || b.outputs.empty()) {
        return true;
    }
    return std::any_of(a.outputs.begin(), a.outputs.end(),
                       [&b](std::string const &name) { return std::find(b.outputs.begin(), b.outputs.end(), name) != b.outputs.end(); });
}

} // namespace

std::string Graph::can_approximate(ApproximationRecord const &candidate) const {
    if (!std::isfinite(candidate.bound) || candidate.bound < 0) {
        return fmt::format("pass '{}' states a bound of {}, which is not a number an accuracy budget can be measured against; a lossy "
                           "rewrite has to say how large its effect is",
                           candidate.pass_name, candidate.bound);
    }

    // Records of DIFFERENT effects are allowed to coexist, and deliberately. They do not
    // convert into one another, but neither do they need to: composition is per effect
    // (@ref accuracy_spent counts one kind at a time) and @ref approximation_tolerance
    // carries the two sides separately, which is the honest representation of "this result
    // is off by so much in norm and so much per element". Refusing the second one would
    // make an ordinary pipeline, a factorization followed by a precision change,
    // unexpressible for no gain.
    //
    // Whether a pass's own error model still holds once something else has perturbed its
    // inputs is a question only that pass can answer, so it is asked of the pass rather
    // than decided here: @ref approximations is readable, and a pass that finds its bound
    // no longer defensible declines through @ref OptimizerPass::approximate like any other
    // refusal.
    if (!_accuracy_budget.has_value()) {
        return {};
    }
    auto const [budget_effect, budget] = *_accuracy_budget;
    if (budget_effect != candidate.effect) {
        return fmt::format("pass '{}' bounds itself {}, and this graph's accuracy budget is stated {}; a budget in other units is not a "
                           "budget this pass can spend against",
                           candidate.pass_name, approximation_effect_name(candidate.effect), approximation_effect_name(budget_effect));
    }

    // A budget IS the one place mixed effects have to be refused, and only because of what a
    // budget claims to be. It caps one kind of error; a record of another kind over the same
    // output is not counted by it and cannot be, so letting this through would leave a
    // caller with a cap they reasonably read as covering everything and that covers part.
    // Saying so is better than capping half of it in silence.
    for (auto const &existing : _approximations) {
        if (existing.effect == budget_effect || !records_overlap(existing, candidate)) {
            continue;
        }
        return fmt::format("pass '{}' would be spent against a {} budget, but '{}' has already been applied to the same outputs with a "
                           "{} bound, which that budget does not cap and cannot; clear the budget or state it in the other units",
                           candidate.pass_name, approximation_effect_name(budget_effect), existing.pass_name,
                           approximation_effect_name(existing.effect));
    }

    // Checked per named output rather than graph-wide, so two passes over DISJOINT outputs
    // each get the whole budget, which is what a budget on one output means.
    std::vector<std::string> const targets = candidate.outputs.empty() ? std::vector<std::string>{std::string{}} : candidate.outputs;
    for (auto const &target : targets) {
        double const composed = compose_approximation(candidate.effect, accuracy_spent(candidate.effect, target), candidate.bound);
        if (composed > budget) {
            return fmt::format("pass '{}' would take {} to {:g} on {}, over this graph's budget of {:g}", candidate.pass_name,
                               approximation_effect_name(candidate.effect), composed,
                               target.empty() ? std::string{"every output"} : fmt::format("'{}'", target), budget);
        }
    }
    return {};
}

void Graph::note_approximation(ApproximationRecord record) {
    if (std::string reason = can_approximate(record); !reason.empty()) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "Graph '{}': {}", _name, reason);
    }
    _approximations.push_back(std::move(record));
    // A record is saved structure, so adding one changes what a save writes and what a
    // content hash covers.
    _structure_version++;
}

void Graph::restore_approximations(std::vector<ApproximationRecord> records) {
    _approximations = std::move(records);
    _structure_version++;
}

double Graph::accuracy_spent(ApproximationEffect effect, std::string const &output) const {
    double spent = 0;
    for (auto const &record : _approximations) {
        if (record.effect == effect && record_covers(record, output)) {
            spent = compose_approximation(effect, spent, record.bound);
        }
    }
    return spent;
}

ApproximationTolerance Graph::approximation_tolerance(std::string const &output) const {
    ApproximationTolerance out;
    for (auto const &record : _approximations) {
        if (!record_covers(record, output)) {
            continue;
        }
        double &side = is_absolute_effect(record.effect) ? out.absolute : out.relative;
        side         = compose_approximation(record.effect, side, record.bound);
    }
    return out;
}

void Graph::set_accuracy_budget(ApproximationEffect effect, double value) {
    if (!std::isfinite(value) || value < 0) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                "Graph '{}': an accuracy budget of {} is not a bound anything can be checked against", _name, value);
    }
    _accuracy_budget = std::pair{effect, value};
}

void Graph::clear_accuracy_budget() {
    _accuracy_budget.reset();
}

void Graph::set_setup_key(std::string key) {
    _setup_key = std::move(key);
    for (auto &node : _nodes) {
        auto *desc = std::get_if<SetupDescriptor>(&node.op_data);
        if (desc == nullptr || desc->state == nullptr) {
            continue;
        }
        desc->state->pending_key = _setup_key;
    }
}

void Graph::update_prefactors(NodeId node_id, PrefactorScalar c_pf, PrefactorScalar ab_pf) {
    for (auto &node : _nodes) {
        if (node.id != node_id) {
            continue;
        }
        auto *desc = std::get_if<EinsumDescriptor>(&node.op_data);
        if (desc == nullptr) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                    "Graph '{}': node {} ({}) is not an einsum; update_prefactors only applies to einsum nodes", _name,
                                    node_id, op_kind_name(node.kind));
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
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "Graph::make_einsum_node: operand dtypes disagree; mixed-precision einsum is not "
                                                       "expressible through one runtime dispatch");
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

        auto *desc = std::get_if<EinsumDescriptor>(&node.op_data);
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

std::vector<Graph::NodeTiming> const &Graph::timing_report() const {
    if (_timing_report_valid) {
        return _timing_report;
    }

    // Attach labels here rather than on the replay path: one string copy per
    // node per REPORT instead of per node per execute().
    std::unordered_map<NodeId, std::string_view> labels;
    labels.reserve(_nodes.size());
    for (auto const &node : _nodes) {
        labels.emplace(node.id, node.label);
    }

    _timing_report.clear();
    _timing_report.reserve(_timing_samples.size());
    for (auto const &sample : _timing_samples) {
        auto const it = labels.find(sample.id);
        _timing_report.push_back({.id          = sample.id,
                                  .label       = it != labels.end() ? std::string(it->second) : fmt::format("node {}", sample.id),
                                  .kind        = sample.kind,
                                  .duration_ms = sample.duration_ms,
                                  .width       = sample.width});
    }
    _timing_report_valid = true;
    return _timing_report;
}

void Graph::print_timing_report(std::ostream &os) const {
    auto const &report = timing_report();
    if (report.empty()) {
        os << "No timing data available. Call execute() first.\n";
        return;
    }

    // Sort by duration descending
    auto sorted = report;
    std::ranges::sort(sorted, [](auto const &a, auto const &b) { return a.duration_ms > b.duration_ms; });

    double total = 0.0;
    for (auto const &t : sorted)
        total += t.duration_ms;

    os << fmt::format("Timing report for graph '{}' ({} nodes, {:.3f} ms total):\n", _name, sorted.size(), total);
    for (auto const &t : sorted) {
        double pct = (total > 0) ? 100.0 * t.duration_ms / total : 0.0;
        os << fmt::format("  {:8.3f} ms ({:5.1f}%)  [{}] {} ({})\n", t.duration_ms, pct, t.id, t.label, op_kind_name(t.kind));
    }
}

void Graph::rebuild_profile_strings() {
    _exec_zone_name = fmt::format("ComputeGraph::execute({})", _name);
    _exec_zone_id   = profile::intern_string(_exec_zone_name);
    _profile_strings.clear();
    _profile_strings.resize(_nodes.size());

    for (size_t idx = 0; idx < _nodes.size(); idx++) {
        auto const         &node  = _nodes[idx];
        NodeProfileStrings &entry = _profile_strings[idx];
        entry.node_id             = node.id;
        entry.zone                = fmt::format("graph:{}/{}", _name, node.label);
        entry.zone_id             = profile::intern_string(entry.zone);

        auto text = [&entry](std::string_view key, std::string_view value) {
            entry.texts.emplace_back(profile::intern_string(key), profile::intern_string(value));
        };
        auto number = [&entry](std::string_view key, int64_t value) { entry.numbers.emplace_back(profile::intern_string(key), value); };

        if (auto const *tdesc = std::get_if<TransferDescriptor>(&node.op_data)) {
            number("transfer_bytes", static_cast<int64_t>(tdesc->size_bytes));
        }

        if (auto const *desc = std::get_if<EinsumDescriptor>(&node.op_data)) {
            text("c_prefactor", to_string(desc->c_prefactor));
            text("ab_prefactor", to_string(desc->ab_prefactor));
            if (!desc->spec.c_indices.empty()) {
                text("c_indices", fmt::format("{}", fmt::join(desc->spec.c_indices, ",")));
                text("a_indices", fmt::format("{}", fmt::join(desc->spec.a_indices, ",")));
                text("b_indices", fmt::format("{}", fmt::join(desc->spec.b_indices, ",")));
            }
        } else if (auto const *sdesc = std::get_if<ScaleDescriptor>(&node.op_data)) {
            entry.reals.emplace_back(profile::intern_string("scale_factor"), as_real<double>(sdesc->factor));
        } else if (auto const *cdesc = std::get_if<CommDescriptor>(&node.op_data)) {
            number("comm_bytes", static_cast<int64_t>(cdesc->size_bytes));
            number("comm_tensor", static_cast<int64_t>(cdesc->tensor_id));
        }

        // A per-operand annotation is worth having for the handful of operands an
        // ordinary node carries, and is worth having for none of the operands of
        // a batched one. The strings are built once, but they are RE-EMITTED on
        // every replay: a 2048-member batched GEMM is 6144 annotation events per
        // execution, which measured 2.55x on the node and wrote a 191 KB report
        // line that nobody can read. Past the threshold the batch is described
        // rather than enumerated - which is the information anyway, since a
        // batch's members agree on their shape by construction.
        constexpr size_t kMaxOperandAnnotations = 16;

        auto annotate_tensors = [&](std::vector<TensorId> const &ids, char const *prefix) {
            if (ids.size() > kMaxOperandAnnotations) {
                number(fmt::format("{}.count", prefix), static_cast<int64_t>(ids.size()));
                std::vector<size_t> const *shared  = nullptr;
                bool                       uniform = true;
                for (TensorId const tid : ids) {
                    auto it = _tensors.find(tid);
                    if (it == _tensors.end() || it->second.dims.empty()) {
                        continue;
                    }
                    if (shared == nullptr) {
                        shared = &it->second.dims;
                    } else if (it->second.dims != *shared) {
                        uniform = false;
                        break;
                    }
                }
                if (shared != nullptr) {
                    text(fmt::format("{}.shape", prefix), uniform ? fmt::format("{}", fmt::join(*shared, "x")) : "mixed");
                }
                return;
            }
            for (TensorId const tid : ids) {
                auto it = _tensors.find(tid);
                if (it != _tensors.end() && !it->second.dims.empty()) {
                    text(fmt::format("{}.{}", prefix, it->second.name), fmt::format("{}", fmt::join(it->second.dims, "x")));
                    if (it->second.is_distributed) {
                        text(fmt::format("{}.{}.distributed", prefix, it->second.name), "true");
                    }
                }
            }
        };
        annotate_tensors(node.inputs, "input");
        annotate_tensors(node.outputs, "output");

        if (node.estimated_flops > 0) {
            number("estimated_flops", static_cast<int64_t>(node.estimated_flops));
        }
    }

    _profile_strings_valid = true;
}

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
                    gpu::memcpy_host_to_device(shadow, handle.data_ptr, tdesc->size_bytes);
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
                    if (shadow) {
                        gpu::memcpy_device_to_host(handle.data_ptr, shadow, tdesc->size_bytes);
                    }
                }
                // Unified memory: no copy needed, result is already in host-accessible memory.
            }
        } else if (node.target == Target::GPU) {
            // GPU node execution.
            std::vector<std::pair<TensorId, void *>> saved_ptrs;

            if constexpr (!gpu::has_unified_memory) {
                // Discrete GPU: swap tensor data pointers to device shadows.
                std::unordered_set<TensorId> swapped;
                auto                         swap_to_shadow = [&](TensorId tid) {
                    if (swapped.count(tid))
                        return;
                    auto &handle = _tensors[tid];
                    void *shadow = _device_shadows.ensure(tid, handle.total_bytes());
                    if (handle.swap_data) {
                        void *old_ptr = handle.swap_data(shadow);
                        saved_ptrs.emplace_back(tid, old_ptr);
                        swapped.insert(tid);
                    }
                };
                for (auto tid : node.inputs)
                    swap_to_shadow(tid);
                for (auto tid : node.outputs)
                    swap_to_shadow(tid);
            }
            // Unified memory: no swap needed, GPU reads tensor.data() directly.
            // MPS wrap_or_copy will create a zero-copy MTLBuffer wrapper.

            // Execute via GPU BLAS dispatch if possible, otherwise CPU fallback.
            bool gpu_dispatched = false;
            try {
                gpu_dispatched = try_gpu_blas_dispatch(node, _tensors, _device_shadows);
                if (gpu_dispatched) {
                    profile::annotate("gpu_dispatch", "gemm");
                }
            } catch (std::exception const &e) {
                EINSUMS_LOG_WARN("GPU GEMM dispatch failed for node {} ({}): {}", node.id, node.label, e.what());
            }

            if (!gpu_dispatched) {
                // Fall back to CPU lambda (with pointers still swapped to shadows).
                profile::annotate("gpu_dispatch", "cpu_fallback");
                if (node.cpu_fallback) {
                    try {
                        node.execute();
                    } catch (std::exception const &e) {
                        EINSUMS_LOG_WARN("GPU execution failed for node {} ({}): {}. Using CPU fallback.", node.id, node.label, e.what());
                        profile::annotate("gpu_fallback", "true");
                        node.cpu_fallback();
                        for (auto tid : node.outputs) {
                            auto it = _tensors.find(tid);
                            if (it != _tensors.end()) {
                                it->second.residency = Residency::Host;
                            }
                        }
                    }
                } else {
                    node.execute();
                }
            }

            // 3. Restore original host pointers (only needed on discrete GPU).
            for (auto const &[tid, old_ptr] : saved_ptrs) {
                auto &handle = _tensors[tid];
                if (handle.swap_data) {
                    handle.swap_data(old_ptr);
                }
            }
        } else {
            // CPU node: execute normally.
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
        // Copy shadows for tensors whose last writer was a GPU node.
        std::unordered_set<TensorId> cpu_written;
        for (auto const &n : _nodes) {
            if (n.target == Target::CPU && n.kind != OpKind::HostToDevice && n.kind != OpKind::DeviceToHost) {
                for (auto tid : n.outputs)
                    cpu_written.insert(tid);
            }
        }

        for (auto &[tid, handle] : _tensors) {
            if (cpu_written.count(tid))
                continue;
            void const *shadow = _device_shadows.get(tid);
            if (shadow && handle.data_ptr) {
                gpu::memcpy_device_to_host(handle.data_ptr, shadow, handle.total_bytes());
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
        profile::ScopedZone const      _zone(site,
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

void Graph::print_dot(std::ostream &os) const {
    os << "digraph \"" << _name << "\" {\n";
    os << "  rankdir=TB;\n";

    // Tensor nodes (rectangles)
    for (auto const &[id, handle] : _tensors) {
        os << fmt::format("  T{} [shape=box, label=\"{}\\n", id, handle.name);
        if (!handle.dims.empty()) {
            os << fmt::format("({})", fmt::join(handle.dims, "x"));
        }
        os << "\"];\n";
    }

    // Operation nodes (ellipses, colored by target/type)
    for (auto const &node : _nodes) {
        std::string style;
        if (node.kind == OpKind::HostToDevice || node.kind == OpKind::DeviceToHost) {
            style = ", style=filled, fillcolor=\"#FFA500\""; // orange for transfers
        } else if (node.target == Target::GPU) {
            style = ", style=filled, fillcolor=\"#6495ED\""; // cornflower blue for GPU
        }

        std::string label = node.label;
        if (auto const *desc = std::get_if<TransferDescriptor>(&node.op_data)) {
            label += fmt::format("\\n({} bytes)", desc->size_bytes);
        }

        os << fmt::format("  N{} [shape=ellipse, label=\"{}\"{}];\n", node.id, label, style);

        for (auto tid : node.inputs) {
            os << fmt::format("  T{} -> N{};\n", tid, node.id);
        }
        for (auto tid : node.outputs) {
            os << fmt::format("  N{} -> T{};\n", node.id, tid);
        }
    }

    os << "}\n";
}

void Graph::print_summary(std::ostream &os) const {
    // The names an operand list reads as, with the placeholder for an id this graph has no
    // handle for. Both lists print the same way, so the join is written once.
    auto const name_list = [this](std::vector<TensorId> const &ids) {
        std::vector<std::string_view> names;
        names.reserve(ids.size());
        for (auto const tid : ids) {
            auto const it = _tensors.find(tid);
            names.emplace_back(it != _tensors.end() ? std::string_view{it->second.name} : std::string_view{"?"});
        }
        return fmt::format("{}", fmt::join(names, ", "));
    };

    os << fmt::format("Graph '{}': {} nodes, {} tensors\n", _name, _nodes.size(), _tensors.size());
    for (auto const &node : _nodes) {
        os << fmt::format("  [{}] {} ({})\n", node.id, node.label, op_kind_name(node.kind));
        if (!node.inputs.empty()) {
            os << fmt::format("    inputs: {}\n", name_list(node.inputs));
        }
        if (!node.outputs.empty()) {
            os << fmt::format("    outputs: {}\n", name_list(node.outputs));
        }
    }
}

// ── JSON serialization ─────────────────────────────────────────────────────

std::string Graph::to_json() const {
    std::scoped_lock const lock(*_content_mutex);
    // Build a ComputeGraphData struct from internal state, then serialize it.
    // This is cleaner than manual JSON string building and uses the shared types
    // that the viewer also understands.

    ComputeGraphData data;
    data.name           = _name;
    data.pipeline_name  = _pipeline_name;
    data.workspace_name = _workspace_name;
    data.stage_name     = _stage_name;
    data.stage_type     = _stage_type;
    data.stage_index    = _stage_index;

    // Tensors
    for (auto const &[id, h] : _tensors) {
        GraphTensorData td;
        td.id              = id;
        td.name            = h.name;
        td.rank            = h.rank;
        td.dims            = h.dims;
        td.element_size    = h.element_size;
        td.dtype           = scalar_type_name(h.dtype);
        td.is_intermediate = h.is_intermediate;
        data.tensors.push_back(std::move(td));
    }

    // Build timing lookup
    std::unordered_map<NodeId, double> timing_map;
    for (auto const &t : _timing_samples)
        timing_map[t.id] = t.duration_ms;

    // Nodes: use array index as ID
    for (size_t ni = 0; ni < _nodes.size(); ni++) {
        auto const   &node = _nodes[ni];
        GraphNodeData nd;
        nd.id        = ni;
        nd.kind      = std::string(op_kind_name(node.kind));
        nd.label     = node.label;
        nd.target    = (node.target == Target::GPU) ? "GPU" : "CPU";
        nd.stream_id = node.stream_id;
        nd.inputs    = node.inputs;
        nd.outputs   = node.outputs;

        // Timing
        auto tit = timing_map.find(node.id);
        if (tit != timing_map.end())
            nd.timing_ms = tit->second;

        // Operation-specific data
        if (auto const *desc = std::get_if<EinsumDescriptor>(&node.op_data)) {
            // GraphNodeData is a viewer-facing snapshot; project complex
            // prefactors to their real part for display. The live value is
            // preserved on the descriptor itself.
            nd.c_prefactor  = as_real<double>(desc->c_prefactor);
            nd.ab_prefactor = as_real<double>(desc->ab_prefactor);
            nd.c_indices    = fmt::format("{}", fmt::join(desc->spec.c_indices, ","));
            nd.a_indices    = fmt::format("{}", fmt::join(desc->spec.a_indices, ","));
            nd.b_indices    = fmt::format("{}", fmt::join(desc->spec.b_indices, ","));
            nd.conj_a       = desc->conj_a;
            nd.conj_b       = desc->conj_b;
        } else if (auto const *desc = std::get_if<ScaleDescriptor>(&node.op_data)) {
            // Same viewer-facing projection as the einsum prefactors above.
            nd.scale_factor = as_real<double>(desc->factor);
        } else if (auto const *desc = std::get_if<PermuteDescriptor>(&node.op_data)) {
            // Same viewer-facing projection as the einsum prefactors above.
            nd.alpha     = desc->alpha.real();
            nd.beta      = desc->beta.real();
            nd.c_indices = fmt::format("{}", fmt::join(desc->c_indices, ","));
            nd.a_indices = fmt::format("{}", fmt::join(desc->a_indices, ","));
        }

        data.nodes.push_back(std::move(nd));
    }

    // Dependency edges
    {
        // Key by the owning buffer: a write through a view of T and a read of T
        // (or of another view of it) share storage, so their dependency edge is
        // only found once aliases resolve to the same id -- the same reason the
        // schedulers resolve_alias before deriving edges.
        std::unordered_map<TensorId, std::vector<size_t>> writers, readers;
        for (size_t i = 0; i < _nodes.size(); i++) {
            for (auto in_tid : _nodes[i].inputs)
                readers[resolve_alias(in_tid)].push_back(i);
            for (auto out_tid : _nodes[i].outputs)
                writers[resolve_alias(out_tid)].push_back(i);
        }

        std::set<std::pair<size_t, size_t>> emitted;
        for (auto const &[tid, reader_list] : readers) {
            auto wit = writers.find(tid);
            if (wit == writers.end())
                continue;

            for (size_t const r : reader_list) {
                size_t best_w = SIZE_MAX;
                for (size_t const w : wit->second)
                    if (w < r && w != r && (best_w == SIZE_MAX || w > best_w))
                        best_w = w;
                if (best_w == SIZE_MAX)
                    for (size_t const w : wit->second)
                        if (w > r && w != r && (best_w == SIZE_MAX || w > best_w))
                            best_w = w;
                if (best_w == SIZE_MAX || best_w == r)
                    continue;

                if (emitted.count({best_w, r}))
                    continue;
                emitted.insert({best_w, r});

                GraphEdgeData edge;
                edge.from      = best_w;
                edge.to        = r;
                edge.tensor_id = tid;
                edge.loop_back = (best_w > r);
                data.edges.push_back(edge);
            }
        }
    }

    // Serialize to JSON manually (structured, no Glaze dependency in library)
    std::string j;
    j.reserve(4096);

    auto esc = [](std::string const &s) -> std::string { return json_escape(s); };

    j += R"({"name":")" + esc(data.name) + "\"";
    if (!data.pipeline_name.empty())
        j += R"(,"pipeline_name":")" + esc(data.pipeline_name) + "\"";
    if (!data.workspace_name.empty())
        j += R"(,"workspace_name":")" + esc(data.workspace_name) + "\"";
    if (!data.stage_name.empty())
        j += R"(,"stage_name":")" + esc(data.stage_name) + "\"";
    if (!data.stage_type.empty())
        j += R"(,"stage_type":")" + esc(data.stage_type) + "\"";
    if (data.stage_index >= 0)
        j += ",\"stage_index\":" + std::to_string(data.stage_index);

    j += ",\"tensors\":[";
    for (size_t i = 0; i < data.tensors.size(); i++) {
        auto const &t = data.tensors[i];
        if (i > 0)
            j += ",";
        j += fmt::format(R"({{"id":{},"name":"{}","rank":{},"dims":[)", t.id, esc(t.name), t.rank);
        j += fmt::format("{}", fmt::join(t.dims, ","));
        j += fmt::format(R"(],"element_size":{},"dtype":"{}","is_intermediate":{}}})", t.element_size, esc(t.dtype),
                         t.is_intermediate ? "true" : "false");
    }
    j += "]";

    j += ",\"nodes\":[";
    for (size_t i = 0; i < data.nodes.size(); i++) {
        auto const &n = data.nodes[i];
        if (i > 0)
            j += ",";
        j += fmt::format(R"({{"id":{},"kind":"{}","label":"{}","target":"{}","stream_id":{})", n.id, esc(n.kind), esc(n.label),
                         esc(n.target), n.stream_id);
        j += fmt::format(R"(,"inputs":[{}])", fmt::join(n.inputs, ","));
        j += fmt::format(R"(,"outputs":[{}])", fmt::join(n.outputs, ","));
        if (n.timing_ms >= 0)
            j += fmt::format(",\"timing_ms\":{:.6f}", n.timing_ms);
        if (n.c_prefactor != 0.0 || n.ab_prefactor != 1.0)
            j += fmt::format(R"(,"c_prefactor":{},"ab_prefactor":{})", n.c_prefactor, n.ab_prefactor);
        if (n.scale_factor != 1.0)
            j += fmt::format(",\"scale_factor\":{}", n.scale_factor);
        if (n.alpha != 1.0 || n.beta != 0.0)
            j += fmt::format(R"(,"alpha":{},"beta":{})", n.alpha, n.beta);
        if (!n.c_indices.empty())
            j += R"(,"c_indices":")" + esc(n.c_indices) + "\"";
        if (!n.a_indices.empty())
            j += R"(,"a_indices":")" + esc(n.a_indices) + "\"";
        if (!n.b_indices.empty())
            j += R"(,"b_indices":")" + esc(n.b_indices) + "\"";
        if (n.conj_a)
            j += ",\"conj_a\":true";
        if (n.conj_b)
            j += ",\"conj_b\":true";
        j += "}";
    }
    j += "]";

    j += ",\"edges\":[";
    for (size_t i = 0; i < data.edges.size(); i++) {
        auto const &e = data.edges[i];
        if (i > 0)
            j += ",";
        j += fmt::format(R"({{"from":{},"to":{},"tensor_id":{})", e.from, e.to, e.tensor_id);
        if (e.loop_back)
            j += ",\"loop_back\":true";
        j += "}";
    }
    j += "]}";

    return j;
}

// ── Global graph registry ──────────────────────────────────────────────────

namespace {

std::mutex           g_registry_mutex;
std::vector<Graph *> g_registered_graphs;

} // namespace

void register_graph(Graph *graph) {
    std::scoped_lock const lock(g_registry_mutex);

    // On first registration, wire up the profiler handler
    static bool handler_registered = false;
    if (!handler_registered) {
#if defined(EINSUMS_HAVE_PROFILER)
        auto *srv = profile::Profiler::instance().server();
        if (srv) {
            srv->register_handler("get_compute_graphs", [](std::string const &) { return registered_graphs_json(); });
        }
#endif
        handler_registered = true;
    }

    // Replace if same name already registered
    for (auto &g : g_registered_graphs) {
        if (g->name() == graph->name()) {
            g = graph;
            return;
        }
    }
    g_registered_graphs.push_back(graph);
}

namespace {
/// Cache of graph JSON for graphs that have been destroyed, keyed by graph
/// name with the same replace-on-collision rule as the live registry. This
/// allows export_session() to include graph data even after the Graph objects
/// go out of scope. Keying by name is what bounds it: a run that builds and
/// destroys a same-named graph repeatedly re-caches one entry instead of
/// appending forever.
std::vector<std::pair<std::string, std::string>> g_cached_graph_jsons;

/// Whether anything can ever read a dead graph's JSON: the shutdown exporter
/// (--einsums:profile:save) or an attached viewer client. Serializing a graph
/// is O(nodes) string building and runs in the destructor, inside whatever
/// phase happens to drop the graph - measured at 41 ms of a 345 ms DLPNO
/// transform phase before it was gated - so it must not happen on the default
/// path, where the profiler server listens but nobody is collecting. A viewer
/// that attaches later still sees every live graph; what it loses is only the
/// post-mortem record of graphs that died before it connected.
bool graph_json_cache_wanted() {
#if defined(EINSUMS_HAVE_PROFILER)
    auto &prof = profile::Profiler::instance();
    if (!prof.enabled())
        return false;
    // Read the config each time rather than latching it: a graph dies once,
    // so this is not a hot path, and the tests flip the key at runtime.
    bool save_configured = false;
    try {
        save_configured = !config::get(option::ProfileSave).empty();
    } catch (...) { // NOLINT
    }
    if (save_configured)
        return true;
    auto const *srv = prof.server();
    return srv != nullptr && srv->has_client();
#else
    return false;
#endif
}
} // namespace

void unregister_graph(Graph *graph) {
    std::scoped_lock const lock(g_registry_mutex);

    auto it = std::ranges::find(g_registered_graphs, graph);
    if (it == g_registered_graphs.end())
        return;
    g_registered_graphs.erase(it);

    if (!graph_json_cache_wanted())
        return;

    // Cache the graph's JSON before it is gone, so it survives destruction.
    auto cached = std::ranges::find_if(g_cached_graph_jsons, [&](auto const &entry) { return entry.first == graph->name(); });
    if (cached != g_cached_graph_jsons.end()) {
        cached->second = graph->to_json();
    } else {
        g_cached_graph_jsons.emplace_back(graph->name(), graph->to_json());
    }
}

std::string registered_graphs_json() {
    std::scoped_lock const lock(g_registry_mutex);
    std::string            result = "{\"graphs\":[";
    bool                   first  = true;

    // Include live graphs.
    for (auto *g : g_registered_graphs) {
        if (!first)
            result += ",";
        first = false;
        result += g->to_json();
    }

    // Include cached JSON from destroyed graphs, skipping any name a live
    // graph has since reclaimed - the live one is the current truth.
    for (auto const &[name, json] : g_cached_graph_jsons) {
        if (std::ranges::any_of(g_registered_graphs, [&](Graph const *g) { return g->name() == name; }))
            continue;
        if (!first)
            result += ",";
        first = false;
        result += json;
    }

    result += "]}";
    return result;
}

EINSUMS_NAMESPACE_END(compute_graph)
