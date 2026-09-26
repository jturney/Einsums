//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file GpuDispatch.cpp
/// @brief The GPU BLAS fast paths, and the shapes each one insists on.
///
/// One predicate per operation, each of which recognizes the descriptor it can
/// serve and declines everything else, so that a node the GPU cannot take is a
/// `false` rather than an error. @ref try_gpu_blas_dispatch is the ordered
/// cascade over them and the only name that leaves this file.
///
/// The operand-pointer question is answered once, in @ref resolve_device_ptr:
/// on unified memory the GPU reads the host tensor directly, and on a discrete
/// device it reads a shadow buffer keyed by tensor id. Every fast path below
/// went through that branch per operand before it was one function.

#include "GpuDispatch.hpp"

#include <Einsums/CXX23/Expected.hpp>
#include <Einsums/ComputeGraph/CaptureContext.hpp>
#include <Einsums/ComputeGraph/Detail/ScalarDispatch.hpp>
#include <Einsums/ComputeGraph/EinsumSpec.hpp>
#include <Einsums/ComputeGraph/Error.hpp>
#include <Einsums/ComputeGraph/ExecutorBuilder.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Options.hpp>
#include <Einsums/ComputeGraph/SpaceRegistryAccess.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/GPU/BLAS.hpp>
#include <Einsums/Profile/Profile.hpp>
#include <Einsums/Tensor/Tensor.hpp>

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

#include "../StridedBatchedEinsum.hpp"

EINSUMS_NAMESPACE_BEGIN(compute_graph::gpu_dispatch)

/// @see GpuDispatch.hpp for why this exists rather than reading
/// TensorHandle::data_ptr.
[[nodiscard]] void *live_host_ptr(TensorHandle const &h) {
    if (!h.impl_fn || h.dtype == packed_gemm::ScalarType::Unknown) {
        return h.data_ptr; // no impl to read through; snapshot is all there is
    }
    void *raw = h.impl_fn();
    if (raw == nullptr) {
        return h.data_ptr;
    }
    void *out = nullptr;
    detail::dispatch_scalar_type(h.dtype, [&]<typename T>(T /*tag*/) {
        auto *impl = static_cast<::einsums::detail::TensorImpl<T> *>(raw);
        out        = static_cast<void *>(impl->data());
    });
    return out;
}

namespace {

/// Live rank-2 geometry, and whether the buffer is plain column-major packed.
///
/// The gpu::blas dispatchers hand cuBLAS a bare pointer plus a leading
/// dimension, which only describes the data when the operand is contiguous
/// column-major. They took M, N, K and lda/ldb/ldc from TensorHandle::dims -
/// registration-time snapshots, with no stride check at all - so a view, a
/// sliced operand, or a tensor whose extents changed after registration was
/// passed to the GEMM with the wrong shape and the wrong leading dimension.
///
/// @return False when the tensor is not a materialized, contiguous,
/// column-major rank-2 buffer, in which case the caller must decline and let
/// the CPU path handle it.
[[nodiscard]] bool live_colmajor_2d(TensorHandle const &h, int64_t &rows, int64_t &cols) {
    if (!h.impl_fn || h.dtype == packed_gemm::ScalarType::Unknown) {
        return false;
    }
    void *raw = h.impl_fn();
    if (raw == nullptr) {
        return false;
    }
    bool ok = false;
    detail::dispatch_scalar_type(h.dtype, [&]<typename T>(T /*tag*/) {
        auto const *impl = static_cast<::einsums::detail::TensorImpl<T> const *>(raw);
        auto const &d    = impl->dims();
        auto const &st   = impl->strides();
        if (d.size() != 2 || st.size() != 2) {
            return;
        }
        rows = static_cast<int64_t>(d[0]);
        cols = static_cast<int64_t>(d[1]);
        ok   = (st[0] == 1) && (static_cast<int64_t>(st[1]) == rows);
    });
    return ok;
}

/// Resolve the device-visible pointer for one operand. On unified memory the GPU
/// reads the host tensor's data directly; on a discrete device the data has been
/// copied into a shadow buffer keyed by tensor id. Shared by every try_gpu_*
/// helper below, which otherwise repeated this if-constexpr branch per operand.
[[nodiscard]] inline void *resolve_device_ptr([[maybe_unused]] TensorHandle const &h, [[maybe_unused]] TensorId id,
                                              [[maybe_unused]] DeviceShadowMap &shadows) {
    if constexpr (gpu::has_unified_memory) {
        // The LIVE buffer, not the one registered at capture: a Free and a later Materialize
        // release and reallocate an eager scratch, and the registration-time pointer then names
        // freed memory that a replayed GEMM wrote into.
        return live_host_ptr(h);
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

    auto const  lists = live_index_lists(desc);
    auto const &ci    = lists.c;    // e.g., ["i", "j"]
    auto const &ai    = lists.a;    // e.g., ["i", "k"]
    auto const &bi    = lists.b;    // e.g., ["k", "j"]
    auto const &li    = lists.link; // e.g., ["k"]

    if (ci.size() != 2 || ai.size() != 2 || bi.size() != 2)
        return false;

    // C is column-major: C[row, col] with dims[0]=rows, dims[1]=cols.
    // M = C rows, N = C cols.
    // Live, contiguity-checked geometry. Declining here costs a CPU fallback;
    // trusting the snapshot costs a wrong answer.
    int64_t a_rows = 0, a_cols = 0, b_rows = 0, b_cols = 0, c_rows = 0, c_cols = 0;
    if (!live_colmajor_2d(ha, a_rows, a_cols) || !live_colmajor_2d(hb, b_rows, b_cols) || !live_colmajor_2d(hc, c_rows, c_cols)) {
        return false;
    }

    auto    M = c_rows;
    auto    N = c_cols;
    int64_t K = 0;

    // Find K from the link index dimension.
    std::string const &link = li[0];
    // K is the dimension of the link index in A (or B).
    for (size_t d = 0; d < 2; d++) {
        if (ai[d] == link) {
            K = (d == 0) ? a_rows : a_cols;
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

    // Verify the operands really have the shapes the index analysis just
    // assigned them. A and B were picked by position in node.inputs, on the
    // assumption that the first non-C input is the spec's A - nothing enforces
    // that, and when the order differs the transpose flags and leading
    // dimensions are computed against the wrong tensor, which produces a
    // confidently wrong GEMM rather than a failure. Cross-check against the
    // live extents and decline if they disagree.
    auto const a_ok = (transa == 'n') ? (a_rows == M && a_cols == K) : (a_rows == K && a_cols == M);
    auto const b_ok = (transb == 'n') ? (b_rows == K && b_cols == N) : (b_rows == N && b_cols == K);
    if (!a_ok || !b_ok) {
        EINSUMS_LOG_DEBUG("try_gpu_gemm: operand extents disagree with the index analysis "
                          "(A={}x{} transa={}, B={}x{} transb={}, M={} N={} K={}); leaving node {} on the host",
                          a_rows, a_cols, transa, b_rows, b_cols, transb, M, N, K, node.id);
        return false;
    }

    auto lda = a_rows; // leading dimension = rows, valid because the operand is packed
    auto ldb = b_rows;
    auto ldc = c_rows;

    // Dispatch based on dtype.
    if (ha.dtype == packed_gemm::ScalarType::Float32) {
        auto alpha = as<float>(live_ab_prefactor(desc));
        auto beta  = as<float>(live_c_prefactor(desc));
        gpu::blas::gemm<float>(transa, transb, M, N, K, alpha, static_cast<float const *>(ptr_a), lda, static_cast<float const *>(ptr_b),
                               ldb, beta, static_cast<float *>(ptr_c), ldc);
        return true;
    } else if (ha.dtype == packed_gemm::ScalarType::Float64) {
        auto alpha = as<double>(live_ab_prefactor(desc));
        auto beta  = as<double>(live_c_prefactor(desc));
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

    auto const  lists = live_index_lists(desc);
    auto const &ai    = lists.a;
    auto const &bi    = lists.b; // "b" is actually x for GEMV
    auto const &ci    = lists.c; // "c" is actually y for GEMV
    auto const &li    = lists.link;

    if (ci.size() != 1 || li.size() != 1)
        return false;

    // A is M×N column-major (dims[0]=M=rows, dims[1]=N=cols).
    int64_t a_rows = 0, a_cols = 0;
    if (!live_colmajor_2d(ha, a_rows, a_cols)) {
        return false;
    }
    auto          M   = a_rows;
    auto          N   = a_cols;
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
        auto alpha = as<float>(live_ab_prefactor(desc));
        auto beta  = as<float>(live_c_prefactor(desc));
        gpu::blas::gemv<float>(trans, M, N, alpha, static_cast<float const *>(ptr_a), lda, static_cast<float const *>(ptr_x), 1, beta,
                               static_cast<float *>(ptr_y), 1);
        return true;
    } else if (ha.dtype == packed_gemm::ScalarType::Float64) {
        auto alpha = as<double>(live_ab_prefactor(desc));
        auto beta  = as<double>(live_c_prefactor(desc));
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

    auto const *desc = node.op_data.get_if<ScaleDescriptor>();
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
    //
    // The live factor, not the capture-time snapshot: a pass that rescales the
    // node writes the params block the host executor reads.
    auto const &factor = live_factor(*desc);
    if (!is_real_valued(factor)) {
        return false;
    }
    if (handle.dtype == packed_gemm::ScalarType::Float32) {
        gpu::blas::scal<float>(n, as_real<float>(factor), static_cast<float *>(ptr), 1);
        return true;
    } else if (handle.dtype == packed_gemm::ScalarType::Float64) {
        gpu::blas::scal<double>(n, as_real<double>(factor), static_cast<double *>(ptr), 1);
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
    auto const *desc = node.op_data.get_if<AxpbyDescriptor>();
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

/// Issue @p plan as one strided-batched GEMM over device pointers, in the einsum's operand order;
/// the plan says whether BLAS takes them swapped.
template <typename T>
void run_gpu_strided_batched(detail::StridedBatchPlan const &plan, T alpha, T beta, void const *ptr_a, void const *ptr_b, void *ptr_c) {
    auto const        *a        = static_cast<T const *>(plan.swap_ab ? ptr_b : ptr_a);
    auto const        *b        = static_cast<T const *>(plan.swap_ab ? ptr_a : ptr_b);
    std::int64_t const stride_a = plan.swap_ab ? plan.stride_b : plan.stride_a;
    std::int64_t const stride_b = plan.swap_ab ? plan.stride_a : plan.stride_b;
    gpu::blas::gemm_strided_batched<T>(plan.trans_a, plan.trans_b, plan.m, plan.n, plan.k, alpha, a, plan.lda, stride_a, b, plan.ldb,
                                       stride_b, beta, static_cast<T *>(ptr_c), plan.ldc, plan.stride_c, plan.batch_count);
}

/// Try an einsum that is a strided batch of matrix products, planned the way the CPU executor plans
/// it (see detail::plan_strided_batch) and issued as one strided-batched GEMM. Reads the live index
/// lists, prefactors and conjugation flags; a conjugated or antisymmetrized contraction has no
/// gemm_batch form.
bool try_gpu_strided_einsum(EinsumDescriptor const &desc, Node const &node, std::unordered_map<TensorId, TensorHandle> const &tensors,
                            DeviceShadowMap &shadows) {
    auto const lists = live_index_lists(desc);
    if (live_conj_a(desc) || live_conj_b(desc) || !lists.operators.empty() || node.inputs.size() < 2 || node.outputs.empty()) {
        return false;
    }
    auto const a_it = tensors.find(node.inputs[0]);
    auto const b_it = tensors.find(node.inputs[1]);
    auto const c_it = tensors.find(node.outputs[0]);
    if (a_it == tensors.end() || b_it == tensors.end() || c_it == tensors.end() || !a_it->second.impl_fn || !b_it->second.impl_fn ||
        !c_it->second.impl_fn || a_it->second.dtype != c_it->second.dtype || b_it->second.dtype != c_it->second.dtype) {
        return false;
    }

    void *ptr_a = resolve_device_ptr(a_it->second, node.inputs[0], shadows);
    void *ptr_b = resolve_device_ptr(b_it->second, node.inputs[1], shadows);
    void *ptr_c = resolve_device_ptr(c_it->second, node.outputs[0], shadows);
    if (!ptr_a || !ptr_b || !ptr_c) {
        return false;
    }

    bool dispatched = false;
    detail::dispatch_scalar_type(c_it->second.dtype, [&]<typename T>(T /*tag*/) {
        using Impl    = ::einsums::detail::TensorImpl<T>;
        auto const *a = static_cast<Impl const *>(a_it->second.impl_fn());
        auto const *b = static_cast<Impl const *>(b_it->second.impl_fn());
        auto const *c = static_cast<Impl const *>(c_it->second.impl_fn());
        if (a == nullptr || b == nullptr || c == nullptr) {
            return;
        }
        auto const plan = detail::plan_strided_batch(lists.c, lists.a, lists.b, lists.link, *a, *b, *c);
        if (!plan) {
            return;
        }
        run_gpu_strided_batched<T>(*plan, as<T>(live_ab_prefactor(desc)), as<T>(live_c_prefactor(desc)), ptr_a, ptr_b, ptr_c);
        dispatched = true;
    });
    return dispatched;
}

} // namespace

/// Top-level GPU BLAS dispatcher: tries GEMM, GEMV, Scale, Axpy, BatchedGemm.
bool try_gpu_blas_dispatch(Node const &node, std::unordered_map<TensorId, TensorHandle> const &tensors, DeviceShadowMap &shadows) {
    // Einsum operations: try GEMM, then GEMV, then a strided batch of GEMMs.
    if (auto const *desc = node.op_data.get_if<EinsumDescriptor>()) {
        if (try_gpu_gemm(*desc, node, tensors, shadows))
            return true;
        if (try_gpu_gemv(*desc, node, tensors, shadows))
            return true;
        if (try_gpu_strided_einsum(*desc, node, tensors, shadows))
            return true;
    }

    // BLAS Level 1 operations.
    if (try_gpu_scale(node, tensors, shadows))
        return true;
    if (try_gpu_axpy(node, tensors, shadows))
        return true;

    return false;
}

EINSUMS_NAMESPACE_END(compute_graph::gpu_dispatch)
