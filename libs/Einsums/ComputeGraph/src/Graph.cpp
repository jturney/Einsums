//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/CXX23/Expected.hpp>
#include <Einsums/ComputeGraph/CaptureContext.hpp>
#include <Einsums/ComputeGraph/Detail/ScalarDispatch.hpp>
#include <Einsums/ComputeGraph/EinsumSpec.hpp>
#include <Einsums/ComputeGraph/Error.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Optimizer.hpp> // For OptimizerPass and PassManager
#include <Einsums/ComputeGraph/StringDispatch.hpp>
#include <Einsums/ComputeGraphTypes/GraphData.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/GPU/BLAS.hpp>
#include <Einsums/LinearAlgebra.hpp>
#include <Einsums/Profile/Profile.hpp>
#include <Einsums/Tensor/Tensor.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <chrono>
#include <complex>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <optional>
#include <ostream>
#include <queue>
#include <set>
#include <unordered_set>

namespace einsums::compute_graph {

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

    if (handle.dtype == packed_gemm::ScalarType::Float32) {
        gpu::blas::scal<float>(n, static_cast<float>(desc->factor), static_cast<float *>(ptr), 1);
        return true;
    } else if (handle.dtype == packed_gemm::ScalarType::Float64) {
        gpu::blas::scal<double>(n, desc->factor, static_cast<double *>(ptr), 1);
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
    PrefactorScalar const &alpha_pf = desc->params ? desc->params->alpha : desc->alpha;
    PrefactorScalar const &beta_pf  = desc->params ? desc->params->beta : desc->beta;

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

void Graph::move_members_from(Graph &&other) noexcept {
    _name                  = std::move(other._name);
    _pipeline_name         = std::move(other._pipeline_name);
    _workspace_name        = std::move(other._workspace_name);
    _stage_name            = std::move(other._stage_name);
    _stage_type            = std::move(other._stage_type);
    _stage_index           = other._stage_index;
    _nodes                 = std::move(other._nodes);
    _tensors               = std::move(other._tensors);
    _next_node_id          = other._next_node_id;
    _next_tensor_id        = other._next_tensor_id;
    _sorted                = other._sorted;
    _executed              = other._executed;
    _deps                  = std::move(other._deps);
    _owned_tensors         = std::move(other._owned_tensors);
    _adopted_cleanups      = std::move(other._adopted_cleanups);
    _params                = std::move(other._params);
    _slot_map              = std::move(other._slot_map);
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
    _usage_version         = other._usage_version;
    _usage                 = std::move(other._usage);
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
        _nodes.insert(_nodes.begin() + static_cast<std::ptrdiff_t>(at), std::make_move_iterator(nodes.begin()),
                      std::make_move_iterator(nodes.end()));
    }
    mark_sorted();
}

namespace {

/// Half-open byte span of a handle's storage, or false when it has none that
/// can be reasoned about (deferred allocation, tiled layout, zero extent).
bool handle_byte_span(TensorHandle const &h, char const *&lo, char const *&hi) {
    if (h.data_ptr == nullptr || h.is_tiled || h.element_size == 0 || h.dims.empty() || h.strides.size() != h.dims.size()) {
        return false;
    }
    size_t last = 0;
    for (size_t d = 0; d < h.dims.size(); ++d) {
        if (h.dims[d] == 0) {
            return false;
        }
        last += (h.dims[d] - 1) * h.strides[d];
    }
    lo = static_cast<char const *>(h.data_ptr);
    hi = lo + (last + 1) * h.element_size;
    return true;
}

/// Recover @p child's region in @p parent's axis space from the pointer offset
/// and the two stride sets. Returns false whenever the layout is not provably a
/// sub-box, which leaves the box empty and the access conservatively
/// whole-tensor. Being wrong here would UNDER-serialize, so every ambiguity
/// declines rather than guesses.
bool derive_alias_box(TensorHandle const &parent, TensorHandle const &child, std::vector<std::pair<std::int64_t, std::int64_t>> &box) {
    size_t const rank = parent.dims.size();
    if (rank == 0 || parent.element_size == 0 || child.element_size != parent.element_size) {
        return false;
    }
    // Equal strides on two traversed axes make the offset decomposition and the
    // axis matching below ambiguous.
    for (size_t a = 0; a < rank; ++a) {
        for (size_t b = a + 1; b < rank; ++b) {
            if (parent.strides[a] == parent.strides[b] && parent.dims[a] > 1 && parent.dims[b] > 1) {
                return false;
            }
        }
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

    // Peel the offset apart largest stride first; unique for a canonical layout.
    std::vector<size_t> order(rank);
    std::iota(order.begin(), order.end(), size_t{0});
    std::ranges::sort(order, [&](size_t a, size_t b) { return parent.strides[a] > parent.strides[b]; });

    std::vector<std::int64_t> start(rank, 0);
    for (size_t const d : order) {
        if (parent.strides[d] == 0) {
            continue;
        }
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

} // namespace

void Graph::link_alias_storage() {
    if (_aliases_linked) {
        return;
    }
    _aliases_linked = true;

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
    std::ranges::sort(spans, [](Entry const &a, Entry const &b) { return a.lo < b.lo; });

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

    // Strictly fewer elements keeps a handle from aliasing itself and makes the
    // relation a partial order, so the chains resolve_alias() walks are acyclic.
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
                if (owner != _tensors.end() && owner->second.total_elems() > self->second.total_elems()) {
                    self->second.aliases = owner->first;
                    if (!derive_alias_box(owner->second, self->second, self->second.alias_box)) {
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
}

TensorId Graph::register_tensor(TensorHandle handle) {
    std::scoped_lock const lock(*_content_mutex);
    TensorId               id = _next_tensor_id++;
    handle.id                 = id;
    auto const &stored        = _tensors.emplace(id, std::move(handle)).first->second;
    if (stored.tensor_ptr != nullptr) {
        _ptr_index.emplace(stored.tensor_ptr, id);
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

void Graph::for_each_subgraph(std::function<void(Graph &)> const &visitor) {
    for (auto &node : _nodes) {
        if (auto *loop = std::get_if<LoopDescriptor>(&node.op_data)) {
            if (loop->body) {
                visitor(*loop->body);
            }
        } else if (auto *cond = std::get_if<ConditionalDescriptor>(&node.op_data)) {
            if (cond->then_branch) {
                visitor(*cond->then_branch);
            }
            if (cond->else_branch) {
                visitor(*cond->else_branch);
            }
        }
    }
}

void Graph::for_each_subgraph(std::function<void(Graph const &)> const &visitor) const {
    for (auto const &node : _nodes) {
        if (auto const *loop = std::get_if<LoopDescriptor>(&node.op_data)) {
            if (loop->body) {
                visitor(*loop->body);
            }
        } else if (auto const *cond = std::get_if<ConditionalDescriptor>(&node.op_data)) {
            if (cond->then_branch) {
                visitor(*cond->then_branch);
            }
            if (cond->else_branch) {
                visitor(*cond->else_branch);
            }
        }
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

    if (node.kind != OpKind::Loop && node.kind != OpKind::Conditional) {
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

    if (auto const *loop = std::get_if<LoopDescriptor>(&node.op_data)) {
        if (loop->body) {
            collect(*loop->body);
        }
    } else if (auto const *cond = std::get_if<ConditionalDescriptor>(&node.op_data)) {
        if (cond->then_branch) {
            collect(*cond->then_branch);
        }
        if (cond->else_branch) {
            collect(*cond->else_branch);
        }
    }

    // Map subtree buffer pointers back to this graph's TensorIds. A buffer used
    // only inside sub-graphs has no parent TensorId yet; register one (a stable
    // shared id) so two control-flow nodes touching the same buffer resolve to
    // the same id and a dependency edge forms between them.
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
    if (node.kind != OpKind::Loop && node.kind != OpKind::Conditional) {
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
    // leaving parallel executors no width. Each eligible view (identity axis
    // order, every bound a compile-time constant) gets a per-parent-axis
    // interval box; two accesses conflict only when their boxes may overlap.
    // Anything unprovable - a runtime bound, a permuted view, a view of a
    // view, a whole-tensor access - keeps a null box, which overlaps
    // everything (the previous behavior). Element-disjoint writes commute
    // bitwise, so relaxing the order cannot change results.
    using Box = std::vector<std::pair<std::int64_t, std::int64_t>>; // per parent axis: [lo, hi)

    std::unordered_map<TensorId, Box>      view_box;    // view tid -> box in parent axis space
    std::unordered_map<TensorId, TensorId> view_parent; // view tid -> immediate parent tid
    for (auto const &nd : _nodes) {
        auto const *vd = std::get_if<ViewDescriptor>(&nd.op_data);
        if (vd == nullptr || nd.kind != OpKind::View || nd.outputs.size() != 1 || !vd->permutation.empty()) {
            continue;
        }
        // The box lives in the immediate parent's axis space; a view of a view
        // would need composing, so require the parent to be the owner.
        if (resolve_alias(vd->parent_id) != vd->parent_id) {
            continue;
        }
        auto const *ph = find_tensor(vd->parent_id);
        if (ph == nullptr || ph->dims.size() != vd->axes.size()) {
            continue;
        }
        Box box;
        box.reserve(vd->axes.size());
        bool constant = true;
        for (size_t d = 0; d < vd->axes.size() && constant; ++d) {
            auto const        &ax  = vd->axes[d];
            std::int64_t const dim = static_cast<std::int64_t>(ph->dims[d]);
            switch (ax.kind) {
            case ViewAxis::Kind::Full:
                box.emplace_back(0, dim);
                break;
            case ViewAxis::Kind::Drop:
                constant = ax.lo.is_const();
                if (constant) {
                    box.emplace_back(ax.lo.const_value(), ax.lo.const_value() + 1);
                }
                break;
            case ViewAxis::Kind::Range:
                constant = ax.lo.is_const() && ax.hi.is_const();
                if (constant) {
                    box.emplace_back(ax.lo.const_value(), ax.hi.const_value());
                }
                break;
            }
        }
        if (!constant) {
            continue;
        }
        view_parent.emplace(nd.outputs[0], vd->parent_id);
        view_box.emplace(nd.outputs[0], std::move(box));
    }

    // Views that reached the graph without a View node (sliced outside a
    // capture, linked by storage containment at registration) carry their box
    // on the handle instead. Without this they would still be ordered against
    // the parent correctly, but as whole-tensor accesses, chaining every slice
    // behind every other and costing the parallel executors their width.
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
            if (auto it = view_box.find(nd.outputs[0]); it != view_box.end() && view_parent.at(nd.outputs[0]) == raw) {
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
            if (box == nullptr) {
                wl.clear(); // a whole-tensor write dominates all prior writers
            }
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

    // Level partition for level-scheduling executors. Edges always point
    // from earlier to later positions (the scan above links prior
    // writers/readers to the current node), so one forward pass suffices.
    std::vector<size_t> level(n, 0);
    size_t              max_level = 0;
    for (size_t i2 = 0; i2 < n; i2++) {
        for (size_t const pred : _deps.predecessors[i2]) {
            level[i2] = std::max(level[i2], level[pred] + 1);
        }
        max_level = std::max(max_level, level[i2]);
    }
    _deps.levels.assign(max_level + 1, {});
    for (size_t i2 = 0; i2 < n; i2++) {
        _deps.levels[level[i2]].push_back(i2);
    }
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

    for_each_hazard_edge(eff_cache, [&](size_t producer, size_t consumer) {
        adj[producer].push_back(consumer);
        in_degree[consumer]++;
    });

    // Kahn's algorithm
    std::queue<size_t> ready;
    for (size_t i = 0; i < n; i++) {
        if (in_degree[i] == 0) {
            ready.push(i);
        }
    }

    std::vector<Node> sorted;
    sorted.reserve(n);

    while (!ready.empty()) {
        size_t const idx = ready.front();
        ready.pop();
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

    // Rebuild dependency info on the sorted node order (positions changed).
    rebuild_deps(eff_cache);

    _sorted     = true;
    _deps_valid = true;
    // Positions changed but the node COUNT did not, so cached position-keyed
    // analyses (UsageAnalysis) must be invalidated explicitly - the count
    // defense in usage() cannot see a same-size reorder.
    _analysis_version++;
}

std::tuple<Graph &, Graph &> Graph::add_conditional(std::string label, std::function<bool()> predicate) {
    auto then_graph = std::make_shared<Graph>(label + "/then");
    auto else_graph = std::make_shared<Graph>(label + "/else");

    ConditionalDescriptor desc;
    desc.predicate   = std::move(predicate);
    desc.then_branch = then_graph;
    desc.else_branch = else_graph;

    Node node;
    node.kind    = OpKind::Conditional;
    node.label   = std::move(label);
    node.execute = [d = desc]() {
        if (d.predicate()) {
            d.then_branch->execute();
        } else if (d.else_branch && d.else_branch->num_nodes() > 0) {
            d.else_branch->execute();
        }
    };
    node.op_data = std::move(desc);

    add_node(std::move(node));

    return {*then_graph, *else_graph};
}

Graph &Graph::add_loop(std::string label, size_t max_iterations, std::function<bool(size_t)> condition) {
    auto body_graph = std::make_shared<Graph>(label + "/body");

    LoopDescriptor desc;
    desc.body           = body_graph;
    desc.max_iterations = max_iterations;
    desc.condition      = std::move(condition);

    Node node;
    node.kind    = OpKind::Loop;
    node.label   = std::move(label);
    node.execute = [d = desc]() mutable {
        d.last_iteration_count = 0;
        for (size_t iter = 0; iter < d.max_iterations; iter++) {
            d.body->execute();
            d.last_iteration_count = iter + 1;
            if (d.condition && !d.condition(iter)) {
                break;
            }
        }
    };
    node.op_data = std::move(desc);

    add_node(std::move(node));

    return *body_graph;
}

void Graph::add_loop(std::string label, size_t max_iterations, std::function<bool(size_t)> condition, std::function<void()> body_fn) {
    auto              &body = add_loop(std::move(label), max_iterations, std::move(condition));
    CaptureGuard const g(body);
    body_fn();
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

    auto find_id = [&](void *ptr) -> TensorId {
        for (auto const &[id, h] : _tensors) {
            if (h.tensor_ptr == ptr)
                return id;
        }
        return 0;
    };

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
            return std::pair{find_id(&t), static_cast<void *>(&t)};
        }
        case 2: {
            auto &t = create_zero_tensor<T, 2>(std::move(name), dims[0], dims[1]);
            return std::pair{find_id(&t), static_cast<void *>(&t)};
        }
        case 3: {
            auto &t = create_zero_tensor<T, 3>(std::move(name), dims[0], dims[1], dims[2]);
            return std::pair{find_id(&t), static_cast<void *>(&t)};
        }
        case 4: {
            auto &t = create_zero_tensor<T, 4>(std::move(name), dims[0], dims[1], dims[2], dims[3]);
            return std::pair{find_id(&t), static_cast<void *>(&t)};
        }
        case 5: {
            auto &t = create_zero_tensor<T, 5>(std::move(name), dims[0], dims[1], dims[2], dims[3], dims[4]);
            return std::pair{find_id(&t), static_cast<void *>(&t)};
        }
        case 6: {
            auto &t = create_zero_tensor<T, 6>(std::move(name), dims[0], dims[1], dims[2], dims[3], dims[4], dims[5]);
            return std::pair{find_id(&t), static_cast<void *>(&t)};
        }
        case 7: {
            auto &t = create_zero_tensor<T, 7>(std::move(name), dims[0], dims[1], dims[2], dims[3], dims[4], dims[5], dims[6]);
            return std::pair{find_id(&t), static_cast<void *>(&t)};
        }
        case 8: {
            auto &t = create_zero_tensor<T, 8>(std::move(name), dims[0], dims[1], dims[2], dims[3], dims[4], dims[5], dims[6], dims[7]);
            return std::pair{find_id(&t), static_cast<void *>(&t)};
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
template <typename Fn>
void dispatch_binary(TensorHandle const &a, TensorHandle const &b, Fn &&fn) {
    if (a.dtype != b.dtype || a.rank != b.rank) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "dispatch_binary: dtype or rank mismatch");
    }
    // A runtime tensor's storage layout differs from Tensor<T, Rank>; casting one
    // handle's tensor_ptr with the other's shape is type confusion. Both operands
    // must be the same kind (callers gate on is_runtime, so this only guards misuse).
    if (a.is_runtime != b.is_runtime) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "dispatch_binary: cannot mix runtime and compile-time tensors");
    }

    auto go = [&]<typename T>(T /*tag*/) {
        // GeneralRuntimeTensor<T> carries its rank dynamically, so one cast covers
        // every rank; branch on the handle kind before the compile-time rank switch.
        if (a.is_runtime) {
            using RT = GeneralRuntimeTensor<T, std::allocator<T>>;
            fn(static_cast<RT *>(a.tensor_ptr), static_cast<RT *>(b.tensor_ptr));
            return;
        }
        detail::dispatch_by_rank(a.rank, [&](auto rank_tag) {
            constexpr std::size_t K = decltype(rank_tag)::value;
            fn(static_cast<Tensor<T, K> *>(a.tensor_ptr), static_cast<Tensor<T, K> *>(b.tensor_ptr));
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
            fn(static_cast<RT *>(a.tensor_ptr));
            return;
        }
        detail::dispatch_by_rank(a.rank, [&](auto rank_tag) {
            constexpr std::size_t K = decltype(rank_tag)::value;
            fn(static_cast<Tensor<T, K> *>(a.tensor_ptr));
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

    auto find_id = [&](void *ptr) -> TensorId {
        for (auto const &[id, h] : _tensors) {
            if (h.tensor_ptr == ptr) {
                return id;
            }
        }
        return 0;
    };

    auto make = [&]<typename T>(T /*tag*/) -> std::pair<TensorId, void *> {
        auto &t = create_zero_runtime_tensor<T, std::allocator<T>>(std::move(name), dims, /*intermediate=*/true);
        return {find_id(&t), static_cast<void *>(&t)};
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
    return [this, a_id, b_id, c_id, alpha, beta]() {
        auto const &a_h = tensor(a_id);
        auto const &b_h = tensor(b_id);
        auto       &c_h = tensor(c_id);

        if (a_h.rank != 2 || b_h.rank != 2 || c_h.rank != 2) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument, "make_gemm_executor: all tensors must be rank 2");
        }

        // Dispatch on dtype (all three must match)
        auto go = [&]<typename T>(T /*tag*/) {
            auto *A = static_cast<Tensor<T, 2> *>(a_h.tensor_ptr);
            auto *B = static_cast<Tensor<T, 2> *>(b_h.tensor_ptr);
            auto *C = static_cast<Tensor<T, 2> *>(c_h.tensor_ptr);
            linear_algebra::gemm<false, false>(static_cast<T>(alpha), *A, *B, static_cast<T>(beta), C);
        };

        detail::dispatch_scalar_type(a_h.dtype, go);
    };
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

    // BLAS batching hint, same gate and same derivation as the capture path in
    // Operations.hpp: three rank-2 operands, exactly one link index, and strides
    // that agree with each tensor's declared layout flag.
    //
    // Building this at pass time is no weaker than building it at capture. The
    // dims come from the same place either way, and GEMMBatching consumes the
    // hint BEFORE DistributionPlanning and Materialization run, so a captured
    // hint on graph-owned scratch is derived from the same shell geometry this
    // is. (A deferred shell carries valid dims and strides; only data() is null.)
    // The extractors are strictly better than the capture path's: resolving by
    // TensorId at call time follows rebind() and survives MemoryPlanning
    // repointing storage through materialize_into, which is exactly why they read
    // data() lazily instead of caching a pointer.
    auto const layout_matches_flag = [](auto const &impl) {
        bool const   row_major = impl.is_row_major();
        size_t const rank      = impl.rank();
        size_t       prev      = 0;
        bool         first     = true;
        for (size_t n = 0; n < rank; ++n) {
            size_t const d = row_major ? rank - 1 - n : n;
            if (impl.dim(d) <= 1) {
                continue; // extent-1 axes are never traversed; ignore their strides
            }
            size_t const st = impl.stride(d);
            if (!first && st < prev) {
                return false;
            }
            prev  = st;
            first = false;
        }
        return true;
    };

    // The batched form is C = op(A) * op(B), so C's FIRST index must be the one
    // the einsum's A operand contributes and its SECOND the one B contributes.
    // "ia <- ma ; mi" has them swapped -- i comes from B -- and m/n/k would then
    // describe a matrix product this einsum does not perform. The generic kernel
    // contracts correctly either way and never consults the hint, so a wrong hint
    // stays invisible until GEMMBatching forms a batch and calls gemm_batch with
    // it. Emit no hint rather than a wrong one.
    auto const gemm_roles_match = [&]() {
        if (spec.a_indices.size() != 2 || spec.b_indices.size() != 2 || spec.c_indices.size() != 2 || desc.spec.link_indices.size() != 1) {
            return false;
        }
        auto const &lnk  = desc.spec.link_indices[0];
        auto const  free = [&lnk](std::vector<std::string> const &idx) { return idx[0] == lnk ? idx[1] : idx[0]; };
        return spec.c_indices[0] == free(spec.a_indices) && spec.c_indices[1] == free(spec.b_indices);
    };

    if (a_h.rank == 2 && b_h.rank == 2 && c_h.rank == 2 && spec.a_indices.size() == 2 && spec.b_indices.size() == 2 &&
        spec.c_indices.size() == 2 && desc.spec.link_indices.size() == 1 && gemm_roles_match()) {
        detail::dispatch_scalar_type(dtype, [&]<typename T>(T /*tag*/) {
            using Impl    = ::einsums::detail::TensorImpl<T>;
            auto const *A = static_cast<Impl const *>(a_h.impl_fn());
            auto const *B = static_cast<Impl const *>(b_h.impl_fn());
            auto const *C = static_cast<Impl const *>(c_h.impl_fn());
            if (A == nullptr || B == nullptr || C == nullptr) {
                return;
            }
            if (!layout_matches_flag(*A) || !layout_matches_flag(*B) || !layout_matches_flag(*C)) {
                return;
            }

            auto hint = std::make_shared<GemmHint>();
            if constexpr (std::is_same_v<T, float>) {
                hint->scalar = BlasScalar::Float;
            } else if constexpr (std::is_same_v<T, double>) {
                hint->scalar = BlasScalar::Double;
            } else if constexpr (std::is_same_v<T, std::complex<float>>) {
                hint->scalar = BlasScalar::ComplexFloat;
            } else {
                hint->scalar = BlasScalar::ComplexDouble;
            }

            std::string const &link = desc.spec.link_indices[0];
            hint->trans_a           = (spec.a_indices[0] == link) ? 'T' : 'N';
            hint->trans_b           = (spec.b_indices[1] == link) ? 'T' : 'N';
            hint->m                 = static_cast<int>(C->dim(0));
            hint->n                 = static_cast<int>(C->dim(1));
            hint->k                 = static_cast<int>(hint->trans_a == 'N' ? A->dim(1) : A->dim(0));

            Graph *g        = this;
            hint->extract_a = [g, a_id]() -> std::pair<void const *, int> {
                auto const *i = static_cast<Impl const *>(g->tensor(a_id).impl_fn());
                return {static_cast<void const *>(i->data()), static_cast<int>(i->get_lda())};
            };
            hint->extract_b = [g, b_id]() -> std::pair<void const *, int> {
                auto const *i = static_cast<Impl const *>(g->tensor(b_id).impl_fn());
                return {static_cast<void const *>(i->data()), static_cast<int>(i->get_lda())};
            };
            hint->extract_c = [g, c_id]() -> std::pair<void *, int> {
                auto *i = static_cast<Impl *>(g->tensor(c_id).impl_fn());
                return {static_cast<void *>(i->data()), static_cast<int>(i->get_lda())};
            };
            desc.gemm_hint = std::move(hint);
        });
    }

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
    auto pg_site = std::make_shared<packed_gemm::ContractionSite>();

    Graph *self  = this;
    node.execute = [self, a_id, b_id, c_id, params, indices, dtype, pg_site]() {
        detail::dispatch_scalar_type(dtype, [&]<typename T>(T /*tag*/) {
            using Impl = ::einsums::detail::TensorImpl<T>;
            // Re-view each operand through its LIVE impl: aliasing, so writes to C
            // land in the real tensor, and current, so rebind(), Materialization and
            // the MemoryPlanning arena are all honored. Works for a typed
            // Tensor<T, Rank> as well, since the impl is rank-erased.
            RuntimeTensorView<T> const A{*static_cast<Impl *>(self->tensor(a_id).impl_fn())};
            RuntimeTensorView<T> const B{*static_cast<Impl *>(self->tensor(b_id).impl_fn())};
            RuntimeTensorView<T>       C{*static_cast<Impl *>(self->tensor(c_id).impl_fn())};
            // The spec is passed BY REFERENCE, so an index rewrite (PermuteFusion)
            // is honored without rebuilding it: the pass writes into this very
            // object. Rebuilding it per call copied three vector<string> for
            // nothing, which a per-tile expansion pays thousands of times a replay.
            dispatch::string_einsum(indices->spec, as<T>(params->c_pf), &C, as<T>(params->ab_pf), A, B, params->conj_a, params->conj_b,
                                    &indices->link_indices, pg_site.get());
        });
    };

    node.op_data = std::move(desc);
    return node;
}

std::function<void()> Graph::make_einsum_executor(TensorId a_id, TensorId b_id, TensorId c_id, ParsedEinsumSpec const &spec, double alpha,
                                                  double beta) {
    // For chain restructuring: the intermediates are rank-2 but the leaf
    // tensors and final output may be higher-rank. We fold them into rank-2
    // by treating contiguous outer dimensions as a single M or N.
    //
    // The effective GEMM is: C_flat(M, N) = alpha * A_flat(M, K) * B_flat(K, N) + beta * C_flat(M, N)
    // where M, K, N are computed from the parsed spec and tensor dims.
    //
    // If all three are already rank-2, this is identical to make_gemm_executor.
    // For higher-rank tensors, we call BLAS gemm directly on the data pointers.

    // Precompute M, K, N from the spec at pass time
    auto const &a_h = tensor(a_id);
    auto const &b_h = tensor(b_id);
    auto const &c_h = tensor(c_id);

    // For the restructured chain, the spec tells us the index structure.
    // Compute link and target dims from the spec.
    auto links   = spec.link_indices();
    auto targets = spec.target_indices();

    // M = product of C dims that are target_a (in A, not in B)
    // N = product of C dims that are target_b (in B, not in A)
    // K = product of link dims
    std::set<std::string> const a_set(spec.a_indices.begin(), spec.a_indices.end());
    std::set<std::string> const b_set(spec.b_indices.begin(), spec.b_indices.end());

    size_t M = 1, N = 1, K = 1;
    for (auto const &idx : spec.c_indices) {
        bool const in_a = a_set.count(idx) > 0;
        bool const in_b = b_set.count(idx) > 0;
        // Find this index's dimension size from whichever tensor has it
        size_t dim_size = 0;
        for (size_t d = 0; d < spec.a_indices.size(); d++) {
            if (spec.a_indices[d] == idx) {
                dim_size = a_h.dims[d];
                break;
            }
        }
        if (dim_size == 0) {
            for (size_t d = 0; d < spec.b_indices.size(); d++) {
                if (spec.b_indices[d] == idx) {
                    dim_size = b_h.dims[d];
                    break;
                }
            }
        }
        // NOLINTBEGIN
        if (in_a && !in_b)
            M *= dim_size;
        else if (in_b && !in_a)
            N *= dim_size;
        else
            M *= dim_size; // Shared/batch: fold into M
        // NOLINTEND
    }
    for (auto const &idx : links) {
        for (size_t d = 0; d < spec.a_indices.size(); d++) {
            if (spec.a_indices[d] == idx) {
                K *= a_h.dims[d];
                break;
            }
        }
    }

    // Per-operand transpose flags, taken from the captured index order.
    //
    // A GEMM reads A as (M,K) and B as (K,N). An operand whose link indices sit
    // at the other end is stored with its axes the other way round and has to
    // be read transposed; BLAS does that in the packing step, so no physical
    // permute is needed. This used to be a comment saying "determine transpose
    // flags" above an unconditional gemm<false, false>, which silently read a
    // transposed operand in the wrong orientation.
    //
    // An operand whose links are neither a prefix nor a suffix (interleaved with
    // its targets) has no flat (M,K) reading at all; the caller is expected to
    // have excluded that, and the executor throws rather than compute garbage.
    auto const a_placement = link_placement(spec.a_indices, links);
    auto const b_placement = link_placement(spec.b_indices, links);

    // Prefer the reading that needs no transpose, which also settles the
    // degenerate all-link / all-target operand where both ends qualify.
    bool const trans_a = !a_placement.suffix && a_placement.prefix;
    bool const trans_b = !b_placement.prefix && b_placement.suffix;

    if (a_placement.split() || b_placement.split()) {
        throw std::invalid_argument(
            fmt::format("Graph '{}': make_einsum_executor cannot express spec '{}' as a GEMM - an operand's link indices are "
                        "interleaved with its target indices, so it has no flat (M,K) or (K,N) reading. Permute it first.",
                        _name, spec.raw));
    }

    return [this, a_id, b_id, c_id, alpha, beta, M, K, N, trans_a, trans_b]() {
        auto const &a_h = tensor(a_id);
        auto const &b_h = tensor(b_id);
        auto       &c_h = tensor(c_id);

        auto go = [&]<typename T>(T /*tag*/) {
            // Runtime flags into the compile-time gemm<TransA, TransB>.
            auto call_gemm = [&](auto const &A, auto const &B, auto *C) {
                if (trans_a) {
                    if (trans_b) {
                        linear_algebra::gemm<true, true>(static_cast<T>(alpha), A, B, static_cast<T>(beta), C);
                    } else {
                        linear_algebra::gemm<true, false>(static_cast<T>(alpha), A, B, static_cast<T>(beta), C);
                    }
                } else {
                    if (trans_b) {
                        linear_algebra::gemm<false, true>(static_cast<T>(alpha), A, B, static_cast<T>(beta), C);
                    } else {
                        linear_algebra::gemm<false, false>(static_cast<T>(alpha), A, B, static_cast<T>(beta), C);
                    }
                }
            };

            // The result is M x N by construction. A destination shaped the
            // other way round would be a silent transpose whenever M == N, so
            // say so instead of writing it.
            auto check_destination = [&](size_t rows, size_t cols) {
                if (rows != M || cols != N) {
                    throw std::invalid_argument(
                        fmt::format("Graph '{}': make_einsum_executor produces a {}x{} result but its destination is {}x{}; the "
                                    "output's index order does not match [A targets..., B targets...].",
                                    _name, M, N, rows, cols));
                }
            };

            bool const rank2_shaped = a_h.rank == 2 && b_h.rank == 2 && c_h.rank == 2;
            bool const any_runtime  = a_h.is_runtime || b_h.is_runtime || c_h.is_runtime;

            // For statically typed rank-2 tensors, use the standard gemm.
            // For higher-rank, create temporary rank-2 views.
            if (rank2_shaped && !any_runtime) {
                auto *A = static_cast<Tensor<T, 2> *>(a_h.tensor_ptr);
                auto *B = static_cast<Tensor<T, 2> *>(b_h.tensor_ptr);
                auto *C = static_cast<Tensor<T, 2> *>(c_h.tensor_ptr);
                check_destination(C->dim(0), C->dim(1));
                call_gemm(*A, *B, C);
            } else if (rank2_shaped && a_h.impl_fn && b_h.impl_fn && c_h.impl_fn) {
                // At least one operand is runtime-rank. Casting to Tensor<T,2>*
                // would be type confusion, so go through the impl each tensor
                // type exposes and let the dynamic-rank gemm overload take it.
                // This is what lets a chain of RuntimeTensors - the Python-facing
                // type - be restructured instead of staying analysis-only.
                using Impl = ::einsums::detail::TensorImpl<T>;
                RuntimeTensorView<T> const A{*static_cast<Impl *>(a_h.impl_fn())};
                RuntimeTensorView<T> const B{*static_cast<Impl *>(b_h.impl_fn())};
                RuntimeTensorView<T>       C{*static_cast<Impl *>(c_h.impl_fn())};
                check_destination(C.dim(0), C.dim(1));
                call_gemm(A, B, &C);
            } else {
                // Higher-rank: use raw BLAS on the data pointers with folded dims.
                // Create temporary rank-2 TensorView-like wrappers. The flat
                // shape follows the transpose flag: a transposed A is stored
                // (K,M), a transposed B is stored (N,K).
                T const     *a_data = static_cast<T *>(a_h.data_ptr ? a_h.data_ptr : const_cast<void *>(a_h.tensor_ptr));
                T const     *b_data = static_cast<T *>(b_h.data_ptr ? b_h.data_ptr : const_cast<void *>(b_h.tensor_ptr));
                T           *c_data = static_cast<T *>(c_h.data_ptr ? c_h.data_ptr : const_cast<void *>(c_h.tensor_ptr));
                Tensor<T, 2> A_view("A_fold", trans_a ? K : M, trans_a ? M : K);
                Tensor<T, 2> B_view("B_fold", trans_b ? N : K, trans_b ? K : N);
                Tensor<T, 2> C_view("C_fold", M, N);
                std::memcpy(A_view.data(), a_data, M * K * sizeof(T));
                std::memcpy(B_view.data(), b_data, K * N * sizeof(T));
                if (beta != 0.0)
                    std::memcpy(C_view.data(), c_data, M * N * sizeof(T));
                call_gemm(A_view, B_view, &C_view);
                std::memcpy(c_data, C_view.data(), M * N * sizeof(T));
            }
        };

        detail::dispatch_scalar_type(a_h.dtype, go);
    };
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
                                  .duration_ms = sample.duration_ms});
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
            entry.reals.emplace_back(profile::intern_string("scale_factor"), sdesc->factor);
        } else if (auto const *cdesc = std::get_if<CommDescriptor>(&node.op_data)) {
            number("comm_bytes", static_cast<int64_t>(cdesc->size_bytes));
            number("comm_tensor", static_cast<int64_t>(cdesc->tensor_id));
        }

        auto annotate_tensors = [&](std::vector<TensorId> const &ids, char const *prefix) {
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
}

void Graph::execute(Executor &executor) {
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
    link_alias_storage();
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

bool Graph::validate_shapes() const {
    // For each registered tensor that has a valid pointer, check that its
    // current dimensions still match what was recorded at capture time.
    // Since tensors are type-erased, we can only check by stored metadata.
    // A more thorough check would re-read dims from the tensor, but that
    // requires type info. For now, we trust the handles are up-to-date.
    // This is a placeholder for future extension.
    return true;
}

void Graph::print_dot(std::ostream &os) const {
    os << "digraph \"" << _name << "\" {\n";
    os << "  rankdir=TB;\n";

    // Tensor nodes (rectangles)
    for (auto const &[id, handle] : _tensors) {
        os << fmt::format("  T{} [shape=box, label=\"{}\\n", id, handle.name);
        if (!handle.dims.empty()) {
            os << "(";
            for (size_t i = 0; i < handle.dims.size(); i++) {
                if (i > 0)
                    os << "x";
                os << handle.dims[i];
            }
            os << ")";
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
    os << fmt::format("Graph '{}': {} nodes, {} tensors\n", _name, _nodes.size(), _tensors.size());
    for (auto const &node : _nodes) {
        os << fmt::format("  [{}] {} ({})\n", node.id, node.label, op_kind_name(node.kind));
        if (!node.inputs.empty()) {
            os << "    inputs: ";
            for (size_t i = 0; i < node.inputs.size(); i++) {
                if (i > 0)
                    os << ", ";
                auto it = _tensors.find(node.inputs[i]);
                os << (it != _tensors.end() ? it->second.name : "?");
            }
            os << "\n";
        }
        if (!node.outputs.empty()) {
            os << "    outputs: ";
            for (size_t i = 0; i < node.outputs.size(); i++) {
                if (i > 0)
                    os << ", ";
                auto it = _tensors.find(node.outputs[i]);
                os << (it != _tensors.end() ? it->second.name : "?");
            }
            os << "\n";
        }
    }
}

// ── JSON serialization ─────────────────────────────────────────────────────

namespace {

std::string escape_json(std::string const &s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char const c : s) {
        switch (c) {
        case '"':
            out += "\\\"";
            break;
        case '\\':
            out += "\\\\";
            break;
        case '\n':
            out += "\\n";
            break;
        case '\r':
            out += "\\r";
            break;
        case '\t':
            out += "\\t";
            break;
        default:
            out += c;
        }
    }
    return out;
}

char const *scalar_type_str(packed_gemm::ScalarType dt) {
    switch (dt) {
    case packed_gemm::ScalarType::Float32:
        return "float32";
    case packed_gemm::ScalarType::Float64:
        return "float64";
    case packed_gemm::ScalarType::Complex64:
        return "complex64";
    case packed_gemm::ScalarType::Complex128:
        return "complex128";
    default:
        return "unknown";
    }
}

} // namespace

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
        td.dtype           = scalar_type_str(h.dtype);
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
            nd.scale_factor = desc->factor;
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

    auto esc = [](std::string const &s) -> std::string { return escape_json(s); };

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
        for (size_t d = 0; d < t.dims.size(); d++) {
            if (d > 0)
                j += ",";
            j += std::to_string(t.dims[d]);
        }
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
        j += ",\"inputs\":[";
        for (size_t k = 0; k < n.inputs.size(); k++) {
            if (k > 0)
                j += ",";
            j += std::to_string(n.inputs[k]);
        }
        j += "]";
        j += ",\"outputs\":[";
        for (size_t k = 0; k < n.outputs.size(); k++) {
            if (k > 0)
                j += ",";
            j += std::to_string(n.outputs[k]);
        }
        j += "]";
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
/// Cache of graph JSON for graphs that have been destroyed.
/// This allows export_session() to include graph data even after the Graph objects go out of scope.
std::vector<std::string> g_cached_graph_jsons;
} // namespace

void unregister_graph(Graph *graph) {
    std::scoped_lock const lock(g_registry_mutex);

    // Cache the graph's JSON before removing it, so it survives destruction.
    auto it = std::ranges::find(g_registered_graphs, graph);
    if (it != g_registered_graphs.end()) {
        g_cached_graph_jsons.push_back(graph->to_json());
        g_registered_graphs.erase(it);
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

    // Include cached JSON from destroyed graphs.
    for (auto const &cached : g_cached_graph_jsons) {
        if (!first)
            result += ",";
        first = false;
        result += cached;
    }

    result += "]}";
    return result;
}

} // namespace einsums::compute_graph
