//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/CXX23/Expected.hpp>
#include <Einsums/ComputeGraph/CaptureContext.hpp>
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
#include <ostream>
#include <queue>
#include <unordered_set>

namespace einsums::compute_graph {

namespace {

/// Try to dispatch a GPU node via gpu::blas (GEMM or GEMV).
/// Returns true if dispatched, false if not applicable (caller should use CPU fallback).
bool try_gpu_blas_dispatch(Node const &node, std::unordered_map<TensorId, TensorHandle> const &tensors, DeviceShadowMap &shadows);

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

    // Get shadow pointers (data pointers have already been swapped to shadows).
    // On unified memory: use tensor data pointers directly (GPU reads host memory).
    // On discrete GPU: use shadow pointers (data was copied to device).
    void *ptr_a, *ptr_b, *ptr_c;
    if constexpr (gpu::has_unified_memory) {
        ptr_a = ha.data_ptr;
        ptr_b = hb.data_ptr;
        ptr_c = hc.data_ptr;
    } else {
        ptr_a = shadows.get(a_id);
        ptr_b = shadows.get(b_id);
        ptr_c = shadows.get(c_id);
    }

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

    void *ptr_a, *ptr_x, *ptr_y;
    if constexpr (gpu::has_unified_memory) {
        ptr_a = ha.data_ptr;
        ptr_x = hx.data_ptr;
        ptr_y = hy.data_ptr;
    } else {
        ptr_a = shadows.get(a_id);
        ptr_x = shadows.get(x_id);
        ptr_y = shadows.get(y_id);
    }

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

    void *ptr;
    if constexpr (gpu::has_unified_memory) {
        ptr = handle.data_ptr;
    } else {
        ptr = shadows.get(tid);
    }
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

/// Try Axpy dispatch: y = alpha * x + y
bool try_gpu_axpy(Node const &node, std::unordered_map<TensorId, TensorHandle> const &tensors, DeviceShadowMap &shadows) {
    if (node.kind != OpKind::Axpy)
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

    void *ptr_x, *ptr_y;
    if constexpr (gpu::has_unified_memory) {
        ptr_x = hx.data_ptr;
        ptr_y = hy.data_ptr;
    } else {
        ptr_x = shadows.get(x_id);
        ptr_y = shadows.get(y_id);
    }
    if (!ptr_x || !ptr_y)
        return false;

    auto n = static_cast<int64_t>(hy.total_bytes() / hy.element_size);

    // Get alpha from the ScaleDescriptor if present, otherwise 1.0
    double alpha = 1.0;
    if (auto const *sdesc = std::get_if<ScaleDescriptor>(&node.op_data)) {
        alpha = sdesc->factor;
    }

    if (hx.dtype == packed_gemm::ScalarType::Float32) {
        gpu::blas::axpy<float>(n, static_cast<float>(alpha), static_cast<float const *>(ptr_x), 1, static_cast<float *>(ptr_y), 1);
        return true;
    } else if (hx.dtype == packed_gemm::ScalarType::Float64) {
        gpu::blas::axpy<double>(n, alpha, static_cast<double const *>(ptr_x), 1, static_cast<double *>(ptr_y), 1);
        return true;
    }
    return false;
}

/// Try strided-batched GEMM dispatch for OpKind::BatchedGemm nodes.
/// Only the strided mode is handled here — that's what the 3D-batch
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

    void *ptr_a, *ptr_b, *ptr_c;
    if constexpr (gpu::has_unified_memory) {
        ptr_a = a_it->second.data_ptr;
        ptr_b = b_it->second.data_ptr;
        ptr_c = c_it->second.data_ptr;
    } else {
        ptr_a = shadows.get(a_id);
        ptr_b = shadows.get(b_id);
        ptr_c = shadows.get(c_id);
    }
    if (!ptr_a || !ptr_b || !ptr_c)
        return false;

    switch (desc.scalar) {
    case BlasScalar::Float: {
        gpu::blas::gemm_strided_batched<float>(
            desc.trans_a, desc.trans_b, desc.m, desc.n, desc.k, static_cast<float>(desc.alpha), static_cast<float const *>(ptr_a), desc.lda,
            desc.batch_stride_a, static_cast<float const *>(ptr_b), desc.ldb, desc.batch_stride_b, static_cast<float>(desc.beta),
            static_cast<float *>(ptr_c), desc.ldc, desc.batch_stride_c, desc.batch_count);
        return true;
    }
    case BlasScalar::Double: {
        gpu::blas::gemm_strided_batched<double>(desc.trans_a, desc.trans_b, desc.m, desc.n, desc.k, desc.alpha,
                                                static_cast<double const *>(ptr_a), desc.lda, desc.batch_stride_a,
                                                static_cast<double const *>(ptr_b), desc.ldb, desc.batch_stride_b, desc.beta,
                                                static_cast<double *>(ptr_c), desc.ldc, desc.batch_stride_c, desc.batch_count);
        return true;
    }
    case BlasScalar::ComplexFloat: {
        std::complex<float> const alpha{static_cast<float>(desc.alpha), 0.0f};
        std::complex<float> const beta{static_cast<float>(desc.beta), 0.0f};
        gpu::blas::gemm_strided_batched<std::complex<float>>(
            desc.trans_a, desc.trans_b, desc.m, desc.n, desc.k, alpha, static_cast<std::complex<float> const *>(ptr_a), desc.lda,
            desc.batch_stride_a, static_cast<std::complex<float> const *>(ptr_b), desc.ldb, desc.batch_stride_b, beta,
            static_cast<std::complex<float> *>(ptr_c), desc.ldc, desc.batch_stride_c, desc.batch_count);
        return true;
    }
    case BlasScalar::ComplexDouble: {
        std::complex<double> const alpha{desc.alpha, 0.0};
        std::complex<double> const beta{desc.beta, 0.0};
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

Graph::Graph(Graph &&other) noexcept
    : _name(std::move(other._name)), _pipeline_name(std::move(other._pipeline_name)), _workspace_name(std::move(other._workspace_name)),
      _stage_name(std::move(other._stage_name)), _stage_type(std::move(other._stage_type)), _stage_index(other._stage_index),
      _nodes(std::move(other._nodes)), _tensors(std::move(other._tensors)), _next_node_id(other._next_node_id),
      _next_tensor_id(other._next_tensor_id), _sorted(other._sorted), _executed(other._executed), _deps(std::move(other._deps)),
      _owned_tensors(std::move(other._owned_tensors)), _adopted_cleanups(std::move(other._adopted_cleanups)),
      _params(std::move(other._params)), _slot_map(std::move(other._slot_map)), _timing_report(std::move(other._timing_report)),
      _params_store(std::move(other._params_store)) {
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
        _name             = std::move(other._name);
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
        _slot_map         = std::move(other._slot_map);
        _timing_report    = std::move(other._timing_report);
        _params_store     = std::move(other._params_store);

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
    node.id         = _next_node_id++;
    NodeId const id = node.id;
    _nodes.push_back(std::move(node));
    _sorted   = false;
    _executed = false;
    return id;
}

TensorId Graph::register_tensor(TensorHandle handle) {
    TensorId id = _next_tensor_id++;
    handle.id   = id;
    _tensors.emplace(id, std::move(handle));
    return id;
}

TensorHandle &Graph::tensor(TensorId id) {
    auto it = _tensors.find(id);
    if (it == _tensors.end()) {
        EINSUMS_THROW_EXCEPTION(std::out_of_range, "Graph '{}': no tensor with id {}", _name, id);
    }
    return it->second;
}

TensorHandle const &Graph::tensor(TensorId id) const {
    auto it = _tensors.find(id);
    if (it == _tensors.end()) {
        EINSUMS_THROW_EXCEPTION(std::out_of_range, "Graph '{}': no tensor with id {}", _name, id);
    }
    return it->second;
}

void Graph::topological_sort() {
    if (_nodes.empty()) {
        _sorted = true;
        return;
    }

    // Build adjacency from data dependencies:
    // If node A writes tensor T and node B reads tensor T (and B comes after A),
    // then A → B (A must execute before B).

    size_t const n = _nodes.size();

    // Resolve any chain of aliases (View outputs) to the underlying owner so
    // reads/writes through a slice register as reads/writes of the parent.
    // Without this, the scheduler would treat ``GEMM(C_occ, …)`` and
    // ``Syev(C, …)`` as independent — they're not, since C_occ aliases C.
    auto resolve_owner = [&](TensorId id) {
        for (int hops = 0; hops < 32; ++hops) {
            auto it = _tensors.find(id);
            if (it == _tensors.end() || it->second.aliases == 0)
                return id;
            id = it->second.aliases;
        }
        return id;
    };

    // Track dependencies: read-after-write, write-after-write, write-after-read.
    // Keyed by *owner* TensorId (after alias resolution) — see above.
    std::unordered_map<TensorId, size_t>              last_writer;
    std::unordered_map<TensorId, std::vector<size_t>> last_readers;
    std::vector<std::vector<size_t>>                  adj(n);
    std::vector<size_t>                               in_degree(n, 0);

    for (size_t i = 0; i < n; i++) {
        // Read-after-write: this node reads T → depends on last writer of T
        for (auto raw : _nodes[i].inputs) {
            TensorId const tid = resolve_owner(raw);
            auto           it  = last_writer.find(tid);
            if (it != last_writer.end() && it->second != i) {
                adj[it->second].push_back(i);
                in_degree[i]++;
            }
            last_readers[tid].push_back(i);
        }
        // Write-after-write + write-after-read: this node writes T
        for (auto raw : _nodes[i].outputs) {
            TensorId const tid = resolve_owner(raw);
            // Write-after-write: depends on last writer
            auto writ = last_writer.find(tid);
            if (writ != last_writer.end() && writ->second != i) {
                adj[writ->second].push_back(i);
                in_degree[i]++;
            }
            // Write-after-read: depends on all previous readers
            auto rdit = last_readers.find(tid);
            if (rdit != last_readers.end()) {
                for (size_t const reader : rdit->second) {
                    if (reader != i) {
                        adj[reader].push_back(i);
                        in_degree[i]++;
                    }
                }
                rdit->second.clear(); // Reset readers since we're writing
            }
            last_writer[tid] = i;
        }
    }

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

    // Rebuild dependency info on the sorted node order.
    // After sorting, node positions changed, so recompute adjacency.
    // Same alias-aware resolution as the first pass.
    _deps.successors.assign(n, {});
    _deps.predecessors.assign(n, {});
    {
        std::unordered_map<TensorId, size_t>              lw;
        std::unordered_map<TensorId, std::vector<size_t>> lr;
        for (size_t i2 = 0; i2 < n; i2++) {
            for (auto raw : _nodes[i2].inputs) {
                TensorId const tid = resolve_owner(raw);
                auto           it2 = lw.find(tid);
                if (it2 != lw.end() && it2->second != i2) {
                    _deps.successors[it2->second].push_back(i2);
                    _deps.predecessors[i2].push_back(it2->second);
                }
                lr[tid].push_back(i2);
            }
            for (auto raw : _nodes[i2].outputs) {
                TensorId const tid = resolve_owner(raw);
                auto           it2 = lw.find(tid);
                if (it2 != lw.end() && it2->second != i2) {
                    _deps.successors[it2->second].push_back(i2);
                    _deps.predecessors[i2].push_back(it2->second);
                }
                auto rdit = lr.find(tid);
                if (rdit != lr.end()) {
                    for (size_t const reader : rdit->second) {
                        if (reader != i2) {
                            _deps.successors[reader].push_back(i2);
                            _deps.predecessors[i2].push_back(reader);
                        }
                    }
                    rdit->second.clear();
                }
                lw[tid] = i2;
            }
        }
    }

    _sorted = true;
}

std::pair<Graph &, Graph &> Graph::add_conditional(std::string label, std::function<bool()> predicate) {
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
        if (node.id == node_id) {
            if (auto *desc = std::get_if<EinsumDescriptor>(&node.op_data)) {
                desc->c_prefactor  = c_pf;
                desc->ab_prefactor = ab_pf;
            }
            // Update all params in the store that match the old values
            // The executor lambda captures a shared_ptr<EinsumParams>,
            // so updating through the store changes the execution behavior.
            for (auto &params : _params_store) {
                // We need to find the params associated with this node.
                // Since we don't have a direct node→params mapping, update
                // all params (this is a simplification — in practice there's
                // typically one params per einsum node).
                // TODO: Add node_id → params mapping for precise updates.
            }
            // For now, update the params via the shared_ptr store.
            // The params are ordered by creation (same as node order during capture).
            // Find the params for this node by matching the node index.
            size_t node_idx  = 0;
            size_t param_idx = 0;
            for (size_t ni = 0; ni < _nodes.size(); ni++) {
                if (_nodes[ni].id == node_id) {
                    node_idx = ni;
                    break;
                }
            }
            // Count einsum nodes before this one to find the params index
            for (size_t ni = 0; ni < node_idx; ni++) {
                if (_nodes[ni].kind == OpKind::Einsum) {
                    param_idx++;
                }
            }
            if (param_idx < _params_store.size()) {
                _params_store[param_idx]->c_pf  = c_pf;
                _params_store[param_idx]->ab_pf = ab_pf;
            }
            return;
        }
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

    auto go = [&]<typename T>(T /*tag*/) {
        switch (a.rank) {
        case 1:
            fn(static_cast<Tensor<T, 1> *>(a.tensor_ptr), static_cast<Tensor<T, 1> *>(b.tensor_ptr));
            break;
        case 2:
            fn(static_cast<Tensor<T, 2> *>(a.tensor_ptr), static_cast<Tensor<T, 2> *>(b.tensor_ptr));
            break;
        case 3:
            fn(static_cast<Tensor<T, 3> *>(a.tensor_ptr), static_cast<Tensor<T, 3> *>(b.tensor_ptr));
            break;
        case 4:
            fn(static_cast<Tensor<T, 4> *>(a.tensor_ptr), static_cast<Tensor<T, 4> *>(b.tensor_ptr));
            break;
        default:
            EINSUMS_THROW_EXCEPTION(std::invalid_argument, "dispatch_binary: unsupported rank {}", a.rank);
        }
    };

    switch (a.dtype) {
    case packed_gemm::ScalarType::Float32:
        go(float{});
        break;
    case packed_gemm::ScalarType::Float64:
        go(double{});
        break;
    case packed_gemm::ScalarType::Complex64:
        go(std::complex<float>{});
        break;
    case packed_gemm::ScalarType::Complex128:
        go(std::complex<double>{});
        break;
    default:
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "dispatch_binary: unknown ScalarType");
    }
}

/// Dispatch a unary operation on one tensor.
template <typename Fn>
void dispatch_unary(TensorHandle const &a, Fn &&fn) {
    auto go = [&]<typename T>(T /*tag*/) {
        switch (a.rank) {
        case 1:
            fn(static_cast<Tensor<T, 1> *>(a.tensor_ptr));
            break;
        case 2:
            fn(static_cast<Tensor<T, 2> *>(a.tensor_ptr));
            break;
        case 3:
            fn(static_cast<Tensor<T, 3> *>(a.tensor_ptr));
            break;
        case 4:
            fn(static_cast<Tensor<T, 4> *>(a.tensor_ptr));
            break;
        default:
            EINSUMS_THROW_EXCEPTION(std::invalid_argument, "dispatch_unary: unsupported rank {}", a.rank);
        }
    };

    switch (a.dtype) {
    case packed_gemm::ScalarType::Float32:
        go(float{});
        break;
    case packed_gemm::ScalarType::Float64:
        go(double{});
        break;
    case packed_gemm::ScalarType::Complex64:
        go(std::complex<float>{});
        break;
    case packed_gemm::ScalarType::Complex128:
        go(std::complex<double>{});
        break;
    default:
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "dispatch_unary: unknown ScalarType");
    }
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

        switch (a_h.dtype) {
        case packed_gemm::ScalarType::Float32:
            go(float{});
            break;
        case packed_gemm::ScalarType::Float64:
            go(double{});
            break;
        case packed_gemm::ScalarType::Complex64:
            go(std::complex<float>{});
            break;
        case packed_gemm::ScalarType::Complex128:
            go(std::complex<double>{});
            break;
        default:
            EINSUMS_THROW_EXCEPTION(std::invalid_argument, "make_gemm_executor: unknown ScalarType");
        }
    };
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

    // Determine transpose flags: does A have the link index first or last?
    // Standard GEMM: C(M,N) = A(M,K) * B(K,N)
    // If A's indices have link before target, A is transposed.
    // For the folded case, we use column-major BLAS with explicit dimensions.
    // Since the data is contiguous and we control M,K,N, we call gemm directly.

    return [this, a_id, b_id, c_id, alpha, beta, M, K, N]() {
        auto const &a_h = tensor(a_id);
        auto const &b_h = tensor(b_id);
        auto       &c_h = tensor(c_id);

        auto go = [&]<typename T>(T /*tag*/) {
            // For rank-2 tensors, use the standard gemm.
            // For higher-rank, create temporary rank-2 views.
            if (a_h.rank == 2 && b_h.rank == 2 && c_h.rank == 2) {
                auto *A = static_cast<Tensor<T, 2> *>(a_h.tensor_ptr);
                auto *B = static_cast<Tensor<T, 2> *>(b_h.tensor_ptr);
                auto *C = static_cast<Tensor<T, 2> *>(c_h.tensor_ptr);
                linear_algebra::gemm<false, false>(static_cast<T>(alpha), *A, *B, static_cast<T>(beta), C);
            } else {
                // Higher-rank: use raw BLAS on the data pointers with folded dims.
                // Create temporary rank-2 TensorView-like wrappers.
                // For column-major: A is stored as (M, K), B as (K, N), C as (M, N).
                T const     *a_data = static_cast<T *>(a_h.data_ptr ? a_h.data_ptr : const_cast<void *>(a_h.tensor_ptr));
                T const     *b_data = static_cast<T *>(b_h.data_ptr ? b_h.data_ptr : const_cast<void *>(b_h.tensor_ptr));
                T           *c_data = static_cast<T *>(c_h.data_ptr ? c_h.data_ptr : const_cast<void *>(c_h.tensor_ptr));
                Tensor<T, 2> A_view("A_fold", M, K);
                Tensor<T, 2> B_view("B_fold", K, N);
                Tensor<T, 2> C_view("C_fold", M, N);
                std::memcpy(A_view.data(), a_data, M * K * sizeof(T));
                std::memcpy(B_view.data(), b_data, K * N * sizeof(T));
                if (beta != 0.0)
                    std::memcpy(C_view.data(), c_data, M * N * sizeof(T));
                linear_algebra::gemm<false, false>(static_cast<T>(alpha), A_view, B_view, static_cast<T>(beta), &C_view);
                std::memcpy(c_data, C_view.data(), M * N * sizeof(T));
            }
        };

        switch (a_h.dtype) {
        case packed_gemm::ScalarType::Float32:
            go(float{});
            break;
        case packed_gemm::ScalarType::Float64:
            go(double{});
            break;
        case packed_gemm::ScalarType::Complex64:
            go(std::complex<float>{});
            break;
        case packed_gemm::ScalarType::Complex128:
            go(std::complex<double>{});
            break;
        default:
            EINSUMS_THROW_EXCEPTION(std::invalid_argument, "make_einsum_executor: unknown ScalarType");
        }
    };
}

std::function<void()> Graph::make_zero_executor(TensorId tensor_id) {
    return [this, tensor_id]() {
        auto &h = tensor(tensor_id);
        dispatch_unary(h, [](auto *t) { t->zero(); });
    };
}

expected<void, GraphError> Graph::validate_tensors() const {
    for (auto const &[id, handle] : _tensors) {
        // Skip validation for deferred tensors — they haven't been materialized yet.
        if (handle.alloc_state == AllocState::Deferred)
            continue;
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
                if (handle.rank != 0 && handle.rank != expected_rank) {
                    EINSUMS_THROW_EXCEPTION(std::runtime_error,
                                            "Graph '{}': shape mismatch in node '{}': "
                                            "output tensor '{}' has rank {} but {} indices specified",
                                            _name, node.label, handle.name, handle.rank, expected_rank);
                }
            }
        }
    }
}

void Graph::print_timing_report(std::ostream &os) const {
    if (_timing_report.empty()) {
        os << "No timing data available. Call execute() first.\n";
        return;
    }

    // Sort by duration descending
    auto sorted = _timing_report;
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

void Graph::execute() {
    if (!_sorted) {
        topological_sort();
    }

    // Validate slot pointers every execution (cheap check).
    // This catches cross-pipeline tensor misuse before it segfaults.
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

    if (!_executed) {
        auto validation = validate_tensors();
        if (!validation) {
            EINSUMS_THROW_EXCEPTION(std::runtime_error, "{}", validation.error().message);
        }
        register_graph(this);
    }

    _timing_report.clear();

    profile::Profiler::instance().push(fmt::format("ComputeGraph::execute({})", _name));

    for (auto &node : _nodes) {
        profile::Profiler::instance().push(fmt::format("graph:{}/{}", _name, node.label));

        // Annotate with operation metadata
        profile::annotate("op_kind", op_kind_name(node.kind));
        profile::annotate("device", node.target == Target::GPU ? "GPU" : "CPU");

        // Transfer node annotations
        if (auto *tdesc = std::get_if<TransferDescriptor>(&node.op_data)) {
            profile::annotate("transfer_bytes", static_cast<int64_t>(tdesc->size_bytes));
        }

        if (auto *desc = std::get_if<EinsumDescriptor>(&node.op_data)) {
            profile::annotate("c_prefactor", to_string(desc->c_prefactor));
            profile::annotate("ab_prefactor", to_string(desc->ab_prefactor));
            if (!desc->spec.c_indices.empty()) {
                profile::annotate("c_indices", fmt::format("{}", fmt::join(desc->spec.c_indices, ",")));
                profile::annotate("a_indices", fmt::format("{}", fmt::join(desc->spec.a_indices, ",")));
                profile::annotate("b_indices", fmt::format("{}", fmt::join(desc->spec.b_indices, ",")));
            }
        } else if (auto *sdesc = std::get_if<ScaleDescriptor>(&node.op_data)) {
            profile::annotate("scale_factor", sdesc->factor);
        } else if (auto *cdesc = std::get_if<CommDescriptor>(&node.op_data)) {
            profile::annotate("comm_bytes", static_cast<int64_t>(cdesc->size_bytes));
            profile::annotate("comm_tensor", static_cast<int64_t>(cdesc->tensor_id));
        }

        // Annotate tensor dimensions and distribution for inputs/outputs
        for (unsigned long long const input : node.inputs) {
            auto it = _tensors.find(input);
            if (it != _tensors.end() && !it->second.dims.empty()) {
                profile::annotate(fmt::format("input.{}", it->second.name), fmt::format("{}", fmt::join(it->second.dims, "x")));
                if (it->second.is_distributed) {
                    profile::annotate(fmt::format("input.{}.distributed", it->second.name), "true");
                }
            }
        }
        for (unsigned long long const output : node.outputs) {
            auto it = _tensors.find(output);
            if (it != _tensors.end() && !it->second.dims.empty()) {
                profile::annotate(fmt::format("output.{}", it->second.name), fmt::format("{}", fmt::join(it->second.dims, "x")));
                if (it->second.is_distributed) {
                    profile::annotate(fmt::format("output.{}.distributed", it->second.name), "true");
                }
            }
        }

        if (node.estimated_flops > 0) {
            profile::annotate("estimated_flops", static_cast<int64_t>(node.estimated_flops));
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
                // Unified memory: no copy needed — GPU reads host memory directly.
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
                // Unified memory: no copy needed — result is already in host-accessible memory.
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
            // Unified memory: no swap needed — GPU reads tensor.data() directly.
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
        _timing_report.push_back({.id = node.id, .label = node.label, .kind = node.kind, .duration_ms = ms});

        profile::Profiler::instance().pop();
    }

    // Final D2H flush (discrete GPU only).
    // On unified memory, GPU wrote directly to host-accessible tensor data — no copy needed.
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

    profile::Profiler::instance().pop();
    _executed = true;
}

void Graph::execute(Executor &executor) {
    if (!_sorted) {
        topological_sort();
    }

    if (!_executed) {
        auto validation = validate_tensors();
        if (!validation) {
            EINSUMS_THROW_EXCEPTION(std::runtime_error, "{}", validation.error().message);
        }
    }

    _timing_report.clear();

    profile::Profiler::instance().push(fmt::format("ComputeGraph::execute({}, executor={})", _name, executor.name()));
    executor.execute(*this);
    profile::Profiler::instance().pop();
    _executed = true;
}

bool Graph::apply(PassManager &pm) {
    bool const modified = pm.run(*this);
    if (modified) {
        _executed = false;
    }
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
    for (auto const &t : _timing_report)
        timing_map[t.id] = t.duration_ms;

    // Nodes — use array index as ID
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
            nd.alpha     = desc->alpha;
            nd.beta      = desc->beta;
            nd.c_indices = fmt::format("{}", fmt::join(desc->c_indices, ","));
            nd.a_indices = fmt::format("{}", fmt::join(desc->a_indices, ","));
        }

        data.nodes.push_back(std::move(nd));
    }

    // Dependency edges
    {
        std::unordered_map<TensorId, std::vector<size_t>> writers, readers;
        for (size_t i = 0; i < _nodes.size(); i++) {
            for (auto in_tid : _nodes[i].inputs)
                readers[in_tid].push_back(i);
            for (auto out_tid : _nodes[i].outputs)
                writers[out_tid].push_back(i);
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

    // Serialize to JSON manually (structured — no Glaze dependency in library)
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
