//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/// @file GpuDispatch.hpp
/// @brief The one entry point into the GPU BLAS fast paths.
///
/// Private to the `Graph` sources. `Execute.cpp` calls @ref try_gpu_blas_dispatch
/// per node and falls back to the CPU executor whenever it declines, so the whole
/// backend is a single predicate from the executor's point of view and the shapes
/// each fast path insists on stay in `GpuDispatch.cpp`.

#include <Einsums/ComputeGraph/DeviceShadowMap.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/Config/Namespace.hpp>

#include <unordered_map>

EINSUMS_NAMESPACE_BEGIN(compute_graph::gpu_dispatch)

/// The tensor's CURRENT host data pointer, read through its rank-erased impl.
///
/// TensorHandle::data_ptr is a registration-time snapshot that nothing
/// refreshes - the header says so, and says not to build an executor on it. It
/// is null for any tensor that was deferred when registered, which is every
/// tensor a Materialize node later allocates. The device-shadow path was built
/// on that field, so for lazily materialized tensors it skipped the upload and,
/// worse, had nowhere to copy a GPU result back to: the answer was computed on
/// the device and then dropped.
///
/// Shared with `Execute.cpp`, which needs the same answer for its host-to-device
/// and device-to-host transfers as the fast paths need for their operands.
///
/// Returns nullptr when the tensor genuinely has no single buffer (tile-wise
/// sparse) or is not materialized yet.
[[nodiscard]] void *live_host_ptr(TensorHandle const &h);

/// Try to dispatch a GPU node via gpu::blas (GEMM or GEMV).
/// Returns true if dispatched, false if not applicable (caller should use CPU fallback).
bool try_gpu_blas_dispatch(Node const &node, std::unordered_map<TensorId, TensorHandle> const &tensors, DeviceShadowMap &shadows);

EINSUMS_NAMESPACE_END(compute_graph::gpu_dispatch)
