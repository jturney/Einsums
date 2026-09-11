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

/// Try to dispatch a GPU node via gpu::blas (GEMM or GEMV).
/// Returns true if dispatched, false if not applicable (caller should use CPU fallback).
bool try_gpu_blas_dispatch(Node const &node, std::unordered_map<TensorId, TensorHandle> const &tensors, DeviceShadowMap &shadows);

EINSUMS_NAMESPACE_END(compute_graph::gpu_dispatch)
