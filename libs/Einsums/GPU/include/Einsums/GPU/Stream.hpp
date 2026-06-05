//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/GPU/Platform.hpp>

// Include vendor headers for type definitions
#if defined(EINSUMS_HAVE_CUDA)
#    include <cublas_v2.h>
#    include <cuda_runtime_api.h>
#    include <cusolverDn.h>
#elif defined(EINSUMS_HAVE_HIP)
#    include <hip/hip_runtime_api.h>
#    include <hipblas/hipblas.h>
#    include <hipsolver/hipsolver.h>
#endif

namespace einsums::gpu {

// ===========================================================================
// Type aliases — resolve to vendor type or mock placeholder.
// ===========================================================================

// NOLINTBEGIN
#if defined(EINSUMS_HAVE_CUDA)
using stream_t        = cudaStream_t;
using blas_handle_t   = cublasHandle_t;
using solver_handle_t = cusolverDnHandle_t;
using event_t         = cudaEvent_t;
#elif defined(EINSUMS_HAVE_HIP)
using stream_t        = hipStream_t;
using blas_handle_t   = hipblasHandle_t;
using solver_handle_t = hipsolverHandle_t;
using event_t         = hipEvent_t;
#else // Mock
using stream_t        = int;
using blas_handle_t   = int;
using solver_handle_t = int;
using event_t         = int;
#endif
// NOLINTEND

// ===========================================================================
// Thread-local stream management (existing pattern from GPUStreams).
// ===========================================================================

EINSUMS_EXPORT stream_t        get_stream();
EINSUMS_EXPORT stream_t        get_stream(int thread_id);
EINSUMS_EXPORT blas_handle_t   get_blas_handle();
EINSUMS_EXPORT solver_handle_t get_solver_handle();

EINSUMS_EXPORT void stream_wait(stream_t stream);
EINSUMS_EXPORT void stream_wait(bool may_skip = false);
EINSUMS_EXPORT void all_stream_wait();

// ===========================================================================
// Stream creation/destruction (for ComputeGraph node-level streams).
// ===========================================================================

EINSUMS_EXPORT stream_t create_stream();
EINSUMS_EXPORT void     destroy_stream(stream_t stream);

// ===========================================================================
// Events for dependency tracking between graph nodes.
// ===========================================================================

EINSUMS_EXPORT event_t create_event();
EINSUMS_EXPORT void    destroy_event(event_t event);
EINSUMS_EXPORT void    record_event(event_t event, stream_t stream);
EINSUMS_EXPORT void    stream_wait_event(stream_t stream, event_t event);
EINSUMS_EXPORT bool    event_completed(event_t event);

} // namespace einsums::gpu
