//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/GPU/Error.hpp>
#include <Einsums/GPU/Stream.hpp>

namespace einsums::gpu {

// ===========================================================================
// Thread-local stream management
// ===========================================================================

#if defined(EINSUMS_HAVE_CUDA) || defined(EINSUMS_HAVE_HIP)

stream_t get_stream() {
    return 0;
}
stream_t get_stream(int) {
    return 0;
}
blas_handle_t get_blas_handle() {
    return 0;
}
solver_handle_t get_solver_handle() {
    return 0;
}

void stream_wait(stream_t) {
}
void stream_wait(bool) {
}
void all_stream_wait() {
}

#else // Mock and MPS backends use int type aliases

stream_t get_stream() {
    return 0;
}
stream_t get_stream(int) {
    return 0;
}
blas_handle_t get_blas_handle() {
    return 0;
}
solver_handle_t get_solver_handle() {
    return 0;
}

void stream_wait(stream_t) {
}
void stream_wait(bool) {
}
void all_stream_wait() {
}

#endif

// ===========================================================================
// Stream creation/destruction
// MPS: creates real MTLCommandQueues internally, returns int handles.
// Mock: returns sequential ints.
// ===========================================================================

stream_t create_stream() {
#if defined(EINSUMS_HAVE_CUDA)
    cudaStream_t s;
    gpu_catch(cudaStreamCreate(&s));
    return s;
#elif defined(EINSUMS_HAVE_HIP)
    hipStream_t s;
    gpu_catch(hipStreamCreate(&s));
    return s;
#else
    // MPS and mock: return sequential int handles.
    // MPS command queues are managed internally by MPSBackend.mm.
    static int stream_counter = 0;
    return ++stream_counter;
#endif
}

void destroy_stream(stream_t stream) {
#if defined(EINSUMS_HAVE_CUDA)
    gpu_catch(cudaStreamDestroy(stream));
#elif defined(EINSUMS_HAVE_HIP)
    gpu_catch(hipStreamDestroy(stream));
#else
    (void)stream;
#endif
}

// ===========================================================================
// Events
// ===========================================================================

event_t create_event() {
#if defined(EINSUMS_HAVE_CUDA)
    cudaEvent_t e;
    gpu_catch(cudaEventCreate(&e));
    return e;
#elif defined(EINSUMS_HAVE_HIP)
    hipEvent_t e;
    gpu_catch(hipEventCreate(&e));
    return e;
#else
    static int event_counter = 0;
    return ++event_counter;
#endif
}

void destroy_event(event_t event) {
#if defined(EINSUMS_HAVE_CUDA)
    gpu_catch(cudaEventDestroy(event));
#elif defined(EINSUMS_HAVE_HIP)
    gpu_catch(hipEventDestroy(event));
#else
    (void)event;
#endif
}

void record_event(event_t event, stream_t stream) {
#if defined(EINSUMS_HAVE_CUDA)
    gpu_catch(cudaEventRecord(event, stream));
#elif defined(EINSUMS_HAVE_HIP)
    gpu_catch(hipEventRecord(event, stream));
#else
    (void)event;
    (void)stream;
#endif
}

void stream_wait_event(stream_t stream, event_t event) {
#if defined(EINSUMS_HAVE_CUDA)
    gpu_catch(cudaStreamWaitEvent(stream, event, 0));
#elif defined(EINSUMS_HAVE_HIP)
    gpu_catch(hipStreamWaitEvent(stream, event, 0));
#else
    (void)stream;
    (void)event;
#endif
}

bool event_completed(event_t event) {
#if defined(EINSUMS_HAVE_CUDA)
    cudaError_t err = cudaEventQuery(event);
    return err == cudaSuccess;
#elif defined(EINSUMS_HAVE_HIP)
    hipError_t err = hipEventQuery(event);
    return err == hipSuccess;
#else
    (void)event;
    return true; // MPS and mock: synchronous execution, always complete.
#endif
}

} // namespace einsums::gpu
