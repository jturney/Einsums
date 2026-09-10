//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config/Namespace.hpp>
#include <Einsums/GPU/Error.hpp>
#include <Einsums/GPU/Runtime.hpp>

#if defined(EINSUMS_HAVE_CUDA)
#    include <cuda_runtime_api.h>
#elif defined(EINSUMS_HAVE_HIP)
#    include <hip/hip_runtime_api.h>
#elif defined(EINSUMS_HAVE_MPS)
#    include <Einsums/GPU/MPSBackend.hpp>
#else
#    include <cstdlib>
#    include <cstring>
#    if defined(EINSUMS_HAVE_GPU_MOCK_DISCRETE)
#        include <sys/mman.h>
#    endif
#    if defined(_WIN32)
#        ifndef WIN32_LEAN_AND_MEAN
#            define WIN32_LEAN_AND_MEAN
#        endif
#        ifndef NOMINMAX
#            define NOMINMAX
#        endif
#        include <windows.h>
#    else
#        include <unistd.h>
#    endif
#endif

#include <algorithm>
#include <atomic>
#include <cstring>
#include <mutex>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(gpu)

#if defined(EINSUMS_HAVE_GPU_MOCK_DISCRETE)
namespace {

/// Page-aligned "device" allocation for the mock-discrete backend.
///
/// The point is that a host dereference must FAULT. A plain malloc'd mock
/// pointer is host-readable, so code that wrongly treats a device pointer as a
/// host pointer - a CPU BLAS call on a device buffer, or the ComputeGraph
/// fallback running its CPU lambda while pointers are still swapped to shadows
/// - silently computes the right answer under the mock and segfaults on CUDA.
/// That asymmetry is exactly what let those bugs sit unnoticed.
///
/// So the region is mapped PROT_NONE and only made accessible to the explicit
/// transfer entry points (memcpy_*_to_*, device_memset), which unprotect it for
/// the duration of the copy. Everything else touching it takes SIGSEGV at the
/// offending instruction, where a debugger can name the culprit.
///
/// Layout, one mmap per allocation:
///   [ guard page ][ payload pages (PROT_NONE at rest) ][ guard page ]
/// with the header holding the mapping base and length so device_free and the
/// unprotect helpers can recover them from the payload pointer alone.
struct MockDiscreteHeader {
    void  *base;   ///< start of the whole mapping (first guard page)
    size_t length; ///< total mapped length
    size_t bytes;  ///< requested payload size
};

std::size_t mock_page_size() {
    static std::size_t const pagesize = static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
    return pagesize;
}

/// Header lives in the last bytes of the leading guard page, so the payload
/// stays page-aligned (device allocators are page-aligned in practice, and
/// alignment assumptions in the tensor layer would otherwise trip).
MockDiscreteHeader *mock_header(void *payload) {
    auto *p = static_cast<unsigned char *>(payload);
    return reinterpret_cast<MockDiscreteHeader *>(p - sizeof(MockDiscreteHeader));
}

void mock_set_payload_prot(void *payload, int prot) {
    MockDiscreteHeader const *h        = mock_header(payload);
    std::size_t const         pagesize = mock_page_size();
    std::size_t const         span     = ((h->bytes + pagesize - 1) / pagesize) * pagesize;
    if (span > 0) {
        mprotect(payload, span, prot);
    }
}

/// Live mock allocations, so MockDeviceKernelScope can unprotect all of them.
/// A device kernel may touch any device buffer, and the mock has no way to know
/// which, so the scope is all-or-nothing.
std::mutex               &mock_registry_mutex() {
    static std::mutex m;
    return m;
}
std::vector<void *> &mock_registry() {
    static std::vector<void *> v;
    return v;
}

void mock_registry_add(void *payload) {
    std::lock_guard<std::mutex> const lock(mock_registry_mutex());
    mock_registry().push_back(payload);
}

void mock_registry_remove(void *payload) {
    std::lock_guard<std::mutex> const lock(mock_registry_mutex());
    auto                             &v  = mock_registry();
    auto                              it = std::find(v.begin(), v.end(), payload);
    if (it != v.end())
        v.erase(it);
}

/// RAII: make a mock device buffer host-accessible for one transfer.
struct MockAccess {
    void *payload;
    explicit MockAccess(void const *p) : payload(const_cast<void *>(p)) {
        if (payload)
            mock_set_payload_prot(payload, PROT_READ | PROT_WRITE);
    }
    ~MockAccess() {
        if (payload)
            mock_set_payload_prot(payload, PROT_NONE);
    }
    MockAccess(MockAccess const &)            = delete;
    MockAccess &operator=(MockAccess const &) = delete;
};

} // namespace
#endif // EINSUMS_HAVE_GPU_MOCK_DISCRETE

expected<void *, GpuError> device_malloc(size_t bytes) {
    void *ptr = nullptr;
#if defined(EINSUMS_HAVE_CUDA)
    auto err = cudaMalloc(&ptr, bytes);
    if (err != cudaSuccess)
        return unexpected(GpuError{fmt::format("cudaMalloc({} bytes) failed: {}", bytes, cudaGetErrorString(err)), static_cast<int>(err)});
#elif defined(EINSUMS_HAVE_HIP)
    auto err = hipMalloc(&ptr, bytes);
    if (err != hipSuccess)
        return unexpected(GpuError{fmt::format("hipMalloc({} bytes) failed: {}", bytes, hipGetErrorString(err)), static_cast<int>(err)});
#elif defined(EINSUMS_HAVE_MPS)
    ptr = mps::device_malloc(bytes);
    if (!ptr && bytes > 0)
        return unexpected(GpuError{.message = fmt::format("MPS device_malloc({} bytes) failed", bytes), .code = -1});
#elif defined(EINSUMS_HAVE_GPU_MOCK_DISCRETE)
    {
        std::size_t const pagesize = mock_page_size();
        std::size_t const payload  = ((bytes + pagesize - 1) / pagesize) * pagesize;
        std::size_t const length   = payload + 2 * pagesize; // guard page either side
        void             *base     = mmap(nullptr, length, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (base == MAP_FAILED)
            return unexpected(GpuError{fmt::format("mock-discrete mmap({} bytes) failed", bytes), -1});

        auto *payload_ptr = static_cast<unsigned char *>(base) + pagesize;
        // Header sits at the tail of the leading guard page, which must stay
        // readable/writable for us; the guard property that matters is that the
        // PAYLOAD faults, and that overruns off either end land outside it.
        MockDiscreteHeader *h = reinterpret_cast<MockDiscreteHeader *>(payload_ptr - sizeof(MockDiscreteHeader));
        h->base               = base;
        h->length             = length;
        h->bytes              = bytes;

        ptr = payload_ptr;
        mock_set_payload_prot(ptr, PROT_NONE);
        mock_registry_add(ptr);
    }
#else
    ptr = std::malloc(bytes);
    if (!ptr && bytes > 0)
        return unexpected(GpuError{fmt::format("malloc({} bytes) failed", bytes), -1});
#endif
    return ptr;
}

void device_free(void *ptr) {
    if (ptr == nullptr)
        return;
#if defined(EINSUMS_HAVE_CUDA)
    gpu_catch(cudaFree(ptr));
#elif defined(EINSUMS_HAVE_HIP)
    gpu_catch(hipFree(ptr));
#elif defined(EINSUMS_HAVE_MPS)
    mps::device_free(ptr);
#elif defined(EINSUMS_HAVE_GPU_MOCK_DISCRETE)
    {
        mock_registry_remove(ptr);
        MockDiscreteHeader const *h = mock_header(ptr);
        munmap(h->base, h->length);
    }
#else
    std::free(ptr);
#endif
}

void memcpy_host_to_device(void *dst, void const *src, size_t bytes) {
#if defined(EINSUMS_HAVE_CUDA)
    gpu_catch(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice));
#elif defined(EINSUMS_HAVE_HIP)
    gpu_catch(hipMemcpy(dst, src, bytes, hipMemcpyHostToDevice));
#elif defined(EINSUMS_HAVE_MPS)
    mps::memcpy_host_to_device(dst, src, bytes);
#elif defined(EINSUMS_HAVE_GPU_MOCK_DISCRETE)
    {
        MockAccess dst_access(dst);
        std::memcpy(dst, src, bytes);
    }
#else
    std::memcpy(dst, src, bytes);
#endif
}

void memcpy_device_to_host(void *dst, void const *src, size_t bytes) {
#if defined(EINSUMS_HAVE_CUDA)
    gpu_catch(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost));
#elif defined(EINSUMS_HAVE_HIP)
    gpu_catch(hipMemcpy(dst, src, bytes, hipMemcpyDeviceToHost));
#elif defined(EINSUMS_HAVE_MPS)
    mps::memcpy_device_to_host(dst, src, bytes);
#elif defined(EINSUMS_HAVE_GPU_MOCK_DISCRETE)
    {
        MockAccess src_access(src);
        std::memcpy(dst, src, bytes);
    }
#else
    std::memcpy(dst, src, bytes);
#endif
}

void memcpy_device_to_device(void *dst, void const *src, size_t bytes) {
#if defined(EINSUMS_HAVE_CUDA)
    gpu_catch(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice));
#elif defined(EINSUMS_HAVE_HIP)
    gpu_catch(hipMemcpy(dst, src, bytes, hipMemcpyDeviceToDevice));
#elif defined(EINSUMS_HAVE_MPS)
    mps::memcpy_device_to_device(dst, src, bytes);
#elif defined(EINSUMS_HAVE_GPU_MOCK_DISCRETE)
    {
        MockAccess dst_access(dst);
        MockAccess src_access(src);
        std::memcpy(dst, src, bytes);
    }
#else
    std::memcpy(dst, src, bytes);
#endif
}

void device_memset(void *ptr, int value, size_t bytes) {
#if defined(EINSUMS_HAVE_CUDA)
    gpu_catch(cudaMemset(ptr, value, bytes));
#elif defined(EINSUMS_HAVE_HIP)
    gpu_catch(hipMemset(ptr, value, bytes));
#elif defined(EINSUMS_HAVE_MPS)
    mps::device_memset(ptr, value, bytes);
#elif defined(EINSUMS_HAVE_GPU_MOCK_DISCRETE)
    {
        MockAccess access(ptr);
        std::memset(ptr, value, bytes);
    }
#else
    std::memset(ptr, value, bytes);
#endif
}

void device_synchronize() {
#if defined(EINSUMS_HAVE_CUDA)
    gpu_catch(cudaDeviceSynchronize());
#elif defined(EINSUMS_HAVE_HIP)
    gpu_catch(hipDeviceSynchronize());
#elif defined(EINSUMS_HAVE_MPS)
    mps::device_synchronize();
#else
    // Mock: everything is synchronous, nothing to wait for.
#endif
}

namespace {

/// Override for available_device_memory (0 = use real value).
/// Works on all backends for testing budget-constrained placement.
std::atomic<size_t> g_memory_override{0};

#if !defined(EINSUMS_HAVE_CUDA) && !defined(EINSUMS_HAVE_HIP) && !defined(EINSUMS_HAVE_MPS)
/// Mock device memory limit (0 = use default: system RAM / 2).
std::atomic<size_t> mock_memory_limit{0};

size_t default_mock_limit() {
#    if defined(_SC_PHYS_PAGES) && defined(_SC_PAGE_SIZE)
    auto pages     = sysconf(_SC_PHYS_PAGES);
    auto page_size = sysconf(_SC_PAGE_SIZE);
    if (pages > 0 && page_size > 0) {
        return static_cast<size_t>(pages) * static_cast<size_t>(page_size) / 2;
    }
#    elif defined(EINSUMS_WINDOWS)
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status) && status.ullTotalPhys > 0) {
        return static_cast<size_t>(status.ullTotalPhys) / 2;
    }
#    endif
    return size_t{4} * 1024 * 1024 * 1024; // 4 GB fallback
}
#endif

} // namespace

size_t available_device_memory() {
    // Check test override first (works on all backends).
    size_t override_val = g_memory_override.load(std::memory_order_relaxed);
    if (override_val > 0)
        return override_val;

#if defined(EINSUMS_HAVE_CUDA)
    size_t free_mem = 0, total_mem = 0;
    gpu_catch(cudaMemGetInfo(&free_mem, &total_mem));
    return free_mem;
#elif defined(EINSUMS_HAVE_HIP)
    size_t free_mem = 0, total_mem = 0;
    gpu_catch(hipMemGetInfo(&free_mem, &total_mem));
    return free_mem;
#elif defined(EINSUMS_HAVE_MPS)
    return mps::available_device_memory();
#else
    return default_mock_limit();
#endif
}

void set_mock_device_memory_limit(size_t bytes) {
    g_memory_override.store(bytes, std::memory_order_relaxed);
}

std::string device_name() {
    // Served from the cached probe rather than re-querying. This used to return
    // the literal "Unknown CUDA Device" when cudaGetDeviceProperties failed,
    // which reads as a real device downstream: ComputeGraph's
    // CostModel::has_gpu() tests !gpu.name.empty(), so a driverless machine
    // convinced the cost model it had a GPU to place work on.
#if defined(EINSUMS_HAVE_MPS)
    return mps::device_name();
#else
    return device_capabilities().name;
#endif
}

namespace {

/// One-shot probe behind device_capabilities(). Never throws and never lets a
/// vendor error escape: a machine with no driver must yield available == false,
/// not an exception out of a query function.
DeviceCapabilities detect_capabilities() {
    DeviceCapabilities caps;

#if defined(EINSUMS_HAVE_CUDA)
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count <= 0) {
        // Clear the sticky error so later calls report their own failures
        // rather than inheriting this one.
        (void)cudaGetLastError();
        return caps;
    }
    cudaDeviceProp props{};
    if (cudaGetDeviceProperties(&props, 0) != cudaSuccess) {
        (void)cudaGetLastError();
        return caps;
    }
    caps.available      = true;
    caps.device_count   = count;
    caps.compute_major  = props.major;
    caps.compute_minor  = props.minor;
    caps.unified_memory = props.unifiedAddressing != 0;
    caps.total_memory   = props.totalGlobalMem;
    caps.name           = props.name;

    // Tensor-core arithmetic gates, by compute capability. Turing (sm_75, the
    // TITAN RTX) has FP16 tensor cores but neither BF16 nor FP8.
    int const cc   = props.major * 10 + props.minor;
    caps.fp16_gemm = cc >= 70;
    caps.bf16_gemm = cc >= 80;
    caps.fp8_gemm  = cc >= 89;

#elif defined(EINSUMS_HAVE_HIP)
    int count = 0;
    if (hipGetDeviceCount(&count) != hipSuccess || count <= 0) {
        (void)hipGetLastError();
        return caps;
    }
    hipDeviceProp_t props{};
    if (hipGetDeviceProperties(&props, 0) != hipSuccess) {
        (void)hipGetLastError();
        return caps;
    }
    caps.available      = true;
    caps.device_count   = count;
    caps.compute_major  = props.major;
    caps.compute_minor  = props.minor;
    caps.unified_memory = props.unifiedAddressing != 0;
    caps.total_memory   = props.totalGlobalMem;
    caps.name           = props.name;

#elif defined(EINSUMS_HAVE_MPS)
    caps.available      = true;
    caps.device_count   = 1;
    caps.unified_memory = true;
    caps.name           = mps::device_name();
    caps.total_memory   = mps::available_device_memory();
    caps.fp16_gemm      = true;

#else
    // Mock: the host is the device, so it is always "available".
    caps.available      = true;
    caps.device_count   = 1;
    caps.unified_memory = has_unified_memory;
    caps.total_memory   = available_device_memory();
    caps.name           = has_mock_discrete ? "Mock GPU (discrete)" : "Mock GPU (unified)";
#endif

    return caps;
}

} // namespace

DeviceCapabilities const &device_capabilities() {
    // Function-local static: thread-safe initialization, probed once. The CUDA
    // probe creates a context, which is far too expensive to repeat per query.
    static DeviceCapabilities const caps = detect_capabilities();
    return caps;
}

bool gpu_available() {
    return device_capabilities().available;
}

#if defined(EINSUMS_HAVE_GPU_MOCK_DISCRETE)

namespace {
/// Nesting depth of device-kernel scopes on this thread. Only the outermost
/// scope flips protection, so the templated gpu::blas wrappers forwarding to
/// the typed ones do not re-protect out from under the inner call.
thread_local int mock_kernel_depth = 0;

void mock_set_all(int prot) {
    std::lock_guard<std::mutex> const lock(mock_registry_mutex());
    for (void *p : mock_registry()) {
        mock_set_payload_prot(p, prot);
    }
}
} // namespace

MockDeviceKernelScope::MockDeviceKernelScope() {
    if (mock_kernel_depth++ == 0) {
        mock_set_all(PROT_READ | PROT_WRITE);
    }
}

MockDeviceKernelScope::~MockDeviceKernelScope() {
    if (--mock_kernel_depth == 0) {
        mock_set_all(PROT_NONE);
    }
}

#endif // EINSUMS_HAVE_GPU_MOCK_DISCRETE

EINSUMS_NAMESPACE_END(gpu)
