//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/BLAS/ThreadControl.hpp>

#if defined(_WIN32)
#    include <windows.h>
#else
#    include <dlfcn.h>
#endif

namespace einsums::blas {

namespace {

/// MKL's thread-local setter. Returns the previous value; we ignore it.
using MklSetNumThreadsLocal = int (*)(int);

/// MKL's reader. Reports what MKL would use in the calling thread's context,
/// so it reflects a thread-local setting rather than only the global one.
using MklGetMaxThreads = int (*)();

/// Look a symbol up in whatever is already loaded into this process. Nothing
/// is loaded on our behalf: if the vendor is not linked in, the lookup simply
/// fails and the caller falls back to doing nothing.
void *find_loaded_symbol(char const *name) {
#if defined(_WIN32)
    // No RTLD_DEFAULT equivalent, so walk the handful of module names the
    // vendors ship under. GetModuleHandle only succeeds for already-loaded
    // modules, which is the intent - it never triggers a load. The null handle
    // covers a statically linked vendor.
    static char const *const modules[] = {nullptr, "mkl_rt.2.dll", "mkl_rt.1.dll", "mkl_rt.dll", "mkl_core.2.dll", "mkl_core.dll"};
    for (char const *module : modules) {
        HMODULE const handle = GetModuleHandleA(module);
        if (handle == nullptr) {
            continue;
        }
        if (FARPROC const symbol = GetProcAddress(handle, name)) {
            // FARPROC -> object pointer needs the trip through void(*)().
            return reinterpret_cast<void *>(symbol);
        }
    }
    return nullptr;
#else
    return dlsym(RTLD_DEFAULT, name);
#endif
}

/// Resolved once: the lookup is the expensive part, and the answer cannot
/// change for the life of the process.
MklSetNumThreadsLocal resolve_mkl_setter() {
    // MKL exports both spellings; which one depends on the interface headers
    // the build used, so try the documented name first and the legacy
    // lowercase alias second.
    for (char const *name : {"MKL_Set_Num_Threads_Local", "mkl_set_num_threads_local"}) {
        if (void *symbol = find_loaded_symbol(name)) {
            return reinterpret_cast<MklSetNumThreadsLocal>(symbol);
        }
    }
    return nullptr;
}

MklGetMaxThreads resolve_mkl_getter() {
    for (char const *name : {"MKL_Get_Max_Threads", "mkl_get_max_threads"}) {
        if (void *symbol = find_loaded_symbol(name)) {
            return reinterpret_cast<MklGetMaxThreads>(symbol);
        }
    }
    return nullptr;
}

MklSetNumThreadsLocal const mkl_setter = resolve_mkl_setter();
MklGetMaxThreads const      mkl_getter = resolve_mkl_getter();

} // namespace

void set_num_threads_this_thread(int nthreads) {
    if (nthreads < 1) {
        return;
    }
    if (mkl_setter != nullptr) {
        mkl_setter(nthreads);
    }
}

bool has_per_thread_control() {
    return mkl_setter != nullptr;
}

int get_num_threads_this_thread() {
    if (mkl_getter != nullptr) {
        return mkl_getter();
    }
    return 0;
}

} // namespace einsums::blas
