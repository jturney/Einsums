//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/BLAS/ThreadControl.hpp>

#if defined(_WIN32)
#    include <windows.h>

// windows.h is kept in its own block above because psapi.h needs its types;
// the blank line is what stops clang-format sorting the two together.
// K32EnumProcessModules is exported by kernel32, so despite being declared in
// psapi.h it needs no psapi.lib.
#    include <algorithm>
#    include <iterator>
#    include <psapi.h>
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
    // Windows has no RTLD_DEFAULT, so walk what is loaded and ask each module.
    // Enumerating beats naming the DLLs directly: MKL alone ships the symbol
    // under mkl_rt or one of the layered mkl_core / mkl_intel_thread modules
    // depending on how it was linked, each with its own version suffix, and a
    // list of guesses silently degrades to a no-op the day one of them is
    // renamed. Nothing is loaded on our behalf - a vendor that is not already
    // in the process simply is not found.
    HMODULE      modules[512];
    DWORD        needed = 0;
    HANDLE const self   = GetCurrentProcess();
    if (K32EnumProcessModules(self, modules, static_cast<DWORD>(sizeof(modules)), &needed) == 0) {
        return nullptr;
    }
    DWORD const count = (std::min)(static_cast<DWORD>(needed / sizeof(HMODULE)), static_cast<DWORD>(std::size(modules)));
    for (DWORD i = 0; i < count; ++i) {
        if (FARPROC const symbol = GetProcAddress(modules[i], name)) {
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
