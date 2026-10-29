//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// Which BLAS this process uses, for the hardware profile (measurements taken against one BLAS do
// not describe another) and for the thread fence, which must know whether OpenMP reaches the
// vendor. The answer comes from the library that actually provides dgemm, found
// from that function's address, not from the vendor the build was configured with: a conda
// environment can swap the implementation behind libblas without rebuilding anything, and a name
// lookup in the global scope finds nothing when Python loads Einsums with RTLD_LOCAL.

#include <Einsums/Config.hpp>

#include <Einsums/BLASVendor/Vendor.hpp>
#include <Einsums/Hardware/Profile.hpp>

#include <filesystem>
#include <string>
#include <string_view>
#include <system_error>

#include "Common.hpp"

#if defined(_WIN32)
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#else
#    include <dlfcn.h>
#endif

extern "C" {
extern void FC_GLOBAL(dgemm, DGEMM)(char *, char *, einsums::blas::int_t *, einsums::blas::int_t *, einsums::blas::int_t *, double *,
                                    double const *, einsums::blas::int_t *, double const *, einsums::blas::int_t *, double *, double *,
                                    einsums::blas::int_t *);
}

EINSUMS_NAMESPACE_BEGIN(blas::vendor)

namespace {

using OpenBlasConfig = char *(*)();
using MklVersion     = void (*)(char *, int);

/// "OpenBLAS 0.3.28" out of a config string that goes on to list build flags.
std::string openblas_version(char const *config) {
    std::string_view const text(config);
    auto const             first = text.find(' ');
    if (first == std::string_view::npos) {
        return std::string(text);
    }
    auto const second = text.find(' ', first + 1);
    return std::string(text.substr(0, second));
}

/// The MKL version string without its trailing padding.
std::string mkl_version(MklVersion get) {
    char buffer[256] = {}; // NOLINT
    get(buffer, static_cast<int>(sizeof(buffer)));
    std::string const text(buffer);
    auto const        end = text.find_last_not_of(" \t\r\n");
    return end == std::string::npos ? std::string{} : text.substr(0, end + 1);
}

/// A name for the library at @p path that is the same in every environment holding it.
std::string library_name(std::string const &path) {
    // Accelerate lives in the shared cache rather than on disk, under a framework path.
    if (path.find("Accelerate.framework") != std::string::npos || path.find("vecLib.framework") != std::string::npos) {
        return "Accelerate";
    }
    // Resolve symlinks such as conda's libblas.so.3, which point at the implementation chosen.
    std::error_code             ec;
    std::filesystem::path const real = std::filesystem::canonical(path, ec);
    return (ec ? std::filesystem::path(path) : real).filename().string();
}

/// The library that provides dgemm, and the vendor entry points it exports.
struct DgemmProvider {
    std::string    path;
    OpenBlasConfig openblas{nullptr};
    MklVersion     mkl{nullptr};
};

DgemmProvider find_dgemm_provider() {
    auto const   *symbol = reinterpret_cast<void const *>(&FC_GLOBAL(dgemm, DGEMM));
    DgemmProvider provider;
#if defined(_WIN32)
    HMODULE module = nullptr;
    if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           static_cast<LPCSTR>(symbol), &module) != 0) {
        char path[MAX_PATH] = {}; // NOLINT
        if (GetModuleFileNameA(module, path, MAX_PATH) != 0) {
            provider.path = path;
        }
        provider.openblas = reinterpret_cast<OpenBlasConfig>(GetProcAddress(module, "openblas_get_config"));
        provider.mkl      = reinterpret_cast<MklVersion>(GetProcAddress(module, "MKL_Get_Version_String"));
    }
#else
    Dl_info info{};
    if (dladdr(symbol, &info) != 0 && info.dli_fname != nullptr) {
        provider.path = info.dli_fname;
        // The library is already loaded; this only obtains a handle to look its entry points up in.
        // Closing it again leaves the process's own reference, so the pointers stay valid.
        if (void *handle = dlopen(info.dli_fname, RTLD_LAZY | RTLD_NOLOAD); handle != nullptr) {
            provider.openblas = reinterpret_cast<OpenBlasConfig>(dlsym(handle, "openblas_get_config"));
            provider.mkl      = reinterpret_cast<MklVersion>(dlsym(handle, "MKL_Get_Version_String"));
            dlclose(handle);
        }
    }
#endif
    return provider;
}

std::string identify() {
    DgemmProvider const provider = find_dgemm_provider();
    std::string const   library  = provider.path.empty() ? std::string{} : library_name(provider.path);
    std::string         version;
    if (provider.openblas != nullptr) {
        version = openblas_version(provider.openblas());
    } else if (provider.mkl != nullptr) {
        version = mkl_version(provider.mkl);
    }
    if (library.empty()) {
        return version.empty() ? std::string{"unidentified"} : version;
    }
    return version.empty() ? library : version + " (" + library + ")";
}

[[maybe_unused]] int const registered = (hardware::set_blas_identity_provider(&identify), 0);

} // namespace

char const *openblas_config() {
    static char const *const config = []() -> char const * {
        OpenBlasConfig const get = find_dgemm_provider().openblas;
        return get != nullptr ? get() : nullptr;
    }();
    return config;
}

EINSUMS_NAMESPACE_END(blas::vendor)
