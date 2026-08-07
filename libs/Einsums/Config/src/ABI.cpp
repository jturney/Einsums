//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config/ABI.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Config/Version.hpp>

#include <algorithm>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#if defined(_WIN32)
#    include <windows.h>
// psapi.h must follow windows.h.
#    include <psapi.h>
#elif defined(__APPLE__)
#    include <dlfcn.h>
#    include <mach-o/dyld.h>
#else
#    include <dlfcn.h>
#    include <link.h>
#endif

EINSUMS_NAMESPACE_BEGIN(sealed)

namespace {

/// Basename of @p path, for both separators: a Windows path reaches this code
/// with backslashes even when the rest of the string looks POSIX.
std::string_view basename_of(std::string_view path) {
    std::size_t const pos = path.find_last_of("/\\");
    return pos == std::string_view::npos ? path : path.substr(pos + 1);
}

/// Whether @p path names a libEinsums rather than something merely linked
/// against one. Prefix-matched so it keeps working once the library carries an
/// ABI tag in its filename (`libEinsums-abi7.so`), and so it does NOT match the
/// Python extension module, which is `_core`.
bool is_einsums_library(std::string_view path) {
    std::string_view const base = basename_of(path);
    return base.starts_with("libEinsums") || base.starts_with("Einsums");
}

/// The address whose identity IS the world identity.
///
/// A function-local static in a non-inline function: exactly one exists per
/// loaded copy of the library, so comparing its address answers "did we resolve
/// to the same copy?" and nothing else. Its value is never read.
void const *identity_address() noexcept {
    static char const marker = 0;
    return &marker;
}

/// Path of the library containing identity_address().
///
/// Resolved from a symbol inside this library, so each caller gets the path of
/// the copy IT bound to, which is the whole point: two callers in two worlds
/// print two different paths.
std::string this_library_path() {
#if defined(_WIN32)
    HMODULE module = nullptr;
    if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           reinterpret_cast<LPCSTR>(identity_address()), &module) != 0) {
        char  buffer[MAX_PATH] = {};
        DWORD n                = GetModuleFileNameA(module, buffer, static_cast<DWORD>(sizeof(buffer)));
        if (n > 0 && n < sizeof(buffer)) {
            return std::string(buffer, n);
        }
    }
    return {};
#else
    Dl_info info{};
    if (dladdr(identity_address(), &info) != 0 && info.dli_fname != nullptr) {
        return info.dli_fname;
    }
    return {};
#endif
}

/// Registered stage-module names, guarded because nothing stops two threads
/// importing modules at once.
struct Registry {
    std::mutex               mutex;
    std::vector<std::string> names;
};

Registry &registry() {
    static Registry r;
    return r;
}

#if !defined(_WIN32) && !defined(__APPLE__)
int collect_elf_image(struct dl_phdr_info *info, std::size_t, void *data) {
    auto *out = static_cast<std::vector<std::string> *>(data);
    if (info->dlpi_name != nullptr && info->dlpi_name[0] != '\0' && is_einsums_library(info->dlpi_name)) {
        out->emplace_back(info->dlpi_name);
    }
    return 0;
}
#endif

} // namespace

WorldInfo const &world() noexcept {
    // Function-local static rather than a namespace-scope object: the path
    // lookup and the layout fingerprint both want to run after the library is
    // fully loaded, not during its static initialization.
    static std::string const path = this_library_path();
    // Config sits below the Version module, so the string is assembled from
    // the macros here rather than borrowed from einsums::full_version_as_string().
    static std::string const version = std::to_string(EINSUMS_VERSION_MAJOR) + "." + std::to_string(EINSUMS_VERSION_MINOR) + "." +
                                       std::to_string(EINSUMS_VERSION_PATCH) + EINSUMS_VERSION_TAG;
    static std::string const commit{einsums::git_commit()};

    static WorldInfo const info = [] {
        WorldInfo w{};
        w.struct_size        = sizeof(WorldInfo);
        w.identity           = identity_address();
        w.config_fingerprint = config_fingerprint();
        // Calling rather than reading a global keeps this clear of static
        // initialization order across modules.
        w.layout_fingerprint = library_layout_fingerprint();
        w.version_major      = EINSUMS_VERSION_MAJOR;
        w.version_minor      = EINSUMS_VERSION_MINOR;
        w.version_patch      = EINSUMS_VERSION_PATCH;
        w.version_string     = version.c_str();
        w.git_commit         = commit.c_str();
#if defined(__clang__)
        w.compiler_id    = "Clang";
        w.compiler_major = __clang_major__;
#elif defined(__GNUC__)
        w.compiler_id    = "GNU";
        w.compiler_major = __GNUC__;
#elif defined(_MSC_VER)
        w.compiler_id    = "MSVC";
        w.compiler_major = _MSC_VER / 100;
#else
        w.compiler_id    = "Unknown";
        w.compiler_major = 0;
#endif
        w.library_path = path.c_str();
        return w;
    }();
    return info;
}

bool register_stage_module(char const *name) {
    if (name == nullptr) {
        return false;
    }
    Registry              &r = registry();
    std::scoped_lock const guard(r.mutex);
    if (std::find(r.names.begin(), r.names.end(), name) != r.names.end()) {
        return false;
    }
    r.names.emplace_back(name);
    return true;
}

bool stage_module_registered(char const *name) {
    if (name == nullptr) {
        return false;
    }
    Registry              &r = registry();
    std::scoped_lock const guard(r.mutex);
    return std::find(r.names.begin(), r.names.end(), name) != r.names.end();
}

std::vector<std::string> registered_stage_modules() {
    Registry              &r = registry();
    std::scoped_lock const guard(r.mutex);
    return r.names;
}

std::vector<std::string> mapped_einsums_libraries() {
    std::vector<std::string> out;
#if defined(_WIN32)
    HMODULE modules[1024];
    DWORD   needed  = 0;
    HANDLE  process = GetCurrentProcess();
    if (EnumProcessModules(process, modules, static_cast<DWORD>(sizeof(modules)), &needed) != 0) {
        DWORD const count = std::min<DWORD>(needed / sizeof(HMODULE), static_cast<DWORD>(std::size(modules)));
        for (DWORD i = 0; i < count; ++i) {
            char  buffer[MAX_PATH] = {};
            DWORD n                = GetModuleFileNameA(modules[i], buffer, static_cast<DWORD>(sizeof(buffer)));
            if (n > 0 && n < sizeof(buffer) && is_einsums_library(std::string_view(buffer, n))) {
                out.emplace_back(buffer, n);
            }
        }
    }
#elif defined(__APPLE__)
    std::uint32_t const count = _dyld_image_count();
    for (std::uint32_t i = 0; i < count; ++i) {
        char const *name = _dyld_get_image_name(i);
        if (name != nullptr && is_einsums_library(name)) {
            out.emplace_back(name);
        }
    }
#else
    dl_iterate_phdr(&collect_elf_image, &out);
#endif
    // Two loads of one path are one library; two paths are two worlds.
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

EINSUMS_NAMESPACE_END(sealed)
