//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// One collector per process. Each copy looks, as it loads, for another copy already loaded; the
// later copy finds the earlier one, tells it, and switches itself off. A profiler problem never
// stops the program: the off copy's entry points just do nothing.

#include "Duplicates.hpp"

#include <Waggle/Config.hpp>

#include <cstdio>
#include <string>
#include <vector>

#include "Profiler.hpp"

#if defined(__APPLE__)
#    include <dlfcn.h>
#    include <mach-o/dyld.h>
#elif defined(__linux__)
#    include <dlfcn.h>
#    include <link.h>
#endif

extern "C" {
uint32_t const waggle_collector_marker_v1 = 0x57414747; // "WAGG"
}

WAGGLE_NAMESPACE_BEGIN

namespace {

#if defined(__APPLE__) || defined(__linux__)

/// An address in this copy's image, by a symbol that is never exported. Comparing images rather
/// than the marker's address matters: another copy's exported marker can interpose this one's,
/// so this copy's own reference to its marker may land in the other image.
void image_anchor() {
}

/// The image holding @p address: its base and its file.
struct Image {
    void const *base = nullptr;
    std::string path;
};

auto image_of(void const *address) -> Image {
    Dl_info info{};
    if (dladdr(address, &info) == 0) {
        return {};
    }
    return {.base = info.dli_fbase, .path = info.dli_fname != nullptr ? info.dli_fname : ""};
}

/// The file of every image loaded in the process; empty for the main program on Linux.
auto loaded_images() -> std::vector<std::string> {
    std::vector<std::string> out;
#    if defined(__APPLE__)
    uint32_t const count = _dyld_image_count();
    for (uint32_t i = 0; i < count; ++i) {
        if (char const *name = _dyld_get_image_name(i)) {
            out.emplace_back(name);
        }
    }
#    else
    dl_iterate_phdr(
        [](dl_phdr_info *info, size_t, void *data) -> int {
            static_cast<std::vector<std::string> *>(data)->emplace_back(info->dlpi_name != nullptr ? info->dlpi_name : "");
            return 0;
        },
        &out);
#    endif
    return out;
}

/// Look @p symbol up in the image at @p path without loading anything new. A handle, not a global
/// lookup, so a library Python loaded with RTLD_LOCAL is searched too.
auto find_symbol(std::string const &path, char const *symbol) -> void * {
    void *handle = dlopen(path.empty() ? nullptr : path.c_str(), RTLD_LAZY | RTLD_NOLOAD);
    if (handle == nullptr) {
        return nullptr;
    }
    void *found = dlsym(handle, symbol); // NOLINT(misc-const-correctness): callers cast it to a function pointer
    dlclose(handle);                     // NOLOAD only took a reference; the image stays loaded
    return found;
}

/// Find a collector loaded before this one. Returns its image, or an empty one.
auto find_other_collector() -> Image {
    Image const self = image_of(reinterpret_cast<void const *>(&image_anchor));
    if (self.base == nullptr) {
        return {};
    }
    for (auto const &path : loaded_images()) {
        // The lookup also searches the image's dependencies, so the marker it finds may live in
        // another image; whichever it is, it is a collector, and only this copy's own is not.
        void const *marker = find_symbol(path, "waggle_collector_marker_v1");
        if (marker == nullptr) {
            continue;
        }
        if (Image other = image_of(marker); other.base != nullptr && other.base != self.base) {
            return other;
        }
    }
    return {};
}

/// Switch off if another collector is loaded: warn, and tell the other one.
auto detect() -> bool {
    Image const other = find_other_collector();
    if (other.base == nullptr) {
        return false;
    }
    Image const self = image_of(reinterpret_cast<void const *>(&image_anchor));

    uint32_t other_major = 0;
    uint32_t other_minor = 0;
    if (auto *version = reinterpret_cast<void (*)(uint32_t *, uint32_t *)>(find_symbol(other.path, "waggle_abi_version"))) {
        version(&other_major, &other_minor);
    }
    std::fprintf(stderr,
                 "waggle: a second copy of the profiler, %s (interface %u.%u), was loaded beside %s (interface %u.%u); it is off, "
                 "and the zones of the libraries that use it are not recorded\n",
                 self.path.c_str(), WAGGLE_ABI_MAJOR, WAGGLE_ABI_MINOR, other.path.c_str(), other_major, other_minor);

    if (auto *note = reinterpret_cast<void (*)(char const *, uint32_t, uint32_t)>(find_symbol(other.path, "waggle_note_duplicate_v1"))) {
        note(self.path.c_str(), WAGGLE_ABI_MAJOR, WAGGLE_ABI_MINOR);
    }
    return true;
}

#else

// Windows lists its modules with EnumProcessModules and looks symbols up with GetProcAddress;
// that arrives with the rest of Windows support. Until then a second copy stays on.
auto detect() -> bool {
    return false;
}

#endif

/// Decided while the library loads: a namespace-scope initializer runs before any client can call
/// in, and nothing writes it afterwards.
bool const g_inactive = detect();

} // namespace

auto collector_inactive() noexcept -> bool {
    return g_inactive;
}

WAGGLE_NAMESPACE_END

extern "C" void waggle_note_duplicate_v1(char const *path, uint32_t major, uint32_t minor) {
    if (waggle::collector_inactive()) {
        return;
    }
    waggle::Profiler::instance().note_duplicate({.path = path != nullptr ? path : "", .abi_major = major, .abi_minor = minor});
}
