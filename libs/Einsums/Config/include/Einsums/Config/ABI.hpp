//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file
/// Identity of the libEinsums a piece of code is bound to.
///
/// A process may legitimately contain more than one copy of libEinsums: a host
/// application that privately links one version, and a developer's newer one
/// alongside it. That is supported, and the rule that makes it safe is that
/// typed Einsums objects never cross between the two; only neutral buffers do.
///
/// What is NOT safe is a compiled extension module that believes it is talking
/// to the same libEinsums as `einsums._core` when it is not. Shared-graph
/// capture, for one, keeps its capture context in thread-local state inside the
/// library, so two copies means two capture contexts and a graph that silently
/// loses half its nodes. This header is how that is detected instead of
/// debugged.
///
/// Three questions, three answers:
///
/// - "Are we bound to the same library?" `world().identity`, plus
///   register_stage_module(), which answers it without trusting a pointer to
///   have survived whatever the loader did.
/// - "Were we built against matching headers?" The fingerprints.
/// - "How many copies are actually in this process?" mapped_einsums_libraries().
///
/// The namespace is `sealed`, after the sealed-worlds rule above, and NOT `abi`:
/// `<cxxabi.h>` declares `namespace abi = __cxxabiv1;` at global scope, so
/// `einsums::abi` is ambiguous in any translation unit that says
/// `using namespace einsums;` and reaches the C++ ABI header, which every test
/// in this tree does.

#pragma once

#include <Einsums/Config/Defines.hpp>
#include <Einsums/Config/ExportDefinitions.hpp>
#include <Einsums/Config/Version.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace einsums::sealed {

/// Everything one copy of libEinsums knows about itself.
///
/// Compared field by field when a stage module registers, so a mismatch can say
/// which axis diverged rather than only that something did. Extended by
/// appending fields and bumping nothing: `struct_size` lets a reader built
/// against older headers recognize that it is looking at a longer record.
struct WorldInfo {
    /// Size of this struct as the *writer* understood it.
    std::size_t struct_size;

    /// Address of a function-local static inside the library. Equal for two
    /// callers exactly when they resolved to the same copy.
    void const *identity;

    /// Fold of the build toggles this library was compiled with. See
    /// config_fingerprint().
    std::uint64_t config_fingerprint;

    /// Fold of the sizes and alignments of the types that cross the boundary.
    /// See layout_fingerprint(), which lives higher up the module graph.
    std::uint64_t layout_fingerprint;

    int version_major;
    int version_minor;
    int version_patch;

    /// Version, git commit, and compiler, as NUL-terminated literals owned by
    /// the library. Valid for the process lifetime.
    char const *version_string;
    char const *git_commit;
    char const *compiler_id;
    int         compiler_major;

    /// Filesystem path of the library this record came from, or "" when the
    /// platform cannot answer. The single most useful field in a mismatch
    /// report: two paths beat two hex numbers.
    char const *library_path;
};

/// The calling code's libEinsums.
///
/// Deliberately out of line and never inline: an inline definition would put a
/// copy of the static in every module that included this header, and the
/// identity would then say "different" for two callers that share a library.
[[nodiscard]] EINSUMS_EXPORT WorldInfo const &world() noexcept;

/// Record that a stage module bound to *this* library, and answer whether it
/// had already.
///
/// The detection trick is the side effect rather than the return: a module that
/// resolved to a DIFFERENT copy of libEinsums writes into that copy's table, so
/// asking this library afterwards reports nothing. That works even when the
/// module never sets a Python attribute, and it does not depend on a pointer
/// comparison surviving the loader.
EINSUMS_EXPORT bool register_stage_module(char const *name);

/// Whether register_stage_module() was called on THIS library for @p name.
[[nodiscard]] EINSUMS_EXPORT bool stage_module_registered(char const *name);

/// Names passed to register_stage_module() on this library, in call order.
[[nodiscard]] EINSUMS_EXPORT std::vector<std::string> registered_stage_modules();

/// Paths of every libEinsums currently mapped into this process.
///
/// More than one entry is the multi-world signal, and it is the first thing
/// worth printing when a handshake fails or a process dies inside a stage
/// module. Returns an empty vector where the platform offers no way to
/// enumerate loaded images.
[[nodiscard]] EINSUMS_EXPORT std::vector<std::string> mapped_einsums_libraries();

namespace detail {

/// FNV-1a. Not a security hash; it just has to differ when the inputs differ
/// and be computable at compile time.
constexpr std::uint64_t fnv1a(char const *s, std::uint64_t h = 14695981039346656037ULL) noexcept {
    for (; s != nullptr && *s != '\0'; ++s) {
        h = (h ^ static_cast<std::uint64_t>(static_cast<unsigned char>(*s))) * 1099511628211ULL;
    }
    return h;
}

constexpr std::uint64_t fnv1a_value(std::uint64_t v, std::uint64_t h) noexcept {
    for (int i = 0; i < 8; ++i) {
        h = (h ^ ((v >> (i * 8)) & 0xFFULL)) * 1099511628211ULL;
    }
    return h;
}

} // namespace detail

/// Fold of every build setting reachable from <Einsums/Config.hpp>.
///
/// Computed in the header, so a stage module computes it from the headers it
/// was compiled against while the library carries the value it saw when IT was
/// compiled. A difference means the two disagree about how the library was
/// built, which is the stale-headers case and by far the likeliest way an
/// out-of-tree build goes wrong.
///
/// Only what Config can see. The sizes of the types that actually cross the
/// boundary live in layout_fingerprint(), higher up the module graph, because
/// Config sits below every module that declares one.
[[nodiscard]] constexpr std::uint64_t config_fingerprint() noexcept {
    std::uint64_t h = detail::fnv1a("einsums.abi.config.1");

    // Version. A patch bump is not an ABI break, but it is worth reporting.
    h = detail::fnv1a_value(EINSUMS_VERSION_FULL, h);

    // Language level and pointer width: both change layouts wholesale.
    h = detail::fnv1a_value(static_cast<std::uint64_t>(__cplusplus), h);
    h = detail::fnv1a_value(sizeof(void *), h);
    h = detail::fnv1a_value(sizeof(long), h);

    // Build toggles. Anything that changes a member, a base, or a code path
    // reachable across the boundary belongs here.
#if defined(EINSUMS_HAVE_PROFILER)
    h = detail::fnv1a("PROFILER", h);
#endif
#if defined(EINSUMS_HAVE_BACKTRACES)
    h = detail::fnv1a("BACKTRACES", h);
#endif
#if defined(EINSUMS_HAVE_MPS)
    h = detail::fnv1a("MPS", h);
#endif
#if defined(EINSUMS_HAVE_MALLOC)
    h = detail::fnv1a(EINSUMS_HAVE_MALLOC, h);
#endif
#if defined(EINSUMS_HAVE_CXX17_ALIGNED_NEW)
    h = detail::fnv1a("CXX17_ALIGNED_NEW", h);
#endif
#if defined(EINSUMS_HAVE_CXX20_NO_UNIQUE_ADDRESS_ATTRIBUTE)
    h = detail::fnv1a("NO_UNIQUE_ADDRESS", h);
#endif
#if defined(EINSUMS_HAVE_CXX23_STATIC_CALL_OPERATOR)
    h = detail::fnv1a("STATIC_CALL_OPERATOR", h);
#endif
#if defined(EINSUMS_HAVE_ELF_HIDDEN_VISIBILITY)
    h = detail::fnv1a("ELF_HIDDEN_VISIBILITY", h);
#endif
    return h;
}

/// The layout fingerprint of the library, as opposed to of the caller's
/// headers.
///
/// Declared here and DEFINED in a translation unit high enough in the module
/// graph to see the types it measures. Config cannot include Tensor or
/// ComputeGraph, so the value has to arrive from above; world() calls this
/// lazily rather than reading a global, which keeps it clear of static
/// initialization order.
[[nodiscard]] EINSUMS_EXPORT std::uint64_t library_layout_fingerprint() noexcept;

} // namespace einsums::sealed
