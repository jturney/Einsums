//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file
/// Identity of the libEinsums a piece of code is bound to.
///
/// A process may hold several copies of libEinsums, provided typed objects never cross between them.
/// An extension module that wrongly believes it shares `einsums._core`'s copy is the danger (e.g. a
/// capture context per copy loses nodes); this header detects it.
///
///
/// - "Are we bound to the same library?" `world().identity`, plus
///   register_stage_module(), which answers it without trusting a pointer to
///   have survived whatever the loader did.
/// - "Were we built against matching headers?" The fingerprints.
/// - "How many copies are actually in this process?" mapped_einsums_libraries().
///
/// The namespace is `sealed`, not `abi`, which `<cxxabi.h>` already declares at global scope.

#pragma once

#include <Einsums/Config/Defines.hpp>
#include <Einsums/Config/ExportDefinitions.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Config/Version.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(sealed)

/// Everything one copy of libEinsums knows about itself.
///
/// Compared field by field, so a mismatch names the field. Extend by appending; `struct_size` tells
/// older readers the record is longer.
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

    /// Path of the library this record came from, or "" when the platform cannot answer.
    char const *library_path;
};

/// The calling code's libEinsums.
///
/// Out of line, so every caller of one library sees one static.
[[nodiscard]] EINSUMS_EXPORT WorldInfo const &world() noexcept;

/// Record that a stage module bound to *this* library, and answer whether it
/// had already.
///
/// The side effect is the check: a module bound to another copy registers in that copy's table.
EINSUMS_EXPORT bool register_stage_module(char const *name);

/// Whether register_stage_module() was called on THIS library for @p name.
[[nodiscard]] EINSUMS_EXPORT bool stage_module_registered(char const *name);

/// Names passed to register_stage_module() on this library, in call order.
[[nodiscard]] EINSUMS_EXPORT std::vector<std::string> registered_stage_modules();

/// Paths of every libEinsums currently mapped into this process.
///
/// More than one means several worlds. Empty where loaded images cannot be enumerated.
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
/// Computed in the header, so a stage module and the library each fold what they were compiled
/// with; a difference means stale headers. Type layouts are in layout_fingerprint(), higher up.
[[nodiscard]] constexpr std::uint64_t config_fingerprint() noexcept {
    std::uint64_t h = detail::fnv1a("einsums.abi.config.1");

    // The ABI generation first: a mismatch there is the answer, not a clue.
    h = detail::fnv1a_value(EINSUMS_ABI_VERSION, h);

    // Version. A patch bump is not an ABI break, but it is worth reporting.
    h = detail::fnv1a_value(EINSUMS_VERSION_FULL, h);

    // Language level and pointer width: both change layouts wholesale.
    h = detail::fnv1a_value(static_cast<std::uint64_t>(__cplusplus), h);
    h = detail::fnv1a_value(sizeof(void *), h);
    h = detail::fnv1a_value(sizeof(long), h);

    // Standard-library modes that change type layouts, and only those: _ITERATOR_DEBUG_LEVEL (MSVC's
    // /MDd vs /MD runtimes) and _GLIBCXX_DEBUG. Not NDEBUG or checking-only modes, which change no
    // layout and would refuse ordinary consumers.
#if defined(_ITERATOR_DEBUG_LEVEL)
    h = detail::fnv1a_value(_ITERATOR_DEBUG_LEVEL, h);
#endif
#if defined(_GLIBCXX_DEBUG)
    h = detail::fnv1a("GLIBCXX_DEBUG", h);
#endif

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
/// Defined higher in the module graph, where the measured types are visible. Called lazily by world().
[[nodiscard]] EINSUMS_EXPORT std::uint64_t library_layout_fingerprint() noexcept;

EINSUMS_NAMESPACE_END(sealed)
