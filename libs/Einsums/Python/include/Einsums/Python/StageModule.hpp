//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file
/// The handshake a compiled stage module performs at import.
///
/// `einsums.stages.load_stage_module` refuses to read anything out of a module
/// that has not done this, and `einsums.sealed.verify_stage_module` is what
/// does the refusing. This header is the other end of that conversation: one
/// macro, called once in a stage module's `PYBIND11_MODULE` body.
///
/// @code
/// #include <Einsums/Python/StageModule.hpp>
///
/// PYBIND11_MODULE(mymethod_stages, m) {
///     EINSUMS_STAGE_MODULE(m, "mymethod_stages");
///     m.def("stage_pno_transform", &mymethod::pno_transform);
/// }
/// @endcode
///
/// Two things about the expansion are load-bearing, and both are easy to
/// "simplify" into uselessness.
///
/// **The registration call is compiled into the stage module, not into
/// libEinsums.** That is the entire cross-world detection mechanism: the call
/// resolves through the module's own symbol lookup and therefore lands in
/// whichever copy of libEinsums the module actually reached. Ask a different
/// copy afterwards and it reports nothing. A convenience helper inside
/// libEinsums that registered on the module's behalf would always reach the
/// library asking the question, and would answer "same world" every time.
///
/// **The fingerprints come from the headers, not from `world()`.** They are
/// `constexpr` functions, so writing them here evaluates them against the
/// headers this module is being compiled with, which is exactly the quantity a
/// stale-headers check needs. Reading them from the runtime `world()` record
/// would make every module agree with every library by construction, which is a
/// guard that passes for the wrong reason.

#pragma once

#include <Einsums/ComputeGraph/ABILayout.hpp>
#include <Einsums/Config/ABI.hpp>

#include <cstdint>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

/// The compiler compiling THIS translation unit, which is the useful one in a
/// mismatch report. Deliberately not `world().compiler_id`, which is the
/// library's and would agree with itself.
#define EINSUMS_DETAIL_STAGE_STR2(x) #x
#define EINSUMS_DETAIL_STAGE_STR(x)  EINSUMS_DETAIL_STAGE_STR2(x)

#if defined(__VERSION__)
#    define EINSUMS_DETAIL_STAGE_COMPILER __VERSION__
#elif defined(_MSC_VER)
#    define EINSUMS_DETAIL_STAGE_COMPILER "MSVC " EINSUMS_DETAIL_STAGE_STR(_MSC_VER)
#else
#    define EINSUMS_DETAIL_STAGE_COMPILER "unknown"
#endif

/// Perform the stage-module handshake on pybind11 module @p m under @p name.
///
/// @p name must be the module's importable name, since that is the key
/// `einsums.sealed` looks up. Call once, first, in the module body.
#define EINSUMS_STAGE_MODULE(m, name)                                                                                                      \
    do {                                                                                                                                   \
        ::einsums::sealed::register_stage_module(name);                                                                                    \
                                                                                                                                           \
        ::einsums::sealed::WorldInfo const &_einsums_w = ::einsums::sealed::world();                                                       \
                                                                                                                                           \
        (m).attr("__einsums_world__") = reinterpret_cast<std::uintptr_t>(_einsums_w.identity);                                             \
                                                                                                                                           \
        ::pybind11::dict _einsums_info;                                                                                                    \
        /* Computed HERE, from the headers this module sees, which is the whole  */                                                        \
        /* point: world() would report the library's own values and always agree. */                                                       \
        _einsums_info["config_fingerprint"] = ::einsums::sealed::config_fingerprint();                                                     \
        _einsums_info["layout_fingerprint"] = ::einsums::sealed::layout_fingerprint();                                                     \
        /* Descriptive, for the error message rather than the decision - but    */                                                         \
        /* header-derived for the same reason the fingerprints are. Reporting   */                                                         \
        /* the library's own version back at it would print the same string on  */                                                         \
        /* both lines of a stale-headers refusal, which is the one place the    */                                                         \
        /* reader most needs them to differ.                                    */                                                         \
        _einsums_info["version"] = ::std::to_string(EINSUMS_VERSION_MAJOR) + "." + ::std::to_string(EINSUMS_VERSION_MINOR) + "." +         \
                                   ::std::to_string(EINSUMS_VERSION_PATCH);                                                                \
        _einsums_info["compiler"]          = ::std::string(EINSUMS_DETAIL_STAGE_COMPILER);                                                 \
        _einsums_info["cplusplus"]         = static_cast<long>(__cplusplus);                                                               \
        _einsums_info["library_path"]      = ::std::string(_einsums_w.library_path);                                                       \
        _einsums_info["module"]            = ::std::string(name);                                                                          \
        (m).attr("__einsums_world_info__") = _einsums_info;                                                                                \
    } while (false)
