//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file
/// The handshake a compiled stage module performs at import.
///
/// `einsums.stages.load_stage_module` refuses a module that has not done it (see
/// `einsums.sealed.verify_stage_module`). Call the macro once in the `PYBIND11_MODULE` body:
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
/// Two things in the expansion are load-bearing:
///
/// - **The registration is compiled into the stage module**, so it lands in whichever libEinsums
///   the module reached. A helper inside libEinsums would always answer "same world".
/// - **The fingerprints come from the headers, not `world()`**, so they describe what this module
///   was compiled against. Read from `world()`, they would always match.

#pragma once

#include <Einsums/ComputeGraph/ABILayout.hpp>
#include <Einsums/Config/ABI.hpp>

#include <cstdint>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

/// The compiler of THIS translation unit, not `world().compiler_id`, which is the library's.
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
#define EINSUMS_STAGE_MODULE(m, name)                                                                                                        \
    do {                                                                                                                                     \
        ::einsums::sealed::register_stage_module(name);                                                                                      \
                                                                                                                                             \
        ::einsums::sealed::WorldInfo const &_einsums_w = ::einsums::sealed::world();                                                         \
                                                                                                                                             \
        (m).attr("__einsums_world__") = reinterpret_cast<std::uintptr_t>(_einsums_w.identity);                                               \
                                                                                                                                             \
        ::pybind11::dict _einsums_info;                                                                                                      \
        /* Computed HERE, from the headers this module sees, which is the whole  */                                                          \
        /* point: world() would report the library's own values and always agree. */                                                         \
        _einsums_info["config_fingerprint"] = ::einsums::sealed::config_fingerprint();                                                       \
        _einsums_info["layout_fingerprint"] = ::einsums::sealed::layout_fingerprint();                                                       \
        /* For the error message only, but header-derived like the fingerprints */                                                           \
        /* so a stale-headers refusal shows two different versions.             */                                                           \
        _einsums_info["version"]           = ::std::to_string(EINSUMS_VERSION_MAJOR) + "." + ::std::to_string(EINSUMS_VERSION_MINOR) + "." + \
                                             ::std::to_string(EINSUMS_VERSION_PATCH);                                                        \
        _einsums_info["compiler"]          = ::std::string(EINSUMS_DETAIL_STAGE_COMPILER);                                                   \
        _einsums_info["cplusplus"]         = static_cast<long>(__cplusplus);                                                                 \
        _einsums_info["library_path"]      = ::std::string(_einsums_w.library_path);                                                         \
        _einsums_info["module"]            = ::std::string(name);                                                                            \
        (m).attr("__einsums_world_info__") = _einsums_info;                                                                                  \
    } while (false)
