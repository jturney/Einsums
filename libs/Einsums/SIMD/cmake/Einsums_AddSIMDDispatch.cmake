#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

#:
#: .. cmake:command:: einsums_add_simd_dispatch_sources
#:
#:    Generate per-instruction-set translation units for a runtime dispatch
#:    ladder (see ``Einsums/SIMD/RuntimeFeatures.hpp``).
#:
#:    .. code-block:: cmake
#:
#:       einsums_add_simd_dispatch_sources(<out_var>
#:         IMPL <impl-file>
#:         [RUNGS <rung>...]        # subset of: baseline v2 v3 v4 (default: all)
#:       )
#:
#:    For each rung, a thin wrapper ``.cpp`` is generated into the current
#:    binary directory that
#:
#:    1. defines ``EINSUMS_SIMD_ARCH_NS`` to ``arch_<rung>`` (the namespace
#:       the implementation file must wrap its arch-dependent code in),
#:    2. defines ``EINSUMS_SIMD_DISPATCH_RUNG`` to the rung's ordinal
#:       (0 = baseline ... 3 = v4), and
#:    3. includes the implementation file,
#:
#:    and is given the compiler flags of that rung (``-march=x86-64-v2/-v3/-v4``
#:    for GCC/Clang, ``/arch:AVX2``/``/arch:AVX512`` for the MSVC driver).
#:    Because the SIMD headers key off compiler-defined feature macros, the
#:    same implementation source widens ``Vec<T>``/``native_lanes``/all
#:    operations to each rung's register width without source changes.
#:
#:    The generated source list is returned in ``<out_var>`` for passing to
#:    ``einsums_add_module``'s ``SOURCES``. The rungs actually generated are
#:    returned in ``<out_var>_RUNGS``, and matching compile definitions of
#:    the form ``EINSUMS_SIMD_HAS_RUNG_<RUNG>=1`` are returned in
#:    ``<out_var>_DEFINITIONS`` so dispatch-table code can declare exactly
#:    the namespaces that exist (add them to the consuming target with
#:    ``target_compile_definitions``).
#:
#:    Single-TU mode: the requested ladder collapses to one ``native`` rung
#:    (namespace ``arch_native``, ambient compiler flags, definition
#:    ``EINSUMS_SIMD_HAS_RUNG_NATIVE``) when any of the following holds,
#:    because a ladder would be meaningless or unreachable:
#:
#:    * ``EINSUMS_WITH_SIMD_DISPATCH`` is OFF,
#:    * the target processor is not x86-64 (the v2/v3/v4 rungs are x86
#:      levels; on aarch64 the toolchain baseline already includes NEON),
#:    * ``EINSUMS_SIMD_NATIVE_ARCH`` or ``EINSUMS_SIMD_TARGET_CPU`` pins the
#:      whole SIMD interface to a specific CPU (the pin raises every TU's
#:      baseline, so a runtime ladder below it can never be selected).
#:
#:    Independent of that, individual rungs are dropped (degrading toward
#:    the always-present baseline) when their flag is unusable: the true
#:    MSVC driver has no flag for ``v2`` (``v3``/``v4`` map to
#:    ``/arch:AVX2`` and ``/arch:AVX512``; clang-cl reaches ``v2`` via
#:    ``/clang:-march=x86-64-v2`` and Intel icx accepts the GCC spelling),
#:    and every rung flag is probed with ``check_cxx_compiler_flag`` first,
#:    so compilers predating ``-march=x86-64-vN`` (GCC < 11, Clang < 12)
#:    build baseline-only instead of failing to configure.
#:
#:    The implementation file is compiled once per surviving rung, so keep
#:    everything arch-independent out of it; heavy shared code belongs in a
#:    regular TU.
include(CheckCXXCompilerFlag)

function(einsums_add_simd_dispatch_sources out_var)
  set(options)
  set(one_value_args IMPL)
  set(multi_value_args RUNGS)
  cmake_parse_arguments(_simd "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

  if(NOT _simd_IMPL)
    message(FATAL_ERROR "einsums_add_simd_dispatch_sources: IMPL is required")
  endif()
  if(NOT _simd_RUNGS)
    set(_simd_RUNGS baseline v2 v3 v4)
  endif()

  get_filename_component(_impl_abs "${_simd_IMPL}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
  if(NOT EXISTS "${_impl_abs}")
    message(FATAL_ERROR "einsums_add_simd_dispatch_sources: IMPL file not found: ${_impl_abs}")
  endif()
  get_filename_component(_impl_name "${_impl_abs}" NAME_WE)

  # Decide whether the x86 ladder applies at all.
  set(_is_x86 FALSE)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(_is_x86 TRUE)
  endif()
  set(_pinned FALSE)
  if(EINSUMS_SIMD_NATIVE_ARCH OR NOT "${EINSUMS_SIMD_TARGET_CPU}" STREQUAL "")
    set(_pinned TRUE)
  endif()
  # Single-TU mode: no ladder. The wrapper compiles at the ambient flags in
  # the arch_native namespace, and consumers get EINSUMS_SIMD_HAS_RUNG_NATIVE
  # instead of the per-rung definitions.
  if(NOT _is_x86
     OR _pinned
     OR NOT EINSUMS_WITH_SIMD_DISPATCH
  )
    set(_simd_RUNGS native)
  endif()

  # True MSVC driver (cl.exe): no flag exists for the v2 level. clang-cl
  # takes the GCC spelling through the /clang: escape hatch, and Intel's
  # icx-cl (IntelLLVM) accepts the GCC spelling directly.
  set(_msvc_true_driver FALSE)
  if(MSVC AND NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang|IntelLLVM")
    set(_msvc_true_driver TRUE)
  endif()

  set(_sources)
  set(_definitions)
  set(_generated_rungs)
  foreach(_rung IN LISTS _simd_RUNGS)
    # Per-rung ordinal and compiler flags.
    if(_rung STREQUAL "native")
      set(_ordinal 0)
      set(_flags "")
    elseif(_rung STREQUAL "baseline")
      set(_ordinal 0)
      set(_flags "")
    elseif(_rung STREQUAL "v2")
      set(_ordinal 1)
      if(_msvc_true_driver)
        message(STATUS "einsums_add_simd_dispatch_sources(${_impl_name}): MSVC cl has no x86-64-v2 flag; dropping the v2 rung")
        continue()
      elseif(MSVC AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang") # clang-cl
        set(_flags "/clang:-march=x86-64-v2")
      else() # GCC/Clang/IntelLLVM (icx accepts the GCC spelling on Windows too)
        set(_flags "-march=x86-64-v2")
      endif()
    elseif(_rung STREQUAL "v3")
      set(_ordinal 2)
      if(MSVC)
        set(_flags "/arch:AVX2")
      else()
        set(_flags "-march=x86-64-v3")
      endif()
    elseif(_rung STREQUAL "v4")
      set(_ordinal 3)
      if(MSVC)
        set(_flags "/arch:AVX512")
      else()
        set(_flags "-march=x86-64-v4")
      endif()
    else()
      message(FATAL_ERROR "einsums_add_simd_dispatch_sources: unknown rung '${_rung}' (expected baseline/v2/v3/v4)")
    endif()

    set(_wrapper "${CMAKE_CURRENT_BINARY_DIR}/simd_dispatch/${_impl_name}_${_rung}.cpp")
    file(
      CONFIGURE
      OUTPUT "${_wrapper}"
      CONTENT
        "// Generated by einsums_add_simd_dispatch_sources - do not edit.
#define EINSUMS_SIMD_ARCH_NS arch_${_rung}
#define EINSUMS_SIMD_DISPATCH_RUNG ${_ordinal}
#include \"${_impl_abs}\"
"
      @ONLY
      NEWLINE_STYLE UNIX
    )

    # Old compilers (pre -march=x86-64-vN: GCC < 11, Clang < 12) or drivers
    # that reject a spelling drop the rung instead of breaking the build -
    # the ladder degrades toward baseline, which always exists.
    if(NOT "${_flags}" STREQUAL "")
      string(TOUPPER "${_rung}" _rung_upper)
      check_cxx_compiler_flag("${_flags}" EINSUMS_SIMD_RUNG_FLAG_${_rung_upper})
      if(NOT EINSUMS_SIMD_RUNG_FLAG_${_rung_upper})
        message(STATUS "einsums_add_simd_dispatch_sources(${_impl_name}): compiler rejects '${_flags}'; dropping the ${_rung} rung")
        continue()
      endif()
      set_source_files_properties("${_wrapper}" PROPERTIES COMPILE_OPTIONS "${_flags}")
    endif()

    list(APPEND _sources "${_wrapper}")
    string(TOUPPER "${_rung}" _rung_upper)
    list(APPEND _definitions "EINSUMS_SIMD_HAS_RUNG_${_rung_upper}=1")
    list(APPEND _generated_rungs "${_rung}")
  endforeach()

  set(${out_var}
      "${_sources}"
      PARENT_SCOPE
  )
  set(${out_var}_RUNGS
      "${_generated_rungs}"
      PARENT_SCOPE
  )
  set(${out_var}_DEFINITIONS
      "${_definitions}"
      PARENT_SCOPE
  )
endfunction()

#:
#: .. cmake:command:: einsums_add_simd_rung_tests
#:
#:    Re-register an existing ``<name>_test`` executable once per x86
#:    dispatch rung, forcing the rung via the ``EINSUMS_SIMD_ARCH``
#:    environment variable:
#:
#:    .. code-block:: cmake
#:
#:       einsums_add_simd_rung_tests("Modules.HPTT" LargeTranspose)
#:
#:    creates ``Tests.Unit.<subcategory>.<name>.simd_baseline`` / ``.simd_v2``
#:    / ``.simd_v3`` / ``.simd_v4``. Each test runs through the
#:    ``simd_rung_guard`` launcher, which exits with the registered
#:    ``SKIP_RETURN_CODE`` (77) when the host CPU cannot execute the rung -
#:    ctest then reports the test as Skipped rather than silently passing at
#:    a clamped lower rung. No-op when the build is single-TU (dispatch OFF,
#:    non-x86, or a compile-time CPU pin), where only arch_native exists.
function(einsums_add_simd_rung_tests subcategory name)
  if(NOT EINSUMS_WITH_SIMD_DISPATCH
     OR NOT CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64"
     OR EINSUMS_SIMD_NATIVE_ARCH
     OR NOT "${EINSUMS_SIMD_TARGET_CPU}" STREQUAL ""
  )
    return()
  endif()
  foreach(_rung baseline v2 v3 v4)
    set(_test_name "Tests.Unit.${subcategory}.${name}.simd_${_rung}")
    # simd_rung_guard exits 77 (SKIP_RETURN_CODE) when the host CPU cannot
    # execute the rung, so ctest reports "Skipped" instead of silently
    # rerunning at the clamped lower rung.
    add_test(NAME ${_test_name}
             COMMAND simd_rung_guard ${_rung} $<TARGET_FILE:${name}_test> "--einsums:debug:no-install-signal-handlers"
                     "--einsums:debug:no-attach-debugger" "--einsums:profile:no-report"
    )
    set_tests_properties(
      ${_test_name} PROPERTIES ENVIRONMENT "EINSUMS_SIMD_ARCH=${_rung}" LABELS "UNIT_ONLY" SKIP_RETURN_CODE 77
    )
  endforeach()
endfunction()
