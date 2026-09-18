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

#:
#: .. cmake:command:: einsums_simd_rung_flags
#:
#:    Resolve one dispatch rung to the flags that raise a translation unit to
#:    it, its ladder ordinal, and whether this toolchain can reach it at all.
#:
#:    .. code-block:: cmake
#:
#:       einsums_simd_rung_flags(<rung> <out_flags> <out_ordinal> <out_ok> <context>)
#:
#:    ``<out_ok>`` is FALSE when the driver has no spelling for the rung (the
#:    true MSVC driver has none for ``v2`` or ``sme``) or when it rejects the
#:    spelling it does have (compilers predating ``-march=x86-64-vN``). The
#:    caller drops that rung rather than failing to configure. ``<context>``
#:    only prefixes the diagnostic.
#:
#:    Both the per-rung kernel TUs and the per-rung compiled tests resolve
#:    flags through here, so a new rung or a new driver spelling is added in
#:    one place.
function(einsums_simd_rung_flags rung out_flags out_ordinal out_ok context)
  # True MSVC driver (cl.exe): no flag exists for the v2 level. clang-cl
  # takes the GCC spelling through the /clang: escape hatch, and Intel's
  # icx-cl (IntelLLVM) accepts the GCC spelling directly.
  set(_msvc_true_driver FALSE)
  if(MSVC AND NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang|IntelLLVM")
    set(_msvc_true_driver TRUE)
  endif()

  set(_flags "")
  set(_ordinal 0)
  set(_ok TRUE)

  if(rung STREQUAL "native" OR rung STREQUAL "baseline")
    set(_ordinal 0)
  elseif(rung STREQUAL "v2")
    set(_ordinal 1)
    if(_msvc_true_driver)
      message(STATUS "${context}: MSVC cl has no x86-64-v2 flag; dropping the v2 rung")
      set(_ok FALSE)
    elseif(MSVC AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang") # clang-cl
      set(_flags "/clang:-march=x86-64-v2")
    else() # GCC/Clang/IntelLLVM (icx accepts the GCC spelling on Windows too)
      set(_flags "-march=x86-64-v2")
    endif()
  elseif(rung STREQUAL "v3")
    set(_ordinal 2)
    if(MSVC)
      set(_flags "/arch:AVX2")
    else()
      set(_flags "-march=x86-64-v3")
    endif()
  elseif(rung STREQUAL "v4")
    set(_ordinal 3)
    if(MSVC)
      set(_flags "/arch:AVX512")
    else()
      set(_flags "-march=x86-64-v4")
    endif()
  elseif(rung STREQUAL "sme")
    set(_ordinal 4)
    if(_msvc_true_driver)
      message(STATUS "${context}: MSVC cl has no SME flag; dropping the sme rung")
      set(_ok FALSE)
    elseif(MSVC AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang") # clang-cl
      set(_flags "/clang:-march=armv8.6-a+sme2+sme-f64f64")
    else() # GCC/Clang/AppleClang
      set(_flags "-march=armv8.6-a+sme2+sme-f64f64")
    endif()
  else()
    message(FATAL_ERROR "einsums_simd_rung_flags: unknown rung '${rung}' (expected baseline/native/v2/v3/v4/sme)")
  endif()

  # Old compilers (pre -march=x86-64-vN: GCC < 11, Clang < 12) or drivers
  # that reject a spelling drop the rung instead of breaking the build -
  # the ladder degrades toward baseline, which always exists.
  if(_ok AND NOT "${_flags}" STREQUAL "")
    string(TOUPPER "${rung}" _rung_upper)
    check_cxx_compiler_flag("${_flags}" EINSUMS_SIMD_RUNG_FLAG_${_rung_upper})
    if(NOT EINSUMS_SIMD_RUNG_FLAG_${_rung_upper})
      message(STATUS "${context}: compiler rejects '${_flags}'; dropping the ${rung} rung")
      set(_ok FALSE)
    endif()
  endif()

  set(${out_flags} "${_flags}" PARENT_SCOPE)
  set(${out_ordinal} "${_ordinal}" PARENT_SCOPE)
  set(${out_ok} "${_ok}" PARENT_SCOPE)
endfunction()

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

  # Decide which architecture's ladder applies.
  set(_is_x86 FALSE)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(_is_x86 TRUE)
  endif()
  set(_is_aarch64 FALSE)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
    set(_is_aarch64 TRUE)
  endif()
  set(_pinned FALSE)
  if(EINSUMS_SIMD_NATIVE_ARCH OR NOT "${EINSUMS_SIMD_TARGET_CPU}" STREQUAL "")
    set(_pinned TRUE)
  endif()
  # Single-TU mode: no ladder. The wrapper compiles at the ambient flags in
  # the arch_native namespace, and consumers get EINSUMS_SIMD_HAS_RUNG_NATIVE
  # instead of the per-rung definitions.
  #
  # On aarch64 the x86 rungs (baseline/v2/v3/v4) collapse to `native`, but a
  # requested `sme` rung survives alongside it: NEON is the toolchain
  # baseline (native), and SME2 is the one optional aarch64 rung.
  if(_pinned OR NOT EINSUMS_WITH_SIMD_DISPATCH)
    set(_simd_RUNGS native)
  elseif(_is_aarch64)
    set(_arm_rungs native)
    if("sme" IN_LIST _simd_RUNGS)
      list(APPEND _arm_rungs sme)
    endif()
    set(_simd_RUNGS ${_arm_rungs})
  elseif(_is_x86)
    list(REMOVE_ITEM _simd_RUNGS sme)
    if(NOT _simd_RUNGS)
      set(_simd_RUNGS native)
    endif()
  else()
    set(_simd_RUNGS native)
  endif()

  set(_sources)
  set(_definitions)
  set(_generated_rungs)
  foreach(_rung IN LISTS _simd_RUNGS)
    # Declared before the call: einsums_simd_rung_flags returns through
    # PARENT_SCOPE, so without these the names would carry the previous
    # iteration's values if it ever returned without setting them.
    set(_flags "")
    set(_ordinal 0)
    set(_rung_ok FALSE)
    einsums_simd_rung_flags(${_rung} _flags _ordinal _rung_ok "einsums_add_simd_dispatch_sources(${_impl_name})")
    if(NOT _rung_ok)
      continue()
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

    # einsums_simd_rung_flags already probed the spelling, so anything that
    # reaches here is a flag this driver accepts.
    if(NOT "${_flags}" STREQUAL "")
      set_source_files_properties("${_wrapper}" PROPERTIES COMPILE_OPTIONS "${_flags}")
    endif()

    # A rung TU is compiled at its own -march, so it can never share the
    # project's baseline precompiled header: GCC rejects it with
    #   warning: ... .gch: created and used with differing settings of '-march='
    # and the TU re-parses the headers anyway. Opting out makes that explicit
    # and silences a warning that is inherent rather than fixable.
    set_source_files_properties("${_wrapper}" PROPERTIES SKIP_PRECOMPILE_HEADERS ON)

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
     OR EINSUMS_SIMD_NATIVE_ARCH
     OR NOT "${EINSUMS_SIMD_TARGET_CPU}" STREQUAL ""
  )
    return()
  endif()
  # Per-arch rung list: x86 gets the psABI ladder, aarch64 gets the sme rung
  # (the guard skips it with SKIP_RETURN_CODE 77 on non-SME hardware).
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(_guard_rungs baseline v2 v3 v4)
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
    set(_guard_rungs baseline sme)
  else()
    return()
  endif()
  foreach(_rung IN LISTS _guard_rungs)
    set(_test_name "Tests.Unit.${subcategory}.${name}.simd_${_rung}")
    # simd_rung_guard exits 77 (SKIP_RETURN_CODE) when the host CPU cannot
    # execute the rung, so ctest reports "Skipped" instead of silently
    # rerunning at the clamped lower rung.
    add_test(NAME ${_test_name}
             COMMAND simd_rung_guard ${_rung} $<TARGET_FILE:${name}_test> "--einsums:debug:no-install-signal-handlers"
                     "--einsums:debug:no-attach-debugger" "--einsums:profile:no-report"
    )
    # Reuse the standard per-test ENVIRONMENT (LLVM_PROFILE_FILE plus the
    # TSAN/LSAN suppression-file paths) so a rung-pinned test is protected by the
    # same suppressions as its base registration, then APPEND the rung override.
    # ENVIRONMENT clobbers rather than merges: setting it to a bare
    # EINSUMS_SIMD_ARCH=${_rung} here would erase the suppression paths, so every
    # rung test on the sanitizer legs would fail on the benign libomp/HPTT/spdlog
    # false positives that tsan.supp exists to silence.
    einsums_set_test_properties(${_test_name} "UNIT_ONLY")
    set_property(
      TEST ${_test_name}
      APPEND
      PROPERTY ENVIRONMENT "EINSUMS_SIMD_ARCH=${_rung}"
    )
    set_tests_properties(${_test_name} PROPERTIES SKIP_RETURN_CODE 77)
  endforeach()
endfunction()

#:
#: .. cmake:command:: einsums_add_simd_rung_compiled_tests
#:
#:    Build an extra copy of a test per dispatch rung, each compiled AT that
#:    rung, and register it under the same ``simd_rung_guard`` launcher:
#:
#:    .. code-block:: cmake
#:
#:       einsums_add_simd_rung_compiled_tests("Modules.SIMD" Shuffle)
#:
#:    This is the counterpart to ``einsums_add_simd_rung_tests`` for code that
#:    resolves at COMPILE time rather than through a runtime dispatch table.
#:    ``EINSUMS_SIMD_ARCH`` moves the rung a dispatch table selects, so it
#:    covers kernels whose rungs were compiled into the library. It cannot
#:    reach an inline function in a header, which is frozen at whatever width
#:    its own translation unit was compiled for: a test that calls
#:    ``einsums::simd`` directly, built once at the ambient flags, exercises
#:    one rung no matter what the environment says and no matter what the
#:    host CPU supports.
#:
#:    That gap is not hypothetical. It is why a wrong AVX-512 ``Vec<double>``
#:    transpose survived in ``Shuffle.hpp`` with a correctness test sitting
#:    directly on top of it: ``__AVX512F__`` was never defined while that test
#:    was compiled, so the test read the AVX2 branch on every machine.
#:
#:    The ambient build already covers the ``baseline`` rung (x86) and NEON
#:    (aarch64), so only the rungs above it get an extra binary: ``v2``,
#:    ``v3`` and ``v4`` on x86, ``sme`` on aarch64. Rungs this toolchain
#:    cannot spell are dropped, as everywhere else on the ladder. No-op when
#:    the build is single-TU (dispatch OFF or a compile-time CPU pin).
function(einsums_add_simd_rung_compiled_tests subcategory name)
  if(NOT EINSUMS_WITH_SIMD_DISPATCH
     OR EINSUMS_SIMD_NATIVE_ARCH
     OR NOT "${EINSUMS_SIMD_TARGET_CPU}" STREQUAL ""
  )
    return()
  endif()
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(_rungs v2 v3 v4)
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
    set(_rungs sme)
  else()
    return()
  endif()

  foreach(_rung IN LISTS _rungs)
    set(_flags "")
    set(_ordinal 0)
    set(_rung_ok FALSE)
    einsums_simd_rung_flags(${_rung} _flags _ordinal _rung_ok "einsums_add_simd_rung_compiled_tests(${name})")
    if(NOT _rung_ok)
      continue()
    endif()

    set(_target "${name}_simd_${_rung}_test")
    einsums_add_executable(
      ${_target} INTERNAL_FLAGS
      SOURCES ${name}.cpp
      NOINSTALL
    )
    # einsums_add_test() is what normally attaches these to a <name>_test
    # target; this registration calls add_test() directly, so wire them here.
    # einsums_testing carries the Catch2 main that parses the einsums:: flags.
    target_link_libraries(${_target} PRIVATE Catch2::Catch2 einsums_testing)
    target_compile_options(${_target} PRIVATE ${_flags})
    # A TU compiled at its own -march cannot share the project's baseline
    # precompiled header; GCC warns and re-parses anyway. Same reasoning as
    # the per-rung kernel TUs.
    set_target_properties(${_target} PROPERTIES DISABLE_PRECOMPILE_HEADERS ON)

    set(_test_name "Tests.Unit.${subcategory}.${name}.simd_${_rung}")
    # The binary IS the rung here, so there is no EINSUMS_SIMD_ARCH to set.
    # The guard still runs first, because a v4-compiled binary would take
    # SIGILL on a host without AVX-512; it exits 77 and ctest says Skipped.
    add_test(NAME ${_test_name}
             COMMAND simd_rung_guard ${_rung} $<TARGET_FILE:${_target}> "--einsums:debug:no-install-signal-handlers"
                     "--einsums:debug:no-attach-debugger" "--einsums:profile:no-report"
    )
    einsums_set_test_properties(${_test_name} "UNIT_ONLY")
    set_tests_properties(${_test_name} PROPERTIES SKIP_RETURN_CODE 77)
  endforeach()
endfunction()
