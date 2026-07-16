#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
# CMAKE_MODULE_PATH is directory-scoped; export the append so sibling modules
# configured after this one (HPTT and other dispatch consumers) can
# include(Einsums_AddSIMDDispatch).
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" PARENT_SCOPE)

set(SIMDHeaders
    Einsums/SIMD/Platform.hpp
    Einsums/SIMD/Vec.hpp
    Einsums/SIMD/Operations.hpp
    Einsums/SIMD/ComplexVec.hpp
    Einsums/SIMD/Shuffle.hpp
    Einsums/SIMD/Gather.hpp
    Einsums/SIMD/Prefetch.hpp
    Einsums/SIMD/RuntimeFeatures.hpp
)

set(SIMDSources RuntimeFeatures.cpp)

# ── Architecture-targeting options ───────────────────────────────────────────
#
# The SIMD module dispatches via compile-time feature macros (`__AVX512F__`,
# `__ARM_FEATURE_BF16_VECTOR_ARITHMETIC`, etc.). Whether those macros are
# defined depends entirely on the compiler's target-CPU baseline. Most
# toolchains default to a conservative baseline that omits newer features —
# e.g., the conda-forge clang for `arm64-apple-darwin` defaults to roughly
# the M1 feature set, hiding M3+ BF16 and M4+ I8MM.
#
# Two opt-in knobs surface higher-tier hardware to the SIMD module's
# consumers without polluting the rest of the project:
#
#   - `EINSUMS_SIMD_TARGET_CPU=<name>`  — pin to a specific CPU
#       (`apple-m4`, `znver4`, `sapphirerapids`, …). Highest precedence.
#       Useful for cross-builds and reproducible binaries.
#
#   - `EINSUMS_SIMD_NATIVE_ARCH=ON`     — use the host CPU
#       (`-mcpu=native` on aarch64, `-march=native` on x86). Lower
#       precedence; ignored when TARGET_CPU is also set.
#
# Both attach the flag to the `Einsums_SIMD` target's INTERFACE compile
# options so it propagates to every module that lists `Einsums_SIMD` as a
# dependency (HPTT, TensorAlgebra, future kernels) — but nothing else.
# Trade-off either knob makes: the resulting binary is no longer portable
# across older CPUs of the same family. Default is OFF — the module
# silently runs at whatever ISA tier the toolchain's baseline exposes.
einsums_option(
    EINSUMS_WITH_SIMD_DISPATCH BOOL
    "Compile SIMD kernel TUs once per instruction-set rung (x86-64 baseline/\
     v2/v3/v4) and pick the best at runtime (see Einsums/SIMD/RuntimeFeatures.hpp). \
     One portable binary, full speed on newer CPUs. OFF compiles a single TU at \
     the toolchain baseline, matching pre-dispatch behavior. Automatically \
     treated as OFF on non-x86 targets and when EINSUMS_SIMD_NATIVE_ARCH or \
     EINSUMS_SIMD_TARGET_CPU pins the build."
    ON
    CATEGORY "Build Targets"
)
einsums_option(
    EINSUMS_SIMD_NATIVE_ARCH BOOL
    "Compile SIMD-using TUs with -mcpu=native / -march=native. Unlocks every \
     SIMD feature the host CPU supports (BF16, I8MM, FP16, AVX-10, etc.) at \
     the cost of binary portability — the build will not run on older CPUs \
     of the same family."
    OFF
    CATEGORY "Build Targets"
)
einsums_option(
    EINSUMS_SIMD_TARGET_CPU STRING
    "Pin SIMD-using TUs to a specific CPU model. Examples: \
     'apple-m4', 'znver4', 'sapphirerapids', 'graviton3'. \
     Empty (default) defers to EINSUMS_SIMD_NATIVE_ARCH or the toolchain \
     baseline. Takes precedence over EINSUMS_SIMD_NATIVE_ARCH when both are \
     set."
    ""
    CATEGORY "Build Targets"
)

include(Einsums_AddModule)
einsums_add_module(
  Einsums SIMD
  SOURCES ${SIMDSources}
  HEADERS ${SIMDHeaders}
  DEPENDENCIES
  MODULE_DEPENDENCIES Einsums_Config Einsums_Logging
  CMAKE_SUBDIRS examples tests
)

# Resolve precedence and emit the right flag for the host architecture.
# `-mcpu=` is the aarch64 spelling; `-march=` is the x86 spelling. MSVC
# uses `/arch:` with a different vocabulary entirely (AVX, AVX2, AVX512)
# and isn't covered here — Einsums on Windows users would have to edit
# CMAKE_CXX_FLAGS directly. Apply via INTERFACE so the flag travels with
# the target, not via add_compile_options which would leak to the entire
# build.
set(_simd_arch_flag "")
if(NOT "${EINSUMS_SIMD_TARGET_CPU}" STREQUAL "")
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64")
        set(_simd_arch_flag "-mcpu=${EINSUMS_SIMD_TARGET_CPU}")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|i.86")
        set(_simd_arch_flag "-march=${EINSUMS_SIMD_TARGET_CPU}")
    endif()
elseif(EINSUMS_SIMD_NATIVE_ARCH)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64")
        set(_simd_arch_flag "-mcpu=native")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|i.86")
        set(_simd_arch_flag "-march=native")
    endif()
endif()
if(NOT "${_simd_arch_flag}" STREQUAL "")
    if(MSVC)
        message(WARNING
            "Einsums_SIMD: ${_simd_arch_flag} is GCC/Clang syntax — MSVC ignores it. "
            "Use /arch:AVX512 or similar in CMAKE_CXX_FLAGS instead.")
    else()
        target_compile_options(Einsums_SIMD INTERFACE ${_simd_arch_flag})
        message(STATUS "Einsums_SIMD: applying ${_simd_arch_flag} to the SIMD INTERFACE")
    endif()
endif()
unset(_simd_arch_flag)
