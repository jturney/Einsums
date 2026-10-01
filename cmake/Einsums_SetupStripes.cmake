#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

# Stripes (https://github.com/Einsums/Stripes), the SIMD library that began as Einsums' SIMD module:
# Vec<T>, the instruction-set dispatch ladder, and stripes_add_dispatch_sources() and the rung test
# functions, which HPTT and PackedGemm build their per-rung kernels with.

include(FetchContent)

# ── Options ──────────────────────────────────────────────────────────────────
#
# Einsums' spellings of Stripes' three build switches, mapped onto Stripes' below. Either pin makes
# the binary unportable to older CPUs of the same family and collapses every dispatch ladder to one
# native copy.
einsums_option(
  EINSUMS_WITH_SIMD_DISPATCH BOOL
  "Compile SIMD kernel TUs once per instruction-set rung (x86-64 baseline/v2/v3/v4, aarch64 sme) and \
   pick the best at runtime. One portable binary, full speed on newer CPUs. OFF compiles a single TU at \
   the toolchain baseline. Automatically treated as OFF when EINSUMS_SIMD_NATIVE_ARCH or \
   EINSUMS_SIMD_TARGET_CPU pins the build."
  ON
  CATEGORY "Build Targets"
)
einsums_option(
  EINSUMS_SIMD_NATIVE_ARCH BOOL
  "Compile SIMD-using TUs with -mcpu=native / -march=native. Unlocks every SIMD feature the host CPU \
   supports at the cost of binary portability."
  OFF
  CATEGORY "Build Targets"
)
einsums_option(
  EINSUMS_SIMD_TARGET_CPU STRING
  "Pin SIMD-using TUs to a specific CPU model, such as 'apple-m4', 'znver4' or 'sapphirerapids'. Empty \
   (default) defers to EINSUMS_SIMD_NATIVE_ARCH or the toolchain baseline."
  ""
  CATEGORY "Build Targets"
)

set(STRIPES_WITH_DISPATCH ${EINSUMS_WITH_SIMD_DISPATCH})
set(STRIPES_NATIVE_ARCH ${EINSUMS_SIMD_NATIVE_ARCH})
set(STRIPES_TARGET_CPU "${EINSUMS_SIMD_TARGET_CPU}")

# The runtime library (CPU detection and rung selection) is built STATIC and folded into
# libEinsums, as spdlog is (see Einsums_SetupSpdlog): a separate shared library on Windows would
# land in build/_deps/, beside none of the test executables. It is compiled hidden, so nothing of
# it is exported from libEinsums, and a library that brings its own Stripes keeps its own choice of
# rung. Position-independent code, since it goes into a shared library.
set(STRIPES_SHARED OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

fetchcontent_declare(
  Stripes
  GIT_REPOSITORY https://github.com/Einsums/Stripes.git
  GIT_TAG ed0e2963fb4472af3d46e5e16dffd0eee0d121b0 # main, 2026-10-01
  # Its headers are compiled at Stripes' own warning level, not Einsums'.
  SYSTEM
  FIND_PACKAGE_ARGS
  1
  CONFIG
)
fetchcontent_makeavailable(Stripes)

# An installed Stripes brings the dispatch functions with its package; a fetched one leaves them in
# its source tree.
if(NOT COMMAND stripes_add_dispatch_sources)
  list(APPEND CMAKE_MODULE_PATH "${stripes_SOURCE_DIR}/cmake")
  include(StripesDispatch)
endif()
