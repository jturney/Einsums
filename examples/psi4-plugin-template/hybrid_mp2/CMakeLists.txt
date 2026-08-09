#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

# One configure, one build, two binaries - each linking only its own world.
#
#   cmake -S . -B build -GNinja \
#     -C /path/to/psi4/stage/share/cmake/psi4/psi4PluginCache.cmake \
#     "-DCMAKE_PREFIX_PATH=/path/to/psi4/stage;/path/to/einsums/install"
#   cmake --build build
#
# produces build/hybrid_mp2.so (psi4 world) and build/cpp/hybrid_mp2_stages
# (Einsums world). The sealed-worlds rule constrains what each TARGET links,
# not how many configures you run; sharing one build tree only means sharing
# one toolchain, and psi4's plugin cache pins the same conda compilers the
# Einsums dev environment documents. If your psi4 was built with a different
# toolchain than your Einsums, that assumption is false: skip this top-level
# build and configure cpp/ standalone with Einsums's toolchain (its
# CMakeLists documents how) - the import-time fingerprint handshake is the
# backstop either way.

cmake_minimum_required(VERSION 3.19...4.0)
project(hybrid_mp2 CXX)

# Conveniences that used to be command-line flags, absorbed so the configure
# line above is the whole incantation.
#
# C++20 module scanning wants clang-scan-deps, which the conda toolchain the
# plugin cache pins does not ship; nothing here uses modules.
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

# Einsums's dependency configs (spdlog, fmt, BLAS) live in the conda env.
if(DEFINED ENV{CONDA_PREFIX})
  list(APPEND CMAKE_PREFIX_PATH $ENV{CONDA_PREFIX})
endif()

# psi4's plugin cache pins Python_EXECUTABLE; the stage module's pybind11
# resolves Python3_EXECUTABLE. Point them at the same interpreter, because a
# module built for the wrong CPython simply fails to import.
if(DEFINED Python_EXECUTABLE AND NOT DEFINED Python3_EXECUTABLE)
  set(Python3_EXECUTABLE
      "${Python_EXECUTABLE}"
      CACHE FILEPATH "Interpreter the stage module is built for")
endif()

# ---- the psi4-world binary: links psi4 and only psi4 ------------------------
find_package(psi4 1.9 REQUIRED)

add_psi4_plugin(hybrid_mp2 plugin.cc)

# ---- the Einsums-world binary: links Einsums and only Einsums ---------------
# cpp/ is promote-generated and self-contained (its own project(), its own
# find_package(Einsums)), so it builds identically here as a subdirectory or
# on its own; add_subdirectory is the whole merge.
#
# One collision the merged tree exposes: with psi4's stage on
# CMAKE_PREFIX_PATH, Einsums's HDF5 lookup finds the TargetHDF5Config that
# psi4 exports, whose tgt::hdf5 references the `hdf5-shared` imported target
# without defining it. Loading HDF5's own config first defines that target,
# so the reference resolves. Harmless when the collision is absent.
option(HYBRID_MP2_CPP_STAGES "Build the Einsums-world stage module under cpp/" ON)
if(HYBRID_MP2_CPP_STAGES)
  find_package(HDF5 CONFIG QUIET)
  add_subdirectory(cpp)
endif()
