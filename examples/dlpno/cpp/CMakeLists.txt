#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

# An out-of-tree stage module, built the way an external method developer would
# build one: against an INSTALLED Einsums, with no knowledge of this repository.
#
#   cmake -S examples/dlpno/cpp -B build-dlpno-stages -GNinja \
#         -DCMAKE_PREFIX_PATH=/path/to/einsums/install
#   cmake --build build-dlpno-stages
#
# Then put the resulting dlpno_stages module on PYTHONPATH and select it:
#
#   python run_fixtures.py --stages --backend compute_pno_overlaps=cpp
#
# Deliberately not wired into the top-level build. A stage module that only
# ever compiles inside the tree that produced libEinsums proves nothing about
# the SDK path, which is the path every external user is on and the one that
# breaks first.

cmake_minimum_required(VERSION 3.22)
project(dlpno_stages LANGUAGES CXX)

# Shared only, and the config refuses a static-only install: shared-graph
# capture keeps its capture context in thread-local state inside libEinsums, so
# a statically linked stage module gets its own copy and silently captures into
# a graph nobody executes.
find_package(Einsums REQUIRED)

# EXACT, and the version comes from Einsums rather than being chosen here.
# pybind11 keys its type registry on the ABI of the pybind that built each
# module; two different pybind11 versions in one process do not share
# RuntimeTensor and every tensor crossing this boundary would be copied through
# the buffer protocol at best, or fail to cast at worst.
find_package(pybind11 ${Einsums_PYBIND11_VERSION} EXACT REQUIRED)

pybind11_add_module(dlpno_stages MODULE bindings.cpp src/PnoOverlaps.cpp)

target_include_directories(dlpno_stages PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(dlpno_stages PRIVATE Einsums::Einsums)

# Match the library's language level. A stage module compiled at a different
# standard computes a different config fingerprint and is refused at import by
# einsums.sealed, which is the guard working rather than a problem - but it is
# a confusing way to discover it, so state the standard instead of inheriting
# whatever the compiler defaults to.
set_target_properties(
  dlpno_stages PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF
)
