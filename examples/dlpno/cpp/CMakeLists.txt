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
#   python run_fixtures.py --backend compute_pno_overlaps=cpp
#
# Deliberately not wired into the top-level build. A stage module that only
# ever compiles inside the tree that produced libEinsums proves nothing about
# the SDK path, which is the path every external user is on and the one that
# breaks first.

cmake_minimum_required(VERSION 3.22)
project(dlpno_stages LANGUAGES CXX)

# COMPONENTS stages asks for what a stage module needs and cannot do without: a
# SHARED libEinsums. Shared-graph capture keeps its capture context in
# thread-local state inside the library, so a statically linked module gets its
# own copy and silently captures into a graph nobody executes.
#
# This also finds pybind11, at the exact version einsums._core was built with.
# Repeating that find_package here would be redundant: pybind11 only shares
# types between modules whose internals agree, so the version is Einsums's to
# dictate rather than ours to choose.
find_package(Einsums REQUIRED COMPONENTS stages)

# Wraps pybind11_add_module, the link, and the language standard - the last of
# which is the one worth not hand-rolling. A module compiled at a different
# standard than the library folds a different __cplusplus into the config
# fingerprint and is refused at import, and may fail to dlopen before the
# handshake can say so.
einsums_add_stage_module(
  dlpno_stages
  SOURCES bindings.cpp src/PnoOverlaps.cpp
  INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include
)
