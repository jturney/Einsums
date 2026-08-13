#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

# Scaffolded once by 'python -m einsums.stages promote'; this file is yours.

# An out-of-tree stage module, built against an INSTALLED Einsums:
#
#   cmake -S . -B build -GNinja -DCMAKE_BUILD_TYPE=Release \
#     "-DCMAKE_PREFIX_PATH=/path/to/einsums/install;$CONDA_PREFIX" \
#     -DPython3_EXECUTABLE=$CONDA_PREFIX/bin/python
#   cmake --build build
#
# The build type matters: a stage module exists to be measured against its
# Python backend, and an unoptimized one measures as a regression
# (einsums_add_stage_module defaults an EMPTY build type to Release for that
# reason, but say what you mean). The conda prefix supplies Einsums's
# dependency configs, and the explicit interpreter pins the CPython the
# module imports into.
#
# Then put the resulting dlpno_stages module on PYTHONPATH and select it:
#
#   python driver.py --backend compute_pno_overlaps=cpp

cmake_minimum_required(VERSION 3.22)
project(dlpno_stages LANGUAGES CXX)

# COMPONENTS stages asks for what a stage module needs and cannot do without: a SHARED libEinsums.
# Shared-graph capture keeps its capture context in thread-local state inside the library, so a
# statically linked module gets its own copy and silently captures into a graph nobody executes.
#
# This also finds pybind11, at the exact version einsums._core was built with. Repeating that
# find_package here would be redundant: pybind11 only shares types between modules whose internals
# agree, so the version is Einsums's to dictate rather than ours to choose.
find_package(Einsums REQUIRED COMPONENTS stages)

# Wraps pybind11_add_module, the link, and the language standard - the last of which is the one
# worth not hand-rolling. A module compiled at a different standard than the library folds a
# different __cplusplus into the config fingerprint and is refused at import, and may fail to dlopen
# before the handshake can say so.
einsums_add_stage_module(
  dlpno_stages
  SOURCES
  bindings.cpp
  src/ComputePnoIntegrals.cpp
  src/ComputePnoOverlaps.cpp
  src/TransformPnos.cpp
  INCLUDE_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
