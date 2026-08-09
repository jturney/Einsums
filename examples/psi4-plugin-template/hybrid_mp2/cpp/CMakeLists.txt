#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

# Scaffolded once by 'python -m einsums.stages promote'; this file is yours.

# An out-of-tree stage module, built against an INSTALLED Einsums. The
# Einsums-world binary of this plugin: it links Einsums and only Einsums,
# never psi4; its psi4-world sibling is built one directory up.
#
#   cmake -S . -B build -GNinja -DCMAKE_BUILD_TYPE=Release \
#     "-DCMAKE_PREFIX_PATH=/path/to/einsums/install;$CONDA_PREFIX" \
#     -DPython3_EXECUTABLE=$CONDA_PREFIX/bin/python
#   cmake --build build
#
# The build type matters: an unoptimized stage module measures as a
# regression against its Python backend (einsums_add_stage_module defaults
# an empty build type to Release for that reason). The conda prefix supplies
# Einsums's dependency configs (spdlog, fmt, BLAS); the explicit
# Python3_EXECUTABLE pins the interpreter the module is built for, which
# must be the one psi4 and einsums run under - a module built for the wrong
# CPython simply fails to import.
#
# The plugin's pymodule adds this build directory to sys.path on demand, so
# STAGE_BACKEND cpp works with the module left where it was built.

cmake_minimum_required(VERSION 3.22)
project(hybrid_mp2_stages LANGUAGES CXX)

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
  hybrid_mp2_stages SOURCES bindings.cpp src/Mp2Energy.cpp INCLUDE_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
