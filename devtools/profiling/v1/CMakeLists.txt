#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

# The J/K benchmark's einsum rows against an Einsums 1.x install, for the v1
# column of the J/K breakdown figure. A standalone project: point it at a 1.x
# install, for instance the conda-forge package in its own environment. Pin the
# BLAS variant the 2.x build uses, so both columns run on the same BLAS (on macOS
# that is "libblas=*=*newaccelerate"; on Linux, *openblas or *mkl):
#
#   conda create -n einsums-v1 -c conda-forge "einsums=1.1.5" \
#       "libblas=*=*<variant>" cxx-compiler cmake ninja llvm-openmp
#   conda activate einsums-v1
#   cmake -S devtools/profiling/v1 -B build-v1 -GNinja -DCMAKE_BUILD_TYPE=Release
#   cmake --build build-v1

cmake_minimum_required(VERSION 3.24)

project(profile_jk_v1 CXX)

find_package(Einsums 1 REQUIRED)

add_executable(profile_jk_v1 profile_jk_v1.cpp)
target_link_libraries(profile_jk_v1 PRIVATE Einsums::Einsums)
