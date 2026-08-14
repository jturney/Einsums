#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Which vendor was resolved is not answerable from EINSUMS_LINALG_VENDOR: that
# property records what the FindLAPACK search ASKED for, and a conda environment
# answers every ask with the same `libblas`/`liblapack` shim, so an OpenBLAS
# environment and an Accelerate one are indistinguishable by name. Ask the
# resolved libraries for a symbol only OpenBLAS has instead. The distinction
# matters because OpenBLAS threads through our own OpenMP runtime and so obeys
# `omp_set_num_threads`, while Accelerate obeys neither that nor any per-thread
# setter.
include(CheckFunctionExists)
set(CMAKE_REQUIRED_LIBRARIES tgt::lapack)
check_function_exists(openblas_get_config EINSUMS_BLAS_FOUND_OPENBLAS)
unset(CMAKE_REQUIRED_LIBRARIES)
if(EINSUMS_BLAS_FOUND_OPENBLAS)
  einsums_add_config_define_namespace(DEFINE EINSUMS_HAVE_OPENBLAS NAMESPACE BLAS)
endif()

set(BLASHeaders Einsums/BLAS.hpp Einsums/BLAS/InitModule.hpp Einsums/BLAS/ThreadControl.hpp)

set(BLASSources BLAS.cpp InitModule.cpp ThreadControl.cpp)

include(Einsums_AddModule)
einsums_add_module(
  Einsums BLAS
  SOURCES ${BLASSources}
  HEADERS ${BLASHeaders}
  # ThreadControl resolves the vendor's thread setter through dlsym, which
  # lives in libdl on glibc older than 2.34. Empty on Windows and macOS.
  DEPENDENCIES ${CMAKE_DL_LIBS}
  MODULE_DEPENDENCIES Einsums_Config Einsums_Concepts Einsums_BLASBase Einsums_BLASVendor
                      Einsums_Runtime
  CMAKE_SUBDIRS examples tests
)
