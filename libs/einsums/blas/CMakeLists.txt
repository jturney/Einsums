#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(blas_headers einsums/blas/blas.hpp)

set(blas_sources blas.cpp)

include(Einsums_AddModule)
einsums_add_module(
  einsums blas
  GLOBAL_HEADER_GEN ON
  SOURCES ${blas_sources}
  HEADERS ${blas_headers}
  DEPENDENCIES
  MODULE_DEPENDENCIES
        einsums_config
        einsums_concepts
        einsums_blas_base
        einsums_blas_vendor
  CMAKE_SUBDIRS examples tests
)
