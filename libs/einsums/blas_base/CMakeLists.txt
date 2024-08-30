#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

if(EINSUMS_LINALG_INT_INTERFACE STREQUAL ilp64)
    einsums_add_config_define_namespace(DEFINE EINSUMS_BLAS_INTERFACE_ILP64 NAMESPACE BLAS_BASE)
elseif (EINSUMS_LINALG_INT_INTERFACE STREQUAL lp64)
    einsums_add_config_define_namespace(DEFINE EINSUMS_BLAS_INTERFACE_LP64 NAMESPACE BLAS_BASE)
endif()

set(blas_base_headers einsums/blas/types.hpp)

set(blas_base_sources)

include(Einsums_AddModule)
einsums_add_module(
  einsums blas_base
  GLOBAL_HEADER_GEN ON
  SOURCES ${blas_base_sources}
  HEADERS ${blas_base_headers}
  DEPENDENCIES
  MODULE_DEPENDENCIES
  CMAKE_SUBDIRS examples tests
)
