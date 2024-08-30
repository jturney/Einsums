#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

if (EINSUMS_LINALG_VENDOR STREQUAL MKL)
    einsums_add_config_define_namespace(DEFINE EINSUMS_HAVE_MKL NAMESPACE BLAS_VENDOR)
endif()

set(blas_vendor_headers einsums/blas/vendor.hpp)

set(blas_vendor_sources vendor.cpp)

include(Einsums_AddModule)
einsums_add_module(
  einsums blas_vendor
  GLOBAL_HEADER_GEN ON
  SOURCES ${blas_vendor_sources}
  HEADERS ${blas_vendor_headers}
  DEPENDENCIES tgt::lapack
  MODULE_DEPENDENCIES einsums_config einsums_profile einsums_print einsums_blas_base
  CMAKE_SUBDIRS examples tests
)
