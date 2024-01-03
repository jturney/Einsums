# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(linear_algebra_headers)

set(linear_algebra_sources Blas.cpp)

include(einsums_add_module)
einsums_add_module(
    einsums linear_algebra
    GLOBAL_HEADER_GEN OFF
    SOURCES ${linear_algebra_sources}
    HEADERS ${linear_algebra_headers}
    MODULE_DEPENDENCIES einsums_config einsums_core einsums_preprocessor einsums_concepts einsums_tensor
    DEPENDENCIES tgt::lapack
    CMAKE_SUBDIRS examples tests
)

include(einsums_extend_target)
einsums_extend_target(
    einsums linear_algebra PUBLIC_DEFINES EINSUMS_LINALG_INT_INTERFACE_ILP64
    DEFINES MKL_INTERFACE=ILP64
    CONDITION (EINSUMS_LINALG_VENDOR STREQUAL MKL AND EINSUMS_LINALG_INT_INTERFACE STREQUAL ilp64)
    FEATURE_INFO "MKL ILP64 (64-bit integers)"
)

einsums_extend_target(
    einsums linear_algebra PUBLIC_DEFINES MKL_LP64
    DEFINES MKL_INTERFACE=LP64
    CONDITION (EINSUMS_LINALG_VENDOR STREQUAL MKL AND EINSUMS_LINALG_INT_INTERFACE STREQUAL lp64)
    FEATURE_INFO "MKL LP64 (32-bit integers)"
)

einsums_extend_target(
    einsums linear_algebra
    SOURCES backends/mkl/mkl.cpp
    DEFINES EINSUMS_HAVE_MKL_LAPACKE_H EINSUMS_HAVE_MKL_CBLAS_H EINSUMS_HAVE_CBLAS EINSUMS_HAVE_LAPACKE
            EINSUMS_LINEAR_ALGEBRA_NAMESPACE=mkl
    CONDITION EINSUMS_LINALG_VENDOR STREQUAL MKL
    FEATURE_INFO "MKL backend"
)

einsums_extend_target(
    einsums linear_algebra
    SOURCES backends/vendor/Vendor.cpp
    DEFINES EINSUMS_LINEAR_ALGEBRA_NAMESPACE=vendor
    CONDITION NOT EINSUMS_LINALG_VENDOR STREQUAL MKL
    FEATURE_INFO "Generic vendor backend"
)

einsums_extend_target(
    einsums linear_algebra
    DEPENDENCIES ittnotify
    DEFINES HAVE_ITTNOTIFY
    CONDITION TARGET ittnotify
    FEATURE_INFO "VTune Instrumentation and Tracing Technology API"
)
