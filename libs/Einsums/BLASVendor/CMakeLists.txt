#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

if(EINSUMS_LINALG_VENDOR STREQUAL MKL)
  einsums_add_config_define_namespace(DEFINE EINSUMS_HAVE_MKL NAMESPACE BLASVENDOR)
endif()

set(BLASVendorHeaders Einsums/BLASVendor/Vendor.hpp)

set(BLASVendorSources
    InitializeFinalize.cpp
    asum.cpp
    axpy.cpp
    copy.cpp
    dot.cpp
    gees.cpp
    geev.cpp
    gelqf.cpp
    gemm.cpp
    gemm_batch.cpp
    gemv.cpp
    geqrf.cpp
    ger.cpp
    gesv.cpp
    gesdd.cpp
    gesvd.cpp
    getri.cpp
    getrf.cpp
    heev.cpp
    lacgv.cpp
    lange.cpp
    lassq.cpp
    orglq.cpp
    orgqr.cpp
    ormqr.cpp
    potrf.cpp
    scal.cpp
    swap.cpp
    iamax.cpp
    symm.cpp
    syrk.cpp
    syr2k.cpp
    syev.cpp
    syevd.cpp
    sygv.cpp
    getrs.cpp
    gels.cpp
    trsm.cpp
    trsv.cpp
    trsyl.cpp
    trtrs.cpp
    trtri.cpp
    direct_prod.c
    direct_prod.cpp
)

if(CMAKE_SYSTEM_PROCESSOR STREQUAL x86_64)
  set(BLASVendorx64Sources direct_prod_kernels/sdirprod-x86_64-avx2.S direct_prod_kernels/ddirprod-x86_64-avx2.S
      direct_prod_kernels/cdirprod-x86_64-avx2.S direct_prod_kernels/zdirprod-x86_64-avx2.S)
  list(APPEND BLASVendorSources ${BLASVendorx64Sources})
endif()

include(Einsums_AddModule)
einsums_add_module(
  Einsums BLASVendor
  SOURCES ${BLASVendorSources}
  HEADERS ${BLASVendorHeaders}
  DEPENDENCIES tgt::lapack
        MODULE_DEPENDENCIES Einsums_Assertion Einsums_Config Einsums_Print Einsums_Profile
                            Einsums_BLASBase Einsums_HPTT Einsums_Errors Einsums_Logging Einsums_BufferAllocator
  CMAKE_SUBDIRS examples tests
)

# BLASVendor has C sources (.c) and assembly (.S) which are incompatible with
# the C++ precompiled header. Disable PCH for this module.
if(EINSUMS_WITH_PRECOMPILED_HEADERS)
  set_target_properties(Einsums_BLASVendor PROPERTIES DISABLE_PRECOMPILE_HEADERS ON)
endif()
