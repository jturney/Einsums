#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

set(linear_algebra_tests
    direct_prod.cpp
    dot.cpp
    geev.cpp
    gemm.cpp
    gemm_runtime.cpp
    GEMMTensorView.cpp
    gesv.cpp
    gesv_runtime.cpp
    gesvd.cpp
    gemv.cpp
    gemv_runtime.cpp
    ger_runtime.cpp
    blas1_runtime.cpp
    gesdd.cpp
    getrf_getri.cpp
    invert.cpp
    heev.cpp
    lassq.cpp
    norm.cpp
    pow.cpp
    syev.cpp
    Scale.cpp
    Decomposition.cpp
    trsm.cpp
    potrf.cpp
    syrk.cpp
    symm.cpp
    symm_dispatch.cpp
    sygv.cpp
    syevd.cpp
    getrs.cpp
    gels.cpp
    swap.cpp
    iamax.cpp
    trsv.cpp
    syr2k.cpp
    trtrs.cpp
    trtri.cpp
)

foreach(test ${linear_algebra_tests})
  # Extract base name of test Example1.cpp -> Example1, Example.2.hip -> Example
  get_filename_component(TestName ${test} NAME_WE)
  # Extract extension of the test Example1.cpp -> .cpp, Example.2.hip -> .2.hip
  get_filename_component(Extension ${test} EXT)

  # Remove the leading dot and replace any periods with underscores
  string(SUBSTRING ${Extension} 1 -1 EXT_NO_DOT) # Remove the leading dot
  string(REPLACE "." "_" TestExtension ${EXT_NO_DOT})

  # Check if the cleaned extension is "cu" or "hip"
  if(NOT EINSUMS_WITH_GPU_SUPPORT AND (TestExtension STREQUAL "cu" OR TestExtension STREQUAL "hip"))
    # Skip this iteration
    continue()
  endif()

  set(Sources ${test})

  source_group("Source Files" FILES ${Sources})

  einsums_add_executable(
    ${TestName}_test INTERNAL_FLAGS
    SOURCES ${Sources} ${${TestName}_${TestExtension}_FLAGS}
    NOINSTALL
  )

  einsums_add_unit_test(
    "Modules.LinearAlgebra" ${TestName} ${${TestName}_${TestExtension}_PARAMETERS}
  )
endforeach()
