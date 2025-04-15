#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

set(TensorAlgebraTests
    Dot.cpp
    DotGPU.hip
    Einsum1.cpp
    Einsum2.cpp
    Einsum3.cpp
    Einsum4.cpp
    Einsum5.cpp
    Einsum6.cpp
    Einsum7.cpp
    Element.cpp
    Hadamard.cpp
    Indices.cpp
    KhatriRao.cpp
    MixedPrecision.cpp
    OuterProduct.cpp
    Sort.cpp
    Transpose.cpp
    UniquePointer.cpp
    GPUTensorAlgebra.hip
    GPUTensorAlgebraTasks.hip
    QChemTests.cpp
    TensorAlgebraTasks.cpp
)

file(GLOB __data_files data/water_sto3g/*)
file(COPY ${__data_files} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/data/water_sto3g)

foreach(test ${TensorAlgebraTests})
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
    ${TestName}_test INTERNAL_FLAGS SOURCES ${Sources} ${${TestName}_${TestExtension}_FLAGS}
                                            NOINSTALL
  )

  einsums_add_unit_test(
    "Modules.TensorAlgebra" ${TestName} ${${TestName}_${TestExtension}_PARAMETERS}
  )
endforeach()
