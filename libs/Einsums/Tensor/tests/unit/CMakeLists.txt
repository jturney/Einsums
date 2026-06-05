#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

set(TensorTests
    ArithmeticTensor.cpp
    BlockTensor.cpp
    RuntimeTensors.cpp
    TensorView.cpp
    FunctionTensor.cpp
    Symmetry.cpp
    TiledTensor.cpp
    TiledRuntimeTensorAlias.cpp
    Tensor.cpp
    Disk.cpp)

foreach(test ${TensorTests})
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

  einsums_add_unit_test("Modules.Tensor" ${TestName} ${${TestName}_${TestExtension}_PARAMETERS})
endforeach()

einsums_add_python_unit_test(
  "Modules.Tensor" RuntimeTensorPython
  SCRIPT test_runtime_tensor_python.py
)

einsums_add_python_unit_test(
  "Modules.Tensor" NumpyInteropPython
  SCRIPT test_numpy_interop_python.py
)

einsums_add_python_unit_test(
  "Modules.Tensor" NumpyErgonomicsPython
  SCRIPT test_numpy_ergonomics_python.py
)

einsums_add_python_unit_test(
  "Modules.Tensor" TiledRuntimeTensorPython
  SCRIPT test_tiled_runtime_tensor_python.py
)
