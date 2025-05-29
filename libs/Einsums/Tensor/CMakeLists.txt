#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(TensorHeaders
    Einsums/Tensor/BlockTensor.hpp
    Einsums/Tensor/DiskTensor.hpp
    Einsums/Tensor/H5.hpp
    Einsums/Tensor/Tensor.hpp
    Einsums/Tensor/TensorForward.hpp
    Einsums/Tensor/ArithmeticTensor.hpp
    Einsums/Tensor/FunctionTensor.hpp
    Einsums/Tensor/RuntimeTensor.hpp
    Einsums/Tensor/TiledTensor.hpp
)

set(TensorSources Memory.cpp TensorDefs.cpp Requirements.cpp)

if(EINSUMS_WITH_GPU_SUPPORT)
  set_source_files_properties(TensorDefs.cpp PROPERTIES LANGUAGE CXX)

  list(APPEND TensorHeaders Einsums/Tensor/DeviceTensor.hpp)
  list(APPEND TensorSources TensorDefs.hip)

  set(TensorGPUModules Einsums_GPUStreams)
  set(TensorGPUDeps hip::host hip::device)

  if(EINSUMS_WITH_CUDA)
  foreach(f IN LISTS TensorSources)
    if(f MATCHES ".*\.hip")
      set_source_files_properties("src/${f}" PROPERTIES LANGUAGE CUDA)
    endif()
  endforeach()
endif()
endif()

find_package(Python3 REQUIRED COMPONENTS Interpreter)

# Path
set(GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
file(MAKE_DIRECTORY ${GENERATED_DIR})
set(GENERATED_STAMP ${GENERATED_DIR}/GeneratedInstantiations.stamp)

# 2. Capture the output of the generation command
execute_process(
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/cmake/scripts/generate_instantiations.py --output_dir ${GENERATED_DIR} ${TensorHeaders} --list_files
        OUTPUT_VARIABLE FILES_OUTPUT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
string(REPLACE "\n" ";" GENERATED_FILES_OUTPUT "${FILES_OUTPUT}")

# 1. Define the generation command
add_custom_command(
        OUTPUT ${GENERATED_FILES_OUTPUT}
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/cmake/scripts/generate_instantiations.py
            --output_dir ${GENERATED_DIR}
            ${TensorHeaders}
        DEPENDS ${CMAKE_SOURCE_DIR}/cmake/scripts/generate_instantiations.py
        COMMENT "Regenerating template instantiations"
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

add_custom_target(generate_tensor_instantiations DEPENDS ${GENERATED_FILE_OUTPUT})

include(Einsums_AddModule)
einsums_add_module(
  Einsums Tensor
  SOURCES ${TensorSources}
  GENERATED_SOURCES ${GENERATED_FILES_OUTPUT}
  HEADERS ${TensorHeaders}
  DEPENDENCIES Einsums::h5cpp ${TensorGPUDeps}
  MODULE_DEPENDENCIES
    Einsums_BLAS
    Einsums_Concepts
    Einsums_Config
    Einsums_Errors
    Einsums_Iterator
    Einsums_Logging
    Einsums_TensorBase
    Einsums_TypeSupport
    Einsums_Profile
    Einsums_Print
    Einsums_Utilities
    ${TensorGPUModules}
  CMAKE_SUBDIRS examples tests
)

add_dependencies(Einsums_Tensor generate_tensor_instantiations)
