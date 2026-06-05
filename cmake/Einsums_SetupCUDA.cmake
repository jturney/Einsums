#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

if(EINSUMS_WITH_CUDA AND NOT TARGET cuda)
  include(Einsums_Utils)
  include(Einsums_AddDefinitions)
  if(EINSUMS_WTIH_HIP)
    einsums_error(
      "Both EINSUMS_WITH_CUDA and EINSUMS_WITH_HIP are ON. Please choose one of them for einsums to work properly"
    )
  endif()

  # Check and set HIP standard
  if(NOT EINSUMS_FIND_PACKAGE)
    if(DEFINED CMAKE_CUDA_STANDARD AND NOT CMAKE_CUDA_STANDARD STREQUAL EINSUMS_WITH_CXX_STANDARD)
      einsums_error(
        "You've set CMAKE_CUDA_STANDARD to ${CMAKE_CUDA_STANDARD} and EINSUMS_WITH_CXX_STANDARD to ${EINSUMS_WITH_CXX_STANDARD}. Please unset CMAKE_CUDA_STANDARD."
      )
    endif()
  endif()

  set(CMAKE_CUDA_STANDARD ${CMAKE_CXX_STANDARD})
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_EXTENSIONS OFF)

  set(CMAKE_CUDA_STANDARD_DEFAULT 98)

  set(HIP_PLATFORM "nvidia")

  if(NOT CMAKE_HIP_COMPILER_ROCM_ROOT AND NOT HIP_ROCM_ROOT)
    message(
      WARNING
        "CMAKE_HIP_COMPILER_ROCM_ROOT is not set. CMake may not be able to find all of the libraries needed for HIP."
        " Please set this variable. As a shorter alternative, HIP_ROCM_ROOT can be set, and this will set the longer macro."
    )

  elseif(NOT CMAKE_HIP_COMPILER_ROCM_ROOT AND HIP_ROCM_ROOT)
    set(CMAKE_HIP_COMPILER_ROCM_ROOT ${HIP_ROCM_ROOT})
  endif()

  if(CMAKE_HIP_COMPILER_ROCM_ROOT)
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_HIP_COMPILER_ROCM_ROOT}/lib/cmake/hip")

    cmake_path(APPEND CMAKE_HIP_COMPILER_ROCM_ROOT "lib" "cmake" OUTPUT_VARIABLE __hip_cmake_dir)

    cmake_path(APPEND __hip_cmake_dir "hip" OUTPUT_VARIABLE hip_DIR)
    set(ENV{hip_DIR} ${hip_DIR})

  endif()

  enable_language(CUDA)

  file(GLOB_RECURSE src_hip "${CMAKE_CURRENT_SOURCE_DIR}/*.hip")
  foreach(X IN ITEMS ${src_hip})
    set_source_files_properties(${X} PROPERTIES LANGUAGE CUDA)
  endforeach()

  list(APPEND CMAKE_CUDA_SOURCE_FILE_EXTENSIONS hip)

  set(HIP_PLATFORM "nvidia")
  set(USE_CUDA ON)

  find_package(hip REQUIRED)

  set(HIP_PLATFORM "nvidia")
  set(USE_CUDA ON)

  include(Einsums_SetuphipBlas)

  set(HIP_PLATFORM "nvidia")
  set(USE_CUDA ON)

  include(Einsums_SetuphipSolver)

  set(HIP_PLATFORM "nvidia")
  set(USE_CUDA ON)

  include(Einsums_ExportTargets)

  if(NOT EINSUMS_FIND_PACKAGE)
    einsums_add_config_define(EINSUMS_HAVE_CUDA)
  endif()

  set(ENABLE_CUDA "ON")

endif()
