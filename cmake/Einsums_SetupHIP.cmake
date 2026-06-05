#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

if(EINSUMS_WITH_HIP AND NOT TARGET roc::rocblas)
  include(Einsums_Utils)
  include(Einsums_AddDefinitions)

  if(EINSUMS_WTIH_CUDA)
    einsums_error(
      "Both EINSUMS_WITH_CUDA and EINSUMS_WITH_HIP are ON. Please choose one of them for einsums to work properly"
    )
  endif()

  # Check and set HIP standard
  if(NOT EINSUMS_FIND_PACKAGE)
    if(DEFINED CMAKE_HIP_STANDARD AND NOT CMAKE_HIP_STANDARD STREQUAL EINSUMS_WITH_CXX_STANDARD)
      einsums_error(
        "You've set CMAKE_HIP_STANDARD to ${CMAKE_HIP_STANDARD} and EINSUMS_WITH_CXX_STANDARD to ${EINSUMS_WITH_CXX_STANDARD}. Please unset CMAKE_HIP_STANDARD."
      )
    endif()
    # set(CMAKE_HIP_STANDARD ${EINSUMS_WITH_CXX_STANDARD})
  endif()

  set(CMAKE_HIP_STANDARD ${CMAKE_CXX_STANDARD})
  set(CMAKE_HIP_STANDARD_REQUIRED ON)
  set(CMAKE_HIP_EXTENSIONS OFF)

  set(CMAKE_HIP_STANDARD_DEFAULT 98)

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
    cmake_path(APPEND CMAKE_HIP_COMPILER_ROCM_ROOT "lib" "cmake" OUTPUT_VARIABLE __hip_cmake_dir)

    cmake_path(APPEND __hip_cmake_dir "AMDDeviceLibs" OUTPUT_VARIABLE AMDDeviceLibs_DIR)
    set(ENV{AMDDeviceLibs_DIR} ${AMDDeviceLibs_DIR})

    cmake_path(APPEND __hip_cmake_dir "amd_comgr" OUTPUT_VARIABLE amd_comgr_DIR)
    set(ENV{amd_comgr_DIR} ${amd_comgr_DIR})

    cmake_path(APPEND __hip_cmake_dir "hip" OUTPUT_VARIABLE hip_DIR)
    set(ENV{hip_DIR} ${hip_DIR})

    cmake_path(APPEND __hip_cmake_dir "hsa-runtime64" OUTPUT_VARIABLE hsa-runtime64_DIR)
    set(ENV{hsa-runtime64_DIR} ${hsa-runtime64_DIR})

    cmake_path(APPEND __hip_cmake_dir "hipblas" OUTPUT_VARIABLE hipblas_DIR)
    set(ENV{hipblas_DIR} ${hipblas_DIR})

    cmake_path(APPEND __hip_cmake_dir "hipblas-common" OUTPUT_VARIABLE hipblas-common_DIR)
    set(ENV{hipblas-common_DIR} ${hipblas-common_DIR})

    cmake_path(APPEND __hip_cmake_dir "hipsolver" OUTPUT_VARIABLE hipsolver_DIR)
    set(ENV{hipsolver_DIR} ${hipsolver_DIR})
  endif()

  enable_language(HIP)

  include(Einsums_SetuphipBlas)
  include(Einsums_SetuphipSolver)

  if(NOT EINSUMS_FIND_PACKAGE)
    einsums_add_config_define(EINSUMS_HAVE_HIP)
  endif()

  set(ENABLE_HIP "ON")

endif()
