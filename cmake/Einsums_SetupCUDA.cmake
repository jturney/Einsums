#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

# CUDA backend setup.
#
# This file previously routed CUDA through ROCm-on-NVIDIA: it did
# find_package(hip/hipblas/hipsolver REQUIRED) with HIP_PLATFORM=nvidia and never
# called find_package(CUDAToolkit). That could not work on a CUDA-only machine -
# there is no ROCm to find - and it never provided the CUDA::cublas / CUDA::cusolver
# / CUDA::cudart imported targets that libs/Einsums/GPU/CMakeLists.txt links against.
# Those targets come from CUDAToolkit and nothing else.
#
# Building with HIP on an NVIDIA card is still possible: set EINSUMS_WITH_HIP=ON
# with a ROCm install configured for the nvidia platform. That is Einsums_SetupHIP's
# job, not this file's.

if(EINSUMS_WITH_CUDA AND NOT TARGET CUDA::cudart)
  include(Einsums_Utils)
  include(Einsums_AddDefinitions)

  if(EINSUMS_WITH_HIP)
    einsums_error(
      "Both EINSUMS_WITH_CUDA and EINSUMS_WITH_HIP are ON. Please choose one of them for einsums to work properly"
    )
  endif()

  # ------------------------------------------------------------------------------
  # CUDA language standard
  # ------------------------------------------------------------------------------
  # Device code has to be compiled at the same standard as host code, or the two
  # halves of a TU disagree about the ABI of anything they share.
  if(NOT EINSUMS_FIND_PACKAGE)
    if(DEFINED CMAKE_CUDA_STANDARD AND NOT CMAKE_CUDA_STANDARD STREQUAL CMAKE_CXX_STANDARD)
      einsums_error(
        "You've set CMAKE_CUDA_STANDARD to ${CMAKE_CUDA_STANDARD} and CMAKE_CXX_STANDARD to ${CMAKE_CXX_STANDARD}. Please unset CMAKE_CUDA_STANDARD."
      )
    endif()
  endif()

  set(CMAKE_CUDA_STANDARD ${CMAKE_CXX_STANDARD})
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_EXTENSIONS OFF)

  # ------------------------------------------------------------------------------
  # Target architectures
  # ------------------------------------------------------------------------------
  # CMP0104 requires CMAKE_CUDA_ARCHITECTURES to be set before enable_language(CUDA);
  # nothing in the tree set it, so a CUDA build had no -arch at all.
  #
  # `native` compiles only for the cards in this machine, which is what a developer
  # wants and is much faster to build. It resolves to the literal string
  # "No CUDA devices found." when CMake cannot see a device, so it is only safe when
  # a driver is actually present - hence the probe rather than a bare default.
  #
  # The probe is skipped once the cache variable exists, so a reconfigure does not
  # shell out again and an explicit -DEINSUMS_WITH_CUDA_ARCHITECTURES always wins.
  set(_einsums_default_cuda_arch "all-major")
  if(NOT DEFINED EINSUMS_WITH_CUDA_ARCHITECTURES)
    execute_process(
      COMMAND nvidia-smi -L
      RESULT_VARIABLE _einsums_nvidia_smi_result
      OUTPUT_VARIABLE _einsums_nvidia_smi_output
      ERROR_QUIET
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    # Otherwise leave the "all-major" fallback set above: no visible device means
    # build a portable fat binary. This is the CI and cross-build case.
    if(_einsums_nvidia_smi_result EQUAL 0 AND _einsums_nvidia_smi_output MATCHES "GPU 0:")
      set(_einsums_default_cuda_arch "native")
    endif()
    unset(_einsums_nvidia_smi_result)
    unset(_einsums_nvidia_smi_output)
  endif()

  einsums_option(
    EINSUMS_WITH_CUDA_ARCHITECTURES STRING
    "CUDA architectures to build for; passed to CMAKE_CUDA_ARCHITECTURES. 'native' targets only the cards in this machine and requires a visible device; 'all-major' builds a portable fat binary."
    "${_einsums_default_cuda_arch}" CATEGORY "Build Targets"
  )
  unset(_einsums_default_cuda_arch)

  set(CMAKE_CUDA_ARCHITECTURES "${EINSUMS_WITH_CUDA_ARCHITECTURES}")

  # ------------------------------------------------------------------------------
  # Enable the language and find the toolkit
  # ------------------------------------------------------------------------------
  # enable_language(CUDA) is needed even though libs/ has no .cu sources today:
  # einsums_check_for_cxx23_static_call_operator_gpu (Einsums_AddConfigTest.cmake)
  # compiles cmake/tests/cxx23_static_call_operator.cu whenever GPU support is on.
  enable_language(CUDA)

  # CUDAToolkit is what defines CUDA::cudart, CUDA::cublas and CUDA::cusolver.
  find_package(CUDAToolkit REQUIRED)

  einsums_info("CUDA backend: toolkit ${CUDAToolkit_VERSION} (${CUDAToolkit_LIBRARY_ROOT})")
  einsums_info("CUDA backend: architectures ${CMAKE_CUDA_ARCHITECTURES}")

  include(Einsums_ExportTargets)

  if(NOT EINSUMS_FIND_PACKAGE)
    einsums_add_config_define(EINSUMS_HAVE_CUDA)
  endif()

endif()
