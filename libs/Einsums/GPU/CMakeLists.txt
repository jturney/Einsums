#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

# GPU abstraction module — always builds.
# Uses mock (CPU) backend when no GPU is available, CUDA or HIP when enabled.

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(GPUHeaders
    Einsums/GPU/Platform.hpp
    Einsums/GPU/Error.hpp
    Einsums/GPU/Types.hpp
    Einsums/GPU/Runtime.hpp
    Einsums/GPU/Stream.hpp
    Einsums/GPU/BLAS.hpp
    Einsums/GPU/Solver.hpp
    Einsums/GPU/DeviceVector.hpp
    Einsums/GPU/MPSBackend.hpp
)

set(GPUSources
    Runtime.cpp
    Stream.cpp
    BLAS.cpp
    Solver.cpp
)

# Vendor-specific dependencies and MPS Objective-C++ sources
if(EINSUMS_WITH_CUDA)
  set(GPU_VENDOR_DEPS CUDA::cublas CUDA::cusolver CUDA::cudart)
elseif(EINSUMS_WITH_HIP)
  set(GPU_VENDOR_DEPS hip::host hip::device roc::hipblas roc::hipsolver)
elseif(EINSUMS_WITH_MPS)
  list(APPEND GPUSources MPSBackend.mm)
  set(GPU_VENDOR_DEPS einsums_mps)
else()
  # Mock backend: no vendor deps, uses CPU BLAS
  set(GPU_VENDOR_DEPS)
endif()

include(Einsums_AddModule)
einsums_add_module(
  Einsums GPU
  SOURCES ${GPUSources}
  HEADERS ${GPUHeaders}
  DEPENDENCIES ${GPU_VENDOR_DEPS}
  MODULE_DEPENDENCIES Einsums_Config Einsums_CXX23 Einsums_Errors Einsums_Logging Einsums_BLAS Einsums_BLASVendor
  CMAKE_SUBDIRS examples tests
)

# MPS: Objective-C++ files cannot use C++ precompiled headers.
# Disable PCH for the .mm file and enable ObjC compilation.
if(EINSUMS_WITH_MPS)
  set_source_files_properties(
    ${CMAKE_CURRENT_SOURCE_DIR}/src/MPSBackend.mm
    PROPERTIES
      SKIP_PRECOMPILE_HEADERS ON
      COMPILE_FLAGS "-fobjc-arc"
  )
endif()
