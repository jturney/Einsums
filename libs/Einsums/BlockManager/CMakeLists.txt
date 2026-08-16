#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(BlockManagerHeaders Einsums/BlockManager/BlockManager.hpp)

set(BlockManagerSources  BlockManager.cpp)

if(EINSUMS_WITH_GPU_SUPPORT)
set(EinsumsBlockAllocatorGPUDeps Einsums_GPUMemory)
endif()

include(Einsums_AddModule)
einsums_add_module(
  Einsums BlockManager
  SOURCES ${BlockManagerSources}
  HEADERS ${BlockManagerHeaders}
  DEPENDENCIES 
  MODULE_DEPENDENCIES Einsums_Runtime Einsums_Logging Einsums_TypeSupport Einsums_Config Einsums_Options Einsums_BufferAllocator ${EinsumsBlockAllocatorGPUDeps}
  CMAKE_SUBDIRS examples tests
  BASE_LIBNAME Einsums
)

        