#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(BufferAllocatorHeaders
    Einsums/BufferAllocator/InitModule.hpp
    Einsums/BufferAllocator/Options.hpp Einsums/BufferAllocator/ModuleVars.hpp
    Einsums/BufferAllocator/BufferAllocator.hpp Einsums/BufferAllocator/MemoryPool.hpp
)

set(BufferAllocatorSources BufferAllocator.cpp InitModule.cpp MemoryPool.cpp ModuleVars.cpp ForwardDeclarations.cpp)

include(Einsums_AddModule)
einsums_add_module(
  Einsums BufferAllocator
  SOURCES ${BufferAllocatorSources}
  HEADERS ${BufferAllocatorHeaders}
  DEPENDENCIES
  MODULE_DEPENDENCIES Einsums_Runtime Einsums_Logging Einsums_TypeSupport Einsums_Config
                      Einsums_Options Einsums_Print
  CMAKE_SUBDIRS examples tests
  BASE_LIBNAME Einsums
  PYBIND
)
# mimalloc is an implementation detail of this module: MemoryPool.hpp keeps the
# mi_* types opaque, so nothing downstream needs the headers or the link.
target_link_libraries(Einsums_BufferAllocator PRIVATE einsums_dependencies_mimalloc)