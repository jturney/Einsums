#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(BufferAllocatorHeaders
    Einsums/BufferAllocator/InitModule.hpp
    Einsums/BufferAllocator/Options.hpp Einsums/BufferAllocator/ModuleVars.hpp
    Einsums/BufferAllocator/BufferAllocator.hpp
)

set(BufferAllocatorSources BufferAllocator.cpp InitModule.cpp ModuleVars.cpp ForwardDeclarations.cpp)

if (TARGET einsums_dependencies_allocator)
    set(extra_deps einsums_dependencies_allocator)
endif()

include(Einsums_AddModule)
einsums_add_module(
  Einsums BufferAllocator
  SOURCES ${BufferAllocatorSources}
  HEADERS ${BufferAllocatorHeaders}
  DEPENDENCIES
  MODULE_DEPENDENCIES Einsums_Runtime Einsums_Logging Einsums_TypeSupport Einsums_Config
                      Einsums_RuntimeConfiguration Einsums_Options Einsums_Print
  CMAKE_SUBDIRS examples tests
  BASE_LIBNAME Einsums
)
target_link_libraries(Einsums_BufferAllocator PRIVATE ${extra_deps})