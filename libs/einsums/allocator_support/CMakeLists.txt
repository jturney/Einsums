#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(allocator_support_headers
    einsums/allocator_support/aligned_allocator.hpp
    einsums/allocator_support/allocator_deleter.hpp
    einsums/allocator_support/internal_allocator.hpp
    einsums/allocator_support/traits/is_allocator.hpp
)

set(allocator_support_sources)

include(Einsums_AddModule)
einsums_add_module(
  einsums allocator_support
  GLOBAL_HEADER_GEN ON
  SOURCES ${allocator_support_sources}
  HEADERS ${allocator_support_headers}
  DEPENDENCIES
  MODULE_DEPENDENCIES
  CMAKE_SUBDIRS examples tests
)
