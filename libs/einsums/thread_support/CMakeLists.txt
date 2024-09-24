#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(thread_support_headers
  einsums/thread_support/assert_owns_lock.hpp
  einsums/thread_support/atomic_count.hpp
        einsums/thread_support/thread_name.hpp
  einsums/thread_support/spinlock.hpp
  einsums/thread_support/unlock_guard.hpp
)

set(thread_support_sources
        thread_name.cpp
  spinlock.cpp
)

include(Einsums_AddModule)
einsums_add_module(
  einsums thread_support
  GLOBAL_HEADER_GEN ON
  SOURCES ${thread_support_sources}
  HEADERS ${thread_support_headers}
  DEPENDENCIES
  MODULE_DEPENDENCIES
    einsums_assertion
    einsums_config
    einsums_concepts
    einsums_type_support
  CMAKE_SUBDIRS examples tests
)
