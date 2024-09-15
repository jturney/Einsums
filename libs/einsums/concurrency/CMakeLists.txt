#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(concurrency_headers
  einsums/concurrency/barrier.hpp
)

set(concurrency_sources barrier.cpp)

include(Einsums_AddModule)
einsums_add_module(
  einsums concurrency
  GLOBAL_HEADER_GEN ON
  SOURCES ${concurrency_sources}
  HEADERS ${concurrency_headers}
  DEPENDENCIES
  MODULE_DEPENDENCIES
        einsums_assertion
        einsums_concepts
        einsums_config
        einsums_execution_base
        einsums_errors
        einsums_lock_registration
        einsums_thread_support
  CMAKE_SUBDIRS examples tests
)
