#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(threading_base_headers
        einsums/threading_base/callback_notifier.hpp
        einsums/threading_base/scheduler_mode.hpp
        einsums/threading_base/scheduler_state.hpp
)

set(threading_base_sources
        scheduler_mode.cpp
)

include(Einsums_AddModule)
einsums_add_module(
  einsums threading_base
  GLOBAL_HEADER_GEN ON
  SOURCES ${threading_base_sources}
  HEADERS ${threading_base_headers}
  DEPENDENCIES
  MODULE_DEPENDENCIES
        einsums_affinity
        einsums_allocator_support
        einsums_assertion
        einsums_config
        einsums_debugging
        einsums_errors
        einsums_functional
        einsums_logging
        einsums_type_support
  CMAKE_SUBDIRS examples tests
)
