#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(execution_base_headers
  einsums/execution_base/agent_base.hpp
  einsums/execution_base/agent_ref.hpp
  einsums/execution_base/context_base.hpp
  einsums/execution_base/detail/spinlock_deadlock_detection.hpp
  einsums/execution_base/resource_base.hpp
  einsums/execution_base/this_thread.hpp
)

set(execution_base_sources
  agent_ref.cpp
  spinlock_deadlock_detection.cpp
  this_thread.cpp
)

include(Einsums_AddModule)
einsums_add_module(
  einsums execution_base
  GLOBAL_HEADER_GEN ON
  SOURCES ${execution_base_sources}
  HEADERS ${execution_base_headers}
  DEPENDENCIES
  MODULE_DEPENDENCIES
    einsums_assertion
    einsums_config
    einsums_errors
    einsums_functional
    einsums_lock_registration
    einsums_logging
    einsums_timing
    einsums_type_support
  CMAKE_SUBDIRS examples tests
)
