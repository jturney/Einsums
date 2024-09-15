#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(lock_registration_headers einsums/lock_registration/detail/register_locks.hpp)

set(lock_registration_sources register_locks.cpp)

include(Einsums_AddModule)
einsums_add_module(
  einsums lock_registration
  GLOBAL_HEADER_GEN ON
  SOURCES ${lock_registration_sources}
  HEADERS ${lock_registration_headers}
  DEPENDENCIES
  MODULE_DEPENDENCIES
    einsums_assertion
    einsums_concepts
    einsums_config
    einsums_errors
    einsums_functional
    einsums_type_support
  CMAKE_SUBDIRS examples tests
)
