#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(runtime_configuration_headers
        einsums/runtime_configuration/init_ini_data.hpp
        einsums/runtime_configuration/runtime_configuration.hpp
        einsums/runtime_configuration/runtime_configuration_fwd.hpp
)

set(runtime_configuration_sources init_ini_data.cpp runtime_configuration.cpp)

include(Einsums_AddModule)
einsums_add_module(
  einsums runtime_configuration
  GLOBAL_HEADER_GEN ON
  SOURCES ${runtime_configuration_sources}
  HEADERS ${runtime_configuration_headers}
  DEPENDENCIES
  MODULE_DEPENDENCIES
        einsums_program_options
        einsums_errors
        einsums_filesystem
        einsums_itt_notify
        einsums_logging
        einsums_ini
        einsums_coroutines
        einsums_version
        einsums_synchronization
  CMAKE_SUBDIRS examples tests
)
