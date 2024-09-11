#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(command_line_handling_headers
        einsums/command_line_handling/command_line_handling.hpp
        einsums/command_line_handling/get_env_var_as.hpp
        einsums/command_line_handling/late_command_line_handling.hpp
        einsums/command_line_handling/parse_command_line.hpp
)

set(command_line_handling_sources command_line_handling.cpp late_command_line_handling.cpp
        parse_command_line.cpp
)

include(Einsums_AddModule)
einsums_add_module(
  einsums command_line_handling
  GLOBAL_HEADER_GEN ON
  SOURCES ${command_line_handling_sources}
  HEADERS ${command_line_handling_headers}
  DEPENDENCIES
  MODULE_DEPENDENCIES
        einsums_config
        einsums_program_options
        einsums_runtime_configuration
        einsums_debugging
        einsums_errors
        einsums_filesystem
        einsums_topology
        einsums_util
        einsums_version
  CMAKE_SUBDIRS examples tests
)
