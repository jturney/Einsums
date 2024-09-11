#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(init_runtime_headers
        einsums/init_runtime/detail/init_logging.hpp
        einsums/init_runtime/init_runtime.hpp
        einsums/init_runtime/scoped_finalize.hpp
        einsums/init.hpp
)

set(init_runtime_sources
        init_logging.cpp
        init_runtime.cpp
        scoped_finalize.cpp
)

include(Einsums_AddModule)
einsums_add_module(
  einsums init_runtime
  GLOBAL_HEADER_GEN ON
  SOURCES ${init_runtime_sources}
  HEADERS ${init_runtime_headers}
  DEPENDENCIES
  MODULE_DEPENDENCIES
        einsums_debugging
        einsums_config
        einsums_command_line_handling
        einsums_errors
        einsums_filesystem
        einsums_functional
        einsums_logging
        einsums_runtime
        einsums_runtime_configuration
        einsums_preprocessor
        einsums_program_options
        einsums_util
  CMAKE_SUBDIRS examples tests
)
