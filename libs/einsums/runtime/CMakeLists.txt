#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(runtime_headers
        einsums/runtime/config_entry.hpp
        einsums/runtime/custom_exception_info.hpp
        einsums/runtime/debugging.hpp
        einsums/runtime/get_os_thread_count.hpp
        einsums/runtime/get_thread_name.hpp
        einsums/runtime/report_error.hpp
        einsums/runtime/runtime.hpp
        einsums/runtime/runtime_fwd.hpp
        einsums/runtime/runtime_handlers.hpp
        einsums/runtime/shutdown_function.hpp
        einsums/runtime/startup_function.hpp
)

set(runtime_sources
        custom_exception_info.cpp
        debugging.cpp
        runtime.cpp
        runtime_handlers.cpp
)

include(Einsums_AddModule)
einsums_add_module(
  einsums runtime
  GLOBAL_HEADER_GEN ON
  SOURCES ${runtime_sources}
  HEADERS ${runtime_headers}
  DEPENDENCIES
  MODULE_DEPENDENCIES
        einsums_config
        einsums_debugging
        einsums_errors
        einsums_runtime_configuration
        einsums_string_util
        einsums_threading_base
        einsums_topology
        einsums_version
  CMAKE_SUBDIRS examples tests
)
