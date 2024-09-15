#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(debugging_headers
        einsums/debugging/attach_debugger.hpp
        einsums/debugging/backtrace.hpp
        einsums/debugging/detail/backtrace.hpp
        einsums/debugging/demangle_helper.hpp
        einsums/debugging/environ.hpp
)

set(debugging_sources attach_debugger.cpp backtrace.cpp)

include(Einsums_AddModule)
einsums_add_module(
  einsums debugging
  GLOBAL_HEADER_GEN ON
  SOURCES ${debugging_sources}
  HEADERS ${debugging_headers}
  DEPENDENCIES
  MODULE_DEPENDENCIES einsums_config
  CMAKE_SUBDIRS examples tests
)
