# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(debugging_headers
    einsums/debugging/demangle_helper.hpp einsums/debugging/backtrace.hpp einsums/debugging/environ.hpp
    einsums/debugging/backtrace/backtrace.hpp einsums/debugging/print.hpp einsums/debugging/attach_debugger.hpp
)

set(debugging_sources attach_debugger.cpp backtrace.cpp print.cpp)

include(einsums_add_module)
einsums_add_module(
    einsums debugging
    GLOBAL_HEADER_GEN ON
    SOURCES ${debugging_sources}
    HEADERS ${debugging_headers}
    EXCLUDE_FROM_GLOBAL_HEADER "einsums/debugging/backtrace/backtrace.hpp"
    MODULE_DEPENDENCIES einsums_config
    DEPENDENCIES
    CMAKE_SUBDIRS examples tests
)
