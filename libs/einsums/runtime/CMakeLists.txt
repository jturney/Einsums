# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(runtime_headers einsums/runtime/Runtime.hpp einsums/runtime/StartupFunction.hpp einsums/runtime/state.hpp)

set(runtime_sources Runtime.cpp)

include(einsums_add_module)
einsums_add_module(
    einsums runtime
    GLOBAL_HEADER_GEN ON
    SOURCES ${runtime_sources}
    HEADERS ${runtime_headers}
    MODULE_DEPENDENCIES einsums_assertion einsums_config einsums_debugging einsums_threading_base
    DEPENDENCIES
    CMAKE_SUBDIRS examples tests
)
