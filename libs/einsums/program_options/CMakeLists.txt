# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(program_options_headers
    einsums/program_options/CmdLine.hpp
    einsums/program_options/Config.hpp
    einsums/program_options/Errors.hpp
    einsums/program_options/Option.hpp
    einsums/program_options/PositionalOptions.hpp
    einsums/program_options/ValueSemantic.hpp
    einsums/program_options/VariablesMap.hpp
)

set(program_options_sources PositionalOptions.cpp)

include(einsums_add_module)
einsums_add_module(
    einsums program_options
    GLOBAL_HEADER_GEN OFF
    SOURCES ${program_options_sources}
    HEADERS ${program_options_headers}
    MODULE_DEPENDENCIES einsums_config einsums_assertion
    DEPENDENCIES
    CMAKE_SUBDIRS examples tests
)
