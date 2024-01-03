# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(hptt_headers
    einsums/hptt/compute_node.hpp
    einsums/hptt/hptt_types.hpp
    einsums/hptt/hptt.hpp
    einsums/hptt/macros.hpp
    einsums/hptt/plan.hpp
    einsums/hptt/transpose.hpp
    einsums/hptt/utils.hpp
)

set(hptt_sources hptt.cpp plan.cpp transpose.cpp utils.cpp)

include(einsums_add_module)
einsums_add_module(
    einsums hptt
    GLOBAL_HEADER_GEN OFF
    SOURCES ${hptt_sources}
    HEADERS ${hptt_headers}
    MODULE_DEPENDENCIES
    DEPENDENCIES
    CMAKE_SUBDIRS examples tests
)
