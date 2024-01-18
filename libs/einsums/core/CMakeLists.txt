# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(core_headers einsums/core/Alias.hpp einsums/core/AlignedAllocator.hpp einsums/OpenMP.hpp einsums/Print.hpp
                 einsums/Section.hpp einsums/Timer.hpp
)

set(core_sources InitializeFinalize.cpp Print.cpp Section.cpp Timer.cpp)

include(einsums_add_module)
einsums_add_module(
    einsums core
    GLOBAL_HEADER_GEN OFF
    SOURCES ${core_sources}
    HEADERS ${core_headers}
    MODULE_DEPENDENCIES einsums_config einsums_preprocessor
    DEPENDENCIES
    CMAKE_SUBDIRS examples tests
)

include(einsums_extend_target)

# Handle OpenMP
einsums_extend_target(
    einsums core
    SOURCES OpenMP.c
    CONDITION NOT TARGET OpenMP::OpenMP_CXX
)

einsums_extend_target(einsums core PUBLIC_DEPENDENCIES OpenMP::OpenMP_CXX CONDITION TARGET OpenMP::OpenMP_CXX)
