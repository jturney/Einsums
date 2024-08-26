#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(profile_headers einsums/profile/section.hpp einsums/profile/timer.hpp)

set(profile_sources section.cpp timer.cpp)

include(Einsums_AddModule)
einsums_add_module(
        einsums profile
        GLOBAL_HEADER_GEN ON
        SOURCES ${profile_sources}
        HEADERS ${profile_headers}
        DEPENDENCIES
        MODULE_DEPENDENCIES einsums_config einsums_print einsums_itt_notify einsums_string_utils
        CMAKE_SUBDIRS examples tests
)
