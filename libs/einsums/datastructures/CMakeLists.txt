#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(datastructures_headers einsums/datastructures/member_pack.hpp einsums/datastructures/small_vector.hpp einsums/datastructures/variant.hpp)

set(datastructures_sources)

include(Einsums_AddModule)
einsums_add_module(
        einsums datastructures
        GLOBAL_HEADER_GEN ON
        SOURCES ${datastructures_sources}
        HEADERS ${datastructures_headers}
        DEPENDENCIES
        MODULE_DEPENDENCIES
        CMAKE_SUBDIRS examples tests
)
