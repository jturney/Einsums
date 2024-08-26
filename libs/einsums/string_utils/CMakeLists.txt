#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(string_utils_headers einsums/string_utils/trim.hpp)

set(string_utils_sources)

include(Einsums_AddModule)
einsums_add_module(
        einsums string_utils
        GLOBAL_HEADER_GEN ON
        SOURCES ${string_utils_sources}
        HEADERS ${string_utils_headers}
        DEPENDENCIES
        CMAKE_SUBDIRS examples tests
)
