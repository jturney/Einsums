#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(tensor_base_headers einsums/tensor_base/common.hpp einsums/tensor_base/tensor_base.hpp)

set(tensor_base_sources)

include(Einsums_AddModule)
einsums_add_module(
        einsums tensor_base
        GLOBAL_HEADER_GEN ON
        SOURCES ${tensor_base_sources}
        HEADERS ${tensor_base_headers}
        DEPENDENCIES
        MODULE_DEPENDENCIES einsums_string_util
        CMAKE_SUBDIRS examples tests
)
