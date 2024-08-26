#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(tensor_headers einsums/tensor/enumerate.hpp einsums/tensor/tensor.hpp)

set(tensor_sources)

include(Einsums_AddModule)
einsums_add_module(
        einsums tensor
        GLOBAL_HEADER_GEN ON
        SOURCES ${tensor_sources}
        HEADERS ${tensor_headers}
        DEPENDENCIES
        MODULE_DEPENDENCIES einsums_concepts einsums_config einsums_tensor_base einsums_profile einsums_print
        CMAKE_SUBDIRS examples tests
)
