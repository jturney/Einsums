# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(tensor_algebra_headers einsums/Sort.hpp einsums/tensor_algebra/Utilities.hpp)

set(tensor_algebra_sources Sort.cpp)

include(einsums_add_module)
einsums_add_module(
    einsums tensor_algebra
    GLOBAL_HEADER_GEN OFF
    SOURCES ${tensor_algebra_sources}
    HEADERS ${tensor_algebra_headers}
    MODULE_DEPENDENCIES
        einsums_core
        einsums_config
        einsums_preprocessor
        einsums_concepts
        einsums_tensor
        einsums_linear_algebra
        einsums_hptt
    DEPENDENCIES
    CMAKE_SUBDIRS examples tests
)
