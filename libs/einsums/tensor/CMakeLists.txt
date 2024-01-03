# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(tensor_headers einsums/Tensor.hpp einsums/tensor/Types.hpp)

set(tensor_sources)

include(einsums_add_module)
einsums_add_module(
    einsums tensor
    GLOBAL_HEADER_GEN OFF
    SOURCES ${tensor_sources}
    HEADERS ${tensor_headers}
    MODULE_DEPENDENCIES einsums_core einsums_config einsums_preprocessor einsums_concepts
    DEPENDENCIES
    CMAKE_SUBDIRS examples tests
)
