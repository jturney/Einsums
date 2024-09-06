#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(tensor_headers
        einsums/tensor/arithmetic_tensor.hpp
        einsums/tensor/block_tensor.hpp
        einsums/tensor/detail/hash.hpp
        einsums/tensor/H5.hpp
        einsums/tensor/tensor.hpp
        einsums/tensor/tensor_fwd.hpp
        einsums/tensor/tiled_tensor.hpp
)

set(tensor_sources)

include(Einsums_AddModule)
einsums_add_module(
        einsums tensor
        GLOBAL_HEADER_GEN ON
        SOURCES ${tensor_sources}
        HEADERS ${tensor_headers}
        EXCLUDE_FROM_GLOBAL_HEADER
            "einsums/tensor/H5.hpp"
            "einsums/tensor/tensor_fwd.hpp"
        DEPENDENCIES Einsums::h5cpp
        MODULE_DEPENDENCIES
            einsums_assertion
            einsums_concepts
            einsums_config
            einsums_errors
            einsums_tensor_base
            einsums_profile
            einsums_print
            einsums_util
            einsums_iterator
        CMAKE_SUBDIRS examples tests
)
