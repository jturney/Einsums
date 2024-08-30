#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(tensor_utilities_headers
        einsums/tensor_utilities/create_incremented_tensor.hpp
        einsums/tensor_utilities/create_random_definite.hpp
        einsums/tensor_utilities/create_random_semidefinite.hpp
        einsums/tensor_utilities/create_random_tensor.hpp
        einsums/tensor_utilities/create_tensor_like.hpp
        einsums/tensor_utilities/diagonal.hpp
)

set(tensor_utilities_sources)

include(Einsums_AddModule)
einsums_add_module(
  einsums tensor_utilities
  GLOBAL_HEADER_GEN ON
  SOURCES ${tensor_utilities_sources}
  HEADERS ${tensor_utilities_headers}
  DEPENDENCIES
  MODULE_DEPENDENCIES einsums_tensor_base einsums_tensor einsums_linear_algebra einsums_concepts
  CMAKE_SUBDIRS examples tests
)
