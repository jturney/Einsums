#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(linear_algebra_headers
        einsums/linear_algebra/base.hpp
        einsums/linear_algebra/block_tensor.hpp
        einsums/linear_algebra/detail/index_utilities.hpp
        einsums/linear_algebra/linear_algebra.hpp
        einsums/linear_algebra/tiled_tensor.hpp
        einsums/linear_algebra/unoptimized.hpp
)

set(linear_algebra_sources linear_algebra.cpp)

include(Einsums_AddModule)
einsums_add_module(
  einsums linear_algebra
  GLOBAL_HEADER_GEN ON
  SOURCES ${linear_algebra_sources}
  HEADERS ${linear_algebra_headers}
  DEPENDENCIES
  MODULE_DEPENDENCIES einsums_config einsums_blas einsums_concepts einsums_errors einsums_print einsums_assertion
  CMAKE_SUBDIRS examples tests
)
