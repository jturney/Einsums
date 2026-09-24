#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(DecompositionHeaders
    Einsums/Decomposition/CP.hpp Einsums/Decomposition/Tucker.hpp Einsums/Decomposition/StringSpec/CP.hpp
    Einsums/Decomposition/StringSpec/Tucker.hpp Einsums/Decomposition/StringSpec/Unfold.hpp
)

set(DecompositionSources)

include(Einsums_AddModule)
einsums_add_module(
  Einsums Decomposition
  SOURCES ${DecompositionSources}
  HEADERS ${DecompositionHeaders}
  DEPENDENCIES
  MODULE_DEPENDENCIES
    Einsums_Config
    Einsums_LinearAlgebra
    Einsums_ComputeGraph
    Einsums_Concepts
    Einsums_Errors
    Einsums_Print
    Einsums_Profile
    Einsums_Tensor
    Einsums_TensorPermute
    Einsums_TensorUtilities
  CMAKE_SUBDIRS examples tests
)
