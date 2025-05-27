#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(TensorUtilitiesHeaders
    Einsums/TensorUtilities/ARange.hpp
    Einsums/TensorUtilities/CreateIdentity.hpp
    Einsums/TensorUtilities/CreateIncrementedTensor.hpp
    Einsums/TensorUtilities/CreateOnesTensor.hpp
    Einsums/TensorUtilities/CreateRandomDefinite.hpp
    Einsums/TensorUtilities/CreateRandomSemidefinite.hpp
    Einsums/TensorUtilities/CreateRandomTensor.hpp
    Einsums/TensorUtilities/CreateTensorLike.hpp
    Einsums/TensorUtilities/CreateZeroTensor.hpp
    Einsums/TensorUtilities/Diagonal.hpp
    Einsums/TensorUtilities/Detail/SetTo.hpp
    Einsums/TensorUtilities/RMSD.hpp
)

set(TensorUtilitiesSources)

include(Einsums_AddModule)
einsums_add_module(
  Einsums TensorUtilities
  SOURCES ${TensorUtilitiesSources}
  HEADERS ${TensorUtilitiesHeaders}
  DEPENDENCIES
  MODULE_DEPENDENCIES Einsums_TensorBase Einsums_Tensor Einsums_LinearAlgebra Einsums_Concepts
                      Einsums_Utilities Einsums_Logging
  CMAKE_SUBDIRS examples tests
)
