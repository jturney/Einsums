#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(TensorAlgebraHeaders
    Einsums/TensorAlgebra.hpp
    Einsums/TensorAlgebra/Permute.hpp
    Einsums/TensorAlgebra/TensorAlgebra.hpp
    Einsums/TensorAlgebra/Backends/BaseAlgebra.hpp
    Einsums/TensorAlgebra/Backends/BlockAlgebra.hpp
    Einsums/TensorAlgebra/Backends/BlockTileAlgebra.hpp
    Einsums/TensorAlgebra/Backends/Dispatch.hpp
    Einsums/TensorAlgebra/Backends/ElementTransform.hpp
    Einsums/TensorAlgebra/Backends/GenericAlgorithm.hpp
    Einsums/TensorAlgebra/Backends/KhatriRao.hpp
    Einsums/TensorAlgebra/Backends/TileAlgebra.hpp
    Einsums/TensorAlgebra/Backends/Unfold.hpp
    Einsums/TensorAlgebra/Detail/HpttPlanCache.hpp
    Einsums/TensorAlgebra/Detail/Index.hpp
    Einsums/TensorAlgebra/Detail/Utilities.hpp
)

set(TensorAlgebraSources Permute.cpp)

include(Einsums_AddModule)
einsums_add_module(
  Einsums TensorAlgebra
  SOURCES ${TensorAlgebraSources}
  HEADERS ${TensorAlgebraHeaders}
  MODULE_DEPENDENCIES
    Einsums_Comm
    Einsums_Config
    Einsums_LinearAlgebra
    Einsums_BufferAllocator
    Einsums_Concepts
    Einsums_Errors
    Einsums_Logging
    Einsums_Print
    Einsums_Profile
    Einsums_Tensor
    Einsums_TensorUtilities
    Einsums_PackedGemm
    Einsums_HPTT
  CMAKE_SUBDIRS examples tests
)
