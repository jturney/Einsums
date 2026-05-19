#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(LinearAlgebraHeaders
    Einsums/LinearAlgebra.hpp Einsums/LinearAlgebra/Base.hpp Einsums/LinearAlgebra/BlockTensor.hpp
    Einsums/LinearAlgebra/TiledTensor.hpp Einsums/LinearAlgebra/Bases/dot.hpp
    Einsums/LinearAlgebra/Bases/direct_product.hpp Einsums/LinearAlgebra/Bases/gemm.hpp Einsums/LinearAlgebra/Bases/gemv.hpp
    Einsums/LinearAlgebra/Bases/ger.hpp Einsums/LinearAlgebra/Bases/compensated_sum.hpp Einsums/LinearAlgebra/Bases/norm.hpp Einsums/LinearAlgebra/Bases/sum_square.hpp
    Einsums/LinearAlgebra/Bases/syev.hpp
    Einsums/LinearAlgebra/DiskAlgebra.hpp
)

set(LinearAlgebraSources LinearAlgebra.cpp ForwardDeclarations.cpp)

include(Einsums_AddModule)
einsums_add_module(
  Einsums LinearAlgebra
  PYBIND
  SOURCES ${LinearAlgebraSources}
  HEADERS ${LinearAlgebraHeaders}
  MODULE_DEPENDENCIES
    Einsums_Assertion
    Einsums_Config
    Einsums_BLAS
    Einsums_Concepts
    Einsums_Errors
    Einsums_Print
    Einsums_Profile
    Einsums_Tensor
    Einsums_TensorUtilities
    Einsums_TensorImpl
    Einsums_TypeSupport
  CMAKE_SUBDIRS examples tests
)
