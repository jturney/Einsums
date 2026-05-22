#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(TensorHeaders
    Einsums/Tensor/BlockTensor.hpp
    Einsums/Tensor/DiskTensor.hpp
    Einsums/Tensor/Tensor.hpp
    Einsums/Tensor/TensorForward.hpp
    Einsums/Tensor/ArithmeticTensor.hpp
    Einsums/Tensor/FunctionTensor.hpp
    Einsums/Tensor/PendingInit.hpp
    Einsums/Tensor/RuntimeTensor.hpp
    Einsums/Tensor/SymmetryOps.hpp
    Einsums/Tensor/TiledTensor.hpp
    Einsums/Tensor/TiledRuntimeTensor.hpp
    Einsums/Tensor/InitModule.hpp
    Einsums/Tensor/ModuleVars.hpp
)

set(TensorSources TensorDefs.cpp Requirements.cpp InitModule.cpp ModuleVars.cpp)

include(Einsums_AddModule)
einsums_add_module(
  Einsums Tensor
  PYBIND
  SOURCES ${TensorSources}
  HEADERS ${TensorHeaders}
  DEPENDENCIES tgt::hdf5 ZLIB::ZLIB
  MODULE_DEPENDENCIES
    Einsums_BLAS
    Einsums_BufferAllocator
    Einsums_Concepts
    Einsums_Config
    Einsums_Errors
    Einsums_GPU
    Einsums_Iterator
    Einsums_Logging
    Einsums_TensorBase
    Einsums_TensorImpl
    Einsums_TypeSupport
    Einsums_Profile
    Einsums_Print
    Einsums_Runtime
    Einsums_Utilities
  CMAKE_SUBDIRS examples tests
)
