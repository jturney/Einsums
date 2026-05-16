#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

set(TensorIOHeaders
    Einsums/TensorIO/Format.hpp
    Einsums/TensorIO/TensorFile.hpp
    Einsums/TensorIO/DistributedTensorFile.hpp
    Einsums/TensorIO/Checkpoint.hpp
    Einsums/TensorIO/GraphIO.hpp
)

set(TensorIOSources
    TensorFile.cpp
    DistributedTensorFile.cpp
    Checkpoint.cpp
    InitModule.cpp
)

include(Einsums_AddModule)
einsums_add_module(
  Einsums TensorIO
  PYBIND
  SOURCES ${TensorIOSources}
  HEADERS ${TensorIOHeaders}
  MODULE_DEPENDENCIES
    Einsums_Comm
    Einsums_ComputeGraph
    Einsums_Config
    Einsums_Errors
    Einsums_Tensor
    Einsums_TensorBase
    Einsums_Profile
  CMAKE_SUBDIRS examples tests
)
