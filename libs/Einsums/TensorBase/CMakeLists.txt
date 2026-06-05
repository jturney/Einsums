#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(TensorBaseHeaders Einsums/TensorBase/Common.hpp Einsums/TensorBase/IndexUtilities.hpp
                      Einsums/TensorBase/SymmetryDescriptor.hpp Einsums/TensorBase/TensorBase.hpp
                      Einsums/TensorBase/HashFunctions.hpp
)

set(TensorBaseSources IndexUtilities.cpp SymmetryDescriptor.cpp)

include(Einsums_AddModule)
einsums_add_module(
  Einsums TensorBase
  SOURCES ${TensorBaseSources}
  HEADERS ${TensorBaseHeaders}
  DEPENDENCIES
  MODULE_DEPENDENCIES Einsums_Errors Einsums_StringUtil Einsums_Iterator
  CMAKE_SUBDIRS examples tests
)
