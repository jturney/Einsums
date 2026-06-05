#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(ErrorsHeaders Einsums/Errors/Error.hpp Einsums/Errors/ThrowException.hpp)

set(ErrorsSources ErrorNames.cpp)

include(Einsums_AddModule)
einsums_add_module(
  Einsums Errors
  SOURCES ${ErrorsSources}
  HEADERS ${ErrorsHeaders}
  MODULE_DEPENDENCIES Einsums_Config Einsums_TypeSupport
  CMAKE_SUBDIRS examples tests
)
