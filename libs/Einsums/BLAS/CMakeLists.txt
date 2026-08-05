#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(BLASHeaders Einsums/BLAS.hpp Einsums/BLAS/InitModule.hpp Einsums/BLAS/ThreadControl.hpp)

set(BLASSources BLAS.cpp InitModule.cpp ThreadControl.cpp)

include(Einsums_AddModule)
einsums_add_module(
  Einsums BLAS
  SOURCES ${BLASSources}
  HEADERS ${BLASHeaders}
  # ThreadControl resolves the vendor's thread setter through dlsym, which
  # lives in libdl on glibc older than 2.34. Empty on Windows and macOS.
  DEPENDENCIES ${CMAKE_DL_LIBS}
  MODULE_DEPENDENCIES Einsums_Config Einsums_Concepts Einsums_BLASBase Einsums_BLASVendor
                      Einsums_Runtime
  CMAKE_SUBDIRS examples tests
)
