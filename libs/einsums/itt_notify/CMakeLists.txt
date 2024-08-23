#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(itt_notify_headers einsums/itt_notify.hpp)

set(itt_notify_sources itt_notify.cpp)

include(Einsums_AddModule)
einsums_add_module(
  einsums itt_notify
  GLOBAL_HEADER_GEN OFF
  SOURCES ${itt_notify_sources}
  HEADERS ${itt_notify_headers}
  DEPENDENCIES
  MODULE_DEPENDENCIES einsums_config
  CMAKE_SUBDIRS examples tests
)
