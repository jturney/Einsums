#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(itt_notify_headers einsums/itt_notify.hpp)

set(itt_notify_sources itt_notify.cpp)

if (EINSUMS_WITH_ITTNOTIFY)
  find_package(Amplifier)

  if (NOT AMPLIFIER_FOUND)
    einsums_error(
            "Intel Amplifier could not be found and EINSUMS_WITH_ITTNOTIFY=On, please specify AMPLIFIER_ROOT to point to the root of your Amplifier installation"
    )
  endif ()

  einsums_add_config_define(EINSUMS_HAVE_ITTNOTIFY)
endif ()

if (TARGET einsums_internal::amplifier)
  set(itt_notify_optional_dependencies einsums_internal::amplifier)
endif ()

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
