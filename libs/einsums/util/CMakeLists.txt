#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(util_headers
        einsums/util/are_all_convertible.hpp
        einsums/util/arguments.hpp
        einsums/util/ios_flags_saver.hpp
        einsums/util/type_name.hpp
        einsums/util/manage_config.hpp
)

set(util_sources manage_config.cpp)

include(Einsums_AddModule)
einsums_add_module(
  einsums util
  GLOBAL_HEADER_GEN ON
  SOURCES ${util_sources}
  HEADERS ${util_headers}
  DEPENDENCIES
  MODULE_DEPENDENCIES
        einsums_config
        einsums_string_util
  CMAKE_SUBDIRS examples tests
)
