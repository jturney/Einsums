#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(config_headers
        einsums/config.hpp
        einsums/config/attributes.hpp
        einsums/config/branch_hints.hpp
        einsums/config/compiler_fence.hpp
        einsums/config/compiler_specific.hpp
        einsums/config/emulate_deleted.hpp
        einsums/config/export_definitions.hpp
        einsums/config/forceinline.hpp
        einsums/config/namespace.hpp
        einsums/config/static_call_operator.hpp
)

set(config_sources version.cpp)

include(Einsums_AddModule)
einsums_add_module(
  einsums config CONFIG_FILES
  GLOBAL_HEADER_GEN OFF
  SOURCES ${config_sources}
  HEADERS ${config_headers}
  DEPENDENCIES
  MODULE_DEPENDENCIES einsums_preprocessor
  CMAKE_SUBDIRS examples tests
)
