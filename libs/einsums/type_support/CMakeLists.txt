#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(type_support_headers
        einsums/type_support/equality.hpp
        einsums/type_support/lazy_conditional.hpp
        einsums/type_support/type_identity.hpp
)

set(type_support_sources)

include(Einsums_AddModule)
einsums_add_module(
  einsums type_support
  GLOBAL_HEADER_GEN ON
  SOURCES ${type_support_sources}
  HEADERS ${type_support_headers}
  DEPENDENCIES
  CMAKE_SUBDIRS examples tests
)
