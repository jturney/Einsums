#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(iterator_support_headers
        einsums/iterator_support/boost_iterator_categories.hpp
        einsums/iterator_support/iterator_facade.hpp
        einsums/iterator_support/traits/is_iterator.hpp
)

set(iterator_support_sources)

include(Einsums_AddModule)
einsums_add_module(
  einsums iterator_support
  GLOBAL_HEADER_GEN ON
  SOURCES ${iterator_support_sources}
  HEADERS ${iterator_support_headers}
  DEPENDENCIES
  MODULE_DEPENDENCIES einsums_config einsums_type_support
  CMAKE_SUBDIRS examples tests
)
