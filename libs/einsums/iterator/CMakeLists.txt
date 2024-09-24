#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(iterator_headers
        einsums/iterator/boost_iterator_categories.hpp
        einsums/iterator/counting_iterator.hpp
        einsums/iterator/counting_shape.hpp
        einsums/iterator/enumerate.hpp
        einsums/iterator/iterator_adaptor.hpp
        einsums/iterator/iterator_facade.hpp
        einsums/iterator/iterator_range.hpp
        einsums/iterator/range.hpp
        einsums/iterator/traits/is_iterator.hpp
        einsums/iterator/traits/is_range.hpp
        einsums/iterator/traits/is_sentinel_for.hpp
)

set(iterator_sources)

include(Einsums_AddModule)
einsums_add_module(
  einsums iterator
  GLOBAL_HEADER_GEN OFF
  SOURCES ${iterator_sources}
  HEADERS ${iterator_headers}
  DEPENDENCIES
  MODULE_DEPENDENCIES einsums_config einsums_type_support
  CMAKE_SUBDIRS examples tests
)
