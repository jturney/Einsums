#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

include(FetchContent)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

fetchcontent_declare(
  fmt
  URL https://github.com/fmtlib/fmt/archive/refs/tags/11.0.2.tar.gz
  URL_HASH SHA256=6cb1e6d37bdcb756dbbe59be438790db409cdb4868c66e888d5df9f13f7c027f
  # Accept a system fmt in [11, 12) but never fmt 12+, which has breaking API
  # changes. fmt ships an AnyNewerVersion config, so a bare "11" would wrongly
  # accept fmt 12; the upper bound forces a fallback to the pinned 11.0.2.
  FIND_PACKAGE_ARGS
  11...<12
)

fetchcontent_makeavailable(fmt)

target_link_libraries(einsums_base_libraries INTERFACE fmt::fmt)
