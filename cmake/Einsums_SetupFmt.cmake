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
  # Accept only a system fmt in the 11.0 series. fmt 12+ has breaking API
  # changes our spdlog cannot consume, and fmt 11.1+ added a detail::allocator
  # that calls bare malloc/free, which fails to compile against the macOS
  # libc++ that no longer exposes ::malloc through <cstdlib>. fmt ships an
  # AnyNewerVersion config, so a bare "11" would wrongly accept those; the
  # tight upper bound keeps us on the pinned 11.0.2 (or a matching system 11.0.x).
  FIND_PACKAGE_ARGS
  11...<11.1
)

fetchcontent_makeavailable(fmt)

target_link_libraries(einsums_base_libraries INTERFACE fmt::fmt)
