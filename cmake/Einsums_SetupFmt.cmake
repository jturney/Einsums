#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

include(FetchContent)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

fetchcontent_declare(
  fmt
  URL https://github.com/fmtlib/fmt/archive/refs/tags/12.2.0.tar.gz
  URL_HASH SHA256=8b852bb5aa6e7d8564f9e81394055395dd1d1936d38dfd3a17792a02bebd7af0
  # Require fmt 12. The conda clang 22 / macOS libc++ toolchain rejects the
  # older series: fmt 11.0.x trips a consteval FMT_STRING error under the new
  # clang, and fmt 11.1-11.2 added a detail::allocator that calls bare
  # malloc/free that fails against the libc++ which no longer exposes ::malloc
  # through <cstdlib>. fmt 12 fixes both. spdlog is built from source against
  # this external fmt (see Einsums_SetupSpdlog), because conda has no
  # fmt-12-compatible spdlog. fmt ships an AnyNewerVersion config, so the range
  # is bounded on both sides to stay on 12.x.
  FIND_PACKAGE_ARGS
  12...<13
)

fetchcontent_makeavailable(fmt)

target_link_libraries(einsums_base_libraries INTERFACE fmt::fmt)
