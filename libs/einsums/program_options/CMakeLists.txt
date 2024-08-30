# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Default location is $EINSUMS_ROOT/libs/program_options/include
set(program_options_headers
    einsums/program_options.hpp
    einsums/modules/program_options.hpp
    einsums/program_options/cmdline.hpp
    einsums/program_options/config.hpp
    einsums/program_options/environment_iterator.hpp
    einsums/program_options/eof_iterator.hpp
    einsums/program_options/errors.hpp
    einsums/program_options/option.hpp
    einsums/program_options/options_description.hpp
    einsums/program_options/parsers.hpp
    einsums/program_options/positional_options.hpp
    einsums/program_options/value_semantic.hpp
    einsums/program_options/variables_map.hpp
    einsums/program_options/version.hpp
    einsums/program_options/detail/cmdline.hpp
    einsums/program_options/detail/config_file.hpp
    einsums/program_options/detail/convert.hpp
    einsums/program_options/detail/parsers.hpp
    einsums/program_options/detail/utf8_codecvt_facet.hpp
    einsums/program_options/detail/value_semantic.hpp
)

# Default location is $EINSUMS_ROOT/libs/program_options/src
set(program_options_sources
    cmdline.cpp
    config_file.cpp
    convert.cpp
    options_description.cpp
    parsers.cpp
    positional_options.cpp
    split.cpp
    utf8_codecvt_facet.cpp
    value_semantic.cpp
    variables_map.cpp
    winmain.cpp
)

include(Einsums_AddModule)
einsums_add_module(
  einsums program_options
  GLOBAL_HEADER_GEN OFF
  SOURCES ${program_options_sources}
  HEADERS ${program_options_headers}
  MODULE_DEPENDENCIES einsums_config einsums_assertion einsums_iterator einsums_string_util
#        einsums_datastructures einsums_iterator
  CMAKE_SUBDIRS examples tests
)
