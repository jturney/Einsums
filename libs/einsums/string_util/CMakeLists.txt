# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(string_util_headers
    einsums/string_util/bad_lexical_cast.hpp
    einsums/string_util/case_conv.hpp
    einsums/string_util/classification.hpp
    einsums/string_util/from_string.hpp
    einsums/string_util/split.hpp
    einsums/string_util/to_string.hpp
    einsums/string_util/trim.hpp
)

set(string_util_sources bad_lexical_cast.cpp)

include(einsums_add_module)
einsums_add_module(
    einsums string_util
    GLOBAL_HEADER_GEN ON
    SOURCES ${string_util_sources}
    HEADERS ${string_util_headers}
    MODULE_DEPENDENCIES einsums_config
    DEPENDENCIES
    CMAKE_SUBDIRS examples tests
)
