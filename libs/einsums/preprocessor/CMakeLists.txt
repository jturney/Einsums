# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(preprocessor_headers
    einsums/preprocess/cat.hpp
    einsums/preprocessor/config.hpp
    einsums/preprocessor/expand.hpp
    einsums/preprocessor/identity.hpp
    einsums/preprocessor/namespace.hpp
    einsums/preprocessor/nargs.hpp
    einsums/preprocessor/stringize.hpp
    einsums/preprocessor/strip_parens.hpp
)

# cmake-format: off
set(preprocessor_compat_headers)
# cmake-format: on

set(preprocessor_sources)

include(einsums_add_module)
einsums_add_module(
    einsums preprocessor
    SOURCES ${preprocessor_sources}
    HEADERS ${preprocessor_headers}
    CMAKE_SUBDIRS examples tests
)
