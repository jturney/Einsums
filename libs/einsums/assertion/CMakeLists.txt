# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(assertion_headers einsums/Assert.hpp einsums/assertion/CurrentFunction.hpp einsums/assertion/EvaluateAssert.hpp
                      einsums/assertion/SourceLocation.hpp
)

set(assertion_sources Assertion.cpp SourceLocation.cpp)

include(einsums_add_module)
einsums_add_module(
    einsums assertion
    GLOBAL_HEADER_GEN ON
    SOURCES ${assertion_sources}
    HEADERS ${assertion_headers}
    MODULE_DEPENDENCIES einsums_config einsums_preprocessor
    CMAKE_SUBDIRS examples tests
)
