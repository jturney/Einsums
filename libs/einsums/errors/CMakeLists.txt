#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(error_headers einsums/errors/error.hpp einsums/errors/error_code.hpp einsums/errors/exception.hpp
        einsums/errors/exception_fwd.hpp einsums/errors/throw_exception.hpp
)

set(error_sources error_code.cpp)

include(Einsums_AddModule)
einsums_add_module(
        einsums errors
        GLOBAL_HEADER_GEN ON
        SOURCES ${error_sources}
        HEADERS ${error_headers}
        DEPENDENCIES einsums_config
        CMAKE_SUBDIRS examples tests
)
