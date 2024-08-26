#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(errors_headers
        einsums/errors/error.hpp
        einsums/errors/error_code.hpp
        einsums/errors/exception.hpp
        einsums/errors/exception_fwd.hpp
        einsums/errors/exception_info.hpp
        einsums/errors/throw_exception.hpp
)

set(errors_sources error_code.cpp exception.cpp throw_exception.cpp)

include(Einsums_AddModule)
einsums_add_module(
  einsums errors
  GLOBAL_HEADER_GEN ON
  SOURCES ${errors_sources}
  HEADERS ${errors_headers}
  MODULE_DEPENDENCIES
        einsums_assertion einsums_config einsums_preprocessor
  CMAKE_SUBDIRS examples tests
)
