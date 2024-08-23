#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

# The goal is to store all EINSUMS_* cache variables in a file, so that they would be forwarded to projects using
# einsums (the file is included in the einsums-config.cmake)

get_cmake_property(cache_vars CACHE_VARIABLES)

# Keep only the EINSUMS_* like variables
list(FILTER cache_vars INCLUDE REGEX "^EINSUMS")
list(FILTER cache_vars EXCLUDE REGEX "Category$")

# Generate einsums_cache_variables.cmake in the BUILD directory
set(_cache_var_file ${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/einsums/EinsumsCacheVariables.cmake)
set(_cache_var_file_template "${EINSUMS_SOURCE_DIR}/cmake/templates/EinsumsCacheVariables.cmake.in")
set(_cache_variables)
foreach(_var IN LISTS cache_vars)
  set(_cache_variables "${_cache_variables}set(${_var} ${${_var}})\n")
endforeach()

configure_file(${_cache_var_file_template} ${_cache_var_file})

# Install the einsums_cache_variables.cmake in the INSTALL directory
install(
  FILES ${_cache_var_file}
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/einsums
  COMPONENT cmake)
