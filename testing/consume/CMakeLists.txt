#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.24)
project(sample)

# The major version this tree produces. The package is generated with
# COMPATIBILITY AnyNewerVersion, so asking for an older major still
# resolves and the test would pass against an install this tree did not
# build, which is the one thing it exists to catch.
find_package(Einsums 2 REQUIRED)

include(CMakePrintHelpers)
cmake_print_properties(
  TARGETS Einsums::Einsums
  PROPERTIES IMPORTED_LINK_DEPENDENT_LIBRARIES
             INTERFACE_LOCATION
             INTERFACE_LINK_LIBRARIES
             INTERFACE_INCLUDE_DIRECTORIES
             INTERFACE_COMPILE_DEFINITIONS
             INTERFACE_COMPILE_OPTIONS
             INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
)

add_executable(HelloWorld HelloWorld.cpp)
target_link_libraries(HelloWorld Einsums::Einsums)
