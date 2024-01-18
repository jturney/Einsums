#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

add_library(einsums_testing STATIC src/testing.cpp)
target_include_directories(einsums_testing PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(einsums_testing PRIVATE einsums_private_flags)
target_link_libraries(einsums_testing PUBLIC einsums::einsums snitch::snitch)
