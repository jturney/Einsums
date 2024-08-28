#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

set(testing_sources src/main.cpp)

add_library(einsums_testing ${testing_sources})
target_include_directories(einsums_testing PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(einsums_testing PRIVATE einsums_private_flags Catch2::Catch2)
target_link_libraries(einsums_testing PUBLIC Einsums::einsums)