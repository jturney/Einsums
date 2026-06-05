#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

# In your subproject's CMakeLists.txt
if (NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(PROFILING_IS_SUBDIR TRUE)
else ()
    set(PROFILING_IS_SUBDIR FALSE)
endif ()

cmake_minimum_required(VERSION 3.24)

project(profile_test C CXX)

# Set Fortran as optional
enable_language(Fortran OPTIONAL)

# Check if the Fortran compiler was found
if (CMAKE_Fortran_COMPILER)
    message(STATUS "Fortran compiler found: ${CMAKE_Fortran_COMPILER}")
    set(EINSUMS_HAVE_FORTRAN TRUE)
else ()
    message(STATUS "No Fortran compiler found; Fortran components will be disabled")
    set(EINSUMS_HAVE_FORTRAN FALSE)
endif ()

if (NOT PROFILING_IS_SUBDIR)
    find_package(Einsums 2 REQUIRED)
endif ()

add_executable(profile_for_loop profile_for_loop.c)
target_link_libraries(profile_for_loop PRIVATE m)

if (EINSUMS_HAVE_FORTRAN)
add_executable(profile_fort_loop profile_fort_loop.f03)
endif ()

# If you get an error about not being able to find libm, uncomment the commented out lines and replace
# the given directory with the appropriate one for your system. Conda's sysroot package's libraries
# point to /lib64 in many cases, which is the wrong directory at least on Debian.
add_executable(profile_einsums profile_einsums.cpp)
#target_link_directories(profile_einsums PRIVATE /usr/lib/x86_64-linux-gnu)
target_link_libraries(profile_einsums PRIVATE Einsums::Einsums)

add_executable(profile_blas profile_blas.cpp)
#target_link_directories(profile_blas PRIVATE /usr/lib/x86_64-linux-gnu)
target_link_libraries(profile_blas PRIVATE Einsums::Einsums)
