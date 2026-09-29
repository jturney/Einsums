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

# Fortran is optional: enable_language(Fortran OPTIONAL) is a documented
# placeholder that still fails without a compiler, so probe first.
include(CheckLanguage)
check_language(Fortran)
if (CMAKE_Fortran_COMPILER)
    enable_language(Fortran)
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
  # The Fortran compiler brings its own OpenMP runtime (libgomp for gfortran);
  # this is a separate executable, so it never meets the library's.
  find_package(OpenMP COMPONENTS Fortran)
  if (OpenMP_Fortran_FOUND)
    target_link_libraries(profile_fort_loop PRIVATE OpenMP::OpenMP_Fortran)
  endif ()
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

add_executable(profile_strategies profile_strategies.cpp)
target_link_libraries(profile_strategies PRIVATE Einsums::Einsums)

add_executable(profile_jk_breakdown profile_jk_breakdown.cpp)
target_link_libraries(profile_jk_breakdown PRIVATE Einsums::Einsums)
