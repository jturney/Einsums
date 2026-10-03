#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

# Waggle, the profiler Einsums and other libraries share. A project of its own, so it builds the
# same way here, added by Einsums, as it will from its own repository.

cmake_minimum_required(VERSION 3.24)

project(
  Waggle
  VERSION 0.1.0
  DESCRIPTION "A low-overhead tracing profiler that several libraries in one process share"
  LANGUAGES CXX
)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

if(NOT TARGET fmt::fmt)
  find_package(fmt 12...<13 REQUIRED)
endif()
find_package(Threads REQUIRED)

# Always shared, whatever BUILD_SHARED_LIBS says: every library in a process must reach the same
# collector, and a static copy linked into each would give each its own.
add_library(
  waggle SHARED
  src/Abi.cpp
  src/Consumer.cpp
  src/CounterBackend.cpp
  src/Diagnostics.cpp
  src/Duplicates.cpp
  src/Server.cpp
  src/Settings.cpp
  src/Snapshot.cpp
  src/Waggle.cpp
)
add_library(Waggle::waggle ALIAS waggle)

target_compile_features(waggle PUBLIC cxx_std_20)
target_include_directories(
  waggle
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_compile_definitions(waggle PRIVATE WAGGLE_EXPORTS)
target_link_libraries(waggle PUBLIC fmt::fmt PRIVATE Threads::Threads ${CMAKE_DL_LIBS})
if(WIN32)
  target_link_libraries(waggle PRIVATE ws2_32)
endif()
set_target_properties(
  waggle
  PROPERTIES CXX_VISIBILITY_PRESET hidden
             VISIBILITY_INLINES_HIDDEN ON
             VERSION ${PROJECT_VERSION}
             SOVERSION ${PROJECT_VERSION_MAJOR}
)

include(WaggleInstrument)

# The collector's private headers (rings, events, the consumer, the server), for Waggle's own
# tests. Never installed or exported: everything else reaches the collector through the C
# interface, so that nothing about these layouts becomes something a library depends on.
add_library(waggle_internal INTERFACE)
target_include_directories(waggle_internal INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(waggle_internal INTERFACE waggle)
add_library(Waggle::internal ALIAS waggle_internal)

# ── Tests ──────────────────────────────────────────────────────────────────
# On when Waggle is the project being built; a project that adds Waggle decides for itself, and may
# set a prefix that fits its own test names.
option(WAGGLE_BUILD_TESTS "Build Waggle's tests" ${PROJECT_IS_TOP_LEVEL})
set(WAGGLE_TEST_PREFIX
    "Waggle."
    CACHE STRING "Prefix of every Waggle test's name"
)
if(WAGGLE_BUILD_TESTS)
  if(NOT TARGET Catch2::Catch2WithMain)
    find_package(Catch2 3 REQUIRED)
  endif()
  if(PROJECT_IS_TOP_LEVEL)
    enable_testing()
  endif()
  add_subdirectory(tests)
endif()

# ── Install ──────────────────────────────────────────────────────────────────
install(
  TARGETS waggle
  EXPORT WaggleTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

set(_waggle_cmake_dir ${CMAKE_INSTALL_LIBDIR}/cmake/Waggle)
install(
  EXPORT WaggleTargets
  NAMESPACE Waggle::
  DESTINATION ${_waggle_cmake_dir}
)
# The build tree is a usable package too, for a project that builds Waggle and exports its own
# build tree.
export(
  EXPORT WaggleTargets
  NAMESPACE Waggle::
  FILE ${CMAKE_CURRENT_BINARY_DIR}/WaggleTargets.cmake
)
configure_file(cmake/WaggleInstrument.cmake ${CMAKE_CURRENT_BINARY_DIR}/WaggleInstrument.cmake COPYONLY)
configure_package_config_file(
  cmake/WaggleConfig.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/WaggleConfig.cmake INSTALL_DESTINATION ${_waggle_cmake_dir}
)
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/WaggleConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/WaggleConfig.cmake ${CMAKE_CURRENT_BINARY_DIR}/WaggleConfigVersion.cmake
              cmake/WaggleInstrument.cmake DESTINATION ${_waggle_cmake_dir}
)
