#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

include(CMakePackageConfigHelpers)
include(Einsums_GeneratePackageUtils)

set(CMAKE_DIR
    "cmake"
    CACHE STRING "directory (in share), where to put FindEinsums cmake module"
)

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/${EINSUMS_PACKAGE_NAME}/EinsumsConfigVersion.cmake"
  VERSION ${EINSUMS_VERSION}
  COMPATIBILITY AnyNewerVersion
)

# Export einsums_internal_targets in the build directory
export(
  TARGETS ${EINSUMS_EXPORT_INTERNAL_TARGETS}
  NAMESPACE EinsumsInternal::
  FILE "${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/${EINSUMS_PACKAGE_NAME}/EinsumsInternalTargets.cmake"
)

# Export einsums_internal_targets in the install directory
install(
  EXPORT einsums_internal_targets
  NAMESPACE EinsumsInternal::
        FILE EinsumsInternalTargets.cmake
  DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${CMAKE_DIR}/${EINSUMS_PACKAGE_NAME}
)

# Export einsums_targets in the build directory
export(
  TARGETS ${EINSUMS_EXPORT_TARGETS}
  NAMESPACE Einsums::
  FILE "${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/${EINSUMS_PACKAGE_NAME}/EinsumsTargets.cmake"
)

# Add aliases with the namespace for use within einsums
foreach(export_target ${EINSUMS_EXPORT_TARGETS})
  add_library(Einsums::${export_target} ALIAS ${export_target})
endforeach()

foreach(export_target ${EINSUMS_EXPORT_INTERNAL_TARGETS})
  add_library(EinsumsInternal::${export_target} ALIAS ${export_target})
endforeach()

# Export einsums_targets in the install directory
install(
  EXPORT einsums_targets
  NAMESPACE Einsums::
  FILE EinsumsTargets.cmake
  DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${CMAKE_DIR}/${EINSUMS_PACKAGE_NAME}
  COMPONENT cmake
)

# Install dir - SPDLOG_INSTALL lands the vendored spdlog under the Einsums root,
# which EinsumsConfig locates relative to itself so the tree stays relocatable.
set(EINSUMS_BUILD_TREE_SPDLOG_DIR "")
configure_file(
  cmake/templates/EinsumsConfig.cmake.in
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/EinsumsConfig.cmake" ESCAPE_QUOTES @ONLY
)
# ... and the build dir, where that same spdlog is off in FetchContent's _deps
# instead. Nothing puts _deps on a consumer's search path, so without this the
# build-tree export falls through to whatever spdlog the surrounding environment
# ships - which is how an out-of-tree stage module ends up compiling against a
# different spdlog than the one already inside the libEinsums it links.
set(EINSUMS_BUILD_TREE_SPDLOG_DIR "${spdlog_BINARY_DIR}")
configure_file(
  cmake/templates/EinsumsConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/${EINSUMS_PACKAGE_NAME}/EinsumsConfig.cmake" ESCAPE_QUOTES
  @ONLY
)

# Configure macros for the install dir ...
set(EINSUMS_CMAKE_MODULE_PATH "\${CMAKE_CURRENT_LIST_DIR}")
configure_file(
  cmake/templates/EinsumsMacros.cmake.in
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/EinsumsMacros.cmake" ESCAPE_QUOTES @ONLY
)
# ... and the build dir
set(EINSUMS_CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
configure_file(
  cmake/templates/EinsumsMacros.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/${EINSUMS_PACKAGE_NAME}/EinsumsMacros.cmake" ESCAPE_QUOTES
  @ONLY
)

install(
  FILES "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/EinsumsConfig.cmake"
        "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/EinsumsMacros.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/${EINSUMS_PACKAGE_NAME}/EinsumsConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${CMAKE_DIR}/${EINSUMS_PACKAGE_NAME}
  COMPONENT cmake
)
