#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

include(CMakePackageConfigHelpers)
include(Einsums_GeneratePackageUtils)

set(CMAKE_DIR
    "cmake-${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}"
    CACHE STRING "directory (in share), where to put Findeinsums cmake module")

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/${EINSUMS_PACKAGE_NAME}/EinsumsConfigVersion.cmake"
  VERSION ${EINSUMS_VERSION}
  COMPATIBILITY AnyNewerVersion)

# Export einsums_internal_targets in the build directory
export(
  TARGETS ${EINSUMS_EXPORT_INTERNAL_TARGETS}
  NAMESPACE EinsumsInternal::
  FILE "${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/${EINSUMS_PACKAGE_NAME}/EinsumsInternalTargets.cmake")

# Export einsums_internal_targets in the install directory
install(
  EXPORT einsums_internal_targets
  NAMESPACE EinsumsInternal::
  FILE einsums_internal_targets.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${EINSUMS_PACKAGE_NAME})

# Export einsums_targets in the build directory
export(
  TARGETS ${EINSUMS_EXPORT_TARGETS}
  NAMESPACE Einsums::
  FILE "${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/${EINSUMS_PACKAGE_NAME}/EinsumsTargets.cmake")

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
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${EINSUMS_PACKAGE_NAME}
  COMPONENT cmake)

# Install dir
configure_file(cmake/templates/EinsumsConfig.cmake.in
               "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/EinsumsConfig.cmake" ESCAPE_QUOTES @ONLY)
# Build dir
configure_file(cmake/templates/EinsumsConfig.cmake.in
               "${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/${EINSUMS_PACKAGE_NAME}/EinsumsConfig.cmake" ESCAPE_QUOTES @ONLY)

# Configure macros for the install dir ...
set(EINSUMS_CMAKE_MODULE_PATH "\${CMAKE_CURRENT_LIST_DIR}")
configure_file(cmake/templates/EinsumsMacros.cmake.in
               "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/EinsumsMacros.cmake" ESCAPE_QUOTES @ONLY)
# ... and the build dir
set(EINSUMS_CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
configure_file(cmake/templates/EinsumsMacros.cmake.in
               "${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/${EINSUMS_PACKAGE_NAME}/EinsumsMacros.cmake" ESCAPE_QUOTES @ONLY)

install(
  FILES "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/EinsumsConfig.cmake"
        "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/EinsumsMacros.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/${EINSUMS_PACKAGE_NAME}/EinsumsConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${EINSUMS_PACKAGE_NAME}
  COMPONENT cmake)
