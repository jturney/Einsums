#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

include(CMakePackageConfigHelpers)
include(einsums_generate_package_utils)

set(CMAKE_DIR
    "cmake-${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}"
    CACHE STRING "directory (in share), where to put Findeinsums cmake module"
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/einsums/einsums-config-version.cmake"
    VERSION ${EINSUMS_VERSION}
    COMPATIBILITY AnyNewerVersion
)

# Export einsums_internal_targets in the build directory
export(
    TARGETS ${EINSUMS_EXPORT_INTERNAL_TARGETS}
    NAMESPACE einsums_internal::
    FILE "${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/einsums/einsums_internal_targets.cmake"
)

# Export einsums_internal_targets in the install directory
install(
    EXPORT einsums_internal_targets
    NAMESPACE einsums_internal::
    FILE einsums_internal_targets.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/einsums
)

# Export einsums_targets in the build directory
export(
    TARGETS ${EINSUMS_EXPORT_TARGETS}
    NAMESPACE einsums::
    FILE "${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/einsums/einsums_targets.cmake"
)

# Add aliases with the namespace for use within einsums
foreach(export_target ${EINSUMS_EXPORT_TARGETS})
    add_library(einsums::${export_target} ALIAS ${export_target})
endforeach()

foreach(export_target ${EINSUMS_EXPORT_INTERNAL_TARGETS})
    add_library(einsums_internal::${export_target} ALIAS ${export_target})
endforeach()

# Export einsums_targets in the install directory
install(
    EXPORT einsums_targets
    NAMESPACE einsums::
    FILE einsums_targets.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/einsums
)

# Install dir
configure_file(
    cmake/templates/einsums-config.cmake.in "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/einsums-config.cmake"
    ESCAPE_QUOTES @ONLY
)
# Build dir
configure_file(
    cmake/templates/einsums-config.cmake.in "${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/einsums/einsums-config.cmake"
    ESCAPE_QUOTES @ONLY
)

# Configure macros for the install dir ...
set(EINSUMS_CMAKE_MODULE_PATH "\${CMAKE_CURRENT_LIST_DIR}")
configure_file(
    cmake/templates/einsums_macros.cmake.in "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/einsums_macros.cmake"
    ESCAPE_QUOTES @ONLY
)
# ... and the build dir
set(EINSUMS_CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
configure_file(
    cmake/templates/einsums_macros.cmake.in "${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/einsums/einsums_macros.cmake"
    ESCAPE_QUOTES @ONLY
)

install(
    FILES "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/einsums-config.cmake"
          "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/einsums_macros.cmake"
          "${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/einsums/einsums-config-version.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/einsums
    COMPONENT cmake
)
