#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

# These are a dummy targets that we add compile flags to. All einsums targets should link to them.
add_library(einsums_private_flags INTERFACE)
add_library(einsums_public_flags INTERFACE)

# Set C++ standard
target_compile_features(einsums_private_flags INTERFACE cxx_std_${EINSUMS_WITH_CXX_STANDARD})
target_compile_features(einsums_public_flags INTERFACE cxx_std_${EINSUMS_WITH_CXX_STANDARD})

# Set other flags that should always be set

# EINSUMS_DEBUG must be set without a generator expression as it determines ABI compatibility. Projects in Release mode
# using einsums in Debug mode must have EINSUMS_DEBUG set, and projects in Debug mode using einsums in Release mode must
# not have EINSUMS_DEBUG set. EINSUMS_DEBUG must also not be set by projects using einsums.
if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
  target_compile_definitions(einsums_private_flags INTERFACE EINSUMS_DEBUG)
  target_compile_definitions(einsums_public_flags INTERFACE EINSUMS_DEBUG)
endif()

target_compile_definitions(
  einsums_private_flags
  INTERFACE $<$<CONFIG:MinSizeRel>:NDEBUG>
  INTERFACE $<$<CONFIG:Release>:NDEBUG>
  INTERFACE $<$<CONFIG:RelWithDebInfo>:NDEBUG>)

# Remaining flags are set through the macros in cmake/einsums_add_compile_flag.cmake

include(Einsums_ExportTargets)
# Modules can't link to this if not exported
install(
  TARGETS einsums_private_flags
  EXPORT einsums_internal_targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT einsums_private_flags)
install(
  TARGETS einsums_public_flags
  EXPORT einsums_internal_targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT einsums_public_flags)
einsums_export_internal_targets(einsums_private_flags)
einsums_export_internal_targets(einsums_public_flags)
