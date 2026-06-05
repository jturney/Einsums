#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

# Metal Performance Shaders (MPS) backend for Apple Silicon GPU.
# Auto-detected on macOS when EINSUMS_WITH_MPS=ON (default on Apple).

if(EINSUMS_WITH_MPS AND NOT TARGET einsums_mps)
  include(Einsums_AddDefinitions)

  # Enable Objective-C++ for .mm files
  enable_language(OBJCXX)

  if(NOT APPLE)
    message(STATUS "MPS backend: not available (not macOS)")
    set(EINSUMS_WITH_MPS OFF CACHE BOOL "MPS backend" FORCE)
    return()
  endif()

  # Check for Metal and MetalPerformanceShaders frameworks
  find_library(METAL_FRAMEWORK Metal)
  find_library(MPS_FRAMEWORK MetalPerformanceShaders)
  find_library(FOUNDATION_FRAMEWORK Foundation)

  if(NOT METAL_FRAMEWORK OR NOT MPS_FRAMEWORK)
    message(STATUS "MPS backend: Metal or MetalPerformanceShaders framework not found")
    set(EINSUMS_WITH_MPS OFF CACHE BOOL "MPS backend" FORCE)
    return()
  endif()

  message(STATUS "MPS backend: enabled (Metal=${METAL_FRAMEWORK}, MPS=${MPS_FRAMEWORK})")

  einsums_add_config_define(EINSUMS_HAVE_MPS)

  # Create an imported target for convenience
  add_library(einsums_mps INTERFACE IMPORTED)
  target_link_libraries(einsums_mps INTERFACE
    ${METAL_FRAMEWORK}
    ${MPS_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK}
  )
endif()
