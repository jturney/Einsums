#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

einsums_option(
  EINSUMS_WITH_PROFILE_VIEWER BOOL
  "Build the ImGui-based profile viewer (requires glfw from conda; imgui docking branch built from source)." OFF
  CATEGORY "Build Targets"
)

if(NOT EINSUMS_WITH_PROFILE_VIEWER)
  return()
endif()

# glfw comes from conda; imgui/implot/glaze are built from source
# inside devtools/profiling/imgui_viewer/CMakeLists.txt
find_package(glfw3 CONFIG REQUIRED)
find_package(OpenGL REQUIRED)

einsums_info("  glfw3 found:   ${glfw3_DIR}")
