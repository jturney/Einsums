#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

einsums_add_executable(
  calibrate_hardware INTERNAL_FLAGS
  SOURCES calibrate_hardware.cpp
  FOLDER "Tools/ComputeGraph"
  NOINSTALL
)
target_link_libraries(calibrate_hardware PRIVATE Einsums_ComputeGraph Einsums_TensorUtilities)
