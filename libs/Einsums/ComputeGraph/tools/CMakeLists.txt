#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

# Installed: a user runs it once per machine after installing Einsums, and every
# later process on that machine finds the profile it writes.
einsums_add_executable(
  einsums-tune INTERNAL_FLAGS
  SOURCES einsums_tune.cpp
  FOLDER "Tools/ComputeGraph"
)
