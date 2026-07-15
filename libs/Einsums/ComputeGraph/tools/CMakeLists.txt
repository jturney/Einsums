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
# NOTE: no direct module links - einsums_add_executable already links the
# umbrella Einsums library. Linking module OBJECT libraries alongside it
# duplicates every symbol under the MSVC ABI (harmlessly merged on ELF).
