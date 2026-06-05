#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

# Header-only performance testing library (no sources — INTERFACE).
add_library(einsums_performance INTERFACE)
target_include_directories(einsums_performance INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
