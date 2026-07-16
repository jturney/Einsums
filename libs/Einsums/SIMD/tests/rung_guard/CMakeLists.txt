#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

# Launcher used by einsums_add_simd_rung_tests: skips (exit 77) rung tests
# the host CPU cannot execute, otherwise execs the wrapped test binary.
einsums_add_executable(simd_rung_guard INTERNAL_FLAGS SOURCES simd_rung_guard.cpp NOINSTALL)
