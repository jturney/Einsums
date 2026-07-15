#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

# Toolchain: cross-compile for x86_64 Windows (MSVC ABI) from Linux with
# clang-cl + lld-link against an xwin-provided CRT/SDK sysroot.
#
# Expects (set by the windows-cross Docker image):
#   XWIN_ROOT    - xwin splat output (crt/ + sdk/)
#   WINLIBS_ROOT - extracted conda-forge win-64 packages (Library/...)
#
# clang-cl >= 15 understands /winsysroot, which resolves the MSVC CRT and
# Windows SDK include/lib paths in one flag; lld-link accepts the same.

set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_SYSTEM_PROCESSOR AMD64)

if(NOT DEFINED ENV{XWIN_ROOT})
    message(FATAL_ERROR "XWIN_ROOT not set - this toolchain is meant to run inside the einsums-windows-cross image")
endif()
set(_XWIN "$ENV{XWIN_ROOT}")
set(_WINLIBS "$ENV{WINLIBS_ROOT}")

set(CMAKE_C_COMPILER clang-cl)
set(CMAKE_CXX_COMPILER clang-cl)
set(CMAKE_C_COMPILER_TARGET x86_64-pc-windows-msvc)
set(CMAKE_CXX_COMPILER_TARGET x86_64-pc-windows-msvc)
set(CMAKE_LINKER lld-link)
set(CMAKE_AR llvm-lib)
set(CMAKE_MT llvm-mt)
set(CMAKE_RC_COMPILER llvm-rc)

# xwin's splat layout (crt/ + sdk/) is not the native Visual Studio layout
# that /winsysroot expects; the split flags accept xwin's directories
# directly.
add_compile_options(/vctoolsdir "${_XWIN}/crt" /winsdkdir "${_XWIN}/sdk")
add_link_options(/vctoolsdir:${_XWIN}/crt /winsdkdir:${_XWIN}/sdk)

# Match the conda-forge Windows convention (and Nathan's CI env): the
# dynamic release CRT, regardless of build type - the prebuilt Windows
# libraries we link were all built /MD.
set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreadedDLL)

# clang-cl cross drivers cannot do C++ module dependency scanning; nothing in
# the tree uses modules, but fetched deps (cpptrace) trip the check.
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

# Third-party Windows libraries (conda-forge win-64 payload trees).
list(APPEND CMAKE_PREFIX_PATH "${_WINLIBS}/Library")

# clang lowers C99 _Complex multiplication to __mulsc3/__muldc3 from
# compiler-rt. Real clang-cl toolchains auto-link clang_rt.builtins; this
# image compiles the two needed builtins into a shim library instead (see
# builtins_shim.c).
add_link_options("${_WINLIBS}/Library/lib/einsums_wincross_builtins.lib")

# Cross builds cannot execute configure-time test binaries. Preseed the
# results that Einsums' config tests would otherwise try_run for.
set(CMAKE_CROSSCOMPILING_EMULATOR "")

# find_package etc. should look in the sysroots, never the Linux host.
set(CMAKE_FIND_ROOT_PATH "${_XWIN}" "${_WINLIBS}/Library")
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
