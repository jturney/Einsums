@echo off
REM Copyright (c) The Einsums Developers. All rights reserved.
REM Licensed under the MIT License. See LICENSE.txt in the project root for license information.
REM
REM Configure Einsums on Windows (clang-cl from conda + Ninja).
REM Prerequisite: conda activate (einsums-dev or equivalent) in cmd.
REM
REM Why this script exists: conda-forge's vs2022 activation exports settings
REM for the Visual Studio CMake generator (conda-build's convention). Ninja
REM rejects any platform specification, and CMake treats an env var set to the
REM empty string as still specified, so a preset cannot neutralize it - the
REM variables must be UNSET in the shell before cmake runs.
REM
REM Shell-agnostic alternative (works in PowerShell too):
REM   cmake -E env --unset=CMAKE_GENERATOR_PLATFORM --unset=CMAKE_GENERATOR_TOOLSET ^
REM         --unset=CMAKE_GENERATOR cmake --preset windows-clang-cl

REM Clear conda feedstock flags so CMAKE_CXX_STANDARD / CMAKE_BUILD_TYPE win.
set "CXXFLAGS="
set "CFLAGS="
set "CPPFLAGS="

REM vs2022_compiler_vars sets these for the VS generator; see header comment.
set "CMAKE_GENERATOR_PLATFORM="
set "CMAKE_GENERATOR_TOOLSET="
set "CMAKE_GENERATOR="

cmake --preset windows-clang-cl
echo.
echo Exit code: %ERRORLEVEL%
exit /b %ERRORLEVEL%
