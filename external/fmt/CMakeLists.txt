#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

include(FetchContent)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

fetchcontent_declare(
    fmt
    URL https://github.com/fmtlib/fmt/archive/10.1.0.tar.gz
    URL_HASH SHA256=deb0a3ad2f5126658f2eefac7bf56a042488292de3d7a313526d667f3240ca0a
    FIND_PACKAGE_ARGS 10.1
)

fetchcontent_makeavailable(fmt)
