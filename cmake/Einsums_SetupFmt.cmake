#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

include(FetchContent)

fetchcontent_declare(
    fmt URL https://github.com/fmtlib/fmt/archive/refs/tags/11.0.1.tar.gz
    URL_HASH SHA256=7d009f7f89ac84c0a83f79ed602463d092fbf66763766a907c97fd02b100f5e9
    FIND_PACKAGE_ARGS 11
)

fetchcontent_makeavailable(fmt)

target_link_libraries(einsums_base_libraries INTERFACE fmt::fmt)
