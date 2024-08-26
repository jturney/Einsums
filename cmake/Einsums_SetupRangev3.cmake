#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

# * note that upstream uses `write_basic_package_version_file(... COMPATIBILITY
#   ExactVersion)` so no version range of compatibilty can be expressed here.

FetchContent_Declare(
        range-v3
        URL https://github.com/ericniebler/range-v3/archive/0.12.0.tar.gz
        URL_HASH SHA256=015adb2300a98edfceaf0725beec3337f542af4915cec4d0b89fa0886f4ba9cb
        #        FIND_PACKAGE_ARGS 0.12.0
)

FetchContent_MakeAvailable(range-v3)

target_link_libraries(einsums_base_libraries INTERFACE range-v3::range-v3)
