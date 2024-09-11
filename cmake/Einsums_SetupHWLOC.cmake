#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

find_package(HWLOC REQUIRED)
if (NOT HWLOC_FOUND)
    einsums_error("HWLOC could not be found, please specify HWLOC_ROOT.")
endif()
