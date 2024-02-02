#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

find_package(Hwloc REQUIRED)
if(NOT HWLOC_FOUND)
    einsums_error("Hwloc could not be found, please specify HWLOC_ROOT to point to the correct location")
endif()
