#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

if (EINSUMS_WITH_HIP AND NOT TARGET roc::rocblas)
    if (EINSUMS_WTIH_CUDA)
        einsums_error("Both EINSUMS_WITH_CUDA and EINSUMS_WITH_HIP are ON. Please choose one of the for einsums to work properly")
    endif()

    # Check and set HIP standard
    if(NOT EINSUMS_FIND_PACKAGE)
        if(DEFINED CMAKE_HIP_STANDARD AND NOT CMAKE_HIP_STANDARD STREQUAL EINSUMS_WITH_CXX_STANDARD)
            einsums_error(
                    "You've set CMAKE_HIP_STANDARD to ${CMAKE_HIP_STANDARD} and EINSUMS_WITH_CXX_STANDARD to ${EINSUMS_WITH_CXX_STANDARD}. Please unset CMAKE_HIP_STANDARD."
            )
        endif()
        set(CMAKE_HIP_STANDARD ${EINSUMS_WITH_CXX_STANDARD})
    endif()

    set(CMAKE_HIP_STANDARD_REQUIRED ON)
    set(CMAKE_HIP_EXTENSIONS OFF)

    enable_language(HIP)

    find_package(rocblas REQUIRED)
    find_package(rocsolver REQUIRED)

    if (NOT EINSUMS_FIND_PACKAGE)
        einsums_add_config_define(EINSUMS_HAVE_HIP)
    endif()

endif()