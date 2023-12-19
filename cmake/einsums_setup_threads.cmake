#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

if(NOT EINSUMS_FIND_PACKAGE)
    target_link_libraries(einsums_base_libraries INTERFACE Threads::Threads)

    einsums_add_compile_flag_if_available(-pthread PUBLIC)
    einsums_add_link_flag_if_available(-pthread PUBLIC)
endif()
