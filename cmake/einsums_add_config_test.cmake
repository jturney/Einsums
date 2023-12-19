#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

set(EINSUMS_ADDCONFIGTEST_LOADED TRUE)

include(CheckLibraryExists)

function(einsums_add_config_test variable)
    set(options FILE EXECUTE GPU NOT_REQUIRED)
    set(one_value_args SOURCE ROOT CMAKECXXFEATURE CHECK_CXXSTD EXTRA_MSG)
    set(multi_value_args
        CXXFLAGS
        INCLUDE_DIRECTORIES
        LINK_DIRECTORIES
        COMPILE_DEFINITIONS
        LIBRARIES
        ARGS
        DEFINITIONS
        REQUIRED
    )
    cmake_parse_arguments(${variable} "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

    if(${variable}_CHECK_CXXSTD AND ${variable}_CHECK_CXXSTD GREATER EINSUMS_WITH_CXX_STANDARD)
        if(DEFINED ${variable})
            unset(${variable} CACHE)
            einsums_info("Unsetting ${variable} because of EINSUMS_WITH_CXX_STANDARD (${EINSUMS_WITH_CXX_STANDARD})")
        endif()
        return()
    endif()

    set(_run_msg)
    # Check CMake feature tests if the user didn't override the value of this variable:
    if(NOT DEFINED ${variable} AND NOT ${variable}_GPU)
        if(${variable}_CMAKECXXFEATURE)
            # We don't have to run our own feature test if there is a corresponding cmake feature test and cmake reports
            # the feature is supported on this platform.
            list(FIND CMAKE_CXX_COMPILE_FEATURES ${${variable}_CMAKECXXFEATURE} __pos)
            if(NOT ${__pos} EQUAL -1)
                set(${variable}
                    TRUE
                    CACHE INTERNAL ""
                )
                set(_run_msg "Success (cmake feature test)")
            endif()
        endif()
    endif()

    if(NOT DEFINED ${variable})
        file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/config_tests")

        string(TOLOWER "${variable}" variable_lc)
        if(${variable}_FILE)
            if(${variable}_ROOT)
                set(test_source "${${variable}_ROOT}/share/einsums/${${variable}_SOURCE}")
            else()
                set(test_source "${PROJECT_SOURCE_DIR}/${${variable}_SOURCE}")
            endif()
        else()
            if(${variable}_GPU)
                set(test_source "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/config_tests/${variable_lc}.cu")
            else()
                set(test_source "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/config_tests/${variable_lc}.cpp")
            endif()
            file(WRITE "${test_source}" "${${variable}_SOURCE}\n")
        endif()
        set(test_binary ${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/config_tests/${variable_lc})

        get_directory_property(CONFIG_TEST_INCLUDE_DIRS INCLUDE_DIRECTORIES)
        get_directory_property(CONFIG_TEST_LINK_DIRS LINK_DIRECTORIES)
        set(COMPILE_DEFINITIONS_TMP)
        set(CONFIG_TEST_COMPILE_DEFINITIONS)
        get_directory_property(COMPILE_DEFINITIONS_TMP COMPILE_DEFINITIONS)
        foreach(def IN LISTS COMPILE_DEFINITIONS_TMP ${variable}_COMPILE_DEFINITIONS)
            set(CONFIG_TEST_COMPILE_DEFINITIONS "${CONFIG_TEST_COMPILE_DEFINITIONS} -D${def}")
        endforeach()
        get_property(EINSUMS_TARGET_COMPILE_OPTIONS_PUBLIC_VAR GLOBAL PROPERTY EINSUMS_TARGET_COMPILE_OPTIONS_PUBLIC)
        get_property(EINSUMS_TARGET_COMPILE_OPTIONS_PRIVATE_VAR GLOBAL PROPERTY EINSUMS_TARGET_COMPILE_OPTIONS_PRIVATE)
        set(EINSUMS_TARGET_COMPILE_OPTIONS_VAR ${EINSUMS_TARGET_COMPILE_OPTIONS_PUBLIC_VAR}
                                               ${EINSUMS_TARGET_COMPILE_OPTIONS_PRIVATE_VAR}
        )
        foreach(_flag ${EINSUMS_TARGET_COMPILE_OPTIONS_VAR})
            if(NOT "${_flag}" MATCHES "^\\$.*")
                set(CONFIG_TEST_COMPILE_DEFINITIONS "${CONFIG_TEST_COMPILE_DEFINITIONS} ${_flag}")
            endif()
        endforeach()

        set(CONFIG_TEST_INCLUDE_DIRS ${CONFIG_TEST_INCLUDE_DIRS} ${${variable}_INCLUDE_DIRECTORIES})
        set(CONFIG_TEST_LINK_DIRS ${CONFIG_TEST_LINK_DIRS} ${${variable}_LINK_DIRECTORIES})

        set(CONFIG_TEST_LINK_LIBRARIES ${${variable}_LIBRARIES})

        set(additional_cmake_flags)
        if(MSVC)
            set(additional_cmake_flags "-WX")
        else()
            set(additional_cmake_flags "-Werror")
        endif()

        if(${variable}_EXECUTE)
            if(NOT CMAKE_CROSSCOMPILING)
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${additional_cmake_flags} ${${variable}_CXXFLAGS}")
                # cmake-format: off
        try_run(
          ${variable}_RUN_RESULT ${variable}_COMPILE_RESULT
          ${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/config_tests
          ${test_source}
          COMPILE_DEFINITIONS ${CONFIG_TEST_COMPILE_DEFINITIONS}
          CMAKE_FLAGS
            "-DINCLUDE_DIRECTORIES=${CONFIG_TEST_INCLUDE_DIRS}"
            "-DLINK_DIRECTORIES=${CONFIG_TEST_LINK_DIRS}"
            "-DLINK_LIBRARIES=${CONFIG_TEST_LINK_LIBRARIES}"
          CXX_STANDARD ${EINSUMS_WITH_CXX_STANDARD}
          CXX_STANDARD_REQUIRED ON
          CXX_EXTENSIONS FALSE
          RUN_OUTPUT_VARIABLE ${variable}_OUTPUT
          ARGS ${${variable}_ARGS}
        )
        # cmake-format: on
                if(${variable}_COMPILE_RESULT AND NOT ${variable}_RUN_RESULT)
                    set(${variable}_RESULT TRUE)
                else()
                    set(${variable}_RESULT FALSE)
                endif()
            else()
                set(${variable}_RESULT FALSE)
            endif()
        else()
            if(EINSUMS_WITH_CUDA AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
                set(cuda_parameters CUDA_STANDARD ${CMAKE_CUDA_STANDARD})
            endif()
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${additional_cmake_flags} ${${variable}_CXXFLAGS}")
            set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${additional_cmake_flags} ${${variable}_CXXFLAGS}")
            # cmake-format: off
      try_compile(
        ${variable}_RESULT
        ${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/config_tests
        ${test_source}
        COMPILE_DEFINITIONS ${CONFIG_TEST_COMPILE_DEFINITIONS}
        CMAKE_FLAGS
          "-DINCLUDE_DIRECTORIES=${CONFIG_TEST_INCLUDE_DIRS}"
          "-DLINK_DIRECTORIES=${CONFIG_TEST_LINK_DIRS}"
          "-DLINK_LIBRARIES=${CONFIG_TEST_LINK_LIBRARIES}"
        OUTPUT_VARIABLE ${variable}_OUTPUT
        CXX_STANDARD ${EINSUMS_WITH_CXX_STANDARD}
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS FALSE
        ${cuda_parameters}
        COPY_FILE ${test_binary}
      )
      # cmake-format: on
            einsums_debug("Compile test: ${variable}")
            einsums_debug("Compilation output: ${${variable}_OUTPUT}")
        endif()

        set(_run_msg "Success")
    else()
        set(${variable}_RESULT ${${variable}})
        if(NOT _run_msg)
            set(_run_msg "pre-set to ${${variable}}")
        endif()
    endif()

    set(_msg "Performing Test ${variable}")
    if(${variable}_EXTRA_MSG)
        set(_msg "${_msg} (${${variable}_EXTRA_MSG})")
    endif()

    if(${variable}_RESULT)
        set(_msg "${_msg} - ${_run_msg}")
    else()
        set(_msg "${_msg} - Failed")
    endif()

    set(${variable}
        ${${variable}_RESULT}
        CACHE INTERNAL ""
    )
    einsums_info(${_msg})

    if(${variable}_RESULT)
        foreach(definition ${${variable}_DEFINITIONS})
            einsums_add_config_define(${definition})
        endforeach()
    elseif(${variable}_REQUIRED AND NOT ${variable}_NOT_REQUIRED)
        einsums_warn("Test failed, detailed output:\n\n${${variable}_OUTPUT}")
        einsums_error(${${variable}_REQUIRED})
    endif()
endfunction()

# ######################################################################################################################
function(einsums_cpuid target variable)
    einsums_add_config_test(
        ${variable}
        SOURCE cmake/tests/cpuid.cpp
        COMPILE_DEFINITIONS "${boost_include_dir}" "${include_dir}"
        FILE EXECUTE
        ARGS "${target}" ${ARGN}
    )
endfunction()

# ######################################################################################################################
function(einsums_check_for_unistd_h)
    einsums_add_config_test(
        EINSUMS_WITH_UNISTD_H
        SOURCE cmake/tests/unistd_h.cpp
        FILE ${ARGN}
    )
endfunction()

# ######################################################################################################################
function(einsums_check_for_libfun_std_experimental_optional)
    einsums_add_config_test(
        EINSUMS_WITH_LIBFUN_EXPERIMENTAL_OPTIONAL
        SOURCE cmake/tests/libfun_std_experimental_optional.cpp
        FILE ${ARGN}
    )
endfunction()

# ######################################################################################################################
function(einsums_check_for_cxx11_std_atomic)
    # Make sure EINSUMS_HAVE_LIBATOMIC is removed from the cache if necessary
    if(NOT EINSUMS_WITH_CXX11_ATOMIC)
        unset(EINSUMS_CXX11_STD_ATOMIC_LIBRARIES CACHE)
    endif()

    # First see if we can build atomics with no -latomics. We make sure to override REQUIRED, if set, with NOT_REQUIRED
    # so that we can use the fallback test further down.
    set(check_not_required)
    if(NOT MSVC)
        set(check_not_required NOT_REQUIRED)
    endif()

    einsums_add_config_test(
        EINSUMS_WITH_CXX11_ATOMIC
        SOURCE cmake/tests/cxx11_std_atomic.cpp
        LIBRARIES ${EINSUMS_CXX11_STD_ATOMIC_LIBRARIES}
        FILE ${ARGN} ${check_not_required}
    )

    if(NOT MSVC)
        # Sometimes linking against libatomic is required, if the platform doesn't support lock-free atomics. We already
        # know that MSVC works
        if(NOT EINSUMS_WITH_CXX11_ATOMIC)
            set(EINSUMS_CXX11_STD_ATOMIC_LIBRARIES
                atomic
                CACHE STRING "std::atomics need separate library" FORCE
            )
            unset(EINSUMS_WITH_CXX11_ATOMIC CACHE)
            einsums_add_config_test(
                EINSUMS_WITH_CXX11_ATOMIC
                SOURCE cmake/tests/cxx11_std_atomic.cpp
                LIBRARIES ${EINSUMS_CXX11_STD_ATOMIC_LIBRARIES}
                FILE ${ARGN} EXTRA_MSG "with -latomic"
            )
            if(NOT EINSUMS_WITH_CXX11_ATOMIC)
                unset(EINSUMS_CXX11_STD_ATOMIC_LIBRARIES CACHE)
                unset(EINSUMS_WITH_CXX11_ATOMIC CACHE)
            endif()
        endif()
    endif()
endfunction()

# Separately check for 128 bit atomics
function(einsums_check_for_cxx11_std_atomic_128bit)
    # First see if we can build atomics with no -latomics. We make sure to override REQUIRED, if set, with NOT_REQUIRED
    # so that we can use the fallback test further down.
    set(check_not_required)
    if(NOT MSVC)
        set(check_not_required NOT_REQUIRED)
    endif()

    einsums_add_config_test(
        EINSUMS_WITH_CXX11_ATOMIC_128BIT
        SOURCE cmake/tests/cxx11_std_atomic_128bit.cpp
        LIBRARIES ${EINSUMS_CXX11_STD_ATOMIC_LIBRARIES}
        FILE ${ARGN} NOT_REQUIRED
    )

    if(NOT MSVC)
        # Sometimes linking against libatomic is required, if the platform doesn't support lock-free atomics. We already
        # know that MSVC works
        if(NOT EINSUMS_WITH_CXX11_ATOMIC_128BIT)
            set(EINSUMS_CXX11_STD_ATOMIC_LIBRARIES
                atomic
                CACHE STRING "std::atomics need separate library" FORCE
            )
            unset(EINSUMS_WITH_CXX11_ATOMIC_128BIT CACHE)
            einsums_add_config_test(
                EINSUMS_WITH_CXX11_ATOMIC_128BIT
                SOURCE cmake/tests/cxx11_std_atomic_128bit.cpp
                LIBRARIES ${EINSUMS_CXX11_STD_ATOMIC_LIBRARIES}
                FILE ${ARGN} EXTRA_MSG "with -latomic"
            )
            if(NOT EINSUMS_WITH_CXX11_ATOMIC_128BIT)
                # Adding -latomic did not help, so we don't attempt to link to it later
                unset(EINSUMS_CXX11_STD_ATOMIC_LIBRARIES CACHE)
                unset(EINSUMS_WITH_CXX11_ATOMIC_128BIT CACHE)
            endif()
        endif()
    endif()
endfunction()

# ######################################################################################################################
function(einsums_check_for_cxx11_std_shared_ptr_lwg3018)
    einsums_add_config_test(
        EINSUMS_WITH_CXX11_SHARED_PTR_LWG3018
        SOURCE cmake/tests/cxx11_std_shared_ptr_lwg3018.cpp
        FILE ${ARGN}
    )
endfunction()

# ######################################################################################################################
function(einsums_check_for_c11_aligned_alloc)
    einsums_add_config_test(
        EINSUMS_WITH_C11_ALIGNED_ALLOC
        SOURCE cmake/tests/c11_aligned_alloc.cpp
        FILE ${ARGN}
    )
endfunction()

function(einsums_check_for_cxx17_std_aligned_alloc)
    einsums_add_config_test(
        EINSUMS_WITH_CXX17_STD_ALIGNED_ALLOC
        SOURCE cmake/tests/cxx17_std_aligned_alloc.cpp
        FILE ${ARGN}
    )
endfunction()

# ######################################################################################################################
function(einsums_check_for_cxx11_std_quick_exit)
    einsums_add_config_test(
        EINSUMS_WITH_CXX11_STD_QUICK_EXIT
        SOURCE cmake/tests/cxx11_std_quick_exit.cpp
        FILE ${ARGN}
    )
endfunction()

# ######################################################################################################################
function(einsums_check_for_cxx17_aligned_new)
    einsums_add_config_test(
        EINSUMS_WITH_CXX17_ALIGNED_NEW
        SOURCE cmake/tests/cxx17_aligned_new.cpp
        FILE ${ARGN}
        REQUIRED
    )
endfunction()

# ######################################################################################################################
function(einsums_check_for_cxx17_std_transform_scan)
    einsums_add_config_test(
        EINSUMS_WITH_CXX17_STD_TRANSFORM_SCAN_ALGORITHMS
        SOURCE cmake/tests/cxx17_std_transform_scan_algorithms.cpp
        FILE ${ARGN}
    )
endfunction()

# ######################################################################################################################
function(einsums_check_for_cxx17_std_scan)
    einsums_add_config_test(
        EINSUMS_WITH_CXX17_STD_SCAN_ALGORITHMS
        SOURCE cmake/tests/cxx17_std_scan_algorithms.cpp
        FILE ${ARGN}
    )
endfunction()

# ######################################################################################################################
function(einsums_check_for_cxx17_copy_elision)
    einsums_add_config_test(
        EINSUMS_WITH_CXX17_COPY_ELISION
        SOURCE cmake/tests/cxx17_copy_elision.cpp
        FILE ${ARGN}
    )
endfunction()

# ######################################################################################################################
function(einsums_check_for_cxx17_memory_resource)
    einsums_add_config_test(
        EINSUMS_WITH_CXX17_MEMORY_RESOURCE
        SOURCE cmake/tests/cxx17_memory_resource.cpp
        FILE ${ARGN}
    )
endfunction()

# ######################################################################################################################
function(einsums_check_for_cxx20_no_unique_address_attribute)
    einsums_add_config_test(
        EINSUMS_WITH_CXX20_NO_UNIQUE_ADDRESS_ATTRIBUTE
        SOURCE cmake/tests/cxx20_no_unique_address_attribute.cpp
        FILE ${ARGN} CHECK_CXXSTD 20
    )
endfunction()

# ######################################################################################################################
function(einsums_check_for_cxx20_std_disable_sized_sentinel_for)
    einsums_add_config_test(
        EINSUMS_WITH_CXX20_STD_DISABLE_SIZED_SENTINEL_FOR
        SOURCE cmake/tests/cxx20_std_disable_sized_sentinel_for.cpp
        FILE ${ARGN} CHECK_CXXSTD 20
    )
endfunction()

# ######################################################################################################################
function(einsums_check_for_cxx20_trivial_virtual_destructor)
    einsums_add_config_test(
        EINSUMS_WITH_CXX20_TRIVIAL_VIRTUAL_DESTRUCTOR
        SOURCE cmake/tests/cxx20_trivial_virtual_destructor.cpp
        FILE ${ARGN}
    )
endfunction()

# ######################################################################################################################
function(einsums_check_for_cxx_lambda_capture_decltype)
    einsums_add_config_test(
        EINSUMS_WITH_CXX_LAMBDA_CAPTURE_DECLTYPE
        SOURCE cmake/tests/cxx_lambda_capture_decltype.cpp
        FILE ${ARGN}
    )
endfunction()
