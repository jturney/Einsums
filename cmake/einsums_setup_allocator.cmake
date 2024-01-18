#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

include(einsums_add_definitions)

# In case find_package(einsums) is called multiple times
if(NOT TARGET einsums_dependencies_allocator)

    if(NOT EINSUMS_WITH_MALLOC)
        set(EINSUMS_WITH_MALLOC
            CACHE STRING
                  "Use the specified allocator. Supported allocators are tcmalloc, jemalloc, tbbmalloc and system."
                  ${DEFAULT_MALLOC}
        )
        set(allocator_error
            "The default allocator for your system is ${DEFAULT_MALLOC}, but ${DEFAULT_MALLOC} could not be found. "
            "The system allocator has poor performance. As such ${DEFAULT_MALLOC} is a strong optional requirement. "
            "Being aware of the performance hit, you can override this default and get rid of this dependency by setting -DEINSUMS_WITH_MALLOC=system. "
            "Valid options for EINSUMS_WITH_MALLOC are: system, tcmalloc, jemalloc, mimalloc, tbbmalloc, and custom"
        )
    else()
        set(allocator_error
            "EINSUMS_WITH_MALLOC was set to ${EINSUMS_WITH_MALLOC}, but ${EINSUMS_WITH_MALLOC} could not be found. "
            "Valid options for EINSUMS_WITH_MALLOC are: system, tcmalloc, jemalloc, mimalloc, tbbmalloc, and custom"
        )
    endif()

    string(TOUPPER "${EINSUMS_WITH_MALLOC}" EINSUMS_WITH_MALLOC_UPPER)

    add_library(einsums_dependencies_allocator INTERFACE IMPORTED)

    if(NOT EINSUMS_WITH_MALLOC_DEFAULT)

        # ##############################################################################################################
        # TCMALLOC
        if("${EINSUMS_WITH_MALLOC_UPPER}" STREQUAL "TCMALLOC")
            find_package(TCMalloc)
            if(NOT TCMALLOC_LIBRARIES)
                einsums_error(${allocator_error})
            endif()

            target_link_libraries(einsums_dependencies_allocator INTERFACE ${TCMALLOC_LIBRARIES})

            if(MSVC)
                target_compile_options(einsums_dependencies_allocator INTERFACE /INCLUDE:__tcmalloc)
            endif()
            set(_use_custom_allocator TRUE)
        endif()

        # ##############################################################################################################
        # JEMALLOC
        if("${EINSUMS_WITH_MALLOC_UPPER}" STREQUAL "JEMALLOC")
            find_package(Jemalloc)
            if(NOT JEMALLOC_LIBRARIES)
                einsums_error(${allocator_error})
            endif()
            target_include_directories(
                einsums_dependencies_allocator INTERFACE ${JEMALLOC_INCLUDE_DIR} ${JEMALLOC_ADDITIONAL_INCLUDE_DIR}
            )
            target_link_libraries(einsums_dependencies_allocator INTERFACE ${JEMALLOC_LIBRARIES})
        endif()

        # ##############################################################################################################
        # MIMALLOC
        if("${EINSUMS_WITH_MALLOC_UPPER}" STREQUAL "MIMALLOC")
            find_package(mimalloc)
            if(NOT mimalloc_FOUND)
                einsums_error(${allocator_error})
            endif()
            target_link_libraries(einsums_dependencies_allocator INTERFACE mimalloc)
            set(einsums_MALLOC_LIBRARY mimalloc)
            if(MSVC)
                target_compile_options(einsums_dependencies_allocator INTERFACE /INCLUDE:mi_version)
            endif()
            set(_use_custom_allocator TRUE)

            einsums_warn(
                "einsums is using mimalloc as the allocator. Typically, exporting the following environment variables will further improve performance: MIMALLOC_EAGER_COMMIT_DELAY=0 and MIMALLOC_LARGE_OS_PAGES=1."
            )
        endif()

        # ##############################################################################################################
        # TBBMALLOC
        if("${EINSUMS_WITH_MALLOC_UPPER}" STREQUAL "TBBMALLOC")
            find_package(TBBmalloc)
            if(NOT TBBMALLOC_LIBRARY AND NOT TBBMALLOC_PROXY_LIBRARY)
                einsums_error(${allocator_error})
            endif()
            if(MSVC)
                target_compile_options(einsums_dependencies_allocator INTERFACE /INCLUDE:__TBB_malloc_proxy)
            endif()
            target_link_libraries(
                einsums_dependencies_allocator INTERFACE ${TBBMALLOC_LIBRARY} ${TBBMALLOC_PROXY_LIBRARY}
            )
        endif()

        if("${EINSUMS_WITH_MALLOC_UPPER}" STREQUAL "CUSTOM")
            set(_use_custom_allocator TRUE)
        endif()

    else()

        set(EINSUMS_WITH_MALLOC ${EINSUMS_WITH_MALLOC_DEFAULT})

    endif(NOT EINSUMS_WITH_MALLOC_DEFAULT)

    if("${EINSUMS_WITH_MALLOC_UPPER}" MATCHES "SYSTEM")
        if(NOT MSVC)
            einsums_warn("einsums will perform poorly without tcmalloc, jemalloc, or mimalloc. See docs for more info.")
        endif()
        set(_use_custom_allocator FALSE)
    endif()

    einsums_info("Using ${EINSUMS_WITH_MALLOC} allocator.")

    # Setup Intel amplifier
    if((NOT EINSUMS_WITH_APEX) AND EINSUMS_WITH_ITTNOTIFY)

        find_package(Amplifier)
        if(NOT AMPLIFIER_FOUND)
            einsums_error(
                "Intel Amplifier could not be found and EINSUMS_WITH_ITTNOTIFY=On, please specify AMPLIFIER_ROOT to point to the root of your Amplifier installation"
            )
        endif()

        einsums_add_config_define(EINSUMS_HAVE_ITTNOTIFY 1)
        einsums_add_config_define(EINSUMS_HAVE_THREAD_DESCRIPTION)
    endif()

    # convey selected allocator type to the build configuration
    if(NOT EINSUMS_FIND_PACKAGE)
        einsums_add_config_define(EINSUMS_HAVE_MALLOC "\"${EINSUMS_WITH_MALLOC}\"")
        if(${EINSUMS_WITH_MALLOC} STREQUAL "jemalloc")
            if(NOT ("${EINSUMS_WITH_JEMALLOC_PREFIX}" STREQUAL "<none>") AND NOT ("${EINSUMS_WITH_JEMALLOC_PREFIX}x"
                                                                                  STREQUAL "x")
            )
                einsums_add_config_define(EINSUMS_HAVE_JEMALLOC_PREFIX ${EINSUMS_WITH_JEMALLOC_PREFIX})
            endif()
        endif()
    endif()

endif()
