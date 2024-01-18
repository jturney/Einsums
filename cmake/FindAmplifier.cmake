if(NOT TARGET einsums_internal::amplifier)
    # find_package(PkgConfig QUIET) pkg_check_modules(PC_AMPLIFIER QUIET amplifier)

    find_program(AMPLIFIER_EXECUTABLE amplxe-cl HINTS /opt/intel/oneapi/vtune/latest)

    if(NOT AMPLIFIER_EXECUTABLE)
        return()
    endif()

    get_filename_component(AMPLIFIER_DIR ${AMPLIFIER_EXECUTABLE} PATH)
    set(AMPLIFIER_ROOT "${AMPLIFIER_DIR}/..")

    find_path(
        AMPLIFIER_INCLUDE_DIR ittnotify.h
        HINTS ${AMPLIFIER_ROOT} ENV AMPLIFIER_ROOT
        PATH_SUFFIXES include
    )

    find_library(
        AMPLIFIER_LIBRARY
        NAMES ittnotify libittnotify
        HINTS ${AMPLIFIER_ROOT} ENV AMPLIFIER_ROOT
        PATH_SUFFIXES lib lib64
    )

    # Set AMPLIFIER_ROOT in case the other hints are used
    if(AMPLIFIER_ROOT)
        # The call to file is for compatibility for windows paths
        file(TO_CMAKE_PATH ${AMPLIFIER_ROOT} AMPLIFIER_ROOT)
    elseif("$ENV{AMPLIFIER_ROOT}")
        file(TO_CMAKE_PATH $ENV{AMPLIFIER_ROOT} AMPLIFIER_ROOT)
    else()
        file(TO_CMAKE_PATH "${AMPLIFIER_INCLUDE_DIR}" AMPLIFIER_INCLUDE_DIR)
        string(REPLACE "/include" "" AMPLIFIER_ROOT "${AMPLIFIER_INCLUDE_DIR}")
    endif()

    set(AMPLIFIER_LIBRARIES ${AMPLIFIER_LIBRARY})
    set(AMPLIFIER_INCLUDE_DIRS ${AMPLIFIER_INCLUDE_DIR})

    find_package_handle_standard_args(Amplifier DEFAULT_MSG AMPLIFIER_LIBRARY AMPLIFIER_INCLUDE_DIR)

    get_property(
        _type
        CACHE AMPLIFIER_ROOT
        PROPERTY TYPE
    )
    if(_type)
        set_property(CACHE AMPLIFIER_ROOT PROPERTY ADVANCED 1)
        if("x${_type}" STREQUAL "xUNINITIALIZED")
            set_property(CACHE AMPLIFIER_ROOT PROPERTY TYPE PATH)
        endif()
    endif()

    mark_as_advanced(AMPLIFIER_ROOT AMPLIFIER_LIBRARY AMPLIFIER_INCLUDE_DIR)

    message(STATUS "Found components for ITTNOTIFY")
    message(STATUS "AMPLIFIER_ROOT  = ${AMPLIFIER_ROOT}")
    message(STATUS "AMPLIFIER_INCLUDE_DIR  = ${AMPLIFIER_INCLUDE_DIR}")
    message(STATUS "AMPLIFIER_LIBRARIES = ${AMPLIFIER_LIBRARIES}")

    add_library(einsums_internal::amplifier INTERFACE IMPORTED)
    target_include_directories(einsums_internal::amplifier SYSTEM INTERFACE ${AMPLIFIER_INCLUDE_DIR})
    target_link_libraries(einsums_internal::amplifier INTERFACE ${AMPLIFIER_LIBRARIES})
endif()
