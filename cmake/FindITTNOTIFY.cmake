# ITTNOTIFY is the instrumentation and tracing technology (ITT) APIs provided by
# the Intel® VTune™ enable your application to generate and control the collection
# of trace data during its execution.
#
# The following variables are set when ITTNOTIFY is found:
#  ITTNOTIFY_FOUND          = Set to true, if all components of ITTNOTIFY have been found.
#  ITTNOTIFY_INCLUDE_DIRS   = Include path for the header files of ITTNOTIFY.
#  ITTNOTIFY_LIBRARY_DIRS   = Library search path for the ITTNOTIFY libraries.
#  ITTNOTIFY_LIBRARIES      = Link these to use ITTNOTIFY.
#  ITTNOTIFY_LFLAGS         = Linker flags (optional).

include(FindPackageHandleStandardArgs)

if(NOT ITTNOTIFY_FOUND)

    find_program(VTUNE_EXECUTABLE amplxe-cl)

    if(NOT VTUNE_EXECUTABLE)
        set(ITTNOTIFY_FOUND false)
        return()
    endif()

    get_filename_component(VTUNE_DIR ${VTUNE_EXECUTABLE} PATH)
    set(ITTNOTIFY_ROOT_DIR "${VTUNE_DIR}/..")

    # Look for header files
    find_path(
        ITTNOTIFY_INCLUDE_DIRS
        NAMES ittnotify.h
        HINTS ${ITTNOTIFY_ROOT_DIR}
        PATHS /usr /usr/local
        PATH_SUFFIXES include
    )

    # Look for the library
    find_library(
        ITTNOTIFY_LIBRARIES ittnotify
        HINTS ${ITTNOTIFY_ROOT_DIR}
        PATHS /usr /usr/local /opt/local
        PATH_SUFFIXES lib64 lib
    )

    find_package_handle_standard_args(ITTNOTIFY DEFAULT_MSG ITTNOTIFY_LIBRARIES ITTNOTIFY_INCLUDE_DIRS)

    if(ITTNOTIFY_FOUND)
        message(STATUS "Found components for ITTNOTIFY")
        message(STATUS "ITTNOTIFY_ROOT_DIR  = ${ITTNOTIFY_ROOT_DIR}")
        message(STATUS "ITTNOTIFY_INCLUDE_DIRS  = ${ITTNOTIFY_INCLUDE_DIRS}")
        message(STATUS "ITTNOTIFY_LIBRARIES = ${ITTNOTIFY_LIBRARIES}")

        if(UNIX)
            list(APPEND ITTNOTIFY_LIBRARIES dl pthread)
        endif()

        add_library(ittnotify INTERFACE)
        target_include_directories(ittnotify INTERFACE ${ITTNOTIFY_INCLUDE_DIRS})
        target_link_libraries(ittnotify INTERFACE ${ITTNOTIFY_LIBRARIES})
        target_compile_definitions(ittnotify INTERFACE EINSUMS_HAVE_ITTNOTIFY)
    endif()
endif()
