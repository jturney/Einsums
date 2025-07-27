function(einsums_print_target_properties TARGET)
    set(props
            IMPORTED
            IMPORTED_LOCATION
            INTERFACE_INCLUDE_DIRECTORIES
            INTERFACE_COMPILE_DEFINITIONS
            INTERFACE_COMPILE_OPTIONS
            INTERFACE_LINK_LIBRARIES
            INTERFACE_LINK_OPTIONS
            LINK_LIBRARIES
            LINK_FLAGS
            LINK_INTERFACE_LIBRARIES
            INCLUDE_DIRECTORIES
            COMPILE_DEFINITIONS
            COMPILE_OPTIONS
            SOURCES
            OUTPUT_NAME
            RUNTIME_OUTPUT_NAME
            LIBRARY_OUTPUT_NAME
            ARCHIVE_OUTPUT_NAME
            POSITION_INDEPENDENT_CODE
            CXX_STANDARD
            C_STANDARD
            CUDA_STANDARD
            LINK_DEPENDS
            BUILD_RPATH
            INSTALL_RPATH
            # Add more as needed
    )

    foreach(prop ${props})
        get_target_property(value ${TARGET} ${prop})
        if(NOT value STREQUAL "NOTFOUND")
            message(STATUS "${TARGET} :: ${prop} = ${value}")
        endif()
    endforeach()
endfunction()
