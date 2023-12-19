#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

# Creates a symbolic link from the destination to the target, if the link doesn't exist yet. Since `create_symlink` is
# only available for unix derivates, we work around that in this macro.
function(einsums_create_symbolic_link SYM_TARGET SYM_DESTINATION)
    if(WIN32)
        if(NOT EXISTS ${SYM_DESTINATION})
            if(IS_DIRECTORY ${SYM_TARGET})
                # Create a directory junction
                execute_process(
                    COMMAND cmd /C "${PROJECT_SOURCE_DIR}/cmake/scripts/einsums_create_symbolic_link_directory.bat"
                            ${SYM_DESTINATION} ${SYM_TARGET}
                )
            else()
                # Create a file link
                execute_process(
                    COMMAND cmd /C "${PROJECT_SOURCE_DIR}/cmake/scripts/einsums_create_symbolic_link_file.bat"
                            ${SYM_DESTINATION} ${SYM_TARGET}
                )
            endif()
        endif()
    else()
        # Only available on unix derivates
        execute_process(COMMAND "${CMAKE_COMMAND}" -E create_symlink ${SYM_TARGET} ${SYM_DESTINATION})
    endif()
endfunction(einsums_create_symbolic_link)
