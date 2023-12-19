#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

function(einsums_force_out_of_tree_build message)
    string(COMPARE EQUAL "${PROJECT_SOURCE_DIR}" "${PROJECT_BINARY_DIR}" insource)
    get_filename_component(parentdir ${PROJECT_SOURCE_DIR} PATH)
    string(COMPARE EQUAL "${PROJECT_SOURCE_DIR}" "${parentdir}" insourcesubdir)
    if(insource OR insourcesubdir)
        einsums_error("in_tree" "${message}")
    endif()
endfunction()
