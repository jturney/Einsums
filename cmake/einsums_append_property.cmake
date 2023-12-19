#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

function(einsums_append_property target property)
    einsums_get_target_property(existing ${target} ${property})
    if(existing)
        set_property(TARGET ${target} PROPERTY ${property} "${existing} ${ARGN}")
    else()
        set_property(TARGET ${target} PROPERTY ${property} "${ARGN}")
    endif()
endfunction()
