#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

set(EINSUMS_ADDPSEUDODEPENDENCIES_LOADED TRUE)

include(einsums_message)

function(einsums_add_pseudo_dependencies)

    if("${EINSUMS_CMAKE_LOGLEVEL}" MATCHES "DEBUG|debug|Debug")
        set(args)
        foreach(arg ${ARGV})
            set(args "${args} ${arg}")
        endforeach()
        einsums_debug("einsums_add_pseudo_dependencies" ${args})
    endif()

    if(EINSUMS_WITH_PSEUDO_DEPENDENCIES)
        set(shortened_args)
        foreach(arg ${ARGV})
            einsums_shorten_pseudo_target(${arg} shortened_arg)
            set(shortened_args ${shortened_args} ${shortened_arg})
        endforeach()
        add_dependencies(${shortened_args})
    endif()
endfunction()

function(einsums_add_pseudo_dependencies_no_shortening)

    if("${EINSUMS_CMAKE_LOGLEVEL}" MATCHES "DEBUG|debug|Debug")
        set(args)
        foreach(arg ${ARGV})
            set(args "${args} ${arg}")
        endforeach()
        einsums_debug("einsums_add_pseudo_dependencies" ${args})
    endif()

    if(EINSUMS_WITH_PSEUDO_DEPENDENCIES)
        add_dependencies(${ARGV})
    endif()
endfunction()
