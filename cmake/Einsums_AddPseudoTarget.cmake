#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

function(einsums_add_pseudo_target)
  einsums_debug("einsums_add_pseudo_target" "adding pseudo target: ${ARGV}")
  if(EINSUMS_WITH_PSEUDO_DEPENDENCIES)
    set(shortened_args)
    foreach(arg ${ARGV})
      einsums_shorten_pseudo_target(${arg} shortened_arg)
      set(shortened_args ${shortened_args} ${shortened_arg})
    endforeach()
    einsums_debug("einsums_add_pseudo_target" "adding shortened pseudo target: ${shortened_args}")
    foreach(target ${shortened_args})
      if(NOT TARGET ${target})
        add_custom_target(${target})
      endif()
    endforeach()
  endif()
endfunction()
