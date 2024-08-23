#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

function(einsums_export_targets)
  foreach(target ${ARGN})
    list(FIND EINSUMS_EXPORT_TARGETS ${target} _found)
    if(_found EQUAL -1)
      set(EINSUMS_EXPORT_TARGETS
          ${EINSUMS_EXPORT_TARGETS} ${target}
          CACHE INTERNAL "" FORCE)
    endif()
  endforeach()
endfunction(einsums_export_targets)

function(einsums_export_internal_targets)
  foreach(target ${ARGN})
    list(FIND EINSUMS_EXPORT_INTERNAL_TARGETS ${target} _found)
    if(_found EQUAL -1)
      set(EINSUMS_EXPORT_INTERNAL_TARGETS
          ${EINSUMS_EXPORT_INTERNAL_TARGETS} ${target}
          CACHE INTERNAL "" FORCE)
    endif()
  endforeach()
endfunction()
