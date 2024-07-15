#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

# Adds einsums_ prefix to give einsums_${name} to libraries and components
function(einsums_set_lib_name target name)
  einsums_debug("einsums_set_lib_name: target: " ${target} " name: " ${name})

  set_target_properties(
    ${target}
    PROPERTIES OUTPUT_NAME einsums_${name}
               DEBUG_OUTPUT_NAME einsums_${name}
               RELEASE_OUTPUT_NAME einsums_${name}
               MINSIZEREL_OUTPUT_NAME einsums_${name}
               RELWITHDEBINFO_OUTPUT_NAME einsums_${name}
  )
endfunction()
