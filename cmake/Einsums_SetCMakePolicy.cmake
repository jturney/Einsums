#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

function(einsums_set_cmake_policy policy value)
  if(POLICY ${policy})
    cmake_policy(SET ${policy} ${value})
  endif()
endfunction()
