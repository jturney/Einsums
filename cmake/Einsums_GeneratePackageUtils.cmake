#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

# Replace the "-NOTFOUND" by empty var incase the property is not found
macro(einsums_get_target_property var target property)
  get_target_property(${var} ${target} ${property})
  list(FILTER ${var} EXCLUDE REGEX "-NOTFOUND$")
endmacro()
