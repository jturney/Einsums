#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

set(EINSUMS_PLATFORM_CHOICES "Choices are: native.")
set(EINSUMS_PLATFORMS_UC "NATIVE")

if(NOT EINSUMS_PLATFORM)
  set(EINSUMS_PLATFORM
      "native"
      CACHE STRING
            "Sets special compilation flags for specific platforms. ${EINSUMS_PLATFORM_CHOICES}"
  )
else()
  set(EINSUMS_PLATFORM
      "${EINSUMS_PLATFORM}"
      CACHE STRING
            "Sets special compilation flags for specific platforms. ${EINSUMS_PLATFORM_CHOICES}"
  )
endif()

if(NOT EINSUMS_PLATFORM STREQUAL "")
  string(TOUPPER ${EINSUMS_PLATFORM} EINSUMS_PLATFORM_UC)
else()
  set(EINSUMS_PLATFORM
      "native"
      CACHE STRING
            "Sets special compilation flags for specific platforms. ${EINSUMS_PLATFORM_CHOICES}"
            FORCE
  )
  set(EINSUMS_PLATFORM_UC "NATIVE")
endif()

string(FIND "${EINSUMS_PLATFORMS_UC}" "${EINSUMS_PLATFORM_UC}" _PLATFORM_FOUND)
if(_PLATFORM_FOUND EQUAL -1)
  einsums_error("Unknown platform in EINSUMS_PLATFORM. ${EINSUMS_PLATFORM_CHOICES}")
endif()
