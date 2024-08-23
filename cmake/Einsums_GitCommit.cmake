#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

# if no git commit is set, try to get it from the source directory
if(NOT EINSUMS_WITH_GIT_COMMIT OR "${EINSUMS_WITH_GIT_COMMIT}" STREQUAL "None")

  find_package(Git)

  if(GIT_FOUND)
    execute_process(
      COMMAND "${GIT_EXECUTABLE}" "log" "--pretty=%H" "-1" "${PROJECT_SOURCE_DIR}"
      WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
      OUTPUT_VARIABLE EINSUMS_WITH_GIT_COMMIT
      ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
  endif()

endif()

if(NOT EINSUMS_WITH_GIT_COMMIT OR "${EINSUMS_WITH_GIT_COMMIT}" STREQUAL "None")
  einsums_warn("GIT commit not found (set to 'unknown').")
  set(EINSUMS_WITH_GIT_COMMIT "unknown")
  set(EINSUMS_WITH_GIT_COMMIT_SHORT "unknown")
else()
  einsums_info("GIT commit is ${EINSUMS_WITH_GIT_COMMIT}.")
  if(NOT EINSUMS_WITH_GIT_COMMIT_SHORT OR "${EINSUMS_WITH_GIT_COMMIT_SHORT}" STREQUAL "None")
    string(SUBSTRING "${EINSUMS_WITH_GIT_COMMIT}" 0 7 EINSUMS_WITH_GIT_COMMIT_SHORT)
  endif()
endif()
