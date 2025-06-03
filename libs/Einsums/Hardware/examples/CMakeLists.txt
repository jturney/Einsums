#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

if(EINSUMS_WITH_EXAMPLES)
  einsums_add_pseudo_target(Examples.Modules.EinsumsExperimental.Hardware)
  einsums_add_pseudo_dependencies(Examples.Modules Examples.Modules.EinsumsExperimental.Hardware)
  if(EINSUMS_WITH_TESTS AND EINSUMS_WITH_TESTS_EXAMPLES)
    einsums_add_pseudo_target(Tests.Examples.Modules.EinsumsExperimental.Hardware)
    einsums_add_pseudo_dependencies(
      Tests.Examples.Modules Tests.Examples.Modules.EinsumsExperimental.Hardware
    )
  endif()
else()
  return()
endif()

set(example_programs topology)

foreach(example_program ${example_programs})
  set(sources ${example_program}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  einsums_add_executable(
    ${example_program} INTERNAL_FLAGS
    SOURCES ${sources} ${${example_program}_FLAGS}
    DEPENDENCIES EinsumsExperimental
    FOLDER "Examples/Modules/EinsumsExperimental/Hardware"
    NOINSTALL
  )

  einsums_add_example_target_dependencies("Modules.EinsumsExperimental.Hardware" ${example_program})

  if(EINSUMS_WITH_TESTS AND EINSUMS_WITH_TESTS_EXAMPLES)
    einsums_add_example_test(
      "Modules.EinsumsExperimental.Hardware" ${example_program} ${${example_program}_PARAMETERS}
    )
  endif()
endforeach()
