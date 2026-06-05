#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

if(EINSUMS_WITH_EXAMPLES)
  einsums_add_pseudo_target(Examples.Modules.Einsums.Profile)
  einsums_add_pseudo_dependencies(Examples.Modules Examples.Modules.Einsums.Profile)
  if(EINSUMS_WITH_TESTS AND EINSUMS_WITH_TESTS_EXAMPLES)
    einsums_add_pseudo_target(Tests.Examples.Modules.Einsums.Profile)
    einsums_add_pseudo_dependencies(Tests.Examples.Modules Tests.Examples.Modules.Einsums.Profile)
  endif()
else()
  return()
endif()

if(EINSUMS_WITH_PROFILER)
  set(example_programs profile1)

  foreach(example_program ${example_programs})
    set(sources ${example_program}.cpp)

    source_group("Source Files" FILES ${sources})

    # add example executable
    einsums_add_executable(
      PROF_${example_program} INTERNAL_FLAGS
      SOURCES ${sources} ${${example_program}_FLAGS}
      FOLDER "Examples/Modules/Profile"
      NOINSTALL
    )

    einsums_add_example_target_dependencies("Modules.Einsums.Profile" PROF_${example_program})

    if(EINSUMS_WITH_TESTS AND EINSUMS_WITH_TESTS_EXAMPLES)
      einsums_add_example_test(
        "Modules.Einsums.Profile" PROF_${example_program} ${${example_program}_PARAMETERS}
      )
    endif()
  endforeach()

  # Long-running example for live TUI viewer testing (not registered as a ctest)
  einsums_add_executable(
    PROF_profile_live INTERNAL_FLAGS
    SOURCES profile_live.cpp
    FOLDER "Examples/Modules/Profile"
    NOINSTALL
  )
  einsums_add_example_target_dependencies("Modules.Einsums.Profile" PROF_profile_live)
endif()
