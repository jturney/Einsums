#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

if(EINSUMS_WITH_EXAMPLES)
  einsums_add_pseudo_target(Examples.Modules.Einsums.TensorIO)
  einsums_add_pseudo_dependencies(Examples.Modules Examples.Modules.Einsums.TensorIO)
  if(EINSUMS_WITH_TESTS AND EINSUMS_WITH_TESTS_EXAMPLES)
    einsums_add_pseudo_target(Tests.Examples.Modules.Einsums.TensorIO)
    einsums_add_pseudo_dependencies(
      Tests.Examples.Modules Tests.Examples.Modules.Einsums.TensorIO
    )
  endif()
else()
  return()
endif()

set(example_programs TensorIOExample)

foreach(example_program ${example_programs})
  einsums_add_executable(
    TIO_${example_program} INTERNAL_FLAGS
    SOURCES ${example_program}.cpp
    FOLDER "Examples/Modules/TensorIO"
    NOINSTALL
  )

  einsums_add_example_target_dependencies("Modules.Einsums.TensorIO" TIO_${example_program})

  if(EINSUMS_WITH_TESTS AND EINSUMS_WITH_TESTS_EXAMPLES)
    einsums_add_example_test(
      "Modules.Einsums.TensorIO" TIO_${example_program}
    )
  endif()
endforeach()
