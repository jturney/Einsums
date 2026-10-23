#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

set(GPUTests Runtime)

# The MPS test exercises the Metal backend itself; a build without it has nothing to run, and a
# test binary with no cases fails.
if(EINSUMS_WITH_MPS AND NOT EINSUMS_WITH_CUDA)
  list(APPEND GPUTests MPSBackend)
endif()

foreach(test ${GPUTests})
  set(sources ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  einsums_add_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources}
    NOINSTALL
  )

  einsums_add_unit_test("Modules.GPU" ${test})
endforeach()
