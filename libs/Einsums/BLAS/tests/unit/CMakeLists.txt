#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

set(BLASTests ThreadControl.cpp GemmBatchGrouped.cpp)

foreach(test ${BLASTests})
  get_filename_component(TestName ${test} NAME_WE)

  set(Sources ${test})

  source_group("Source Files" FILES ${Sources})

  einsums_add_executable(
    ${TestName}_test
    INTERNAL_FLAGS
    SOURCES ${Sources}
    NOINSTALL
  )

  einsums_add_unit_test("Modules.BLAS" ${TestName})
endforeach()

# Tell the test which vendor it is linked against. Without this the thread
# control test passes either way - when MKL's per-thread setter resolves, and
# when it is not found and the test skips itself - so a green run would say
# nothing about whether the mechanism works on the one platform that needs it.
# FindTargetLAPACK records the vendor it settled on.
if(TARGET tgt::lapack)
  get_target_property(_einsums_blas_vendor tgt::lapack VENDOR)
  if(_einsums_blas_vendor STREQUAL "MKL")
    target_compile_definitions(ThreadControl_test PRIVATE EINSUMS_TEST_BLAS_IS_MKL)
  endif()
endif()
