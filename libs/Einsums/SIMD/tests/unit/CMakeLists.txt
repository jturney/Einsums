#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

set(SIMDTests
    VecBasic
    VecInteger
    VecHalf
    VecInt8
    VecMath
    VecMask
    VecReduce
    VecPartial
    VecConvert
    StreamStore
    Shuffle
    GatherScatter
    ComplexVec
    RuntimeFeatures
)

# Every suite but RuntimeFeatures calls the einsums::simd headers directly, so
# what it exercises is fixed when the test itself is compiled, not by runtime
# dispatch. Built once at the ambient flags these read one rung on every
# machine; the helper adds a copy per higher rung so the wider paths are
# covered too. RuntimeFeatures tests CPU detection, which does not widen.
set(SIMDWidthDependentTests
    VecBasic
    VecInteger
    VecHalf
    VecInt8
    VecMath
    VecMask
    VecReduce
    VecPartial
    VecConvert
    StreamStore
    Shuffle
    GatherScatter
    ComplexVec
)

include(Einsums_AddSIMDDispatch)

foreach(test ${SIMDTests})
  set(sources ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  einsums_add_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources}
    NOINSTALL
  )

  einsums_add_unit_test("Modules.SIMD" ${test})

  if(${test} IN_LIST SIMDWidthDependentTests)
    einsums_add_simd_rung_compiled_tests("Modules.SIMD" ${test})
  endif()
endforeach()

# The ladder tests pin the sme gate against what the rung's flags enable on
# this compiler, the same answer RuntimeFeatures.cpp was built with.
einsums_simd_rung_enables(sme _sme_sve _sme_sve2)
target_compile_definitions(
  RuntimeFeatures_test PRIVATE EINSUMS_SIMD_SME_RUNG_ENABLES_SVE=$<BOOL:${_sme_sve}>
                               EINSUMS_SIMD_SME_RUNG_ENABLES_SVE2=$<BOOL:${_sme_sve2}>
)
