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
    GenericKernel
    VecWide
    VecExp
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
    GenericKernel
    VecWide
    VecExp
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

# IsaNamespace_test links a baseline translation unit with one compiled at the v3
# rung's flags, to check that the SIMD headers' instruction-set namespace keeps
# their entities apart. Only x86 has a rung whose lane count differs from the
# baseline's.
if(EINSUMS_WITH_SIMD_DISPATCH
   AND NOT EINSUMS_SIMD_NATIVE_ARCH
   AND "${EINSUMS_SIMD_TARGET_CPU}" STREQUAL ""
   AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64"
)
  einsums_simd_rung_flags(v3 _isa_wide_flags _isa_wide_ordinal _isa_wide_ok "IsaNamespace_test")
  if(_isa_wide_ok)
    einsums_add_executable(
      IsaNamespace_test INTERNAL_FLAGS
      SOURCES IsaNamespace.cpp IsaNamespaceWide.cpp
      NOINSTALL
    )
    set_source_files_properties(IsaNamespaceWide.cpp PROPERTIES COMPILE_OPTIONS "${_isa_wide_flags}")
    # The wide unit cannot share the baseline precompiled header.
    set_target_properties(IsaNamespace_test PROPERTIES DISABLE_PRECOMPILE_HEADERS ON)
    einsums_add_unit_test("Modules.SIMD" IsaNamespace)
  endif()
endif()

# GenericKernel compares the vector and scalar instantiations of one kernel bit for bit. GCC contracts a
# separate multiply and add into an FMA by default (-ffp-contract=fast), even in ISO mode and in
# intrinsic code, and decides per instantiation, so with it on the comparison would test the
# compiler's contraction choices rather than the library's operations.
if(NOT MSVC)
  foreach(_target GenericKernel_test GenericKernel_simd_v2_test GenericKernel_simd_v3_test GenericKernel_simd_v4_test
                  GenericKernel_simd_sme_test
  )
    if(TARGET ${_target})
      target_compile_options(${_target} PRIVATE -ffp-contract=off)
    endif()
  endforeach()
endif()
