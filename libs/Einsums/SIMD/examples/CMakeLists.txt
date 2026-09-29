#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

if(EINSUMS_WITH_EXAMPLES)
  einsums_add_pseudo_target(Examples.Modules.Einsums.SIMD)
  einsums_add_pseudo_dependencies(Examples.Modules Examples.Modules.Einsums.SIMD)
  if(EINSUMS_WITH_TESTS AND EINSUMS_WITH_TESTS_EXAMPLES)
    einsums_add_pseudo_target(Tests.Examples.Modules.Einsums.SIMD)
    einsums_add_pseudo_dependencies(
      Tests.Examples.Modules Tests.Examples.Modules.Einsums.SIMD
    )
  endif()
else()
  return()
endif()

set(example_programs
    SIMDBasics
    PlatformInfo
    Saxpy
    DotProduct
    ComplexAxpy
    ColumnSum
    BlockTranspose
    ClampAndSanitize
    IntegerBits
    QuantizedGemv
    StreamingScale
    RuntimeDispatch
    GemmMicroKernel
    OperatorsAndFunctions
)

# RuntimeDispatchKernel.cpp is not a program: it is compiled once per instruction-set rung, each
# copy in its own arch_<rung> namespace, and RuntimeDispatch picks one at run time. The x86 rungs
# only; the SME rung would buy a streaming loop nothing. On other targets this is one native copy.
include(Einsums_AddSIMDDispatch)
einsums_add_simd_dispatch_sources(SIMDExampleKernelRungs IMPL RuntimeDispatchKernel.cpp RUNGS baseline v2 v3 v4)

foreach(example_program ${example_programs})
  set(sources ${example_program}.cpp)

  source_group("Source Files" FILES ${sources})

  einsums_add_executable(
    SIMD_${example_program} INTERNAL_FLAGS
    SOURCES ${sources}
    FOLDER "Examples/Modules/SIMD"
    NOINSTALL
  )

  einsums_add_example_target_dependencies("Modules.Einsums.SIMD" SIMD_${example_program})

  # OperatorsAndFunctions shows that a * b - p is zero where fmadd(a, b, -p) is not, which holds
  # only while the compiler does not fuse the expression itself. GCC does by default under GNU
  # extensions, even through the intrinsics; Clang fuses only within one source expression.
  if(example_program STREQUAL "OperatorsAndFunctions")
    target_compile_options(SIMD_OperatorsAndFunctions PRIVATE $<$<CXX_COMPILER_ID:GNU>:-ffp-contract=off>)
  endif()

  if(example_program STREQUAL "RuntimeDispatch")
    target_sources(SIMD_RuntimeDispatch PRIVATE ${SIMDExampleKernelRungs})
    # Tells EINSUMS_SIMD_LADDER which arch_<rung> copies exist.
    set_source_files_properties(
      RuntimeDispatch.cpp PROPERTIES COMPILE_DEFINITIONS "${SIMDExampleKernelRungs_DEFINITIONS}"
    )
  endif()

  if(EINSUMS_WITH_TESTS AND EINSUMS_WITH_TESTS_EXAMPLES)
    einsums_add_example_test(
      "Modules.Einsums.SIMD" SIMD_${example_program}
    )
  endif()
endforeach()
