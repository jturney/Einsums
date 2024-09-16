#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(coroutines_headers
        einsums/coroutines/coroutine.hpp
        einsums/coroutines/coroutine_fwd.hpp
        einsums/coroutines/stackless_coroutine.hpp
        einsums/coroutines/detail/combined_tagged_state.hpp
        einsums/coroutines/detail/context_base.hpp
        einsums/coroutines/detail/context_generic_context.hpp
        einsums/coroutines/detail/context_impl.hpp
        einsums/coroutines/detail/context_linux_x86.hpp
        einsums/coroutines/detail/context_posix.hpp
        einsums/coroutines/detail/context_windows_fibers.hpp
        einsums/coroutines/detail/coroutine_accessor.hpp
        einsums/coroutines/detail/coroutine_impl.hpp
        einsums/coroutines/detail/coroutine_self.hpp
        einsums/coroutines/detail/coroutine_stackful_self.hpp
        einsums/coroutines/detail/coroutine_stackless_self.hpp
        einsums/coroutines/detail/get_stack_pointer.hpp
        einsums/coroutines/detail/posix_utility.hpp
        einsums/coroutines/detail/swap_context.hpp
        einsums/coroutines/detail/tss.hpp
        einsums/coroutines/thread_enums.hpp
        einsums/coroutines/thread_id_type.hpp
)

set(coroutines_sources thread_enums.cpp)

if (MSVC)
  if (EINSUMS_WITH_SWAP_CONTEXT_EMULATION)
    # ##############################################################################################
    # Emulation of SwapContext on Windows
    # ##############################################################################################
    enable_language(ASM_MASM)
    if (NOT CMAKE_ASM_MASM_COMPILER)
      einsums_error(
              "SwitchToFiber emulation can not be enabled. The masm compiler \
         could not be found. Try setting the ASM_MASM environment variable to the \
         assembler executable (ml.exe/ml64.exe) or disable the emulation by setting \
         EINSUMS_WITH_SWAP_CONTEXT_EMULATION to Off"
      )
    endif ()

    einsums_add_config_define_namespace(
            DEFINE EINSUMS_HAVE_SWAP_CONTEXT_EMULATION NAMESPACE COROUTINES
    )

    set(switch_to_fiber_source "${CMAKE_CURRENT_SOURCE_DIR}/src/switch_to_fiber.asm")
    set(switch_to_fiber_object "${CMAKE_CURRENT_BINARY_DIR}/switch_to_fiber.obj")
    add_custom_command(
            OUTPUT "${switch_to_fiber_object}"
            COMMAND "${CMAKE_ASM_MASM_COMPILER}" /Fo "${switch_to_fiber_object}" /nologo /c
            "${switch_to_fiber_source}"
            DEPENDS "${switch_to_fiber_source}"
            VERBATIM
    )
  endif ()
elseif (EINSUMS_WITH_SWAP_CONTEXT_EMULATION)
  einsums_error(
          "The option EINSUMS_WITH_SWAP_CONTEXT_EMULATION is not supported on "
          "this platform, please disable the emulation by setting it to Off"
  )
endif ()

set_source_files_properties(
        src/detail/coroutine_impl.cpp PROPERTIES SKIP_UNITY_BUILD_INCLUSION TRUE
)
set_source_files_properties(
        src/detail/coroutine_self.cpp PROPERTIES SKIP_UNITY_BUILD_INCLUSION TRUE
)

include(Einsums_AddModule)
einsums_add_module(
  einsums coroutines
  GLOBAL_HEADER_GEN ON
  SOURCES ${coroutines_sources}
  HEADERS ${coroutines_headers}
        OBJECTS "${switch_to_fiber_object}"
        DEPENDENCIES $<TARGET_NAME_IF_EXISTS:einsums_internal::valgrind>
  MODULE_DEPENDENCIES
        einsums_assertion
        einsums_config
        einsums_errors
        einsums_functional
        einsums_memory
        einsums_thread_support
        einsums_type_support
        einsums_util
  CMAKE_SUBDIRS examples tests
)

if (MSVC AND EINSUMS_WITH_SWAP_CONTEXT_EMULATION)
  einsums_info(
          "    SwitchToFiber emulation is enabled, using compiler: '${CMAKE_ASM_MASM_COMPILER}'"
  )
endif ()
