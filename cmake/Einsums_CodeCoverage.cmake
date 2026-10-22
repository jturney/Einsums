#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

# Setup coverage for specific target
function(einsums_append_coverage_compiler_flags_to_target name private_interface)
  if(EINSUMS_WITH_COVERAGE)
    target_compile_options(
      ${name}
      ${private_interface}
      $<$<COMPILE_LANG_AND_ID:CXX,AppleClang,Clang,IntelLLVM>:-fprofile-instr-generate
      -fcoverage-mapping>
      $<$<COMPILE_LANG_AND_ID:CXX,GNU>:--coverage>
    )
    target_link_options(
      ${name}
      ${private_interface}
      $<$<COMPILE_LANG_AND_ID:CXX,AppleClang,Clang,IntelLLVM>:-fprofile-instr-generate
      -fcoverage-mapping>
      $<$<COMPILE_LANG_AND_ID:CXX,GNU>:-lgcov
      --coverage>
    )
  endif()
endfunction()
