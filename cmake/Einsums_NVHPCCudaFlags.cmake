#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

#:
#: einsums_add_nvhpc_cuda_flags
#:
#:    Compile one ``.cu`` source with the NVHPC host compiler instead of nvcc.
#:
#:    ``nvc++`` accepts CUDA C++ directly when given ``-cuda``; it does not want the
#:    source routed through a separate CUDA language toolchain. So the source is
#:    marked ``LANGUAGE CXX`` and given ``-cuda``, rather than being left to CMake's
#:    default ``.cu`` -> CUDA mapping.
#:
#:    **Signature**
#:    ``einsums_add_nvhpc_cuda_flags(<source>)``
#:
#:    .. note::
#:
#:       This function is called from ``einsums_add_library`` and
#:       ``einsums_add_executable`` under ``CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC"``.
#:       Both call sites predate this definition - the function did not exist
#:       anywhere in the tree, so any NVHPC build that reached a ``.cu`` source
#:       failed at configure time with "Unknown CMake command". Nothing reached it
#:       because ``libs/`` has no ``.cu`` sources, but the call was live.
#:
#:       This definition is therefore **untested**: no NVHPC toolchain has built this
#:       project. It is written to be the obvious correct thing rather than left as a
#:       guaranteed configure error. Revisit when someone actually builds with NVHPC.
#:
function(einsums_add_nvhpc_cuda_flags source)
  set_source_files_properties(${source} PROPERTIES LANGUAGE CXX)
  set_property(
    SOURCE ${source}
    APPEND
    PROPERTY COMPILE_OPTIONS "-cuda"
  )
endfunction()
