#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

include(CheckLibraryExists)

#:
#: einsums_add_config_test
#:
#:    Configure–time probe that compiles (and optionally runs) a tiny test to set a cache variable.
#:
#:    This helper performs a feature/availability check by generating or loading a small source,
#:    compiling it (and optionally executing it), then setting the named cache variable to the
#:    boolean result. It can also inject follow‑up compile definitions on success and raise a
#:    fatal error when required checks fail.
#:
#:    **Signature**
#:    ``einsums_add_config_test(<variable>
#:        [FILE] [EXECUTE] [GPU] [NOT_REQUIRED]
#:        [SOURCE <code-or-path>] [ROOT <dir>]
#:        [CMAKECXXFEATURE <feature>] [CHECK_CXXSTD <int>] [EXTRA_MSG <text>]
#:        [CXXFLAGS <...>]
#:        [INCLUDE_DIRECTORIES <...>]
#:        [LINK_DIRECTORIES <...>]
#:        [COMPILE_DEFINITIONS <...>]
#:        [LIBRARIES <...>]
#:        [ARGS <...>]
#:        [DEFINITIONS <...>]
#:        [REQUIRED <error-message>]
#:    )``
#:
#:    **Positional Argument**
#:    - ``variable`` *(required)*:
#:      The name of the cache variable to set (boolean). Also used as a prefix for
#:      auto‑generated artifact names and per‑test variables (e.g., ``<variable>_RESULT``).
#:
#:    **Boolean Options**
#:    - ``FILE``: Treat ``SOURCE`` as a file path (see *Source selection*). Otherwise, the
#:      literal contents of ``SOURCE`` are written into a generated test file.
#:    - ``EXECUTE``: After compiling, run the test program via ``try_run`` and
#:      only succeed if it **compiles** and **returns 0**. Without this flag, only compilation
#:      via ``try_compile`` is required.
#:    - ``GPU``: Generate a ``.cu`` test file instead of ``.cpp`` when writing from literal
#:      source (affects only the file extension and CUDA parameters).
#:    - ``NOT_REQUIRED``: Suppress hard failure even if the check is marked ``REQUIRED``.
#:
#:    **One‑value Keywords**
#:    - ``SOURCE``:
#:      * If ``FILE`` is set, a relative path (to ``ROOT`` or project) of a source file to test.
#:      * Otherwise, a literal source **snippet** (string) to be written to a temp file.
#:    - ``ROOT``: Root directory used with ``FILE`` mode. If given, the source is resolved as
#:      ``<ROOT>/share/${PROJECT_NAME}/<SOURCE>``; otherwise ``${PROJECT_SOURCE_DIR}/<SOURCE>``.
#:    - ``CMAKECXXFEATURE``: Name of a CMake C++ compile feature (e.g., ``cxx_constexpr``).
#:      If present in ``CMAKE_CXX_COMPILE_FEATURES`` and the variable is not
#:      pre‑set and ``GPU`` is not used, the check is considered an immediate success without
#:      compiling.
#:    - ``CHECK_CXXSTD``: Minimum C++ standard required (integer). If this value is **greater**
#:      than ``EINSUMS_WITH_CXX_STANDARD``, the test is skipped: the cache variable
#:      (if set) is unset and the function returns early.
#:    - ``EXTRA_MSG``: Extra text appended to the status line printed for this test.
#:
#:    **Multi‑value Keywords**
#:    - ``CXXFLAGS``: Extra compiler flags for the test; combined with a strictness flag
#:      (``/WX`` on MSVC or ``-Werror`` otherwise).
#:    - ``INCLUDE_DIRECTORIES``: Additional include search paths.
#:    - ``LINK_DIRECTORIES``: Additional link directories.
#:    - ``COMPILE_DEFINITIONS``: Extra preprocessor definitions (without ``-D``).
#:    - ``LIBRARIES``: Libraries/targets to link the test against.
#:    - ``ARGS``: Runtime arguments passed when ``EXECUTE`` is enabled.
#:    - ``DEFINITIONS``: On **success**, each item is forwarded to
#:      ``einsums_add_config_define`` to set global config defines.
#:    - ``REQUIRED``: If provided and the test fails, emit its string as a fatal error (unless
#:      ``NOT_REQUIRED`` is set).
#:
#:    **Behavior**
#:    1. **Early standard gate** — If ``CHECK_CXXSTD`` is set and exceeds
#:       ``EINSUMS_WITH_CXX_STANDARD``, unset any pre‑existing cache for
#:       ``<variable>``, log a note, and return.
#:    2. **CMake feature fast‑path** — If ``<variable>`` is **not** already defined, ``GPU`` is
#:       **off**, and ``CMAKECXXFEATURE`` is present in
#:       ``CMAKE_CXX_COMPILE_FEATURES``, set the result to success without compiling.
#:    3. **Source selection**
#:       - *FILE mode*: Resolve ``SOURCE`` via ``ROOT`` (if provided) or project source dir.
#:       - *Literal mode*: Write ``SOURCE`` into a generated test file under
#:         ``${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/config_tests/``, using ``.cu`` if
#:         ``GPU`` is set, otherwise ``.cpp``.
#:    4. **Compilation environment**
#:       - Collect current directory ``INCLUDE_DIRECTORIES``, ``LINK_DIRECTORIES``,
#:         and ``COMPILE_DEFINITIONS``; merge with user‑supplied lists.
#:       - Append global compile **flags** from the Einsums properties
#:         ``EINSUMS_TARGET_COMPILE_OPTIONS_PUBLIC`` and ``EINSUMS_TARGET_COMPILE_OPTIONS_PRIVATE``
#:         (excluding generator expressions) into the definitions string passed to the test
#:         compiler invocation.
#:       - Apply strict warnings as errors (``/WX`` or ``-Werror``) and append ``CXXFLAGS``.
#:       - When not executing, pass CUDA standard parameters if
#:         ``EINSUMS_WITH_CUDA`` is enabled and the compiler is **not** NVHPC.
#:    5. **Run the check**
#:       - With ``EXECUTE`` (and not cross‑compiling): use ``try_run``; success
#:         requires both successful compilation and exit code 0.
#:       - Without ``EXECUTE``: use ``try_compile``; success requires compilation.
#:    6. **Result handling**
#:       - Set ``<variable>`` (CACHE INTERNAL BOOL) to the outcome.
#:       - Print a concise status line (with ``EXTRA_MSG`` when provided).
#:       - On **success**, forward each item in ``DEFINITIONS`` to
#:         ``einsums_add_config_define``.
#:       - On **failure** and if ``REQUIRED`` is provided (and ``NOT_REQUIRED`` is **not** set),
#:         log the compiler output via ``einsums_warn`` and then abort with ``einsums_error``.
#:
#:    **Side effects**
#:    - Creates files under ``${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/config_tests``.
#:    - May update/consume Einsums global properties of compile flags.
#:    - Adjusts ``CMAKE_CXX_FLAGS`` (and ``CMAKE_CUDA_FLAGS`` / ``CMAKE_HIP_FLAGS`` stanzas) for
#:      the duration of the probe invocation.
#:
#:    **Examples**
#:    Compile‑only check with a literal snippet:
#:    .. code-block:: cmake
#:
#:       einsums_add_config_test(HAS_STD_BIT_CAST
#:         SOURCE "#include <bit>\nint main(){ (void)std::bit_cast<unsigned>(0u); return 0; }"
#:         CHECK_CXXSTD 20
#:         CMAKECXXFEATURE cxx_std_20
#:         DEFINITIONS EINSUMS_HAS_STD_BIT_CAST
#:         REQUIRED "C++20 bit_cast is required for this build configuration."
#:       )
#:
#:    Execute a tiny program and use its exit status:
#:    .. code-block:: cmake
#:
#:       einsums_add_config_test(HAS_WORKING_PTHREADS
#:         EXECUTE
#:         SOURCE "int main(){ return 0; }"
#:         LIBRARIES Threads::Threads
#:         EXTRA_MSG "link and run"
#:         REQUIRED "Pthreads support is required."
#:       )
#:
#:    Test from a file path under project sources:
#:    .. code-block:: cmake
#:
#:       einsums_add_config_test(HAS_VENDOR_FOO FILE
#:         SOURCE "cmake/tests/has_vendor_foo.cpp"
#:         CXXFLAGS "-DUSE_VENDOR_FOO"
#:       )
#:
#:    **See also**
#:    - ``try_compile``, ``try_run``
#:    - ``CMAKE_CXX_COMPILE_FEATURES``
#:    - ``einsums_add_config_define``
#:    - ``EINSUMS_WITH_CXX_STANDARD``
function(einsums_add_config_test variable)
  set(options FILE EXECUTE GPU NOT_REQUIRED)
  set(one_value_args SOURCE ROOT CMAKECXXFEATURE CHECK_CXXSTD EXTRA_MSG)
  set(mutlti_value_args
      CXXFLAGS
      INCLUDE_DIRECTORIES
      LINK_DIRECTORIES
      COMPILE_DEFINITIONS
      LIBRARIES
      ARGS
      DEFINITIONS
      REQUIRED
  )
  cmake_parse_arguments(${variable} "${options}" "${one_value_args}" "${mutlti_value_args}" ${ARGN})

  if(${variable}_CHECK_CXXSTD AND ${variable}_CHECK_CXXSTD GREATER EINSUMS_WITH_CXX_STANDARD)
    if(DEFINED ${variable})
      unset(${variable} CACHE)
      einsums_info(
        "Unsetting ${variable} because of EINSUMS_WITH_CXX_STANDARD (${EINSUMS_WITH_CXX_STANDARD})"
      )
    endif()
    return()
  endif()

  set(_run_msg)
  # Check CMake feature tests if the user didn't override the value of this variable:
  if(NOT DEFINED ${variable} AND NOT ${variable}_GPU)
    if(${variable}_CMAKECXXFEATURE)
      # We don't have to run out own feature test if there is a corresponding cmake feature test and
      # cmake reports that the feature is available.
      list(FIND CMAKE_CXX_COMPILE_FEATURES ${${variable}_CMAKECXXFEATURE} _pos)
      if(NOT ${__pos} EQUAL -1)
        set(${variable}
            TRUE
            CACHE INTERNAL ""
        )
        set(_run_msg "Success (cmake feature test)")
      endif()
    endif()
  endif()

  if(NOT DEFINED ${variable})
    file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/config_tests")

    string(TOLOWER "${variable}" variable_lc)
    if(${variable}_FILE)
      if(${variable}_ROOT)
        set(test_source "${${variable}_ROOT}/share/${PROJECT_NAME}/${${variable}_SOURCE}")
      else()
        set(test_source "${PROJECT_SOURCE_DIR}/${${variable}_SOURCE}")
      endif()
    else()
      if(${variable}_GPU)
        set(test_source
            "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/config_tests/${variable_lc}.cu"
        )
      else()
        set(test_source
            "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/config_tests/${variable_lc}.cpp"
        )
      endif()
      file(WRITE "${test_source}" "${${variable}_SOURCE}\n")
    endif()
    set(test_binary ${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/config_tests/${variable_lc})

    get_directory_property(CONFIG_TEST_INCLUDE_DIRS INCLUDE_DIRECTORIES)
    get_directory_property(CONFIG_TEST_LINK_DIRS LINK_DIRECTORIES)
    set(COMPILE_DEFINITIONS_TMP)
    set(CONFIG_TEST_COMPILE_DEFINITIONS)
    get_directory_property(COMPILE_DEFINITIONS_TMP COMPILE_DEFINITIONS)
    foreach(def IN LISTS COMPILE_DEFINITIONS_TMP ${variable}_COMPILE_DEFINITIONS)
      set(CONFIG_TEST_COMPILE_DEFINITIONS "${CONFIG_TEST_COMPILE_DEFINITIONS} -D${def}")
    endforeach()
    get_property(
      EINSUMS_TARGET_COMPILE_OPTIONS_PUBLIC_VAR GLOBAL
      PROPERTY EINSUMS_TARGET_COMPILE_OPTIONS_PUBLIC
    )
    get_property(
      EINSUMS_TARGET_COMPILE_OPTIONS_PRIVATE_VAR GLOBAL
      PROPERTY EINSUMS_TARGET_COMPILE_OPTIONS_PRIVATE
    )
    set(EINSUMS_TARGET_COMPILE_OPTIONS_VAR ${EINSUMS_TARGET_COMPILE_OPTIONS_PUBLIC_VAR}
                                           ${EINSUMS_TARGET_COMPILE_OPTIONS_PRIVATE_VAR}
    )
    foreach(_flag ${EINSUMS_TARGET_COMPILE_OPTIONS_VAR})
      if(NOT "${_flag}" MATCHES "^\\$.*")
        set(CONFIG_TEST_COMPILE_DEFINITIONS "${CONFIG_TEST_COMPILE_DEFINITIONS} ${_flag}")
      endif()
    endforeach()

    set(CONFIG_TEST_INCLUDE_DIRS ${CONFIG_TEST_INCLUDE_DIRS} ${${variable}_INCLUDE_DIRECTORIES})
    set(CONFIG_TEST_LINK_DIRS ${CONFIG_TEST_LINK_DIRS} ${${variable}_LINK_DIRECTORIES})

    set(CONFIG_TEST_LINK_LIBRARIES ${${variable}_LIBRARIES})

    set(additional_cmake_flags)
    if(MSVC)
      set(additional_cmake_flags "-WX")
    else()
      set(additional_cmake_flags "-Werror")
    endif()

    if(${variable}_EXECUTE)
      if(NOT CMAKE_CROSSCOMPILING)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${additional_cmake_flags} ${${variable}_CXXFLAGS}")
        # cmake-format: off
        try_run(
            ${variable}_RUN_RESULT ${variable}_COMPILE_RESULT
            ${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/config_tests
            ${test_source}
            COMPILE_DEFINITIONS ${CONFIG_TEST_COMPILE_DEFINITIONS}
            CMAKE_FLAGS
            "-DINCLUDE_DIRECTORIES=${CONFIG_TEST_INCLUDE_DIRS}"
            "-DLINK_DIRECTORIES=${CONFIG_TEST_LINK_DIRS}"
            "-DLINK_LIBRARIES=${CONFIG_TEST_LINK_LIBRARIES}"
            CXX_STANDARD ${EINSUMS_WITH_CXX_STANDARD}
            CXX_STANDARD_REQUIRED ON
            CXX_EXTENSIONS FALSE
            RUN_OUTPUT_VARIABLE ${variable}_OUTPUT
            ARGS ${${variable}_ARGS}
        )
        # cmake-format: on
        if(${variable}_COMPILE_RESULT AND NOT ${variable}_RUN_RESULT)
          set(${variable}_RESULT TRUE)
        else()
          set(${variable}_RESULT FALSE)
        endif()
      else()
        set(${variable}_RESULT FALSE)
      endif()
    else()
      if(EINSUMS_WITH_CUDA AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
        set(cuda_parameters CUDA_STANDARD ${CMAKE_CUDA_STANDARD})
      endif()
      # if(EINSUMS_WITH_HIP) set(hip_parameters HIP_STANDARD "${CMAKE_HIP_STANDARD}") endif()
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${additional_cmake_flags} ${${variable}_CXXFLAGS}")
      set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${additional_cmake_flags} ${${variable}_CXXFLAGS}")
      set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} ${additional_cmake_flags} ${${variable}_CXXFLAGS}")
      # cmake-format: off
      try_compile(
          ${variable}_RESULT
          ${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/config_tests
          ${test_source}
          COMPILE_DEFINITIONS ${CONFIG_TEST_COMPILE_DEFINITIONS}
          CMAKE_FLAGS
          "-DINCLUDE_DIRECTORIES=${CONFIG_TEST_INCLUDE_DIRS}"
          "-DLINK_DIRECTORIES=${CONFIG_TEST_LINK_DIRS}"
          "-DLINK_LIBRARIES=${CONFIG_TEST_LINK_LIBRARIES}"
          OUTPUT_VARIABLE ${variable}_OUTPUT
          CXX_STANDARD ${EINSUMS_WITH_CXX_STANDARD}
          CXX_STANDARD_REQUIRED ON
          CXX_EXTENSIONS FALSE
          ${cuda_parameters}
          ${hip_parameters}
          COPY_FILE ${test_binary}
      )
      # cmake-format: on
      einsums_debug("Compile test: ${variable}")
      einsums_debug("Compilation output: ${${variable}_OUTPUT}")
    endif()

    set(_run_msg "Success")
  else()
    set(${variable}_RESULT ${${variable}})
    if(NOT _run_msg)
      set(_run_msg "pre-set to ${${variable}}")
    endif()
  endif()

  set(_msg "Performing Test ${variable}")
  if(${variable}_EXTRA_MSG)
    set(_msg "${_msg} (${${variable}_EXTRA_MSG})")
  endif()

  if(${variable}_RESULT)
    set(_msg "${_msg} - ${_run_msg}")
  else()
    set(_msg "${_msg} - Failed")
  endif()

  set(${variable}
      ${${variable}_RESULT}
      CACHE INTERNAL ""
  )
  einsums_info(${_msg})

  if(${variable}_RESULT)
    foreach(definition ${${variable}_DEFINITIONS})
      einsums_add_config_define(${definition})
    endforeach()
  elseif(${variable}_REQUIRED AND NOT ${variable}_NOT_REQUIRED)
    einsums_warn("Test failed, detailed output:\n\n${${variable}_OUTPUT}")
    einsums_error(${${variable}_REQUIRED})
  endif()
endfunction()

# ##################################################################################################
function(einsums_cpuid target variable)
  einsums_add_config_test(
    ${variable}
    SOURCE cmake/tests/cpuid.cpp
    COMPILE_DEFINITIONS "${boost_include_dir}" "${include_dir}"
    FILE EXECUTE
    ARGS "${target}" ${ARGN}
  )
endfunction()

# ##################################################################################################
function(einsums_check_for_unistd_h)
  einsums_add_config_test(EINSUMS_WITH_UNISTD_H SOURCE cmake/tests/unistd_h.cpp FILE ${ARGN})
endfunction()

# ##################################################################################################
function(einsums_check_for_omp_tools_h)
  einsums_add_config_test(EINSUMS_WITH_OMP_TOOLS_H SOURCE cmake/tests/omp_tools.cpp FILE ${ARGN})
endfunction()

# ##################################################################################################
function(einsums_check_for_libfun_std_experimental_optional)
  einsums_add_config_test(
    EINSUMS_WITH_LIBFUN_EXPERIMENTAL_OPTIONAL
    SOURCE cmake/tests/libfun_std_experimental_optional.cpp FILE ${ARGN}
  )
endfunction()

# ##################################################################################################
function(einsums_check_for_cxx11_std_atomic)
  # Make sure EINSUMS_HAVE_LIBATOMIC is removed from the cache if necessary
  if(NOT EINSUMS_WITH_CXX11_ATOMIC)
    unset(EINSUMS_CXX11_STD_ATOMIC_LIBRARIES CACHE)
  endif()

  # First see if we can build atomics with no -latomics. We make sure to override REQUIRED, if set,
  # with NOT_REQUIRED so that we can use the fallback test further down.
  set(check_not_required)
  if(NOT MSVC)
    set(check_not_required NOT_REQUIRED)
  endif()

  einsums_add_config_test(
    EINSUMS_WITH_CXX11_ATOMIC
    SOURCE cmake/tests/cxx11_std_atomic.cpp
    LIBRARIES ${EINSUMS_CXX11_STD_ATOMIC_LIBRARIES} FILE ${ARGN} ${check_not_required}
  )

  if(NOT MSVC)
    # Sometimes linking against libatomic is required, if the platform doesn't support lock-free
    # atomics. We already know that MSVC works
    if(NOT EINSUMS_WITH_CXX11_ATOMIC)
      set(EINSUMS_CXX11_STD_ATOMIC_LIBRARIES
          atomic
          CACHE STRING "std::atomics need separate library" FORCE
      )
      unset(EINSUMS_WITH_CXX11_ATOMIC CACHE)
      einsums_add_config_test(
        EINSUMS_WITH_CXX11_ATOMIC
        SOURCE cmake/tests/cxx11_std_atomic.cpp
        LIBRARIES ${EINSUMS_CXX11_STD_ATOMIC_LIBRARIES} FILE ${ARGN} EXTRA_MSG "with -latomic"
      )
      if(NOT EINSUMS_WITH_CXX11_ATOMIC)
        unset(EINSUMS_CXX11_STD_ATOMIC_LIBRARIES CACHE)
        unset(EINSUMS_WITH_CXX11_ATOMIC CACHE)
      endif()
    endif()
  endif()
endfunction()

# Separately check for 128 bit atomics
function(einsums_check_for_cxx11_std_atomic_128bit)
  # First see if we can build atomics with no -latomics. We make sure to override REQUIRED, if set,
  # with NOT_REQUIRED so that we can use the fallback test further down.
  set(check_not_required)
  if(NOT MSVC)
    set(check_not_required NOT_REQUIRED)
  endif()

  einsums_add_config_test(
    EINSUMS_WITH_CXX11_ATOMIC_128BIT
    SOURCE cmake/tests/cxx11_std_atomic_128bit.cpp
    LIBRARIES ${EINSUMS_CXX11_STD_ATOMIC_LIBRARIES} FILE ${ARGN} NOT_REQUIRED
  )

  if(NOT MSVC)
    # Sometimes linking against libatomic is required, if the platform doesn't support lock-free
    # atomics. We already know that MSVC works
    if(NOT EINSUMS_WITH_CXX11_ATOMIC_128BIT)
      set(EINSUMS_CXX11_STD_ATOMIC_LIBRARIES
          atomic
          CACHE STRING "std::atomics need separate library" FORCE
      )
      unset(EINSUMS_WITH_CXX11_ATOMIC_128BIT CACHE)
      einsums_add_config_test(
        EINSUMS_WITH_CXX11_ATOMIC_128BIT
        SOURCE cmake/tests/cxx11_std_atomic_128bit.cpp
        LIBRARIES ${EINSUMS_CXX11_STD_ATOMIC_LIBRARIES} FILE ${ARGN} EXTRA_MSG "with -latomic"
      )
      if(NOT EINSUMS_WITH_CXX11_ATOMIC_128BIT)
        # Adding -latomic did not help, so we don't attempt to link to it later
        unset(EINSUMS_CXX11_STD_ATOMIC_LIBRARIES CACHE)
        unset(EINSUMS_WITH_CXX11_ATOMIC_128BIT CACHE)
      endif()
    endif()
  endif()
endfunction()

# ##################################################################################################
function(einsums_check_for_cxx11_std_shared_ptr_lwg3018)
  einsums_add_config_test(
    EINSUMS_WITH_CXX11_SHARED_PTR_LWG3018 SOURCE cmake/tests/cxx11_std_shared_ptr_lwg3018.cpp FILE
                                                 ${ARGN}
  )
endfunction()

# ##################################################################################################
function(einsums_check_for_c11_aligned_alloc)
  einsums_add_config_test(
    EINSUMS_WITH_C11_ALIGNED_ALLOC SOURCE cmake/tests/c11_aligned_alloc.cpp FILE ${ARGN}
  )
endfunction()

function(einsums_check_for_cxx17_std_aligned_alloc)
  einsums_add_config_test(
    EINSUMS_WITH_CXX17_STD_ALIGNED_ALLOC SOURCE cmake/tests/cxx17_std_aligned_alloc.cpp FILE
                                                ${ARGN}
  )
endfunction()

# ##################################################################################################
function(einsums_check_for_cxx11_std_quick_exit)
  einsums_add_config_test(
    EINSUMS_WITH_CXX11_STD_QUICK_EXIT SOURCE cmake/tests/cxx11_std_quick_exit.cpp FILE ${ARGN}
  )
endfunction()

# ##################################################################################################
function(einsums_check_for_cxx17_aligned_new)
  einsums_add_config_test(
    EINSUMS_WITH_CXX17_ALIGNED_NEW
    SOURCE cmake/tests/cxx17_aligned_new.cpp FILE ${ARGN}
    REQUIRED
  )
endfunction()

# ##################################################################################################
function(einsums_check_for_cxx17_std_transform_scan)
  einsums_add_config_test(
    EINSUMS_WITH_CXX17_STD_TRANSFORM_SCAN_ALGORITHMS
    SOURCE cmake/tests/cxx17_std_transform_scan_algorithms.cpp FILE ${ARGN}
  )
endfunction()

# ##################################################################################################
function(einsums_check_for_cxx17_std_scan)
  einsums_add_config_test(
    EINSUMS_WITH_CXX17_STD_SCAN_ALGORITHMS SOURCE cmake/tests/cxx17_std_scan_algorithms.cpp FILE
                                                  ${ARGN}
  )
endfunction()

# ##################################################################################################
function(einsums_check_for_cxx17_copy_elision)
  einsums_add_config_test(
    EINSUMS_WITH_CXX17_COPY_ELISION SOURCE cmake/tests/cxx17_copy_elision.cpp FILE ${ARGN}
  )
endfunction()

# ##################################################################################################
function(einsums_check_for_cxx17_memory_resource)
  einsums_add_config_test(
    EINSUMS_WITH_CXX17_MEMORY_RESOURCE SOURCE cmake/tests/cxx17_memory_resource.cpp FILE ${ARGN}
  )
endfunction()

# ##################################################################################################
function(einsums_check_for_cxx20_no_unique_address_attribute)
  einsums_add_config_test(
    EINSUMS_WITH_CXX20_NO_UNIQUE_ADDRESS_ATTRIBUTE
    SOURCE cmake/tests/cxx20_no_unique_address_attribute.cpp FILE ${ARGN} CHECK_CXXSTD 20
  )
endfunction()

# ##################################################################################################
function(einsums_check_for_cxx20_std_disable_sized_sentinel_for)
  einsums_add_config_test(
    EINSUMS_WITH_CXX20_STD_DISABLE_SIZED_SENTINEL_FOR
    SOURCE cmake/tests/cxx20_std_disable_sized_sentinel_for.cpp FILE ${ARGN} CHECK_CXXSTD 20
  )
endfunction()

# ##################################################################################################
function(einsums_check_for_cxx20_trivial_virtual_destructor)
  einsums_add_config_test(
    EINSUMS_WITH_CXX20_TRIVIAL_VIRTUAL_DESTRUCTOR
    SOURCE cmake/tests/cxx20_trivial_virtual_destructor.cpp FILE ${ARGN}
  )
endfunction()

# ##################################################################################################
function(einsums_check_for_cxx23_static_call_operator)
  einsums_add_config_test(
    EINSUMS_WITH_CXX23_STATIC_CALL_OPERATOR SOURCE cmake/tests/cxx23_static_call_operator.cpp FILE
                                                   ${ARGN}
  )
endfunction()

# ##################################################################################################
function(einsums_check_for_cxx23_static_call_operator_gpu)
  if(EINSUMS_WITH_GPU_SUPPORT)
    set(static_call_operator_test_extension "cpp")
    if(EINSUMS_WITH_CUDA AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
      set(static_call_operator_test_extension "cu")
    elseif(EINSUMS_WITH_HIP)
      set(static_call_operator_test_extension "hip")
    endif()

    set(extra_cxxflags)
    if(EINSUMS_WITH_CUDA AND CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
      set(extra_cxxflags "-x cu")
    endif()

    einsums_add_config_test(
      EINSUMS_WITH_CXX23_STATIC_CALL_OPERATOR_GPU
      SOURCE cmake/tests/cxx23_static_call_operator.${static_call_operator_test_extension} GPU FILE
             ${ARGN}
    )
  endif()
endfunction()

# ##################################################################################################
function(einsums_check_for_cxx_lambda_capture_decltype)
  einsums_add_config_test(
    EINSUMS_WITH_CXX_LAMBDA_CAPTURE_DECLTYPE SOURCE cmake/tests/cxx_lambda_capture_decltype.cpp
                                                    FILE ${ARGN}
  )
endfunction()
