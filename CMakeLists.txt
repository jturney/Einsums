#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.25.2 FATAL_ERROR) # 3.25.2 for CUDA20 support.

# Overrides must go before project() statement, otherwise they are ignored

# ##################################################################################################
# C++ overrides
# ##################################################################################################
# Not currently used but provided as an example for the future.
#
# set(CMAKE_USER_MAKE_RULES_OVERRIDE_CXX
# "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Einsums_CXXOverrides.cmake" )

# ##################################################################################################
# Build type (needs to be handled before project command below)
# ##################################################################################################
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE
      "RelWithDebInfo"
      CACHE STRING "Configuration type (one of Debug, RelWithDebInfo, Release, MinSizeRel)" FORCE
  )
endif()

project(Einsums LANGUAGES C CXX ASM)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# ##################################################################################################
# Special handling for building tests/examples against a previously installed version of Einsums
# ##################################################################################################
if(EINSUMS_WITH_INSTALLED_VERSION)
  include(cmake/installed_einsums.cmake)
  return()
endif()

# ##################################################################################################
# Build all of Einsums
# ##################################################################################################
set(EINSUMS_VERSION_MAJOR 2)
set(EINSUMS_VERSION_MINOR 0)
set(EINSUMS_VERSION_PATCH 0)
string(TIMESTAMP EINSUMS_VERSION_DATE "%Y%m%d" UTC)
set(EINSUMS_VERSION_TAG "-trunk")

set(EINSUMS_VERSION "${EINSUMS_VERSION_MAJOR}.${EINSUMS_VERSION_MINOR}.${EINSUMS_VERSION_PATCH}")
set(EINSUMS_VERSION_FULL
    "${EINSUMS_VERSION_MAJOR}.${EINSUMS_VERSION_MINOR}.${EINSUMS_VERSION_PATCH}${EINSUMS_VERSION_TAG}"
)

set(EINSUMS_LIBRARY_VERSION "${EINSUMS_VERSION}")
set(EINSUMS_SOVERSION ${EINSUMS_VERSION_MAJOR})
set(EINSUMS_PACKAGE_NAME Einsums)
# To keep track of the einsums_root when other subprojects are declared
set(EINSUMS_SOURCE_DIR "${PROJECT_SOURCE_DIR}")
set(EINSUMS_BINARY_DIR "${PROJECT_BINARY_DIR}")

# ##################################################################################################
# CMake configuration
# ##################################################################################################
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake")

include(GNUInstallDirs)
include(Einsums_Utils)
include(CheckLanguage)

# Explicitly set certain policies
einsums_set_cmake_policy(CMP0144 NEW)
einsums_set_cmake_policy(CMP0155 OLD)

set(CMAKE_CXX_FLAGS_SAFE ${CMAKE_CXX_FLAGS})
include(Einsums_CheckCXXStandard)

include(CheckCXXCompilerFlag)
include(CMakeDependentOption)

# include additional macro definitions
include(Einsums_AddConfigTest)
include(Einsums_AddDefinitions)
include(Einsums_CreateSymbolicLink)

einsums_force_out_of_tree_build(
  "This project requires an out-of-source-tree build. See README.rst. Clean your CMake cache and CMakeFiles if this message persists."
)

if(NOT EINSUMS_CMAKE_LOGLEVEL)
  set(EINSUMS_CMAKE_LOGLEVEL "WARN")
endif()

# print initial diagnostics
einsums_info("CMake version:   ${CMAKE_VERSION}")
einsums_info("Einsums version: ${EINSUMS_VERSION}")

# ##################################################################################################
# Reset cached variables that need to be re-filled
unset(EINSUMS_EXPORT_TARGETS CACHE)
unset(EINSUMS_EXPORT_INTERNAL_TARGETS CACHE)
unset(EINSUMS_LIBS CACHE)
unset(EINSUMS_ENABLED_MODULES CACHE)

# ##################################################################################################
# Set up dummy compiler flags targets
include(Einsums_CompilerFlagsTargets)

# ##################################################################################################
# Use shared libraries by default
einsums_option(
  BUILD_SHARED_LIBS BOOL "Build using shared libraries (default: ON)" ON CATEGORY "Build Targets"
)

if(MSVC)
  einsums_option(
    EINSUMS_WITH_VS_STARTUP_PROJECT STRING
    "Defines the startup project for the Einsums solution (default: ALL_BUILD)." "ALL_BUILD"
    ADVANCED
  )
  if(EINSUMS_WITH_VS_STARTUP_PROJECT)
    set(VS_STARTUP_PROJECT ${EINSUMS_WITH_VS_STARTUP_PROJECT})
  endif()
endif()

# ##################################################################################################
# Set our build options cache variables which are customizable by users

einsums_option(
  EINSUMS_WITH_DEPRECATION_WARNINGS BOOL "Enable warnings for deprecated facilities. (default: ON)"
  ON ADVANCED CATEGORY "Debugging"
)

if(EINSUMS_WITH_DEPRECATION_WARNINGS)
  einsums_add_config_cond_define(EINSUMS_HAVE_DEPRECATION_WARNINGS 1)
endif()

# Ultimately this will be something different
set(DEFAULT_MALLOC "system")
einsums_option(
  EINSUMS_WITH_MALLOC
  STRING
  "Define which allocator should be linked in. Options are: system and mimalloc. (default: ${DEFAULT_MALLOC})"
  ${DEFAULT_MALLOC}
  STRINGS "system;mimalloc"
  CATEGORY "Build Targets" ADVANCED
)

# Compiler related build options
einsums_option(
  EINSUMS_WITH_COMPILER_WARNINGS BOOL "Enable compiler warnings (default: ON)" ON ADVANCED
  CATEGORY "Debugging"
)

einsums_option(
  EINSUMS_WITH_COMPILER_WARNINGS_AS_ERROR BOOL "Turn compiler warnings into error (default: OFF)"
  OFF ADVANCED CATEGORY "Debugging"
)

einsums_option(
  EINSUMS_WITH_EXECUTABLE_PREFIX STRING
  "Sets a string to be prepended to executable names (default: none)." ""
  CATEGORY "Build Targets" ADVANCED
  CATEGORY "Build Targets"
)

einsums_option(
  EINSUMS_WITH_DOCUMENTATION BOOL "Build the Einsums documentation (default OFF)." OFF
  CATEGORY "Build Targets"
)

if(EINSUMS_WITH_DOCUMENTATION)
  set(valid_output_formats html)
  einsums_option(
    EINSUMS_WITH_DOCUMENTATION_OUTPUT_FORMATS
    STRING
    "List of documentation output formats to generate. Valid options are ${valid_output_formats}. Multiple values can be separated with semicolons. (default html)."
    "html"
    CATEGORY "Build Targets"
  )

  foreach(output_format ${EINSUMS_WITH_DOCUMENTATION_OUTPUT_FORMATS})
    if(NOT ${output_format} IN_LIST valid_output_formats)
      einsums_error(
        "${output_format} is not a valid value for EINSUMS_WITH_DOCUMENTATION_OUTPUT_FORMATS. Valid output format are: ${valid_output_formats}."
      )
    endif()
  endforeach()
endif()

einsums_option(
  EINSUMS_WITH_BINARY_DIR
  PATH
  "Root directory for libraries and executables for the build. This does not affect the installed location of libraries and executables."
  "${PROJECT_BINARY_DIR}"
  CATEGORY "Build Targets" ADVANCED
)

einsums_option(
  EINSUMS_WITH_COVERAGE BOOL "Enable code coverage generation" OFF ADVANCED CATEGORY "Debugging"
)

# Provide public options to globally control RTTI and EH
einsums_option(EINSUMS_ENABLE_EH BOOL "Enable Exception handling" OFF ADVANCED CATEGORY "Compiling")
einsums_option(EINSUMS_ENABLE_RTTI BOOL "Enable run time type information" OFF ADVANCED CATEGORY "Compiling")
if(EINSUMS_ENABLE_EH AND NOT EINSUMS_ENABLE_RTTI)
  einsums_error("Exception handling requires RTTI. You must set EINSUMS_ENABLE_RTTI to ON")
endif()

# ##################################################################################################

if(WIN32)
  set(EINSUMS_WITH_PSEUDO_DEPENDENCIES
      OFF
      CACHE INTERNAL "" FORCE
  )
else()
  set(EINSUMS_WITH_PSEUDO_DEPENDENCIES
      ON
      CACHE INTERNAL "" FORCE
  )
endif()

einsums_option(
  EINSUMS_WITH_UNITY_BUILD BOOL "Enable unity build for certain build targets (default: OFF)" OFF
  ADVANCED CATEGORY "Build Targets"
)
if(EINSUMS_WITH_UNITY_BUILD)
  set(EINSUMS_WITH_UNITY_BUILD_OPTION UNITY_BUILD)
endif()

einsums_option(
  EINSUMS_WITH_PRECOMPILED_HEADERS BOOL
  "Enabled precompiled headers for certain build targets (experimental) (default: OFF)" OFF ADVANCED
  CATEGORY "Build Targets"
)
if(EINSUMS_WITH_PRECOMPILED_HEADERS)
  set(EINSUMS_WITH_PRECOMPILED_HEADERS_INTERNAL ON)
  # Only create the targets here. They will be configured later once all the modules are known.
  add_library(einsums_precompiled_headers OBJECT libs/src/dummy.cpp)
  add_executable(einsums_exe_precompiled_headers libs/src/dummy_main.cpp)

  set_target_properties(einsums_precompiled_headers PROPERTIES FOLDER "Core")
  set_target_properties(einsums_exe_precompiled_headers PROPERTIES FOLDER "Core")
endif()

# ##################################################################################################
einsums_option(
  EINSUMS_WITH_EXAMPLES BOOL "Build examples (default: ON)" ON CATEGORY "Build Targets"
)
einsums_option(EINSUMS_WITH_TESTS BOOL "Build tests (default: ON)" ON CATEGORY "Build Targets")
einsums_option(
  EINSUMS_WITH_TESTS_BENCHMARKS BOOL "Build benchmark tests (default: ON)" ON DEPENDS
  "EINSUMS_WITH_TESTS" CATEGORY "Build Targets"
)
einsums_option(
  EINSUMS_WITH_TESTS_REGRESSIONS BOOL "Build regression tests (default: ON)" ON DEPENDS
  "EINSUMS_WITH_TESTS" CATEGORY "Build Targets"
)
einsums_option(
  EINSUMS_WITH_TESTS_UNIT BOOL "Build unit tests (default: ON)" ON DEPENDS "EINSUMS_WITH_TESTS"
  CATEGORY "Build Targets"
)
einsums_option(
  EINSUMS_WITH_TESTS_HEADERS
  BOOL
  "Build header tests (default: ON)"
  ON
  ADVANCED
  DEPENDS
  "EINSUMS_WITH_TESTS"
  CATEGORY "Build Targets"
)
einsums_option(
  EINSUMS_WITH_TESTS_EXTERNAL_BUILD
  BOOL
  "Build external CMake build tests (default: ON)"
  ON
  ADVANCED
  DEPENDS
  "EINSUMS_WITH_TESTS"
  CATEGORY "Build Targets"
)
einsums_option(
  EINSUMS_WITH_TESTS_EXAMPLES
  BOOL
  "Add examples as tests (default: ON)"
  ON
  DEPENDS
  "EINSUMS_WITH_TESTS;EINSUMS_WITH_EXAMPLES"
  ADVANCED
  CATEGORY "Build Targets"
)
einsums_option(
  EINSUMS_WITH_COMPILE_ONLY_TESTS BOOL
  "Create build system support for compile time only tests (default: ON)" ON DEPENDS
  "EINSUMS_WITH_TESTS" CATEGORY "Build Targets"
)
einsums_option(
  EINSUMS_WITH_FAIL_COMPILE_TESTS BOOL
  "Create build system support for fail compile tests (default: ON)" ON DEPENDS "EINSUMS_WITH_TESTS"
  CATEGORY "Build Targets"
)

einsums_option(
  EINSUMS_WITH_EXPERIMENTAL BOOL "If true, build experimental Einsums modules (default: OFF)" OFF
  CATEGORY "Build Targets"
)

set(EINSUMS_FULL_RPATH_DEFAULT ON)
if(APPLE OR WIN32)
  set(EINSUMS_FULL_RPATH_DEFAULT OFF)
endif()
einsums_option(
  EINSUMS_WITH_FULL_RPATH
  BOOL
  "Build and link libraries and executables with full RPATHs (default: ${EINSUMS_FULL_RPATH_DEFAULT})"
  ${EINSUMS_FULL_RPATH_DEFAULT}
  ADVANCED
  CATEGORY "Build Targets"
)

# ##################################################################################################
# CUDA/HIP configuration
# ##################################################################################################
set(CUDA_OPTION_STRING "Enable support for CUDA (default: OFF)")
set(HIP_OPTION_STRING "Enable support for HIP (default: OFF)")
einsums_option(EINSUMS_WITH_CUDA BOOL "${CUDA_OPTION_STRING}" OFF ADVANCED CATEGORY "Build Targets")

# No need for the user to specify the option explicitly
einsums_option(EINSUMS_WITH_HIP BOOL "${HIP_OPTION_STRING}" OFF ADVANCED CATEGORY "Build Targets")
if("${CMAKE_CXX_COMPILER}" MATCHES "hipcc$")
  set(EINSUMS_WITH_HIP
      ON
      CACHE BOOL "${HIP_OPTION_STRING}" FORCE
  )
endif()

if(EINSUMS_WITH_CUDA AND EINSUMS_WITH_HIP)
  einsums_error(
    "EINSUMS_WITH_CUDA=ON and EINSUMS_WITH_HIP=ON. Only one of them can be on at the same time.\
 Note: EINSUMS_WITH_HIP is automatically enabled when compiling with hipcc."
  )
endif()

if(EINSUMS_WITH_CUDA OR EINSUMS_WITH_HIP)
  einsums_add_config_define(EINSUMS_HAVE_GPU_SUPPORT)
  set(EINSUMS_WITH_GPU_SUPPORT ON)
  set(EINSUMS_WITH_GPU_SUPPORT
      ON
      CACHE INTERNAL "" FORCE
  )

else()
  set(EINSUMS_WITH_GPU_SUPPORT
      OFF
      CACHE INTERNAL "" FORCE
  )
endif()

# ##################################################################################################
# Utility configuration
# ##################################################################################################
set(EINSUMS_HIDDEN_VISIBILITY_DEFAULT ON)
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  set(EINSUMS_HIDDEN_VISIBILITY_DEFAULT OFF)
endif()
if(APPLE)
  set(EINSUMS_HIDDEN_VISIBILITY_DEFAULT OFF)
endif()

einsums_option(
  EINSUMS_WITH_HIDDEN_VISIBILITY
  BOOL
  "Use -fvisibility=hidden for builds on platforms which support it (default: ${EINSUMS_HIDDEN_VISIBILITY_DEFAULT})"
  ${EINSUMS_HIDDEN_VISIBILITY_DEFAULT}
  ADVANCED
  CATEGORY "Build Targets"
)

einsums_option(
  EINSUMS_WITH_BACKTRACES BOOL "Attach backtraces to einsums exceptions (default: ON)" ON
  CATEGORY "Debugging" ADVANCED
)

# ##################################################################################################
# We create a target to contain libraries like rt, dl, etc. in order to remove variables
add_library(einsums_base_libraries INTERFACE)

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
  target_link_libraries(einsums_base_libraries INTERFACE imf svml irng intlc)
endif()

if(EINSUMS_WITH_BACKTRACES)
  einsums_info("Back traces are enabled.")
  einsums_add_config_define(EINSUMS_HAVE_BACKTRACES)
  include(Einsums_SetupCppTrace)

  einsums_option(
    EINSUMS_WITH_THREAD_BACKTRACE_DEPTH STRING
    "Thread stack back trace depth being captured (default: 20)" "20" CATEGORY "Debugging" ADVANCED
  )
  einsums_add_config_define(
    EINSUMS_HAVE_THREAD_BACKTRACE_DEPTH ${EINSUMS_WITH_THREAD_BACKTRACE_DEPTH}
  )

  if(APPLE)
    find_program(dsymutil_EXECUTABLE NAMES dsymutil)
  endif()

endif()

einsums_option(
  EINSUMS_WITH_PROFILER BOOL "Enable Einsums profiling. (Default: ON)" ON CATEGORY "Profiling"
                                                                                   ADVANCED
)
einsums_option(
  EINSUMS_WITH_PROFILER_INTERNAL BOOL "Enable profiling internal Einsums code. (Default: OFF)" OFF
  DEPENDS "EINSUMS_WITH_PROFILER" CATEGORY "Profiling" ADVANCED
)
if(EINSUMS_WITH_PROFILER)
  einsums_add_config_define(EINSUMS_HAVE_PROFILER)
  if(EINSUMS_WITH_PROFILER_INTERNAL)
    einsums_add_config_define(EINSUMS_HAVE_PROFILER_INTERNAL)
  endif()
endif()

einsums_option(
  EINSUMS_WITH_ITTNOTIFY BOOL "Enable Amplifier (ITT) instrumentation support." OFF
  CATEGORY "Profiling" ADVANCED
)

# Debugging related build options
einsums_option(
  EINSUMS_WITH_VALGRIND BOOL "Enable Valgrind instrumentation support." OFF CATEGORY "Debugging"
                                                                                     ADVANCED
)

einsums_option(
  EINSUMS_WITH_TESTS_VALGRIND BOOL "Run selected tests with valgrind." OFF CATEGORY "Debugging"
                                                                                    ADVANCED
)

einsums_option(
  EINSUMS_WITH_TESTS_VALGRIND_OPTIONS STRING "Use these options when running tests under valgrind."
  "--error-exitcode=1" CATEGORY "Debugging" ADVANCED
)

einsums_option(
  EINSUMS_WITH_SANITIZERS STRING "Configure with sanitizer instrumentation support." OFF
  STRINGS "OFF;;address;leak;memory;thread;undefined"
  CATEGORY "Debugging" ADVANCED
)

# Tuning and other things.
einsums_option(
  EINSUMS_ROW_MAJOR_DEFAULT BOOL "If true, tensors will be created in row-major by default. If false, they will be column-major." ON
  CATEGORY "Generic" ADVANCED
)

if(EINSUMS_ROW_MAJOR_DEFAULT)
  einsums_add_config_define(EINSUMS_ROW_MAJOR_DEFAULT)
endif()

# ##################################################################################################
# Special diagnostic flags, do not enable by default, only if needed
einsums_option(
  EINSUMS_WITH_CHECK_MODULE_DEPENDENCIES BOOL
  "Verify that no modules are cross-referenced from a different module category (default: OFF)" OFF
  ADVANCED CATEGORY "Debugging"
)

einsums_option(
  EINSUMS_BUILD_PYTHON
  STRING
  "Build the Python module. A version for the Python build may be specified, or the value may be set to 'ON' for your current Python interpreter.
  Disable by specifying 'OFF'. Defaults to 'ON'."
  ON
  CATEGORY "Features" ADVANCED
)

if(EINSUMS_WITH_ITTNOTIFY)
  find_package(Amplifier)

  if(NOT AMPLIFIER_FOUND)
    einsums_error(
      "Intel Amplifier could not be found and EINSUMS_WITH_ITTNOTIFY=On, please specify AMPLIFIER_ROOT to point to the root of your Amplifier installation"
    )
  endif()

  einsums_add_config_define(EINSUMS_HAVE_ITTNOTIFY)
endif()

# ##################################################################################################
# Detect BLAS/LAPACK
find_package(TargetLAPACK REQUIRED)
get_property(
  EINSUMS_LINALG_VENDOR
  TARGET tgt::lapack
  PROPERTY VENDOR
)
get_property(
  EINSUMS_LINALG_INT_INTERFACE
  TARGET tgt::lapack
  PROPERTY INT_INTERFACE
)
einsums_info("EINSUMS_LINALG_INT_INTERFACE ${EINSUMS_LINALG_INT_INTERFACE}")

find_package(OpenMP REQUIRED MODULE COMPONENTS CXX)
target_link_libraries(einsums_base_libraries INTERFACE OpenMP::OpenMP_CXX)
# einsums_info("OpenMP Version ${OpenMP_CXX_MAJOR_VERSION}")
einsums_check_for_omp_tools_h(DEFINITIONS EINSUMS_HAVE_OMP_TOOLS_H)

# ##################################################################################################
# C++ feature tests
# ##################################################################################################
# Need to set up CUDA/HIP before feature tests to enable the respective languages
include(Einsums_SetupCUDA)
include(Einsums_SetupHIP)
include(Einsums_SetupFFT)
include(Einsums_SetupTracy)
# include(Einsums_SetupHighway)   ## This is currently found in LinearAlgebra/examples
include(Einsums_PerformCXXFeatureTests)
einsums_perform_cxx_feature_tests()

# ##################################################################################################
# Check for misc system headers
# ##################################################################################################
einsums_check_for_unistd_h(DEFINITIONS EINSUMS_HAVE_UNISTD_H)

if(NOT WIN32)
  # ################################################################################################
  # Macro definitions for system headers
  # ################################################################################################
  add_definitions(-D_GNU_SOURCE)

  if(APPLE)
    einsums_add_compile_flag_if_available(-ftemplate-depth=256)
  endif()

else()
  add_definitions(-D_ITERATOR_DEBUG_LEVEL=0)
endif(NOT WIN32)

# Configure Warnings
if(EINSUMS_WITH_COMPILER_WARNINGS)
  if(MSVC) # Adding special warning settings for the MSVC compiler ...
    einsums_add_compile_flag(-W3)

    # MSVC2012/2013 are overeager to report 'qualifier applied to function type has no meaning;
    # ignored'
    einsums_add_compile_flag(-wd4180)

    # Boost.Lockfree triggers 'warning C4307: '+' : integral constant overflow' which is benign
    einsums_add_compile_flag(-wd4307)

    # object allocated on the heap may not be aligned
    einsums_add_compile_flag(-wd4316)

    # max symbol length exceeded
    einsums_add_compile_flag(-wd4503)

    # 'int': forcing value to bool 'true' or 'false' (performance warning)
    einsums_add_compile_flag(-wd4800)

    # vcpkg enables the /utf-8 option which causes (benign) warnings in the Spirit headers: The file
    # contains a character starting at offset ... that is illegal in the current source character
    # set
    einsums_add_compile_flag(-wd4828)

  else() # Trial and error approach for any other compiler ...
    einsums_add_compile_flag_if_available(-Wall)
    einsums_add_compile_flag_if_available(-Wextra)
    einsums_add_compile_flag_if_available(-Wpedantic)
    einsums_add_compile_flag_if_available(-Wno-strict-aliasing)
    einsums_add_compile_flag_if_available(-Wno-sign-promo)
    einsums_add_compile_flag_if_available(-Wno-attributes)
    einsums_add_compile_flag_if_available(-Wno-cast-align)

    # We do not in general guarantee ABI compatibility between C++ standards, so we ignore this
    # warning
    einsums_add_compile_flag_if_available(-Wno-noexcept-type)

    # Be extra strict about format checks Boost.Logging is built on fprintf, sadly
    einsums_add_compile_flag_if_available(-Wformat=2)
    einsums_add_compile_flag_if_available(-Wno-format-nonliteral)

    # Self initialization is dangerous
    einsums_add_compile_flag_if_available(-Winit-self)

    # For portability
    einsums_add_compile_flag_if_available(-Wdouble-promotion)

    # Warn about casting that violates qualifiers or alignment
    einsums_add_compile_flag_if_available(-Wcast-qual)
    if(NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
      # Clang is overeager in reporting cast alignment problems in Boost
      einsums_add_compile_flag_if_available(-Wcast-align)
    endif()

    einsums_add_compile_flag_if_available(-Werror=parentheses)
    einsums_add_compile_flag_if_available(-Werror=reorder)
    einsums_add_compile_flag_if_available(-Werror=return-type)
    einsums_add_compile_flag_if_available(-Werror=sequence-point)
    einsums_add_compile_flag_if_available(-Werror=uninitialized)
    einsums_add_compile_flag_if_available(-Werror=format)
    einsums_add_compile_flag_if_available(-Werror=missing-braces)
    einsums_add_compile_flag_if_available(-Werror=sign-compare)
  endif()
endif()

# Configure compiler warnings as errors
if(EINSUMS_WITH_COMPILER_WARNINGS_AS_ERRORS)
  if(MSVC)
    einsums_add_compile_flag(-WX)
  else()
    einsums_add_compile_flag_if_available(-Werror)
  endif()
endif()

# Diagnostics
if(MSVC)
  # Display full paths in diagnostics
  einsums_add_compile_flag(-FC)
  if(CMAKE_CL_64)
    set(__target_arch "x86_64")
  else()
    set(__target_arch "x86")
  endif()
  einsums_info("Architecture detected: ${__target_arch}")
else()
  # Show the flags that toggle each warning
  einsums_add_compile_flag_if_available(-fdiagnostics-show-option)

  # VLAs are a GNU extensions that we forbid as they are not supported on MSVC
  einsums_add_compile_flag_if_available(-Werror=vla)
  # No return statement in a non-void function can lead to garbage return values in GCC.
  einsums_add_compile_flag_if_available(-Werror=return-type)

  # We get false positives all over the place with this.
  if(CMAKE_COMPILER_IS_GNUCXX)
    einsums_add_compile_flag_if_available(-Wno-unused-but-set-parameter)
    einsums_add_compile_flag_if_available(-Wno-unused-but-set-variable)
    # Uninitialized variables are bad, earlier compilers issue spurious warnings
    einsums_add_compile_flag_if_available(-Werror=uninitialized)
    einsums_add_compile_flag_if_available(-Wno-unused-local-typedefs)
    # -Werror=maybe-uninitialized leads to false positives.
    einsums_add_compile_flag_if_available(-Wno-maybe-uninitialized)
  endif()

  # Silence warning about __sync_fetch_and_nand changing semantics
  einsums_add_compile_flag_if_available(-Wno-sync-nand)

  # Silence warnings about deleting polymorphic objects with non-virtual dtors. These come from
  # within Boost.
  if(CMAKE_COMPILER_IS_GNUCXX)
    einsums_add_compile_flag_if_available(-Wno-delete-non-virtual-dtor)
  endif()

  # Check if our libraries have unresolved symbols if(NOT APPLE AND NOT WIN32 AND NOT
  # EINSUMS_WITH_SANITIZERS ) einsums_add_link_flag_if_available(-Wl,-z,defs) endif()
  if(WIN32)
    target_link_libraries(einsums_base_libraries INTERFACE psapi WS2_32 mswsock)
  endif()

  if("${EINSUMS_PLATFORM_UC}" STREQUAL "BLUEGENEQ")
    einsums_add_compile_flag_if_available(-Wno-deprecated-register)
  endif()

  if(EINSUMS_WITH_HIDDEN_VISIBILITY)
    einsums_add_compile_flag_if_available(-fvisibility=hidden)
    einsums_add_link_flag_if_available(-fvisibility=hidden)
    einsums_add_config_define(EINSUMS_HAVE_ELF_HIDDEN_VISIBILITY)
  endif()

  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    einsums_add_compile_flag_if_available(-Wno-cast-align)
  endif()

  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    # Disable the following warnings: #1170: invalid redeclaration of nested class
    einsums_add_compile_flag_if_available(-wd1170)
    # #858: type qualifier on return type is meaningless
    einsums_add_compile_flag_if_available(-wd858)
    # #1098: the qualifier on this friend declaration is ignored
    einsums_add_compile_flag_if_available(-wd1098)
    # #488: template parameter not used in declaring the parameter type
    einsums_add_compile_flag_if_available(-wd488)
    # #2203: cast discards qualifiers from target type (needed for mvapich2 mpi header)
    einsums_add_compile_flag_if_available(-wd2203)
    # #2536: cannot specify explicit initializer for arrays
    einsums_add_compile_flag_if_available(-wd2536)
    # #1286: invalid attribute
    einsums_add_compile_flag_if_available(-wd1286)
  endif()

  include(Einsums_TargetArch)

  einsums_target_architecture(__target_arch)

  einsums_info("Architecture detected: ${__target_arch}")
endif()

# store target architecture for later use
set(EINSUMS_WITH_TARGET_ARCHITECTURE
    ${__target_arch}
    CACHE INTERNAL "" FORCE
)

# ##################################################################################################
# Find Our dependencies: These are all dependencies needed to build the core library. Dependencies
# that are only needed by plugins, examples or tests should be found separately in the appropriate
# subdirectory. add_subdirectory(external)

# ##################################################################################################
# Find other dependencies.
include(Einsums_SetupFmt)
include(Einsums_SetupCatch)
include(Einsums_SetupHDF5)
include(Einsums_SetupSpdlog)
include(Einsums_SetupAllocator)

if(EINSUMS_USE_LIBRETT AND EINSUMS_WITH_GPU_SUPPORT)
  include(Einsums_SetupLibreTT)
endif()

if(EINSUMS_BUILD_PYTHON OR EINSUMS_BUILD_PYTHON MATCHES "3(\.[0-9]+)*")
  if(EINSUMS_BUILD_PYTHON MATCHES "3(\.[0-9]+)*")
    find_package(
      Python "${EINSUMS_BUILD_PYTHON}" EXACT COMPONENTS Interpreter Development NumPy REQUIRED
    )
  else()
    find_package(Python 3.10 COMPONENTS Interpreter Development NumPy REQUIRED)
  endif()

  include(Einsums_SetupPybind11)

  find_package(pybind11)
endif()

if(EINSUMS_WITH_SANITIZERS AND NOT MSVC)
  einsums_info("Adding sanitizer flags for ${EINSUMS_WITH_SANITIZERS}")
  target_compile_options(einsums_public_flags INTERFACE -fsanitize=${EINSUMS_WITH_SANITIZERS})
  target_link_options(einsums_public_flags INTERFACE -fsanitize=${EINSUMS_WITH_SANITIZERS})
  einsums_add_config_define(EINSUMS_HAVE_SANITIZERS)
endif()

# Add exception handling flags based on the compiler and platform
#if(MSVC)
#  # For MSVC, use /EHsc or /EHa depending on the level of SEH required
#  add_compile_options($<$<CXX_COMPILER_ID:MSVC>:/EHsc>)
#elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
#  # For Clang and ClangCL
#  if(WIN32)
#    # ClangCL on Windows
#    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-fcxx-exceptions -fseh-exceptions>)
#  else()
#    # Clang on macOS/Linux
#    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-fcxx-exceptions>)
#  endif()
#elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
#  # For GCC (Linux/macOS)
#  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-fexceptions>)
#endif()

if (EINSUMS_ENABLE_EH)
  target_compile_options(einsums_public_flags INTERFACE $<$<CXX_COMPILER_ID:MSVC>:/EHsc>)
else()
  target_compile_definitions(einsums_public_flags INTERFACE $<$<CXX_COMPILER_ID:MSVC>:_HAS_EXCEPTIONS=0>)
  target_compile_options(einsums_public_flags INTERFACE $<$<CXX_COMPILER_ID:MSVC>:/EHs-c->)

  target_compile_options(einsums_public_flags INTERFACE $<$<BOOL:${EINSUMS_COMPILER_IS_GCC_COMPATIBLE}>:-fno-exceptions>)
endif()

if (EINSUMS_ENABLE_RTTI)
  target_compile_options(einsums_public_flags INTERFACE $<$<CXX_COMPILER_ID:MSVC>:/GR>)
else()
  target_compile_options(einsums_public_flags INTERFACE $<$<CXX_COMPILER_ID:MSVC>:/GR->)
  target_compile_options(einsums_public_flags INTERFACE $<$<BOOL:${EINSUMS_COMPILER_IS_GCC_COMPATIBLE}>:-fno-rtti>)
endif()

  # Note: on windows systems the ':' will be converted to a ';' at runtime
einsums_add_config_cond_define(
  EINSUMS_DEFAULT_COMPONENT_PATH_SUFFIXES
  "\"/${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME}:/${CMAKE_INSTALL_BINDIR}/${PROJECT_NAME}:/lib/${PROJECT_NAME}:/bin/${PROJECT_NAME}\""
)

# ##################################################################################################
# search path configuration
# ##################################################################################################
if(EINSUMS_WITH_FULL_RPATH)
  include(Einsums_SetFullRPATH)
endif()

# ##################################################################################################
# Git commit detection
# ##################################################################################################
include(Einsums_GitCommit)
einsums_add_config_define(EINSUMS_HAVE_GIT_COMMIT "\"${EINSUMS_WITH_GIT_COMMIT}\"")

include(Einsums_SetOutputPaths)

# ##################################################################################################
# Add custom targets for tests
# ##################################################################################################
if(EINSUMS_WITH_TESTS)
  einsums_add_pseudo_target(Tests)
  if(EINSUMS_WITH_TESTS_UNIT)
    einsums_add_pseudo_target(Tests.Unit)
    einsums_add_pseudo_dependencies(Tests Tests.Unit)
  endif()
  if(EINSUMS_WITH_TESTS_REGRESSIONS)
    einsums_add_pseudo_target(Tests.Regressions)
    einsums_add_pseudo_dependencies(Tests Tests.Regressions)
  endif()
  if(EINSUMS_WITH_TESTS_BENCHMARKS)
    einsums_add_pseudo_target(Tests.Performance)
    einsums_add_pseudo_dependencies(Tests Tests.Performance)
  endif()
  if(EINSUMS_WITH_TESTS_HEADERS)
    einsums_add_pseudo_target(Tests.Headers)
    einsums_add_pseudo_dependencies(Tests Tests.Headers)
  endif()
  if(EINSUMS_WITH_EXAMPLES AND EINSUMS_WITH_TESTS_EXAMPLES)
    einsums_add_pseudo_target(Tests.Examples)
    einsums_add_pseudo_dependencies(Tests Tests.Examples)
  endif()

  enable_testing()
  include(CTest)

  add_subdirectory(testing)
endif()

if(EINSUMS_WITH_EXAMPLES)
  einsums_add_pseudo_target(Examples)
endif()

# ##################################################################################################
# Debug library postfix
# ##################################################################################################
set(CMAKE_DEBUG_POSTFIX "d")
set(EINSUMS_DEBUG_POSTFIX "d")

# ##################################################################################################
# Add libraries
# ##################################################################################################
add_subdirectory(libs)
if(EINSUMS_BUILD_PYTHON)
  add_subdirectory(einsums)
endif()

# ##################################################################################################
# Documentation toolchain (Sphinx, Doxygen, Breathe)
# ##################################################################################################
include(Einsums_Documentation)

if(EINSUMS_WITH_DOCUMENTATION)
  add_subdirectory(docs)
endif()

# ##################################################################################################
# Simple timing tools
# ##################################################################################################
einsums_option(EINSUMS_WITH_PROFILING_DEVTOOLS BOOL "Enable profile programs in devtools/profiling" OFF CATEGORY "Advanced" ADVANCED)
if (EINSUMS_WITH_PROFILING_DEVTOOLS)
  add_subdirectory(devtools/profiling)
endif ()

# ##################################################################################################
# installation instructions
# ##################################################################################################
# Install all einsums header that have been configured using various CMake options
install(
  DIRECTORY "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/${PROJECT_NAME}/"
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}
  COMPONENT core
  FILES_MATCHING
  PATTERN "*.hpp"
  PATTERN "*.h"
  PATTERN ".git" EXCLUDE
  PATTERN "CMakeFiles" EXCLUDE
  PATTERN "CTestFiles" EXCLUDE
)

install(
  # Install all einsums CMake utility files
  DIRECTORY cmake/
  DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/cmake/${PROJECT_NAME}
  COMPONENT core
  PATTERN "templates" EXCLUDE
  PATTERN "packaging" EXCLUDE
  PATTERN ".git" EXCLUDE
)

install(
  FILES "${PROJECT_SOURCE_DIR}/LICENSE.txt"
  DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}
  COMPONENT license
)

# ##################################################################################################
# print overall configuration summary
include(Einsums_PrintSummary)
einsums_create_configuration_summary("Configuration summary:\n--" "einsums")

include(Einsums_ExportTargets)
# Modules can't link to this if not exported
install(
  TARGETS einsums_base_libraries
  EXPORT einsums_internal_targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT einsums_base_libraries
)
einsums_export_internal_targets(einsums_base_libraries)

# ##################################################################################################
# store cache vars and their values in order for them to be forwarded to the projects (needs to be
# before the EINSUMS_GeneratePackage call)
include(Einsums_ForwardCacheVariables)

# ##################################################################################################
# External build system support.
include(Einsums_GeneratePackage)

message("")
message("Einsums will be installed to ${CMAKE_INSTALL_PREFIX}")
message("")
