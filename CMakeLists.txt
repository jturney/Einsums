#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.24 FATAL_ERROR) # 3.24 for FetchContent_Declare(FIND_PACKAGE_ARGS

# Overrides must go before project() statement, otherwise they are ignored

# ######################################################################################################################
# C++ overrides
# ######################################################################################################################
# Not currently used but provided as an exmaple for the future.
#
# set(CMAKE_USER_MAKE_RULES_OVERRIDE_CXX "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Einsums_CXXOverrides.cmake" )

# ######################################################################################################################
# Build type (needs to be handled before project command below)
# ######################################################################################################################
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE
      "RelWithDebInfo"
      CACHE STRING "Configuration type (one of Debug, RelWithDebInfo, Release, MinSizeRel)" FORCE
  )
endif()

project(Einsums LANGUAGES CXX)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# ######################################################################################################################
# Special handling for building tests/examples against a previously installed version of Einsums
# ######################################################################################################################
if(EINSUMS_WITH_INSTALLED_VERSION)
  include(cmake/installed_einsums.cmake)
  return()
endif()

# ######################################################################################################################
# Build all of Einsums
# ######################################################################################################################
set(EINSUMS_VERSION_MAJOR 1)
set(EINSUMS_VERSION_MINOR 0)
set(EINSUMS_VERSION_SUBMINOR 0)
set(EINSUMS_VERSION_DATE 20240822)
set(EINSUMS_VERSION_TAG "-trunk")

set(EINSUMS_VERSION "${EINSUMS_VERSION_MAJOR}.${EINSUMS_VERSION_MINOR}.${EINSUMS_VERSION_SUBMINOR}")
set(EINSUMS_VERSION_FULL
    "${EINSUMS_VERSION_MAJOR}.${EINSUMS_VERSION_MINOR}.${EINSUMS_VERSION_SUBMINOR}${EINSUMS_VERSION_TAG}"
)

set(EINSUMS_LIBRARY_VERSION "${EINSUMS_VERSION}")
set(EINSUMS_SOVERSION ${EINSUMS_VERSION_MAJOR})
set(EINSUMS_PACKAGE_NAME EINSUMS)
# To keep track of the einsums_root when other subprojects are declared
set(EINSUMS_SOURCE_DIR "${PROJECT_SOURCE_DIR}")
set(EINSUMS_BINARY_DIR "${PROJECT_BINARY_DIR}")

# ######################################################################################################################
# CMake configuration
# ######################################################################################################################
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake")

include(GNUInstallDirs)
include(Einsums_Utils)

# Explicitly set certain policies
einsums_set_cmake_policy(CMP0144 NEW)
einsums_set_cmake_policy(CMP0155 OLD)

include(Einsums_CheckCXXStandard)

include(CheckCXXCompilerFlag)
include(CMakeDependentOption)

# include additional macro definitions
include(Einsums_AddConfigTest)
include(Einsums_AddDefinitions)

einsums_force_out_of_tree_build(
  "This project requires an out-of-source-tree build. See README.rst. Clean your CMake cache and CMakeFiles if this message persists."
)

if(NOT EINSUMS_CMAKE_LOGLEVEL)
  set(EINSUMS_CMAKE_LOGLEVEL "WARN")
endif()

# print initial diagnostics
einsums_info("CMake version:   ${CMAKE_VERSION}")
einsums_info("Einsums version: ${EINSUMS_VERSION}")

# ######################################################################################################################
# Reset cached variables that need to be re-filled
unset(EINSUMS_EXPORT_TARGETS CACHE)
unset(EINSUMS_EXPORT_INTERNAL_TARGETS CACHE)
unset(EINSUMS_LIBS CACHE)
unset(EINSUMS_ENABLED_MODULES CACHE)

# ######################################################################################################################
# Set up dummy compiler flags targets
# ######################################################################################################################
include(Einsums_CompilerFlagsTargets)

if(MSVC)
  einsums_option(
    EINSUMS_WITH_VS_STARTUP_PROJECT STRING "Defined the startup project for the einsums solution (default: ALL_BUILD)."
    "ALL_BUILD" ADVANCED
  )
  if(EINSUMS_WITH_VS_STARTUP_PROJECT)
    set(VS_STARTUP_PROJECT ${EINSUMS_WITH_VS_STARTUP_PROJECT})
  endif()
endif()

# ######################################################################################################################

einsums_option(
        EINSUMS_WITH_BINARY_DIR
        PATH
        "Root directory for libraries and executables for the build. This does not affect the installed location of libraries and executables."
        "${PROJECT_BINARY_DIR}"
        CATEGORY "Build Targets"
        ADVANCED
)

# ######################################################################################################################

einsums_option(
  EINSUMS_WITH_DEPRECATION_WARNINGS BOOL "Enable warnings for deprecated features. (default: ON)" ON ADVANCED
)

if(EINSUMS_WITH_DEPRECATION_WARNINGS)
  einsums_add_config_cond_define(EINSUMS_WITH_DEPRECATION_WARNINGS 1)
endif()

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

einsums_option(EINSUMS_WITH_UNITY_BUILD BOOL "Enable unity build for certain build targets (default: OFF)" OFF ADVANCED)
if(EINSUMS_WITH_UNITY_BUILD)
  set(EINSUMS_WITH_UNITY_BUILD_OPTION UNITY_BUILD)
endif()

# ######################################################################################################################
einsums_option(EINSUMS_WITH_EXAMPLES BOOL "Build examples (default: OFF)" OFF CATEGORY "Build Targets")
einsums_option(EINSUMS_WITH_TESTS BOOL "Build tests (default: ON)" ON CATEGORY "Build Targets")
einsums_option(
  EINSUMS_WITH_TESTS_BENCHMARKS BOOL "Build benchmark tests (default: ON)" ON DEPENDS "EINSUMS_WITH_TESTS"
  CATEGORY "Build Targets"
)
einsums_option(
  EINSUMS_WITH_TESTS_REGRESSIONS BOOL "Build regression tests (default: ON)" ON DEPENDS "EINSUMS_WITH_TESTS"
  CATEGORY "Build Targets"
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
  EINSUMS_WITH_COMPILE_ONLY_TESTS BOOL "Create build system support for compile time only tests (default: ON)" ON
  DEPENDS "EINSUMS_WITH_TESTS" CATEGORY "Build Targets"
)
einsums_option(
  EINSUMS_WITH_FAIL_COMPILE_TESTS BOOL "Create build system support for fail compile tests (default: ON)" ON DEPENDS
  "EINSUMS_WITH_TESTS" CATEGORY "Build Targets"
)

set(EINSUMS_FULL_RPATH_DEFAULT ON)
if(APPLE OR WIN32)
  set(EINSUMS_FULL_RPATH_DEFAULT OFF)
endif()
einsums_option(
  EINSUMS_WITH_FULL_RPATH BOOL
  "Build and link libraries and executables with full RPATHs (default: ${EINSUMS_FULL_RPATH_DEFAULT})"
  ${EINSUMS_FULL_RPATH_DEFAULT} ADVANCED
)

einsums_option(
  EINSUMS_WITH_ITTNOTIFY BOOL "Enable Amplifier (ITT) instrumentation support." ON
  CATEGORY "Profiling"
  ADVANCED
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

# ######################################################################################################################
# CUDA/HIP configuration
# ######################################################################################################################
set(CUDA_OPTION_STRING "Enable support for CUDA (default: OFF)")
set(HIP_OPTION_STRING "Enable support for HIP (default: OFF)")
einsums_option(EINSUMS_WITH_CUDA BOOL "${CUDA_OPTION_STRING}" OFF ADVANCED)

# No need for the user to specify the option explicitly
einsums_option(EINSUMS_WITH_HIP BOOL "${HIP_OPTION_STRING}" OFF ADVANCED)
if ("${CMAKE_CXX_COMPILER}" MATCHES "hipcc$")
  set(EINSUMS_WITH_HIP
          ON
          CACHE BOOL "${HIP_OPTION_STRING}" FORCE
  )
endif ()

if (EINSUMS_WITH_CUDA AND EINSUMS_WITH_HIP)
  einsums_error(
          "EINSUMS_WITH_CUDA=ON and EINSUMS_WITH_HIP=ON. Only one of them can be on at the same time.\
 Note: EINSUMS_WITH_HIP is automatically enabled when compiling with hipcc."
  )
endif ()

if (EINSUMS_WITH_CUDA OR EINSUMS_WITH_HIP)
  einsums_add_config_define(EINSUMS_HAVE_GPU_SUPPORT)
  set(EINSUMS_WITH_GPU_SUPPORT ON)
  set(EINSUMS_WITH_GPU_SUPPORT
          ON
          CACHE INTERNAL "" FORCE
  )
else ()
  set(EINSUMS_WITH_GPU_SUPPORT
          OFF
          CACHE INTERNAL "" FORCE
  )
endif ()

# ######################################################################################################################
# We create a target to contain libraries like rt, dl, etc. in order to remove variables
add_library(einsums_base_libraries INTERFACE)

# ######################################################################################################################
# Find Our dependencies: These are all dependencies needed to build the core library. Dependencies that are only needed
# by plugins, examples or tests should be found separately in the appropriate subdirectory. add_subdirectory(external)

# ######################################################################################################################
# Find other dependencies.
include(Einsums_SetupFmt)
include(Einsums_SetupHIP)

# ######################################################################################################################
# search path configuration
# ######################################################################################################################
if(EINSUMS_WITH_FULL_RPATH)
  include(Einsums_SetFullRPATH)
endif()

# ######################################################################################################################
# Git commit detection
# ######################################################################################################################
include(Einsums_GitCommit)
einsums_add_config_define(EINSUMS_HAVE_GIT_COMMIT "\"${EINSUMS_WITH_GIT_COMMIT}\"")

include(Einsums_SetOutputPaths)

# ######################################################################################################################
# Add custom targets for tests
# ######################################################################################################################
if(EINSUMS_WITH_TESTS)
  einsums_add_pseudo_target(tests)
  if(EINSUMS_WITH_TESTS_UNIT)
    einsums_add_pseudo_target(tests.unit)
    einsums_add_pseudo_dependencies(tests tests.unit)
  endif()
  if(EINSUMS_WITH_TESTS_REGRESSIONS)
    einsums_add_pseudo_target(tests.regressions)
    einsums_add_pseudo_dependencies(tests tests.regressions)
  endif()
  if(EINSUMS_WITH_TESTS_BENCHMARKS)
    einsums_add_pseudo_target(tests.performance)
    einsums_add_pseudo_dependencies(tests tests.performance)
  endif()
  if(EINSUMS_WITH_TESTS_HEADERS)
    einsums_add_pseudo_target(tests.headers)
    einsums_add_pseudo_dependencies(tests tests.headers)
  endif()
  if(EINSUMS_WITH_EXAMPLES AND EINSUMS_WITH_TESTS_EXAMPLES)
    einsums_add_pseudo_target(tests.examples)
    einsums_add_pseudo_dependencies(tests tests.examples)
  endif()

  enable_testing()
  include(CTest)

  # TODO: Uncomment add_subdirectory(testing)
endif()

if(EINSUMS_WITH_EXAMPLES)
  add_subdirectory(examples)
endif()

# ######################################################################################################################
# Debug library postfix
# ######################################################################################################################
set(CMAKE_DEBUG_POSTFIX "d")
set(EINSUMS_DEBUG_POSTFIX "d")

# ######################################################################################################################
# Add libraries
# ######################################################################################################################
add_subdirectory(libs)

# ######################################################################################################################
# Print overall configuration summary
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

# ######################################################################################################################
# store cache vars and their values in order for them to be forwarded to the projects (needs to be before the
# EINSUMS_GeneratePackage call)
include(Einsums_ForwardCacheVariables)

# ######################################################################################################################
# External build system supportP.
include(Einsums_GeneratePackage)

message("")
message("Einsums will be installed to ${CMAKE_INSTALL_PREFIX}")
message("")

# include(FeatureSummary) include(EinsumsAPI)

# # Force C++ standard globally, do not fall back, do not use compiler extensions set(CMAKE_CXX_STANDARD 20)
# set(CMAKE_CXX_STANDARD_REQUIRED ON) set(CMAKE_CXX_EXTENSIONS OFF) set(RANGES_CXX_STD 20)

# # Force position independent code globally. set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# include(EinsumsBuildTypeASAN) include(EinsumsBuildTypeMSAN) include(EinsumsBuildTypeUBSAN)

# set(ein Einsums) # Namespace

# #
# ######################################################################################################################
# # User input options                                                           # #
# ######################################################################################################################
# include(einsums_option)

# # borrowed from pybind11 since einsums_option doesn't work if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
# message(STATUS "Setting tests build type to Release as none was specified") set(CMAKE_BUILD_TYPE Release CACHE STRING
# "Choose the type of build." FORCE) set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel"
# "RelWithDebInfo") endif()

# include(autocmake_safeguards)

# einsums_option(EINSUMS_CONTINUOUSLY_TEST_EINSUM BOOL "Every call to TensorAlgebra::einsum is tested" OFF)
# einsums_option(EINSUMS_TEST_EINSUM_ABORT BOOL "Abort execution if an error is found" ON)
# einsums_option(EINSUMS_WARN_NOT_IMPLEMENTED BOOL "If a specific case is not implemented in einsum then warn the user"
# OFF) einsums_option(EINSUMS_SHOW_WARNING BOOL "Does the code report warnings to the user or not" ON)
# einsums_option(EINSUMS_RUNTIME_INDICES_CHECK BOOL "Check the sizes of corresponding indices at runtime" ON)
# einsums_option(EINSUMS_TEST_NANS BOOL "Test the result of every einsum for nans" OFF) einsums_option(EINSUMS_USE_HPTT
# BOOL "Use the HPTT package for tensor transpositions" ON) einsums_option(EINSUMS_ENABLE_TESTING BOOL "Build testing
# (requires Catch2)" ${PROJECT_IS_TOP_LEVEL}) einsums_option(EINSUMS_INSTALL_CMAKEDIR STRING "Directory to which Einsums
# CMake config files installed." share/cmake/Einsums) einsums_option(EINSUMS_H5CPP_USE_OMP_ALIGNED_ALLOC BOOL "On Mac,
# use omp_aligned_alloc instead of aligned_alloc. Helpful for old SDK." OFF) einsums_option(EINSUMS_BUILD_HIP BOOL
# "Build HIP GPU code." OFF) einsums_option(EINSUMS_COVERAGE BOOL "Enable code coverage checking" OFF) einsums_option(
# EINSUMS_FAST_INTRINSICS BOOL "When generating code for GPU, use the fast math intrinsics (__expf, __sinf, etc.) \
# rather than the normal math functions (expf, sinf, etc.) for single-precision floats and single-precision complex
# floats." OFF)

# add_feature_info(EINSUMS_CONTINUOUSLY_TEST_EINSUM ${EINSUMS_CONTINUOUSLY_TEST_EINSUM} "Test every call to einsum.
# EXPENSIVE!") add_feature_info(EINSUMS_TEST_EINSUM_ABORT ${EINSUMS_TEST_EINSUM_ABORT} "Abort execution if an error is
# found") add_feature_info(EINSUMS_WARN_NOT_IMPLEMENTED ${EINSUMS_WARN_NOT_IMPLEMENTED} "Report if a specific case is
# not implemented") add_feature_info(EINSUMS_SHOW_WARNING ${EINSUMS_SHOW_WARNING} "Does the code report warnings to the
# user or not") add_feature_info(EINSUMS_TEST_NANS ${EINSUMS_TEST_NANS} "Test the result of every einsum for nans, infs,
# and large values") add_feature_info(EINSUMS_RUNTIME_INDICES_CHECK ${EINSUMS_RUNTIME_INDICES_CHECK} "Check the sizes of
# corresponding indices at runtime") add_feature_info(EINSUMS_USE_HPTT ${EINSUMS_USE_HPTT} "Build with internal HPTT
# version")

# if(EINSUMS_BUILD_HIP) if(NOT HIP_INSTALL_DIR) message(FATAL_ERROR "Please set HIP_INSTALL_DIR.") endif()

# cmake_path(APPEND HIP_INSTALL_DIR "lib" "cmake" OUTPUT_VARIABLE __hip_cmake_dir)

# cmake_path(APPEND __hip_cmake_dir "AMDDeviceLibs" OUTPUT_VARIABLE AMDDeviceLibs_DIR) set(ENV{AMDDeviceLibs_DIR}
# ${AMDDeviceLibs_DIR})

# cmake_path(APPEND __hip_cmake_dir "amd_comgr" OUTPUT_VARIABLE amd_comgr_DIR) set(ENV{amd_comgr_DIR} ${amd_comgr_DIR})

# cmake_path(APPEND __hip_cmake_dir "hipblas" OUTPUT_VARIABLE hipblas_DIR) set(ENV{hipblas_DIR} ${hipblas_DIR})

# cmake_path(APPEND __hip_cmake_dir "hip" OUTPUT_VARIABLE hip_DIR) set(ENV{hip_DIR} ${hipblas_DIR})

# cmake_path(APPEND __hip_cmake_dir "hsa-runtime64" OUTPUT_VARIABLE hsa-runtime64_DIR) set(ENV{hsa-runtime64_DIR}
# ${hsa-runtime64_DIR})

# cmake_path(APPEND __hip_cmake_dir "hipsolver" OUTPUT_VARIABLE hipsolver_DIR) set(ENV{hipsolver_DIR} ${hipsolver_DIR})

# enable_language(HIP) find_package(hipblas REQUIRED) find_package(hipsolver REQUIRED) set(CURSES_NEED_NCURSES True)
# find_package(Curses)

# set(ENABLE_HIP ON) endif()

# # include(cmake/DetectHostArch.cmake) detect_host_arch() message(STATUS # "${CMAKE_CXX_COMPILER_ID} flags:
# ${HOST_FLAGS}")

# include(cmake/FindITTNOTIFY.cmake) # include(cmake/FindIntelSYCL.cmake)

# # Prefer to use MKL find_package(TargetLAPACK REQUIRED) get_property( EINSUMS_LINALG_VENDOR TARGET tgt::lapack
# PROPERTY VENDOR) get_property( EINSUMS_LINALG_INT_INTERFACE TARGET tgt::lapack PROPERTY INT_INTERFACE)

# # Set default FFT library to mkl, if mkl was found, otherwise FFTW3 set(EINSUMS_FFT_LIBRARY_DEFAULT "fftw3")
# if(${EINSUMS_LINALG_VENDOR} MATCHES "[mM][kK][lL]") set(EINSUMS_FFT_LIBRARY_DEFAULT "mkl") endif()

# einsums_option(EINSUMS_FFT_LIBRARY STRING "FFT library" ${EINSUMS_FFT_LIBRARY_DEFAULT} STRINGS "fftw3;mkl;off")

# include(EinsumsManageFFTLibraries)

# find_package(OpenMP REQUIRED MODULE COMPONENTS CXX) if(OpenMP_FOUND) message(STATUS "OpenMP Version
# ${OpenMP_CXX_VERSION_MAJOR}.${OpenMP_CXX_VERSION_MINOR}") endif()

# add_subdirectory(external) add_subdirectory(src)

# if(EINSUMS_COVERAGE AND NOT MSVC) target_compile_options(einsums PUBLIC
# $<$<COMPILE_LANG_AND_ID:CXX,Clang>:-fprofile-instr-generate -fcoverage-mapping>) target_compile_options(einsums PUBLIC
# $<$<COMPILE_LANGUAGE:HIP>:-fprofile-instr-generate -fcoverage-mapping>) target_compile_options(einsums PUBLIC
# $<$<COMPILE_LANG_AND_ID:CXX,GNU>:--coverage>) target_link_options(einsums PUBLIC -lgcov --coverage) set(COV_CMD gcov)
# endif()

# if(EINSUMS_ENABLE_TESTING AND NOT MSVC) target_compile_options(einsums BEFORE PUBLIC
# $<$<CONFIG:Debug>:$<$<COMPILE_LANG_AND_ID:CXX,Clang>:-gdwarf-4 -O0 -g3 -ggdb>>) target_compile_options(einsums BEFORE
# PUBLIC $<$<CONFIG:Debug>:$<$<COMPILE_LANG_AND_ID:HIP,Clang>:-gdwarf-4 -O0 -g3 -ggdb>>) target_compile_options(einsums
# BEFORE PUBLIC $<$<CONFIG:Debug>:$<$<COMPILE_LANG_AND_ID:CXX,GNU>:-Og -g3 -ggdb>>) endif()

# if(EINSUMS_FAST_INTRINSICS) target_compile_definitions(einsums PUBLIC -DEINSUMS_FAST_INTRINSICS) endif()

# if(EINSUMS_ENABLE_TESTING) enable_testing() add_subdirectory(tests) endif()

# add_subdirectory(timing) # add_subdirectory(experiments)

# feature_summary(INCLUDE_QUIET_PACKAGES WHAT PACKAGES_FOUND PACKAGES_NOT_FOUND ENABLED_FEATURES DISABLED_FEATURES)
