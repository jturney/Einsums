#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.24 FATAL_ERROR) # 3.24 for FetchContent_Declare(FIND_PACKAGE_ARGS

# Overrides must go before project() statement, otherwise they are ignored

# ##################################################################################################
# C++ overrides
# ##################################################################################################
# Not currently used but provided as an exmaple for the future.
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

project(Einsums LANGUAGES C CXX)

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
set(EINSUMS_VERSION_MAJOR 1)
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
set(EINSUMS_PACKAGE_NAME einsums)
# To keep track of the einsums_root when other subprojects are declared
set(EINSUMS_SOURCE_DIR "${PROJECT_SOURCE_DIR}")
set(EINSUMS_BINARY_DIR "${PROJECT_BINARY_DIR}")

# ##################################################################################################
# CMake configuration
# ##################################################################################################
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake")

include(GNUInstallDirs)
include(Einsums_Utils)

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
    "Defined the startup project for the einsums solution (default: ALL_BUILD)." "ALL_BUILD"
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
  ON ADVANCED
)

if(EINSUMS_WITH_DEPRECATION_WARNINGS)
  einsums_add_config_cond_define(EINSUMS_HAVE_DEPRECATIO_WARNINGS 1)
endif()

# Generic build options
if("${CMAKE_SYSTEM_NAME}" STREQUAL "Linux")
  set(EINSUMS_WITH_STACKOVERFLOW_DETECTION_DEFAULT OFF)
  string(TOUPPER ${CMAKE_BUILD_TYPE} CMAKE_BUILD_TYPE_UC)
  if("${CMAKE_BUILD_TYPE_UC}" STREQUAL "DEBUG")
    set(EINSUMS_WITH_STACKOVERFLOW_DETECTION_DEFAULT ON)
  endif()
  einsums_option(
    EINSUMS_WITH_STACKOVERFLOW_DETECTION BOOL
    "Enable stackoverflow detection for einsums threads/coroutines. (default: OFF, debug: ON)"
    ${EINSUMS_WITH_STACKOVERFLOW_DETECTION_DEFAULT} ADVANCED
  )
  if(EINSUMS_WITH_STACKOVERFLOW_DETECTION)
    einsums_add_config_define(EINSUMS_HAVE_STACKOVERFLOW_DETECTION)
  endif()
endif()

# Ultimately this will be something different
set(DEFAULT_MALLOC "system")
einsums_option(
  EINSUMS_WITH_MALLOC
  STRING
  "Define which allocator should be linked in. Options are: system, tcmalloc, jemalloc, mimalloc, tbbmalloc, and custom. (default: ${DEFAULT_MALLOC})"
  ${DEFAULT_MALLOC}
  STRINGS "system;tcmalloc;jemalloc;mimalloc;tbbmalloc;custom"
)

# On some systems, jemalloc requires an explicit prefix for the API functions (i.e. 'malloc' is
# called 'je_malloc', etc.)
if(${EINSUMS_WITH_MALLOC} STREQUAL "jemalloc")
  if(MSVC)
    set(EINSUMS_WITH_JEMALLOC_PREFIX_DEFAULT "je_")
  else()
    set(EINSUMS_WITH_JEMALLOC_PREFIX_DEFAULT "<none>")
  endif()
  einsums_option(
    EINSUMS_WITH_JEMALLOC_PREFIX STRING "Optional naming prefix for jemalloc API functions"
    ${EINSUMS_WITH_JEMALLOC_PREFIX_DEFAULT} ADVANCED
  )
endif()

# Compiler related build options
einsums_option(
  EINSUMS_WITH_COMPILER_WARNINGS BOOL "Enable compiler warnings (default: ON)" ON ADVANCED
)

einsums_option(
  EINSUMS_WITH_COMPILER_WARNINGS_AS_ERROR BOOL "Turn compiler warnings into error (default: OFF"
  OFF ADVANCED
)

einsums_option(
  EINSUMS_WITH_EXECUTABLE_PREFIX STRING "Executable prefix (default: none)." ""
  CATEGORY "Build Targets"
  ADVANCED
)

einsums_option(
  EINSUMS_WITH_BINARY_DIR
  PATH
  "Root directory for libraries and executables for the build. This does not affect the installed location of libraries and executables."
  "${PROJECT_BINARY_DIR}"
  CATEGORY "Build Targets"
  ADVANCED
)

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
  ADVANCED
)
if(EINSUMS_WITH_UNITY_BUILD)
  set(EINSUMS_WITH_UNITY_BUILD_OPTION UNITY_BUILD)
endif()

einsums_option(
  EINSUMS_WITH_PRECOMPILED_HEADERS BOOL
  "Enabled precompiled headers for certain build targets (experimental) (default: OFF)" OFF
  ADVANCED
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
  EINSUMS_WITH_EXAMPLES BOOL "Build examples (default: OFF)" OFF CATEGORY "Build Targets"
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
)

# ##################################################################################################
# CUDA/HIP configuration
# ##################################################################################################
set(CUDA_OPTION_STRING "Enable support for CUDA (default: OFF)")
set(HIP_OPTION_STRING "Enable support for HIP (default: OFF)")
einsums_option(EINSUMS_WITH_CUDA BOOL "${CUDA_OPTION_STRING}" OFF ADVANCED)

# No need for the user to specify the option explicitly
einsums_option(EINSUMS_WITH_HIP BOOL "${HIP_OPTION_STRING}" OFF ADVANCED)
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
# Threadlevel Nice option
# ##################################################################################################
einsums_option(
  EINSUMS_WITH_NICE_THREADLEVEL BOOL
  "Set einsums worker threads to have high NICE level (may impact performance) (default: OFF)" OFF
  ADVANCED
)
if(EINSUMS_WITH_NICE_THREADLEVEL)
  einsums_info("Nice threadlevel is enabled.")
  einsums_add_config_define(EINSUMS_HAVE_NICE_THREADLEVEL)
else()
  einsums_info("Nice threadlevel is disabled.")
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
)

# ##################################################################################################
# Thread Manager related build options
# ##################################################################################################
set(EINSUMS_MAX_CPU_COUNT_DEFAULT "64")
einsums_option(
  EINSUMS_WITH_MAX_CPU_COUNT
  STRING
  "Einsums applications will not use more than this number of OS threads (default: \"${EINSUMS_MAX_CPU_COUNT_DEFAULT}\")"
  "${EINSUMS_MAX_CPU_COUNT_DEFAULT}"
  CATEGORY "Thread Manager"
  ADVANCED
)
einsums_add_config_define(EINSUMS_MAX_CPU_COUNT ${EINSUMS_WITH_MAX_CPU_COUNT})
if(EINSUMS_WITH_MAX_CPU_COUNT GREATER 64)
  einsums_add_config_define(EINSUMS_HAVE_MORE_THAN_64_THREADS)
endif()

set(EINSUMS_MAX_NUMA_DOMAIN_COUNT_DEFAULT "8")
einsums_option(
  EINSUMS_WITH_MAX_NUMA_DOMAIN_COUNT
  STRING
  "einsums applications will not run on machines with more NUMA domains (default: ${EINSUMS_MAX_NUMA_DOMAIN_COUNT_DEFAULT})"
  ${EINSUMS_MAX_NUMA_DOMAIN_COUNT_DEFAULT}
  CATEGORY "Thread Manager"
  ADVANCED
)
einsums_add_config_define(EINSUMS_HAVE_MAX_NUMA_DOMAIN_COUNT ${EINSUMS_WITH_MAX_NUMA_DOMAIN_COUNT})

einsums_option(
  EINSUMS_WITH_THREAD_STACK_MMAP BOOL "Use mmap for stack allocation on appropriate platforms" ON
  CATEGORY "Thread Manager"
  ADVANCED
)

einsums_option(
  EINSUMS_WITH_THREAD_MANAGER_IDLE_BACKOFF BOOL
  "einsums scheduler threads do exponential backoff on idle queues (default: ON)" ON
  CATEGORY "Thread Manager"
  ADVANCED
)

einsums_option(
  EINSUMS_WITH_STACKTRACES BOOL "Attach backtraces to einsums exceptions (default: ON)" ON
  CATEGORY "Thread Manager"
  ADVANCED
)

einsums_option(
  EINSUMS_WITH_THREAD_BACKTRACE_ON_SUSPENSION BOOL
  "Enable thread stack back trace being captured on suspension (default: OFF)" OFF
  CATEGORY "Thread Manager"
  ADVANCED
)

einsums_option(
  EINSUMS_WITH_SWAP_CONTEXT_EMULATION BOOL
  "Emulate SwapContext API for coroutines (Windows only, default: OFF)" OFF
  CATEGORY "Thread Manager"
  ADVANCED
)

# ##################################################################################################
# We create a target to contain libraries like rt, dl, etc. in order to remove variables
add_library(einsums_base_libraries INTERFACE)

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
  target_link_libraries(einsums_base_libraries INTERFACE imf svml irng intlc)
endif()

if(EINSUMS_WITH_STACKTRACES OR EINSUMS_WITH_THREAD_BACKTRACE_ON_SUSPENSION)
  einsums_info("Stack traces are enabled.")
  einsums_add_config_define(EINSUMS_HAVE_STACKTRACES)
  if(WIN32)
    target_link_libraries(einsums_base_libraries INTERFACE dbghelp)
  endif()

  einsums_option(
    EINSUMS_WITH_THREAD_BACKTRACE_DEPTH STRING
    "Thread stack back trace depth being captured (default: 20)" "20"
    CATEGORY "Thread Manager"
    ADVANCED
  )
  einsums_add_config_define(
    EINSUMS_HAVE_THREAD_BACKTRACE_DEPTH ${EINSUMS_WITH_THREAD_BACKTRACE_DEPTH}
  )

  if(("${CMAKE_SYSTEM_NAME}" STREQUAL "Linux") AND ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang"
                                                    OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  )

    # This option is OFF by default as we have seen random segfaults out in the wild if this is
    # enabled.
    einsums_option(
      EINSUMS_WITH_STACKTRACES_STATIC_SYMBOLS BOOL
      "Thread stack back trace will resolve static symbols (default: OFF)" OFF
      CATEGORY "Thread Manager"
      ADVANCED
    )
    einsums_add_config_define(
      EINSUMS_HAVE_STACKTRACES_STATIC_SYMBOLS ${EINSUMS_WITH_STACKTRACES_STATIC_SYMBOLS}
    )

    # Demangling can segfault in certain configurations.
    einsums_option(
      EINSUMS_WITH_STACKTRACES_DEMANGLE_SYMBOLS BOOL
      "Thread stack back trace symbols will be demangled (default: ON)" ON
      CATEGORY "Thread Manager"
      ADVANCED
    )
    if(EINSUMS_WITH_STACKTRACES_DEMANGLE_SYMBOLS)
      einsums_add_config_define(EINSUMS_HAVE_STACKTRACES_DEMANGLE_SYMBOLS)
    endif()
  endif()
endif()

if(EINSUMS_WITH_THREAD_BACKTRACE_ON_SUSPENSION)
  einsums_add_config_define(EINSUMS_HAVE_THREAD_BACKTRACE_ON_SUSPENSION)

  einsums_option(
    EINSUMS_WITH_THREAD_FULLBACKTRACE_ON_SUSPENSION BOOL
    "Enable thread stack back trace being captured on suspension (default: OFF)" OFF
    CATEGORY "Thread Manager"
    ADVANCED
  )
  if(EINSUMS_WITH_THREAD_FULLBACKTRACE_ON_SUSPENSION)
    einsums_add_config_define(EINSUMS_HAVE_THREAD_FULLBACKTRACE_ON_SUSPENSION)
  endif()
endif()

einsums_option(
  EINSUMS_WITH_THREAD_QUEUE_WAITTIME BOOL
  "Enable collecting queue wait times for threads (default: OFF)" OFF
  CATEGORY "Thread Manager"
  ADVANCED
)

if(EINSUMS_WITH_THREAD_QUEUE_WAITTIME)
  einsums_add_config_define(EINSUMS_HAVE_THREAD_QUEUE_WAITTIME)
endif()

einsums_option(
  EINSUMS_WITH_THREAD_IDLE_RATES BOOL
  "Enable measuring the percentage of overhead times spent in the scheduler (default: OFF)" OFF
  CATEGORY "Thread Manager"
  ADVANCED
)

einsums_option(
  EINSUMS_WITH_THREAD_CREATION_AND_CLEANUP_RATES BOOL
  "Enable measuring thread creation and cleanup times (default: OFF)" OFF
  CATEGORY "Thread Manager"
  ADVANCED
)

if(EINSUMS_WITH_THREAD_IDLE_RATES)
  einsums_add_config_define(EINSUMS_HAVE_THREAD_IDLE_RATES)
  if(EINSUMS_WITH_THREAD_CREATION_AND_CLEANUP_RATES)
    einsums_add_config_define(EINSUMS_HAVE_THREAD_CREATION_AND_CLEANUP_RATES)
  endif()
endif()

einsums_option(
  EINSUMS_WITH_THREAD_CUMULATIVE_COUNTS BOOL
  "Enable keeping track of cumulative thread counts in the schedulers (default: OFF)" OFF
  CATEGORY "Thread Manager"
  ADVANCED
)

if(EINSUMS_WITH_THREAD_CUMULATIVE_COUNTS)
  einsums_add_config_define(EINSUMS_HAVE_THREAD_CUMULATIVE_COUNTS)
endif()

einsums_option(
  EINSUMS_WITH_THREAD_STEALING_COUNTS BOOL
  "Enable keeping track of counts of thread stealing incidents in the schedulers (default: OFF)" OFF
  CATEGORY "Thread Manager"
  ADVANCED
)

if(EINSUMS_WITH_THREAD_STEALING_COUNTS)
  einsums_add_config_define(EINSUMS_HAVE_THREAD_STEALING_COUNTS)
endif()

einsums_option(
  EINSUMS_WITH_COROUTINE_COUNTERS BOOL
  "Enable keeping track of coroutine creation and rebind counts (default: OFF)" OFF
  CATEGORY "Thread Manager"
  ADVANCED
)
if(EINSUMS_WITH_COROUTINE_COUNTERS)
  einsums_add_config_define(EINSUMS_HAVE_COROUTINE_COUNTERS)
endif()

einsums_option(
  EINSUMS_WITH_THREAD_LOCAL_STORAGE BOOL
  "Enable thread local storage for all einsums threads (default: OFF)" OFF
  CATEGORY "Thread Manager"
  ADVANCED
)

if(EINSUMS_WITH_THREAD_LOCAL_STORAGE)
  einsums_add_config_define(EINSUMS_HAVE_THREAD_LOCAL_STORAGE)
endif()

einsums_option(
  EINSUMS_WITH_SCHEDULER_LOCAL_STORAGE BOOL
  "Enable scheduler local storage for all einsums schedulers (default: OFF)" OFF
  CATEGORY "Thread Manager"
  ADVANCED
)

if(EINSUMS_WITH_SCHEDULER_LOCAL_STORAGE)
  einsums_add_config_define(EINSUMS_HAVE_SCHEDULER_LOCAL_STORAGE)
endif()

einsums_option(
  EINSUMS_WITH_SPINLOCK_POOL_NUM STRING "Number of elements a spinlock pool manages (default: 128)"
  128
  CATEGORY "Thread Manager"
  ADVANCED
)

einsums_add_config_define(EINSUMS_HAVE_SPINLOCK_POOL_NUM ${EINSUMS_WITH_SPINLOCK_POOL_NUM})

einsums_option(
  EINSUMS_WITH_SPINLOCK_DEADLOCK_DETECTION BOOL "Enable spinlock deadlock detection (default: OFF)"
  OFF
  CATEGORY "Thread Manager"
  ADVANCED
)

if(EINSUMS_WITH_SPINLOCK_DEADLOCK_DETECTION)
  einsums_add_config_define(EINSUMS_HAVE_SPINLOCK_DEADLOCK_DETECTION)
endif()

einsums_option(
  EINSUMS_WITH_ITTNOTIFY BOOL "Enable Amplifier (ITT) instrumentation support." OFF
  CATEGORY "Profiling"
  ADVANCED
)

einsums_option(
  EINSUMS_WITH_MPI BOOL
  "Enable support for returning futures from MPI asynchronous calls (default: OFF)" OFF
  CATEGORY "MPI"
)

if(EINSUMS_WITH_MPI)
  einsums_add_config_define(EINSUMS_HAVE_MPI)

  # mpi settings
  einsums_option(
    EINSUMS_WITH_MPI_ENV
    STRING
    "List of environment variables checked to detect MPI (default: MV2_COMM_WORLD_RANK;PMI_RANK;OMPI_COMM_WORLD_SIZE;ALPS_APP_PE;PMIX_RANK;PALS_NODEID)."
    "MV2_COMM_WORLD_RANK;PMI_RANK;OMPI_COMM_WORLD_SIZE;ALPS_APP_PE;PMIX_RANK;PALS_NODEID"
    CATEGORY "MPI"
    ADVANCED
  )

  # This list is to detect whether we run inside an mpi environment.
  if(EINSUMS_WITH_MPI_ENV)
    string(REPLACE ";" "," einsums_mpi_env_ "${EINSUMS_WITH_MPI_ENV}")
    einsums_add_config_define(EINSUMS_HAVE_MPI_ENV "\"${einsums_mpi_env_}\"")
  endif()

  einsums_option(
    EINSUMS_WITH_MPI_MULTITHREADED BOOL "Turn on MPI multithreading support (default: ON)." ON
    CATEGORY "MPI"
    ADVANCED
  )
  if(EINSUMS_WITH_MPI_MULTITHREADED)
    einsums_add_config_define(EINSUMS_HAVE_MPI_MULTITHREADED)
  endif()

  if(MSVC)
    # FIXME: add OpenMPI specific flag here for now as the einsums_add_compile_flag() below does not
    # add the extra options to the top level directory
    einsums_add_config_define(OMPI_IMPORTS)
  endif()
endif()

# External libraries/frameworks used by sme of the examples and benchmarks
einsums_option(
  EINSUMS_WITH_EXAMPLES_OPENMP BOOL "Enable examples requiring OpenMP support (default: OFF)." OFF
  CATEGORY "Build Targets" DEPENDS "EINSUMS_WITH_EXAMPLES"
  ADVANCED
)
if(EINSUMS_WITH_EXAMPLES_OPENMP)
  find_package(OpenMP REQUIRED)
endif()
einsums_option(
  EINSUMS_WITH_EXAMPLES_TBB BOOL "Enable examples requiring TBB support (default: OFF)." OFF
  CATEGORY "Build Targets" DEPENDS "EINSUMS_WITH_EXAMPLES"
  ADVANCED
)
if(EINSUMS_WITH_EXAMPLES_TBB)
  find_package(TBB)
  if(NOT TBB_FOUND)
    set(EINSUMS_WITH_EXAMPLES_TBB OFF)
  endif()
endif()
einsums_option(
  EINSUMS_WITH_EXAMPLES_QTHREADS BOOL "Enable examples requiring QThreads support (default: OFF)."
  OFF
  CATEGORY "Build Targets" DEPENDS "EINSUMS_WITH_EXAMPLES"
  ADVANCED
)
if(EINSUMS_WITH_EXAMPLES_QTHREADS)
  find_package(QThreads)
  if(NOT QTHREADS_FOUND)
    set(EINSUMS_WITH_EXAMPLES_QTHREADS OFF)
  endif()
endif()
einsums_option(
  EINSUMS_WITH_EXAMPLES_HDF5 BOOL "Enable examples requiring HDF5 support (default: OFF)." OFF
  CATEGORY "Build Targets" DEPENDS "EINSUMS_WITH_EXAMPLES"
  ADVANCED
)
if(EINSUMS_WITH_EXAMPLES_HDF5)
  find_package(HDF5 COMPONENTS CXX)
  if(NOT HDF5_FOUND)
    set(EINSUMS_WITH_EXAMPLES_HDF5 OFF)
  endif()
endif()

# Debugging related build options
einsums_option(
  EINSUMS_WITH_VALGRIND BOOL "Enable Valgrind instrumentation support." OFF
  CATEGORY "Debugging"
  ADVANCED
)

einsums_option(
  EINSUMS_WITH_TESTS_VALGRIND BOOL "Run selected tests with valgrind." OFF
  CATEGORY "Debugging"
  ADVANCED
)

einsums_option(
  EINSUMS_WITH_TESTS_VALGRIND_OPTIONS STRING "Use these options when running tests under valgrind."
  "--error-exitcode=1"
  CATEGORY "Debugging"
  ADVANCED
)

einsums_option(
  EINSUMS_WITH_SANITIZERS BOOL "Configure with sanitizer instrumentation support." OFF
  CATEGORY "Debugging"
  ADVANCED
)

set(EINSUMS_WITH_VERIFY_LOCKS_DEFAULT OFF)
if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
  set(EINSUMS_WITH_VERIFY_LOCKS_DEFAULT ON)
endif()

einsums_option(
  EINSUMS_WITH_VERIFY_LOCKS BOOL
  "Enable lock verification code (default: OFF, enabled in debug builds)"
  ${EINSUMS_WITH_VERIFY_LOCKS_DEFAULT}
  CATEGORY "Debugging"
  ADVANCED
)
einsums_option(
  EINSUMS_WITH_VERIFY_LOCKS_BACKTRACE
  BOOL
  "Enable thread stack back trace being captured on lock registration (to be used in combination with EINSUMS_WITH_VERIFY_LOCKS=ON, default: OFF)"
  OFF
  CATEGORY "Debugging"
  ADVANCED
)

einsums_option(
  EINSUMS_WITH_THREAD_DEBUG_INFO BOOL "Enable thread debugging information (default: OFF)" OFF
  CATEGORY "Debugging"
  ADVANCED
)

if(EINSUMS_WITH_VERIFY_LOCKS)
  einsums_add_config_define(EINSUMS_HAVE_VERIFY_LOCKS)
  if(EINSUMS_WITH_VERIFY_LOCKS_BACKTRACE)
    einsums_add_config_define(EINSUMS_HAVE_VERIFY_LOCKS_BACKTRACE)
  endif()
endif()

# Additional debug support
if(NOT WIN32 AND EINSUMS_WITH_THREAD_STACK_MMAP)
  einsums_add_config_define(EINSUMS_HAVE_THREAD_STACK_MMAP)
endif()

if(EINSUMS_WITH_THREAD_MANAGER_IDLE_BACKOFF)
  einsums_add_config_define(EINSUMS_HAVE_THREAD_MANAGER_IDLE_BACKOFF)
endif()

einsums_option(
  EINSUMS_WITH_THREAD_DESCRIPTION_FULL BOOL
  "Use function address for thread description (default: OFF)" OFF
  CATEGORY "Debugging"
  ADVANCED
)

einsums_option(
  EINSUMS_WITH_TESTS_DEBUG_LOG BOOL "Turn on debug logs for tests (default: OFF)" OFF
  CATEGORY "Debugging"
  ADVANCED
)

einsums_option(
  EINSUMS_WITH_TESTS_DEBUG_LOG_DESTINATION STRING "Destination for test debug logs (default: cout)"
  "cout"
  CATEGORY "Debugging"
  ADVANCED
)

einsums_option(
  EINSUMS_WITH_TESTS_MAX_THREADS
  STRING
  "Maximum number of threads to use for tests (default: 0, use the number of threads specified by the test)"
  0
  CATEGORY "Debugging"
  ADVANCED
)

einsums_option(
  EINSUMS_WITH_PARALLEL_TESTS_BIND_NONE BOOL
  "Pass --einsums:bind=none to tests that may run in parallel (cmake -j flag) (default: OFF)" OFF
  CATEGORY "Debugging"
  ADVANCED
)

if(EINSUMS_WITH_THREAD_DEBUG_INFO)
  einsums_add_config_define(EINSUMS_HAVE_THREAD_PARENT_REFERENCE)
  einsums_add_config_define(EINSUMS_HAVE_THREAD_PHASE_INFORMATION)
  einsums_add_config_define(EINSUMS_HAVE_THREAD_DESCRIPTION)
  einsums_add_config_define(EINSUMS_HAVE_THREAD_DEADLOCK_DETECTION)
  if(EINSUMS_HAVE_THREAD_DESCRIPTION_FULL)
    einsums_add_config_define(EINSUMS_HAVE_THREAD_DESCRIPTION_FULL)
  endif()
endif()

einsums_option(EINSUMS_WITH_STDEXEC BOOL "Use stdexec for sender/receiver functionality." OFF CATEGORY "Advanced")
if (EINSUMS_WITH_STDEXEC)
    einsums_add_config_define(EINSUMS_HAVE_STDEXEC)
endif ()

# ##################################################################################################
# Special diagnostic flags, do not enable by default, only if needed
einsums_option(
  EINSUMS_WITH_CHECK_MODULE_DEPENDENCIES BOOL
  "Verify that no modules are cross-referenced from a different module category (default: OFF)" OFF
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

find_package(OpenMP REQUIRED MODULE COMPONENTS CXX)
target_link_libraries(einsums_base_libraries INTERFACE OpenMP::OpenMP_CXX)

# ##################################################################################################
# C++ feature tests
# ##################################################################################################
# Need to set up CUDA/HIP before feature tests to enable the respective languages
include(Einsums_SetupCUDA)
include(Einsums_SetupHIP)
include(Einsums_PerformCXXFeatureTests)
einsums_perform_cxx_feature_tests()

# ##################################################################################################
# Check for misc system headers
# ##################################################################################################
einsums_check_for_unistd_h(DEFINITIONS EINSUMS_HAVE_UNISTD_H)

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

  # Check if our libraries have unresolved symbols
  if(NOT APPLE
     AND NOT WIN32
     AND NOT EINSUMS_WITH_SANITIZERS
  )
    einsums_add_link_flag_if_available(-Wl,-z,defs)
  endif()
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

  set(__has_timestamp_support OFF)
  include(Einsums_TargetArch)

  einsums_target_architecture(__target_arch)
  if("${__target_arch}" STREQUAL "i386"
     OR "${__target_arch}" STREQUAL "ix86"
     OR "${__target_arch}" STREQUAL "x86_64"
     OR "${__target_arch}" STREQUAL "ia64"
  )

    # rdtsc is an x86 instruction that reads the value of a CPU time stamp counter. rdtscp is an
    # extension to rdtsc. The difference is that rdtscp is a serializing instruction.
    einsums_cpuid("rdtsc" EINSUMS_WITH_RDTSC DEFINITIONS EINSUMS_HAVE_RDTSC)

    # XeonPhi's do not support RDTSCP
    if(NOT ("${EINSUMS_PLATFORM_UC}" STREQUAL "XEONPHI"))
      einsums_cpuid("rdtscp" EINSUMS_WITH_RDTSCP DEFINITIONS EINSUMS_HAVE_RDTSCP)
    endif()

    if(EINSUMS_WITH_RDTSC OR EINSUMS_WITH_RDTSCP)
      set(__has_timestamp_support ON)
    endif()
  elseif(
    "${__target_arch}" STREQUAL "arm"
    OR "${__target_arch}" STREQUAL "armv5"
    OR "${__target_arch}" STREQUAL "armv6"
    OR "${__target_arch}" STREQUAL "armv7"
    OR "${__target_arch}" STREQUAL "arm64"
    OR "${__target_arch}" STREQUAL "ppc"
    OR "${__target_arch}" STREQUAL "ppc64"
    OR "${__target_arch}" STREQUAL "bgq"
    OR "${__target_arch}" STREQUAL "s390x"
  )
    set(__has_timestamp_support ON)
  endif()

  einsums_info("Architecture detected: ${__target_arch}")
  if(NOT __has_timestamp_support)
    einsums_warn(
      "No timestamp support is available; some performance counters may report incorrect results"
    )
  endif()
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
include(Einsums_SetupRangev3)
include(Einsums_SetupH5cpp)
include(Einsums_SetupHWLOC)
include(Einsums_SetupSpdlog)
# include(Einsums_SetupCppTrace)

if(EINSUMS_WITH_SANITIZERS)
  einsums_add_config_define(EINSUMS_HAVE_SANITIZERS)
endif()

# Note: on windows systems the ':' will be converted to a ';' at runtime
einsums_add_config_cond_define(
  EINSUMS_DEFAULT_COMPONENT_PATH_SUFFIXES
  "\"/${CMAKE_INSTALL_LIBDIR}/einsums:/${CMAKE_INSTALL_BINDIR}/einsums:/lib/einsums:/bin/einsums\""
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

  add_subdirectory(testing)
endif()

if(EINSUMS_WITH_EXAMPLES)
  add_subdirectory(examples)
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

# ##################################################################################################
# installation instructions
# ##################################################################################################
# Install all einsums header that have been configured using various CMake options
install(
  DIRECTORY "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/einsums/"
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/einsums
  COMPONENT core
  FILES_MATCHING
  PATTERN "*.hpp"
  PATTERN ".git" EXCLUDE
  PATTERN "CMakeFiles" EXCLUDE
  PATTERN "CTestFiles" EXCLUDE
)

install(
  # Install all einsums CMake utility files
  DIRECTORY cmake/
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/einsums
  COMPONENT core
  PATTERN "templates" EXCLUDE
  PATTERN "packaging" EXCLUDE
  PATTERN ".git" EXCLUDE
)

install(
  FILES "${PROJECT_SOURCE_DIR}/LICENSE.txt"
  DESTINATION ${CMAKE_INSTALL_DATADIR}/einsums
  COMPONENT license
)

# Install einsums-bind script
configure_file(
  ${PROJECT_SOURCE_DIR}/scripts/einsums-bind.sh.in ${PROJECT_BINARY_DIR}/scripts/einsums-bind @ONLY
)

install(PROGRAMS ${PROJECT_BINARY_DIR}/scripts/einsums-bind DESTINATION ${CMAKE_INSTALL_BINDIR})

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
message("einsums will be installed to ${CMAKE_INSTALL_PREFIX}")
message("")
