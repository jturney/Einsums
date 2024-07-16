cmake_minimum_required(VERSION 3.24) # 3.24 for FetchContent_Declare(FIND_PACKAGE_ARGS

# Overrides must go before the project() statement, otherwise they are ignored.

# ##################################################################################################
# C++ overrides
# ##################################################################################################
# set(CMAKE_USER_MAKE_RULES_OVERRIDE_CXX
# "${CMAKE_CURRENT_SOURCE_DIR}/cmake/einsums_cxx_overrides.cmake")

# ##################################################################################################
# Build type (needs to be handled before project command below)
# ##################################################################################################
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE
        "Debug"
        CACHE STRING
              "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel."
              FORCE
    )
endif()

# ##################################################################################################
# Project metadata
# ##################################################################################################
project(Einsums CXX)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# ##################################################################################################
# Build all of Einsums
# ##################################################################################################
set(EINSUMS_VERSION_MAJOR 0)
set(EINSUMS_VERSION_MINOR 99)
set(EINSUMS_VERSION_PATCH 0)
set(EINSUMS_VERSION_DATE 20240715)
set(EINSUMS_VERSION_TAG "-trunk")

set(EINSUMS_VERSION "${EINSUMS_VERSION_MAJOR}.${EINSUMS_VERSION_MINOR}.${EINSUMS_VERSION_PATCH}")
set(EINSUMS_VERSION_FULL
    "${EINSUMS_VERSION_MAJOR}.${EINSUMS_VERSION_MINOR}.${EINSUMS_VERSION_PATCH}${EINSUMS_VERSION_TAG}"
)
set(EINSUMS_LIBRARY_VERSION "${EINSUMS_VERSION}")
set(EINSUMS_SOVERSION ${EINSUMS_VERSION_MAJOR})
set(EINSUMS_PACKAGE_NAME Einsums)

# To keep track of the einsums root when other subprojects are declared
set(EINSUMS_SOURCE_DIR ${PROJECT_SOURCE_DIR})
set(EINSUMS_BINARY_DIR ${PROJECT_BINARY_DIR})

# ##################################################################################################
# CMake configuration
# ##################################################################################################
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake")

include(GNUInstallDirs)
include(Einsums_Utils)

# explicitly set certain policies
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
einsums_info("CMake version: ${CMAKE_VERSION}")
einsums_info("Einsums version: ${EINSUMS_VERSION}")

# ##################################################################################################
# Reset cached variables that need to be re-filled
unset(EINSUMS_EXPORT_TARGETS CACHE)
unset(EINSUMS_EXPORT_INTERNAL_TARGETS CACHE)
unset(EINSUMS_LIBS CACHE)
unset(EINSUMS_ENABLED_MODULES CACHE)

# ##################################################################################################
# Set up dummy compiler flags targets
# ##################################################################################################
include(Einsums_CompilerFlagsTargets)

# ##################################################################################################
# Set up platform for which einsums should be compiled.
include(Einsums_SetPlatform)

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

einsums_option(
    EINSUMS_WITH_DEPRECATION_WARNINGS BOOL "Enable warnings for deprecated features. (default: ON)"
    ON ADVANCED
)

if(EINSUMS_WITH_DEPRECATION_WARNINGS)
    einsums_add_config_cond_define(EINSUMS_WITH_DEPRECATION_WARNINGS 1)
endif()

# Logging configuration
einsums_option(
    EINSUMS_WITH_LOGGING BOOL "Build einsums with logging enabled. (default: ON)" ON ADVANCED
)
if(EINSUMS_WITH_LOGGING)
    einsums_add_config_define(EINSUMS_HAVE_LOGGING)
endif()

einsums_option(
    EINSUMS_WITH_COMPILER_WARNINGS BOOL "Enable compiler warnings. (default: ON)" ON ADVANCED
)

einsums_option(
    EINSUMS_WITH_COMPILER_WARNINGS_AS_ERRORS BOOL
    "Turn compiler warnings into errors (default: OFF)" OFF ADVANCED
)

einsums_option(
    EINSUMS_WITH_EXECUTABLE_PREFIX STRING "Executable prefix (default: none)." ""
    CATEGORY "Build Targets" ADVANCED
)

einsums_option(
    EINSUMS_WITH_BINARY_DIR
    PATH
    "Root directory for libraries and executables for the build. This does not affect the installed location of libraries and executables."
    "${PROJECT_BINARY_DIR}"
    CATEGORY "Build Targets"
    ADVANCED
)

if(WIN32)
    set(EINSUMS_WITH_PSEUDO_DEPENDENCIES OFF CACHE INTERNAL "" FORCE)
else()
    set(EINSUMS_WITH_PSEUDO_DEPENDENCIES ON CACHE INTERNAL "" FORCE)
endif()

einsums_option(
    EINSUMS_WITH_UNITY_BUILD BOOL "Enable unity build for certain build targets (default: ON)" ON
    ADVANCED
)
if(EINSUMS_WITH_UNITY_BUILD)
    set(EINSUMS_WITH_UNITY_BUILD_OPTION UNITY_BUILD)
endif()

einsums_option(
    EINSUMS_WITH_PRECOMPILED_HEADERS BOOL
    "Enable precompiled headers for certain build targets (experimental) (default: OFF)" OFF
    ADVANCED
)
if(EINSUMS_WITH_PRECOMPILED_HEADERS)
    set(EINSUMS_WITH_PRECOMPILED_HEADERS_INTERNAL ON)
    # Only create the targets here. They will be set up later once all modules are known.
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
    "Create build system support for fail compile tests (default: ON)" ON DEPENDS
    "EINSUMS_WITH_TESTS" CATEGORY "Build Targets"
)

einsums_option(
    EINSUMS_WITH_TOOLS BOOL "Build tools (default: OFF)" OFF ADVANCED CATEGORY "Build Targets"
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

einsums_option(
    EINSUMS_WITH_STACKTRACES BOOL "Attach backtraces to einsums exceptions (default: ON)" ON
    CATEGORY "Thread Manager" ADVANCED
)

# ##################################################################################################
# We create a target to contain libraries like rt, dl, etc. in order to remove variables
add_library(einsums_base_libraries INTERFACE)

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    target_link_libraries(einsums_base_libraries INTERFACE imf svml irng intlc)
endif()

if(EINSUMS_WITH_STACKTRACES)
    einsums_info("Stack traces are enabled.")
    einsums_add_config_define(EINSUMS_HAVE_STACKTRACES)

    einsums_option(
        EINSUMS_WITH_THREAD_BACKTRACE_DEPTH STRING
        "Thread stack back trace depth being captured (default: 20)" "20" CATEGORY "Thread Manager"
        ADVANCED
    )
    einsums_add_config_define(
        EINSUMS_HAVE_THREAD_BACKTRACE_DEPTH ${EINSUMS_WITH_THREAD_BACKTRACE_DEPTH}
    )

endif()

# ##################################################################################################
# Find Our dependencies: These are all dependencies needed to build the core library. Dependencies
# that are only needed by plugins, examples or tests should be found separately in the appropriate
# subdirectory. add_subdirectory(external)

include(Einsums_SetupFmt)

# ##################################################################################################
# Find other dependencies.

# ##################################################################################################
# Check for misc system headers
# ##################################################################################################

einsums_check_for_unistd_h(DEFINITIONS EINSUMS_HAVE_UNISTD_H)

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

    # TODO: Uncomment add_subdirectory(testing)
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

# ##################################################################################################
# store cache vars and their values in order for them to be forwarded to the projects (needs to be
# before the EINSUMS_GeneratePackage call)
include(Einsums_ForwardCacheVariables)

# ##################################################################################################
# External build system supportP.
include(Einsums_GeneratePackage)

message("")
message("Einsums will be installed to ${CMAKE_INSTALL_PREFIX}")
message("")
