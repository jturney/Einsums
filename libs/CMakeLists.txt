#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

include(Einsums_Message)
include(Einsums_AddPseudoDependencies)
include(Einsums_AddPseudoTarget)

# add example pseudo targets needed for modules
if(EINSUMS_WITH_EXAMPLES)
  einsums_add_pseudo_target(Examples.Modules)
  einsums_add_pseudo_dependencies(Examples Examples.Modules)
endif()

# add test pseudo targets needed for modules
if(EINSUMS_WITH_TESTS)
  if(EINSUMS_WITH_TESTS_UNIT)
    einsums_add_pseudo_target(Tests.Unit.Modules)
    einsums_add_pseudo_dependencies(Tests.Unit Tests.Unit.Modules)
  endif()

  if(EINSUMS_WITH_EXAMPLES AND EINSUMS_WITH_TESTS_EXAMPLES)
    einsums_add_pseudo_target(Tests.Examples.Modules)
    einsums_add_pseudo_dependencies(Tests.Examples Tests.Examples.Modules)
  endif()

  if(EINSUMS_WITH_TESTS_REGRESSIONS)
    einsums_add_pseudo_target(Tests.Regressions.Modules)
    einsums_add_pseudo_dependencies(Tests.Regressions Tests.Regressions.Modules)
  endif()

  if(EINSUMS_WITH_TESTS_BENCHMARKS)
    einsums_add_pseudo_target(Tests.Performance.Modules)
    einsums_add_pseudo_dependencies(Tests.Performance Tests.Performance.Modules)
  endif()

  if(EINSUMS_WITH_TESTS_HEADERS)
    einsums_add_pseudo_target(Tests.Headers.Modules)
    einsums_add_pseudo_dependencies(Tests.Headers Tests.Headers.Modules)
  endif()
endif()

# Generate a file that lists all enabled modules for checks that might be optional based on the
# presence of a module or not
set(MODULE_ENABLED_MODULE_DEFINES)

# variables needed for config_strings_modules.hpp
set(CONFIG_STRINGS_MODULES_INCLUDES)
set(CONFIG_STRINGS_MODULES_ENTRIES)

# List of sub-directories to work through
set(EINSUMS_LIBS
    Einsums
    CACHE INTERNAL "List of Einsums libraries" FORCE
)

if(EINSUMS_WITH_EXPERIMENTAL)
  list(APPEND EINSUMS_LIBS EinsumsExperimental)
endif()

include(Einsums_CodeCoverage)
set(einsums_libs_public_static SHARED)
if(EINSUMS_WITH_COVERAGE OR NOT BUILD_SHARED_LIBS)
  set(einsums_libs_public_static STATIC)
endif()

foreach(lib ${EINSUMS_LIBS})
  add_library(${lib} ${einsums_libs_public_static} src/dummy.cpp)
  einsums_append_coverage_compiler_flags_to_target(${lib} PUBLIC)

  # Add underscores before uppercase letters, except the first one
  string(REGEX REPLACE "([A-Z])" "_\\1" transformed_string ${lib})
  # Remove the leading underscore if it exists
  string(REGEX REPLACE "^_" "" transformed_string ${transformed_string})
  # Convert to uppercase
  string(TOUPPER ${transformed_string} LIB_NAME)
  string(MAKE_C_IDENTIFIER ${LIB_NAME} MACRO_LIB_NAME)

  target_compile_definitions(${lib} PRIVATE ${MACRO_LIB_NAME}_EXPORTS)

  if(UNIX AND (CMAKE_CXX_COMPILER_ID MATCHES Intel OR CMAKE_CXX_COMPILIER_ID MATCHES IntelLLVM))
    target_compile_options(${lib} PUBLIC -ffp-model=precise -Rno-debug-disables-optimization)
  endif()

  if(EINSUMS_HAVE_BACKTRACES)
    # Copy the cpptrace.dll on windows to the same directory as the executable for your_target. Not
    # required if static linking.
    if(WIN32)
      add_custom_command(
        TARGET ${lib}
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:cpptrace::cpptrace>
                $<TARGET_FILE_DIR:${lib}>
      )
    endif()

    # Create a .dSYM file on macOS
    if(APPLE AND dsymutil_EXECUTABLE)
      add_custom_command(
        TARGET ${lib}
        POST_BUILD
        COMMENT "Calling dsymutil on $<TARGET_FILE:${lib}>"
        COMMAND dsymutil $<TARGET_FILE:${lib}>
      )
    endif()
  endif()

  set_target_properties(
    ${lib} PROPERTIES VERSION ${EINSUMS_VERSION} SOVERSION ${EINSUMS_SOVERSION} OUTPUT_NAME ${lib}
                      FOLDER "Core"
  )
  install(
    TARGETS ${lib}
    EXPORT einsums_targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT ${lib}
  )
  einsums_export_targets(${lib})

  add_subdirectory(${lib})
endforeach()

set(PYTHON_LIBS
    EinsumsPy
    CACHE INTERNAL "List of Python libraries for Einsums" FORCE
)

if(EINSUMS_BUILD_PYTHON OR EINSUMS_BUILD_PYTHON MATCHES "3(\.[0-9]+)*")

  set(_old_default_comp "${CMAKE_INSTALL_DEFAULT_COMPONENT_NAME}")
  set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME EinsumsPy)

  file(RELATIVE_PATH _ein_pymod_relDir ${CMAKE_INSTALL_PREFIX}/${EINSUMS_INSTALL_PYMODDIR}/einsums
                                       ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR})
  include(Einsums_ExtendWithPython)

  foreach(lib ${PYTHON_LIBS})
    set(EINSUMS_PYTHON_LIB_NAME ${lib})
    pybind11_add_module(${lib} MODULE src/dummy.cpp)

    set_target_properties(${lib} PROPERTIES OUTPUT_NAME ${lib} EXPORT_NAME ${lib} FOLDER "Core")

    # adds some RPATH and link flags
    einsums_extend_with_python(${lib})

    if(${BLA_VENDOR} MATCHES "[mM][kK][lL]")
      target_link_libraries(${lib} $<$<CXX_COMPILER_ID:GNU>:atomic>)
    endif()

    install(
      TARGETS ${lib}
      EXPORT einsums_targets
      LIBRARY DESTINATION ${EINSUMS_INSTALL_PYMODDIR}/einsums
      ARCHIVE DESTINATION ${EINSUMS_INSTALL_PYMODDIR}/einsums
      RUNTIME DESTINATION ${EINSUMS_INSTALL_PYMODDIR}/einsums
      COMPONENT ${lib}
    )
    einsums_export_targets(${lib})

    add_subdirectory(${lib})
  endforeach()

  install(
    DIRECTORY
      ../tests
    DESTINATION
      ${EINSUMS_INSTALL_PYMODDIR}/einsums
    USE_SOURCE_PERMISSIONS
    COMPONENT EinsumsPy
    FILES_MATCHING
      PATTERN "test_*.py"
  )
  install(
    FILES
      ../pyeinsums/__init__.py
      ../pyeinsums/utils.py
    DESTINATION
      ${EINSUMS_INSTALL_PYMODDIR}/einsums
    COMPONENT EinsumsPy
  )
  set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME "${_old_default_comp}")
endif()

# einsums_add_module populates EINSUMS_ENABLED_MODULES
foreach(module ${EINSUMS_ENABLED_MODULES})
  string(TOUPPER ${module} uppercase_module)

  set(MODULE_ENABLED_MODULE_DEFINES
      "${MODULE_ENABLED_MODULE_DEFINES}#define EINSUMS_HAVE_MODULE_${uppercase_module}\n"
  )
endforeach()

configure_file(
  "${PROJECT_SOURCE_DIR}/cmake/templates/ModulesEnabled.hpp.in"
  "${PROJECT_BINARY_DIR}/libs/Einsums/Config/include/Einsums/Config/ModulesEnabled.hpp" @ONLY
)

if(EINSUMS_WITH_ITTNOTIFY)
  target_link_libraries(Einsums PUBLIC einsums_internal::amplifier)
endif()
