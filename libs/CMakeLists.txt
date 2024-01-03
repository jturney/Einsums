#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

include(einsums_message)
include(einsums_add_pseudo_dependencies)
include(einsums_add_pseudo_target)

# add example pseudo targets needed for modules
if(EINSUMS_WITH_EXAMPLES)
    einsums_add_pseudo_target(examples.modules)
    einsums_add_pseudo_dependencies(examples examples.modules)
endif()

# add test pseudo targets needed for modules einsums_warn("EINSUMS_WITH_TESTS=${EINSUMS_WITH_TESTS}")
if(EINSUMS_WITH_TESTS)
    if(EINSUMS_WITH_TESTS_UNIT)
        einsums_add_pseudo_target(tests.unit.modules)
        einsums_add_pseudo_dependencies(tests.unit tests.unit.modules)
    endif()

    if(EINSUMS_WITH_EXAMPLES AND EINSUMS_WITH_TESTS_EXAMPLES)
        einsums_add_pseudo_target(tests.examples.modules)
        einsums_add_pseudo_dependencies(tests.examples tests.examples.modules)
    endif()

    if(EINSUMS_WITH_TESTS_REGRESSIONS)
        einsums_add_pseudo_target(tests.regressions.modules)
        einsums_add_pseudo_dependencies(tests.regressions tests.regressions.modules)
    endif()

    if(EINSUMS_WITH_TESTS_BENCHMARKS)
        einsums_add_pseudo_target(tests.performance.modules)
        einsums_add_pseudo_dependencies(tests.performance tests.performance.modules)
    endif()

    if(EINSUMS_WITH_TESTS_HEADERS)
        einsums_add_pseudo_target(tests.headers.modules)
        einsums_add_pseudo_dependencies(tests.headers tests.headers.modules)
    endif()
endif()

# Generate a file that lists all enabled modules for checks that might be optional based on the presence of a module or
# not
set(MODULE_ENABLED_MODULE_DEFINES)

# variables needed for config_strings_modules.hpp
set(CONFIG_STRINGS_MODULES_INCLUDES)
set(CONFIG_STRINGS_MODULES_ENTRIES)

if(MSVC AND MSVC14)
    # add natvis files to solution
    einsums_add_library_sources(einsums_natvis_files GLOB GLOBS "${PROJECT_SOURCE_DIR}/tools/VS/*.natvis")
    source_group("Natvis Files" FILES ${einsums_natvis_files_SOURCES})
endif()

set(einsums_libs einsums)

foreach(lib ${einsums_libs})
    add_library(${lib} SHARED src/dummy.cpp ${einsums_natvis_files_SOURCES})
    set_target_properties(
        ${lib}
        PROPERTIES VERSION ${EINSUMS_VERSION}
                   SOVERSION ${EINSUMS_SOVERSION}
                   CLEAN_DIRECT_OUTPUT 1
                   OUTPUT_NAME ${lib}
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

# einsums_add_module populates EINSUMS_ENABLED_MODULES
foreach(module ${EINSUMS_ENABLED_MODULES})
    string(TOUPPER ${module} uppercase_module)

    set(MODULE_ENABLED_MODULE_DEFINES
        "${MODULE_ENABLED_MODULE_DEFINES}#define EINSUMS_HAVE_MODULE_${uppercase_module}\n"
    )
endforeach()

configure_file(
    "${PROJECT_SOURCE_DIR}/cmake/templates/ModulesEnabled.hpp.in"
    "${PROJECT_BINARY_DIR}/libs/einsums/config/include/einsums/config/ModulesEnabled.hpp" @ONLY
)

if(MSVC AND EINSUMS_WITH_SWAP_CONTEXT_EMULATION)
    target_link_options(einsums PRIVATE "/EXPORT:switch_to_fiber")
endif()

if(EINSUMS_WITH_ITTNOTIFY)
    target_link_libraries(einsums PUBLIC einsums_internal::amplifier)
endif()

include(einsums_setup_h5cpp)
