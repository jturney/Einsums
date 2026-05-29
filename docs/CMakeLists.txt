#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

# used while generating sphinx config file
set(doxygen_definition_list
    "DOXYGEN:=1"
    "EINSUMS_EXPORT="
    "EINSUMS_FORCEINLINE="
    "__HIP__="
    "__device_ptr__="
    "__host_ptr__="
    "EINSUMS_HOST="
    "EINSUMS_DEVICE="
    "EINSUMS_HOSTDEV="
)

set(SPHINX_PYTHON_DEPENDS)

if (NOT EINSUMS_BUILD_PYTHON)
  einsums_error("Building documentation requires Python to be enabled")
endif()

set(SPHINX_PYTHON_DEPENDS PyEinsums)

foreach(doxygen_predef ${doxygen_definition_list})
  set(doxygen_definitions "${doxygen_definitions} \"${doxygen_predef}\"")
endforeach()

# Generate rst files with CMake variables and toolchains
set(EINSUMS_CMAKE_OPTIONS_RST "")
foreach(_cat ${EINSUMS_OPTION_CATEGORIES})
  string(REPLACE " " "_" _cat_link "${_cat}")
  set(EINSUMS_CMAKE_OPTIONS_RST
      "${EINSUMS_CMAKE_OPTIONS_RST}* :ref:`${_cat} options <${_cat_link}>`\n"
  )
endforeach()
set(EINSUMS_CMAKE_OPTIONS_RST "${EINSUMS_CMAKE_OPTIONS_RST}\n")

get_cmake_property(_variableNames CACHE_VARIABLES)
foreach(_cat ${EINSUMS_OPTION_CATEGORIES})
  string(REPLACE " " "_" _cat_anchor "${_cat}")
  set(_cat_title "${_cat} options")
  string(REGEX REPLACE "." "-" _cat_title_underline "${_cat_title}")
  set(EINSUMS_CMAKE_OPTIONS_RST
      "${EINSUMS_CMAKE_OPTIONS_RST}.. _${_cat_anchor}:\n\n${_cat_title}\n${_cat_title_underline}\n\n"
  )
  foreach(_variableName ${_variableNames})
    if(${_variableName}Category)
      if(_cat STREQUAL ${_variableName}Category)
        string(REPLACE " " "_" _variableName_anchor "${_variableName}")

        get_property(
          _doc
          CACHE "${_variableName}"
          PROPERTY HELPSTRING
        )
        get_property(
          _type
          CACHE "${_variableName}"
          PROPERTY TYPE
        )
        if(NOT _type STREQUAL "UNINITIALIZED")
          set(_variableName "${_variableName}:${_type}")
        endif()
        set(EINSUMS_CMAKE_OPTIONS_RST "${EINSUMS_CMAKE_OPTIONS_RST}* :option:`${_variableName}`\n")
      endif()
    endif()
  endforeach()
  set(EINSUMS_CMAKE_OPTIONS_RST "${EINSUMS_CMAKE_OPTIONS_RST}\n")

  foreach(_variableName ${_variableNames})
    if(${_variableName}Category)
      if(_cat STREQUAL ${_variableName}Category)
        string(REPLACE " " "_" _variableName_anchor "${_variableName}")

        get_property(
          _doc
          CACHE "${_variableName}"
          PROPERTY HELPSTRING
        )
        get_property(
          _type
          CACHE "${_variableName}"
          PROPERTY TYPE
        )
        if(NOT _type STREQUAL "UNINITIALIZED")
          set(_variableName "${_variableName}:${_type}")
        endif()
        set(EINSUMS_CMAKE_OPTIONS_RST
            "${EINSUMS_CMAKE_OPTIONS_RST}.. option:: ${_variableName}\n\n   ${_doc}\n\n"
        )
      endif()
    endif()
  endforeach()
endforeach()

set(SPHINX_CMAKE_VARIABLES_RST_DEST
    "${CMAKE_CURRENT_BINARY_DIR}/sphinx/generated/CMakeVariables.rst"
)
configure_file(
  "${PROJECT_SOURCE_DIR}/cmake/templates/CMakeVariables.rst.in" ${SPHINX_CMAKE_VARIABLES_RST_DEST}
  @ONLY
)

# Configure Sphinx conf.py
set(EINSUMS_CONF_DOCUMENTATION_SHORT_VERSION "${EINSUMS_VERSION}")
if(EINSUMS_WITH_GIT_TAG)
  set(EINSUMS_CONF_DOCUMENTATION_VERSION "${EINSUMS_WITH_GIT_TAG}")
elseif(EINSUMS_WITH_GIT_BRANCH)
  set(EINSUMS_CONF_DOCUMENTATION_VERSION "${EINSUMS_WITH_GIT_BRANCH}")
else()
  # EINSUMS_VERSION is always available, use as fallback
  set(EINSUMS_CONF_DOCUMENTATION_VERSION "${EINSUMS_VERSION_FULL}")
endif()

set(SPHINX_CONF_PY_DEST "${CMAKE_CURRENT_BINARY_DIR}/sphinx/conf.py")
configure_file("${PROJECT_SOURCE_DIR}/cmake/templates/conf.py.in" ${SPHINX_CONF_PY_DEST} @ONLY)

# Copy the Sphinx source files to build directory
file(
  GLOB_RECURSE sphinx_source_files
  LIST_DIRECTORIES false
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}/"
  "sphinx/*"
)

# exclude api.rst...
list(FILTER sphinx_source_files EXCLUDE REGEX "sphinx/api/full_api.rst")

foreach(sphinx_source_file ${sphinx_source_files})
  add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${sphinx_source_file}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${sphinx_source_file}"
    COMMAND
      ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/${sphinx_source_file}"
      "${CMAKE_CURRENT_BINARY_DIR}/${sphinx_source_file}"
  )
endforeach()

foreach(sphinx_source_file ${sphinx_source_files})
  set(sphinx_source_files_build ${sphinx_source_files_build}
                                "${CMAKE_CURRENT_BINARY_DIR}/${sphinx_source_file}"
  )
endforeach()

# Copy the Sphinx source files of the libs directory to build directory
file(
  GLOB_RECURSE sphinx_libs_source_files
  LIST_DIRECTORIES false
  RELATIVE "${PROJECT_SOURCE_DIR}/libs"
  "${PROJECT_SOURCE_DIR}/libs/*/*/docs/*"
)

list(APPEND sphinx_libs_source_files "index.rst" "overview.rst")

foreach(lib IN LISTS EINSUMS_LIBS PYTHON_LIBS)
  file(
    GLOB _sphinx_sources
    RELATIVE "${PROJECT_SOURCE_DIR}/libs"
    "${PROJECT_SOURCE_DIR}/libs/${lib}/*.rst"
  )
  list(APPEND sphinx_libs_source_files ${_sphinx_sources})
endforeach()

foreach(sphinx_source_file ${sphinx_libs_source_files})
  add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/sphinx/libs/${sphinx_source_file}"
    DEPENDS "${PROJECT_SOURCE_DIR}/libs/${sphinx_source_file}"
    COMMAND
      ${CMAKE_COMMAND} -E copy_if_different "${PROJECT_SOURCE_DIR}/libs/${sphinx_source_file}"
      "${CMAKE_CURRENT_BINARY_DIR}/sphinx/libs/${sphinx_source_file}"
  )
endforeach()

foreach(sphinx_source_file ${sphinx_libs_source_files})
  set(sphinx_source_files_build ${sphinx_source_files_build}
                                "${CMAKE_CURRENT_BINARY_DIR}/sphinx/libs/${sphinx_source_file}"
  )
endforeach()

# Create links to source files so that they are visible to sphinx
# create_symbolic_link("${PROJECT_SOURCE_DIR}/src" "${CMAKE_CURRENT_BINARY_DIR}/src")
# create_symbolic_link("${PROJECT_SOURCE_DIR}/Einsums" "${CMAKE_CURRENT_BINARY_DIR}/Einsums")
create_symbolic_link("${PROJECT_SOURCE_DIR}/examples" "${CMAKE_CURRENT_BINARY_DIR}/examples")
create_symbolic_link("${PROJECT_SOURCE_DIR}/tests" "${CMAKE_CURRENT_BINARY_DIR}/tests")
create_symbolic_link("${PROJECT_SOURCE_DIR}/libs" "${CMAKE_CURRENT_BINARY_DIR}/libs")

add_custom_target(docs)
add_custom_target(docs-html)
add_custom_target(docs-singlehtml)
add_custom_target(docs-latexpdf)
add_custom_target(docs-man)

set(SPHINX_DOCS_HTML_OUTPUT_FILE "index.html")
set(SPHINX_DOCS_SINGLEHTML_OUTPUT_FILE "index.html")
set(SPHINX_DOCS_PDF_OUTPUT_FILE "latex/Einsums.pdf")
set(SPHINX_DOCS_MAN_OUTPUT_FILE "Einsums.1")

foreach(output_format ${EINSUMS_WITH_DOCUMENTATION_OUTPUT_FORMATS})
  string(TOUPPER ${output_format} output_format_upper_case)
  set(SPHINX_DOCS_OUTPUT_DIR "${PROJECT_BINARY_DIR}/share/Einsums/docs/${output_format}")
  set(SPHINX_DOCS_OUTPUT_FILE
      "${SPHINX_DOCS_OUTPUT_DIR}/${SPHINX_DOCS_${output_format_upper_case}_OUTPUT_FILE}"
  )

  # NOTE: The PDF does not build without errors but still produces a reasonable PDF. If there are
  # real errors this will ignore them. Fixing the errors is at the moment not worth the effort.
  if(${output_format} STREQUAL "latexpdf")
    set(SPHINX_DOCS_BUILD_COMMAND
        ${Sphinx_EXECUTABLE} -M ${output_format} "${CMAKE_CURRENT_BINARY_DIR}/sphinx"
        "${SPHINX_DOCS_OUTPUT_DIR}" || (exit 0)
    )
  else()
    set(SPHINX_DOCS_BUILD_COMMAND
        ${Sphinx_EXECUTABLE} -b ${output_format} -n -d "${CMAKE_CURRENT_BINARY_DIR}/doctree"
        "${CMAKE_CURRENT_BINARY_DIR}/sphinx" "${SPHINX_DOCS_OUTPUT_DIR}"
    )
  endif()

  add_custom_command(
    OUTPUT "${SPHINX_DOCS_OUTPUT_FILE}"
    DEPENDS "${sphinx_source_files_build}" Einsums ${SPHINX_PYTHON_DEPENDS}
    COMMAND ${SPHINX_DOCS_BUILD_COMMAND}
  )

  add_custom_target(docs-${output_format}-file ALL DEPENDS ${SPHINX_DOCS_OUTPUT_FILE})
  add_dependencies(docs-${output_format} docs-${output_format}-file)

  add_dependencies(docs docs-${output_format})
endforeach()


add_custom_target(
  git_docs
  COMMAND
    "${CMAKE_COMMAND}" -DEINSUMS_BINARY_DIR:PATH=${PROJECT_BINARY_DIR}
    -DEINSUMS_VERSION:STRING=${EINSUMS_VERSION}
    -DEINSUMS_WITH_GIT_BRANCH=${EINSUMS_WITH_GIT_BRANCH}
    -DEINSUMS_WITH_GIT_TAG=${EINSUMS_WITH_GIT_TAG}
    -DEINSUMS_WITH_DOCUMENTATION_OUTPUT_FORMATS="${EINSUMS_WITH_DOCUMENTATION_OUTPUT_FORMATS}" -P
    "${PROJECT_SOURCE_DIR}/cmake/Einsums_UpdateGitDocs.cmake"
)

set_target_properties(git_docs PROPERTIES FOLDER "Documentation/")

add_dependencies(git_docs docs)
