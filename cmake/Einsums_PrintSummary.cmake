#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

function(einsums_create_configuration_summary message module_name)

  set(einsums_config_information)
  set(upper_cfg_name "EINSUMS")
  set(upper_option_suffix "")

  string(TOUPPER ${module_name} module_name_uc)
  if(NOT "${module_name_uc}x" STREQUAL "EINSUMSx")
    set(upper_cfg_name "EINSUMS_${module_name_uc}")
    set(upper_option_suffix "_${module_name_uc}")
  endif()

  get_property(DEFINITIONS_VARS GLOBAL PROPERTY EINSUMS_CONFIG_DEFINITIONS${upper_option_suffix})
  if(DEFINED DEFINITIONS_VARS)
    list(SORT DEFINITIONS_VARS)
    list(REMOVE_DUPLICATES DEFINITIONS_VARS)
  endif()

  get_property(_variableNames GLOBAL PROPERTY EINSUMS_MODULE_CONFIG_${module_name_uc})
  list(SORT _variableNames)

  # Only print the module configuration if options specified
  list(LENGTH _variableNames _length)
  if(${_length} GREATER_EQUAL 1)
    einsums_info("")
    einsums_info(${message})

    foreach(_variableName ${_variableNames})
      if(${_variableName}Category)

        # handle only options which start with EINSUMS_WITH_
        string(FIND ${_variableName} "${upper_cfg_name}_WITH_" __pos)

        if(${__pos} EQUAL 0)
          get_property(
            _value
            CACHE "${_variableName}"
            PROPERTY VALUE
          )
          einsums_info("    ${_variableName}=${_value}")
        endif()

        string(REPLACE "_WITH_" "_HAVE_" __variableName ${_variableName})
        list(FIND DEFINITIONS_VARS ${__variableName} __pos)
        if(NOT ${__pos} EQUAL -1)
          set(einsums_config_information "${einsums_config_information}"
                                         "\n        \"${_variableName}=${_value}\","
          )
        elseif(NOT ${_variableName}Category STREQUAL "Generic" AND NOT ${_variableName}Category
                                                                   STREQUAL "Build Targets"
        )
          get_property(
            _type
            CACHE "${_variableName}"
            PROPERTY TYPE
          )
          if(_type STREQUAL "BOOL")
            set(einsums_config_information "${einsums_config_information}"
                                           "\n        \"${_variableName}=OFF\","
            )
          endif()
        endif()
      endif()
    endforeach()
  endif()

  if(einsums_config_information)
    string(REPLACE ";" "" einsums_config_information ${einsums_config_information})
  endif()

  if("${module_name_uc}" STREQUAL "EINSUMS")
    # Write via a temp file + compare_files so we only touch ConfigStrings.hpp
    # when the content actually changes. Without this, Einsums_AddModule's
    # zombie cleanup deletes ConfigStrings.hpp on every regen (it's generated
    # by this function, not add_module), then configure_file recreates it from
    # scratch — bumping mtime each regen and cascading a full Einsums rebuild.
    set(_out "${EINSUMS_BINARY_DIR}/libs/${PROJECT_NAME}/Config/include/${PROJECT_NAME}/Config/ConfigStrings.hpp")
    set(_tmp "${_out}.tmp")
    get_filename_component(_out_dir "${_out}" DIRECTORY)
    if(NOT IS_DIRECTORY "${_out_dir}")
      file(MAKE_DIRECTORY "${_out_dir}")
    endif()
    configure_file(
      "${EINSUMS_SOURCE_DIR}/cmake/templates/ConfigDefinesStrings.hpp.in"
      "${_tmp}"
      @ONLY
    )
    execute_process(
      COMMAND ${CMAKE_COMMAND} -E compare_files "${_out}" "${_tmp}"
      RESULT_VARIABLE _cmp_res
      ERROR_QUIET
    )
    if(_cmp_res)
      file(RENAME "${_tmp}" "${_out}")
    else()
      file(REMOVE "${_tmp}")
    endif()
  endif()
endfunction()
