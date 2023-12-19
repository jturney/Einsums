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
                                                   "\n    \"${_variableName}=${_value}\","
                    )
                elseif(NOT ${_variableName}Category STREQUAL "Generic" AND NOT ${_variableName}Category STREQUAL
                                                                           "Build Targets"
                )
                    get_property(
                        _type
                        CACHE "${_variableName}"
                        PROPERTY TYPE
                    )
                    if(_type STREQUAL "BOOL")
                        set(einsums_config_information "${einsums_config_information}"
                                                       "\n    \"${_variableName}=OFF\","
                        )
                    endif()
                endif()
            endif()
        endforeach()

        einsums_info("")
    endif()

    if(einsums_config_information)
        string(REPLACE ";" "" einsums_config_information ${einsums_config_information})
    endif()

    set(_base_dir_local "einsums/config")
    set(_base_dir "einsums/config")
    set(_template "ConfigDefinesStrings.hpp.in")

    if("${module_name_uc}" STREQUAL "EINSUMS")
        configure_file(
            "${EINSUMS_SOURCE_DIR}/cmake/templates/${_template}"
            "${EINSUMS_BINARY_DIR}/${_base_dir_local}/ConfigStrings.hpp" @ONLY
        )
        configure_file(
            "${EINSUMS_SOURCE_DIR}/cmake/templates/${_template}"
            "${EINSUMS_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/${_base_dir}/ConfigStrings.hpp" @ONLY
        )
    endif()
endfunction()
