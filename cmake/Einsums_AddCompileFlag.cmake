#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

include(CMakeParseArguments)
include(CheckCXXCompilerFlag)

#:
#: .. cmake:command:: einsums_add_target_compile_option
#:
#:    Add a compile flag conditionally based on configuration and language.
#:
#:    This helper parses arguments so callers can later apply the flag with
#:    the desired visibility (e.g., ``PUBLIC``) and constraints.
#:
#:    **Signature**
#:    ``einsums_add_target_compile_option(<flag> [PUBLIC] [CONFIGURATIONS ...] [LANGUAGES ...])``
#:
#:    **Arguments**
#:
#:    - ``<flag>`` *(required)*:
#:      The compiler flag to be considered (e.g., ``-Wall``, ``/W4``, ``-Werror``).
#:
#:    - ``PUBLIC`` *(optional, boolean switch)*:
#:      Indicates the flag should be applied with **PUBLIC** visibility when
#:      used with ``target_compile_options()`` by the caller. If omitted, you
#:      can treat it as **PRIVATE** by default (up to the caller’s policy).
#:
#:    - ``CONFIGURATIONS`` *(optional)*:
#:      List of build configs the flag should apply to (e.g., ``Debug``, ``Release``).
#:
#:    - ``LANGUAGES`` *(optional)*:
#:      List of languages to which the flag applies (e.g., ``C``, ``CXX``, ``CUDA``).
#:
#:    **Example**
#:
#:    .. code-block:: cmake
#:
#:       # Parse intent
#:       einsums_add_target_compile_option(
#:         -Werror
#:         PUBLIC
#:         CONFIGURATIONS Debug
#:         LANGUAGES C CXX
#:       )
#:
function(einsums_add_target_compile_option FLAG)
  set(options PUBLIC)
  set(one_value_args)
  set(multi_value_args CONFIGURATIONS LANGUAGES)
  cmake_parse_arguments(
    target_compile_option "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN}
  )

  if(target_compile_option_PUBLIC)
    set(_dest einsums_public_flags)
  else()
    set(_dest einsums_private_flags)
  endif()

  set(_configurations "none")
  if(target_compile_option_CONFIGURATIONS)
    set(_configurations ${target_compile_option_CONFIGURATIONS})
  endif()

  set(_languages "CXX" "C" "ASM")
  if(target_compile_option_LANGUAGES)
    set(_languages ${target_compile_option_LANGUAGES})
  endif()

  foreach(_lang ${_languages})
    foreach(_config ${_configurations})
      if(NOT _config STREQUAL "none")
        set(_config "$<$<AND:$<CONFIG:${_config}>,$<COMPILE_LANGUAGE:${_lang}>>:${FLAG}>")
      else()
        set(_config "$<$<COMPILE_LANGUAGE:${_lang}>:${FLAG}>")
      endif()
      target_compile_options(${_dest} INTERFACE "${_config}")
    endforeach()
  endforeach()
endfunction()

#:
#: .. cmake:command:: einsums_add_target_compile_option_if_available
#:
#:    Add a compile flag only if it is supported by the compiler.
#:
#:    This function checks whether a given compiler flag is available for the
#:    specified language(s), and conditionally applies it to the target using
#:    Einsums' `einsums_add_target_compile_option()` wrapper.
#:
#:    **Signature**
#:    ``einsums_add_target_compile_option_if_available(<flag> [NAME <name>] [PUBLIC] [CONFIGURATIONS ...] [LANGUAGES ...])``
#:
#:    **Arguments**
#:
#:    - ``<flag>>`` *(required)*:
#:      The compiler flag to test and possibly add, e.g., ``-march=native``, ``-Wno-unused-variable``.
#:
#:    - ``NAME`` *(optional)*:
#:      Override the generated variable name used for caching the result of `check_cxx_compiler_flag()`.
#:      By default, the variable name is derived from the sanitized form of the flag itself.
#:
#:    - ``PUBLIC`` *(optional, boolean switch)*:
#:      If set, the flag will be added with `PUBLIC` visibility in `target_compile_options()`.
#:      Otherwise, defaults to `PRIVATE`.
#:
#:    - ``CONFIGURATIONS`` *(optional)*:
#:      List of CMake build configurations (e.g., ``Debug``, ``Release``) to restrict the flag to.
#:
#:    - ``LANGUAGES`` *(optional)*:
#:      List of languages to test the flag against (currently only supports ``CXX``).
#:      Defaults to ``CXX`` if omitted.
#:
#:    **Behavior**
#:
#:    For each specified language (default: ``CXX``), this function:
#:
#:    1. Checks if the compiler accepts the given flag using `check_cxx_compiler_flag()`.
#:
#:    2. If supported:
#:       - Calls :cmake:command:`einsums_add_target_compile_option` with the flag and other parsed arguments.
#:
#:    3. If not supported:
#:       - Logs an informational message using `einsums_info()`.
#:
#:    Unsupported languages result in an error via `einsums_error()`.
#:
#:    **Caching**
#:
#:    A CMake cache variable is created per language/flag combo:
#:    ``EINSUMS_WITH_<LANG>_FLAG_<SANITIZED_NAME>``, where the name is auto-generated
#:    from the flag or overridden with `NAME`.
#:
#:    **Sanitization Rules for Variable Names**
#:
#:    - Remove leading `-` characters
#:    - Convert `=`, `-`, `,` → `_`
#:    - Convert `+` → `X`
#:
#:    **Example**
#:
#:    .. code-block:: cmake
#:
#:       einsums_add_target_compile_option_if_available(
#:         -Wno-unused-parameter
#:         PUBLIC
#:         CONFIGURATIONS Debug
#:         LANGUAGES CXX
#:       )
#:
#:       einsums_add_target_compile_option_if_available(
#:         -march=native
#:         NAME march_native
#:         CONFIGURATIONS Release
#:       )
#:
#:    In the second case, the check result is cached in:
#:    ``EINSUMS_WITH_CXX_FLAG_MARCH_NATIVE``
#:
#:    **See also**
#:
#:    - :cmake:command:`einsums_add_target_compile_option`
#:    - :cmake:command:`check_cxx_compiler_flag`
#:    - :cmake:command:`target_compile_options`
function(einsums_add_target_compile_option_if_available FLAG)
  set(options PUBLIC)
  set(one_value_args NAME)
  set(multi_value_args CONFIGURATIONS LANGUAGES)
  cmake_parse_arguments(
    target_compile_option_ia "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN}
  )

  if(target_compile_option_ia_PUBLIC)
    set(_modifier PUBLIC)
  else()
    set(_modifier PRIVATE)
  endif()

  if(target_compile_option_ia_NAME)
    string(TOUPPER ${target_compile_option_ia_NAME} _name)
  else()
    string(TOUPPER ${FLAG} _name)
  endif()

  string(REGEX REPLACE "^-+" "" _name ${_name})
  string(REGEX REPLACE "[=\\-]" "_" _name ${_name})
  string(REGEX REPLACE "," "_" _name ${_name})
  string(REGEX REPLACE "\\+" "X" _name ${_name})

  set(_languages "CXX")
  if(target_compile_option_ia_LANGUAGES)
    set(_languages ${target_compile_option_ia_LANGUAGES})
  endif()

  foreach(_lang ${_languages})
    if(_lang STREQUAL "CXX")
      check_cxx_compiler_flag(${FLAG} EINSUMS_WITH_${_lang}_FLAG_${_name})
    else()
      einsums_error("Unsupported language: ${_lang}.")
    endif()
    if(EINSUMS_WITH_${_lang}_FLAG_${_name})
      einsums_add_target_compile_option(
        ${FLAG} ${_modifier}
        CONFIGURATIONS ${target_compile_option_ia_CONFIGURATIONS}
        LANGUAGES ${_lang}
      )
    else()
      einsums_info("\"${FLAG}\" not available for language ${_lang}.")
    endif()
  endforeach()
endfunction()

#:
#: .. cmake:command:: einsums_remove_target_compile_option
#:
#:    Remove a previously registered compile flag from global Einsums compile options.
#:
#:    This function removes a compile flag (with optional configuration constraints)
#:    from the set of global compile options that Einsums tracks using global
#:    properties. These are typically applied to targets later via custom logic
#:    (e.g., when building Einsums or its backends).
#:
#:    **Signature**
#:    ``einsums_remove_target_compile_option(<flag> [PUBLIC] [CONFIGURATIONS ...])``
#:
#:    **Arguments**
#:
#:    - ``<flag>`` *(required)*:
#:      The compile flag string to remove (e.g., ``-march=native`` or ``-Werror``).
#:
#:    - ``PUBLIC`` *(optional, boolean switch)*:
#:      If passed, the flag is removed from the `PUBLIC` set of global compile options.
#:      If omitted, it is removed from the `PRIVATE` set. Flags are stored separately.
#:
#:    - ``CONFIGURATIONS`` *(optional)*:
#:      A list of build configurations (e.g., ``Debug``, ``Release``) from which to remove the flag.
#:      If omitted, the flag is matched unconditionally.
#:
#:    **Behavior**
#:
#:    The Einsums build system tracks global compile options using two global properties:
#:
#:    - ``EINSUMS_TARGET_COMPILE_OPTIONS_PUBLIC``
#:    - ``EINSUMS_TARGET_COMPILE_OPTIONS_PRIVATE``
#:
#:    This function removes any exact matches to the given flag (optionally wrapped in
#:    generator expressions for specific configs) from these properties. For example:
#:
#:    - If ``CONFIGURATIONS Debug`` is passed, the function matches and removes:
#:      ``$<$<CONFIG:Debug>:-Werror>``
#:    - If no configurations are passed, the function matches the plain flag string.
#:
#:    The implementation:
#:
#:    1. Retrieves current values of the `PUBLIC` and `PRIVATE` global properties.
#:
#:    2. Clears the existing properties.
#:
#:    3. Re-adds only those flags that do **not** match the given ``FLAG`` + config.
#:
#:    **Example**
#:
#:    .. code-block:: cmake
#:
#:       einsums_remove_target_compile_option(-Werror CONFIGURATIONS Debug)
#:
#:       einsums_remove_target_compile_option(-march=native PUBLIC)
#:
#:    This removes the `-Werror` flag from the Debug configuration and
#:    `-march=native` from the PUBLIC global flag list unconditionally.
#:
#:    **See also**
#:
#:    - :cmake:command:`einsums_add_target_compile_option`
#:    - :cmake:command:`einsums_add_target_compile_option_if_available`
#:    - :cmake:command:`target_compile_options`
#:    - :cmake:command:`set_property(GLOBAL ...)`
function(einsums_remove_target_compile_option FLAG)
  set(options PUBLIC)
  set(one_value_args)
  set(multi_value_args CONFIGURATIONS)
  cmake_parse_arguments(
    EINSUMS_ADD_TARGET_COMPILE_OPTION "${options}" "${one_value_args}" "${multi_value_args}"
    ${ARGN}
  )

  set(_configurations "none")
  if(EINSUMS_ADD_TARGET_COMPILE_OPTION_CONFIGURATIONS)
    set(_configurations "${EINSUMS_ADD_TARGET_COMPILE_OPTION_CONFIGURATIONS}")
  endif()

  get_property(
    EINSUMS_TARGET_COMPILE_OPTIONS_PUBLIC_VAR GLOBAL PROPERTY EINSUMS_TARGET_COMPILE_OPTIONS_PUBLIC
  )
  get_property(
    EINSUMS_TARGET_COMPILE_OPTIONS_PRIVATE_VAR GLOBAL
    PROPERTY EINSUMS_TARGET_COMPILE_OPTIONS_PRIVATE
  )
  set_property(GLOBAL PROPERTY EINSUMS_TARGET_COMPILE_OPTIONS_PUBLIC "")
  set_property(GLOBAL PROPERTY EINSUMS_TARGET_COMPILE_OPTIONS_PRIVATE "")

  foreach(_config ${_configurations})
    set(_conf "${FLAG}")
    if(NOT _config STREQUAL "none")
      set(_conf "$<$<CONFIG:${_config}>:${FLAG}>")
    endif()
    foreach(_flag ${EINSUMS_TARGET_COMPILE_OPTIONS_PUBLIC_VAR})
      if(NOT ${_flag} STREQUAL ${_conf})
        set_property(GLOBAL APPEND PROPERTY EINSUMS_TARGET_COMPILE_OPTIONS_PUBLIC "${_flag}")
      endif()
    endforeach()
    foreach(_flag ${EINSUMS_TARGET_COMPILE_OPTIONS_PRIVATE_VAR})
      if(NOT ${_flag} STREQUAL ${_conf})
        set_property(GLOBAL APPEND PROPERTY EINSUMS_TARGET_COMPILE_OPTIONS_PRIVATE "${_flag}")
      endif()
    endforeach()
  endforeach()
endfunction()

#:
#: .. cmake:command:: einsums_add_target_compile_definition
#:
#:    Add a preprocessor definition to Einsums' global interface targets.
#:
#:    This function registers a `#define` (compile definition) on one of the
#:    Einsums interface targets (`einsums_public_flags` or `einsums_private_flags`)
#:    for a given configuration.
#:
#:    **Signature**
#:    ``einsums_add_target_compile_definition(<definition> [PUBLIC] [CONFIGURATIONS ...])``
#:
#:    **Arguments**
#:
#:    - ``<definition>`` *(required)*:
#:      The compile-time macro to define, e.g., ``EINSUMS_USE_MIMALLOC`` or ``VERSION=42``.
#:      This is passed directly to :cmake:command:`target_compile_definitions`.
#:
#:    - ``PUBLIC`` *(optional, boolean switch)*:
#:      If passed, adds the definition to the `einsums_public_flags` target.
#:      Otherwise, it is added to `einsums_private_flags`.
#:
#:    - ``CONFIGURATIONS`` *(optional)*:
#:      A list of build configurations (e.g., ``Debug``, ``Release``).
#:      If specified, the flag is wrapped in a generator expression to only apply for those configurations.
#:
#:    **Behavior**
#:
#:    - Determines whether to apply to `einsums_public_flags` or `einsums_private_flags`
#:      based on presence of the `PUBLIC` switch.
#:
#:    - Wraps the definition in a config-specific generator expression if
#:      `CONFIGURATIONS` are specified.
#:
#:    - Adds the result to the target via:
#:      ``target_compile_definitions(<target> INTERFACE <definition>)``
#:
#:    The two global interface targets (`einsums_public_flags` and `einsums_private_flags`) are
#:    used by Einsums to aggregate global compile options and definitions and attach them to
#:    real targets later (e.g., in backend setup macros).
#:
#:    **Example**
#:
#:    .. code-block:: cmake
#:
#:       # Define a macro in all configs
#:       einsums_add_target_compile_definition(EINSUMS_USE_MIMALLOC PUBLIC)
#:
#:       # Define a macro only in Release builds
#:       einsums_add_target_compile_definition(
#:         VERSION=42
#:         CONFIGURATIONS Release
#:       )
#:
#:    **See also**
#:
#:    - :cmake:command:`target_compile_definitions`
#:    - :cmake:command:`einsums_add_target_compile_option`
#:    - ``einsums_public_flags`` and ``einsums_private_flags`` interface targets
function(einsums_add_target_compile_definition FLAG)
  set(options PUBLIC)
  set(one_value_args)
  set(multi_value_args CONFIGURATIONS)
  cmake_parse_arguments(
    EINSUMS_ADD_TARGET_COMPILE_DEFINITION "${options}" "${one_value_args}" "${multi_value_args}"
    ${ARGN}
  )

  if(EINSUMS_ADD_TARGET_COMPILE_DEFINITION_PUBLIC)
    set(_dest einsums_public_flags)
  else()
    set(_dest einsums_private_flags)
  endif()

  set(_configurations "none")
  if(EINSUMS_ADD_TARGET_COMPILE_DEFINITION_CONFIGURATIONS)
    set(_configurations "${EINSUMS_ADD_TARGET_COMPILE_DEFINITION_CONFIGURATIONS}")
  endif()

  foreach(_config ${_configurations})
    set(_conf "${FLAG}")
    if(NOT _config STREQUAL "none")
      set(_conf "$<$<CONFIG:${_config}>:${FLAG}>")
    endif()
    target_compile_definitions(${_dest} INTERFACE "${_conf}")
  endforeach()
endfunction()

#:
#: .. cmake:command:: einsums_add_compile_flag
#:
#:    Shorthand for :cmake:command:`einsums_add_target_compile_option` applied to global flags.
#:
#:    This function is a convenience wrapper around
#:    :cmake:command:`einsums_add_target_compile_option`, used to register a
#:    compiler flag globally on the Einsums interface targets (e.g., for
#:    `einsums_public_flags` or `einsums_private_flags`).
#:
#:    **Signature**
#:    ``einsums_add_compile_flag(<flag> [PUBLIC] [CONFIGURATIONS ...] [LANGUAGES ...])``
#:
#:    **Arguments**
#:
#:    - All arguments are forwarded directly to
#:      :cmake:command:`einsums_add_target_compile_option`.
#:
#:    This includes:
#:
#:    - ``<flag>``: The compiler flag (e.g., ``-Wall``, ``-march=native``)
#:    - ``PUBLIC``: Optional switch for interface visibility.
#:    - ``CONFIGURATIONS``: Optional list of build types (e.g., ``Debug``).
#:    - ``LANGUAGES``: Optional list of languages (default: ``CXX``).
#:
#:    **Example**
#:
#:    .. code-block:: cmake
#:
#:       einsums_add_compile_flag(-Wall PUBLIC CONFIGURATIONS Debug LANGUAGES CXX)
#:
#:    This behaves identically to:
#:
#:    .. code-block:: cmake
#:
#:       einsums_add_target_compile_option(
#:         -Wall
#:         PUBLIC
#:         CONFIGURATIONS Debug
#:         LANGUAGES CXX
#:       )
#:
#:    **See also**
#:
#:    - :cmake:command:`einsums_add_target_compile_option`
#:    - :cmake:command:`einsums_add_compile_definition`
function(einsums_add_compile_flag)
  einsums_add_target_compile_option(${ARGN})
endfunction()

#:
#: .. cmake:command:: einsums_add_compile_flag_if_available
#:
#:    Conditionally adds a compile flag if the compiler supports it.
#:
#:    This function wraps :cmake:command:`einsums_add_compile_flag`, performing a check
#:    to determine if the given compiler flag is supported by the current compiler.
#:    If supported, the flag is added with optional visibility, configuration, and language constraints.
#:
#:    **Signature**
#:    ``einsums_add_compile_flag_if_available(<flag> [NAME <name>] [PUBLIC] [CONFIGURATIONS ...] [LANGUAGES ...])``
#:
#:    **Arguments**
#:
#:    - ``<flag>`` *(required)*:
#:      The compile flag to test and potentially add (e.g., ``-march=native`` or ``-Werror``).
#:
#:    - ``NAME`` *(optional)*:
#:      Explicit name for the cache variable used to store the flag check result.
#:      If omitted, one is auto-generated from the flag string using sanitized uppercase.
#:
#:    - ``PUBLIC`` *(optional, boolean switch)*:
#:      If provided, the flag will be registered as a `PUBLIC` interface option.
#:      Otherwise, it defaults to `PRIVATE`.
#:
#:    - ``CONFIGURATIONS`` *(optional)*:
#:      A list of build configurations (e.g., ``Debug``, ``Release``) to restrict the flag's effect to.
#:
#:    - ``LANGUAGES`` *(optional)*:
#:      A list of programming languages to check and apply the flag for. Defaults to ``CXX``.
#:
#:    **Behavior**
#:
#:    For each specified language (currently only ``CXX`` is supported), the function:
#:
#:    1. Sanitizes a variable name from the flag or user-specified ``NAME``, replacing:
#:       - Leading `-` with nothing
#:       - `=`, `-`, `,` → `_`
#:       - `+` → `X`
#:
#:    2. Performs a compiler feature check using :cmake:command:`check_cxx_compiler_flag`.
#:
#:    3. If the flag is supported:
#:       - Calls :cmake:command:`einsums_add_compile_flag` with the parsed args.
#:
#:    4. If unsupported:
#:       - Logs an informational message using `einsums_info()`.
#:
#:    5. If an unsupported language is specified, the function errors via `einsums_error()`.
#:
#:    The result of the flag check is cached using the variable:
#:    ``EINSUMS_WITH_<LANG>_FLAG_<SANITIZED_NAME>``
#:
#:    **Example**
#:
#:    .. code-block:: cmake
#:
#:       einsums_add_compile_flag_if_available(
#:         -Wno-unused-parameter
#:         PUBLIC
#:         CONFIGURATIONS Debug
#:       )
#:
#:       einsums_add_compile_flag_if_available(
#:         -march=native
#:         NAME march_native
#:         LANGUAGES CXX
#:       )
#:
#:    **See also**
#:
#:    - :cmake:command:`einsums_add_compile_flag`
#:    - :cmake:command:`einsums_add_target_compile_option_if_available`
#:    - :cmake:command:`check_cxx_compiler_flag`
#:    - :cmake:command:`target_compile_options`
function(einsums_add_compile_flag_if_available FLAG)
  set(options PUBLIC)
  set(one_value_args NAME)
  set(multi_value_args CONFIGURATIONS LANGUAGES)
  cmake_parse_arguments(
    EINSUMS_ADD_COMPILE_FLAG_IA "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN}
  )

  set(_public)
  if(EINSUMS_ADD_COMPILE_FLAG_IA_PUBLIC)
    set(_public PUBLIC)
  endif()

  if(EINSUMS_ADD_COMPILE_FLAG_IA_NAME)
    string(TOUPPER ${EINSUMS_ADD_COMPILE_FLAG_IA_NAME} _name)
  else()
    string(TOUPPER ${FLAG} _name)
  endif()

  string(REGEX REPLACE "^-+" "" _name ${_name})
  string(REGEX REPLACE "[=\\-]" "_" _name ${_name})
  string(REGEX REPLACE "," "_" _name ${_name})
  string(REGEX REPLACE "\\+" "X" _name ${_name})

  set(_languages "CXX")
  if(EINSUMS_ADD_COMPILE_FLAG_IA_LANGUAGES)
    set(_languages ${EINSUMS_ADD_COMPILE_FLAG_IA_LANGUAGES})
  endif()

  foreach(_lang ${_languages})
    if(_lang STREQUAL "CXX")
      check_cxx_compiler_flag(${FLAG} EINSUMS_WITH_${_lang}_FLAG_${_name})
    else()
      einsums_error("Unsupported language ${_lang}.")
    endif()
    if(EINSUMS_WITH_${_lang}_FLAG_${_name})
      einsums_add_compile_flag(
        ${FLAG} CONFIGURATIONS ${EINSUMS_ADD_COMPILE_FLAG_IA_CONFIGURATIONS} LANGUAGES ${_lang}
        ${_public}
      )
    else()
      einsums_info("\"${FLAG}\" not available for language ${_lang}.")
    endif()
  endforeach()
endfunction()
