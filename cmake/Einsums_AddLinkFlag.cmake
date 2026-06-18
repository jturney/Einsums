#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

include(CMakeParseArguments)
include(CheckCXXCompilerFlag)

#:
#: einsums_add_link_flag
#:
#:    Register a linker flag on Einsums’ global interface targets, with optional config/target gating.
#:
#:    Adds a link option to either ``einsums_public_flags`` (when ``PUBLIC`` is set) or
#:    ``einsums_private_flags`` (default) using ``target_link_options``. The flag can be
#:    conditionally applied via generator expressions based on **build configuration** and/or the
#:    consuming target’s **TYPE** (e.g., ``EXECUTABLE``, ``SHARED_LIBRARY``).
#:
#:    **Signature**
#:    ``einsums_add_link_flag(<flag> [PUBLIC] [TARGETS <types...>] [CONFIGURATIONS <configs...>])``
#:
#:    **Arguments**
#:
#:    - ``<flag>`` *(required)*:
#:      The raw linker flag string, e.g. ``-Wl,--as-needed`` or ``/INCREMENTAL:NO``.
#:
#:    - ``PUBLIC`` *(optional switch)*:
#:      Attach to the ``einsums_public_flags`` interface target. If omitted, attaches to
#:      ``einsums_private_flags``.
#:
#:    - ``TARGETS <types...>`` *(optional, multi-value)*:
#:      Restrict application to consuming targets whose ``TYPE`` matches any of the
#:      given values (e.g., ``EXECUTABLE``, ``SHARED_LIBRARY``, ``STATIC_LIBRARY``, ``MODULE_LIBRARY``).
#:
#:    - ``CONFIGURATIONS <configs...>`` *(optional, multi-value)*:
#:      Restrict application to specific build configs (e.g., ``Debug``, ``Release``).
#:
#:    **Behavior**
#:    - Builds a generator expression for each combination of (config, type):
#:      - both given → ``$<$<AND:$<CONFIG:cfg>,$<STREQUAL:$<TARGET_PROPERTY:TYPE>,type>>:flag>``
#:      - only type → ``$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,type>:flag>``
#:      - only config → ``$<$<CONFIG:cfg>:flag>``
#:      - neither → ``flag`` (always on)
#:    - Appends each expression to the chosen interface target with
#:      ``target_link_options(<iface> INTERFACE <expr>)``.
#:
#:    **Examples**
#:    Apply a flag to all executables in Release:
#:    .. code-block:: cmake
#:
#:       einsums_add_link_flag(-Wl,--gc-sections TARGETS EXECUTABLE CONFIGURATIONS Release)
#:
#:    Publicly expose an MSVC linker setting to all consumers:
#:    .. code-block:: cmake
#:
#:       einsums_add_link_flag(/INCREMENTAL:NO PUBLIC)
#:
#:    **See also**
#:    - ``target_link_options``
#:    - ``TYPE``
macro(einsums_add_link_flag FLAG)
  set(options PUBLIC)
  set(one_value_args)
  set(multi_value_args TARGETS CONFIGURATIONS)
  cmake_parse_arguments(
    EINSUMS_ADD_LINK_FLAG "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN}
  )

  if(EINSUMS_ADD_LINK_FLAG_PUBLIC)
    set(_dest einsums_public_flags)
  else()
    set(_dest einsums_private_flags)
  endif()

  set(_targets "none")
  if(EINSUMS_ADD_LINK_FLAG_TARGETS)
    set(_targets ${EINSUMS_ADD_LINK_FLAG_TARGETS})
  endif()

  set(_configurations "none")
  if(EINSUMS_ADD_LINK_FLAG_CONFIGURATIONS)
    set(_configurations "${EINSUMS_ADD_LINK_FLAG_CONFIGURATIONS}")
  endif()

  foreach(_config ${_configurations})
    foreach(_target ${_targets})
      if(NOT _config STREQUAL "none" AND NOT _target STREQUAL "none")
        set(_flag
            "$<$<AND:$<CONFIG:${_config}>,$<STREQUAL:$<TARGET_PROPERTY:TYPE>,${_target}>:${FLAG}>"
        )
      elseif(_config STREQUAL "none" AND NOT _target STREQUAL "none")
        set(_flag "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,${_target}>:${FLAG}>")
      elseif(NOT _config STREQUAL "none" AND _target STREQUAL "none")
        set(_flag "$<$<CONFIG:${_config}>:${FLAG}>")
      else()
        set(_flag "${FLAG}")
      endif()
      target_link_options(${_dest} INTERFACE "${_flag}")
    endforeach()
  endforeach()
endmacro()

#:
#: einsums_add_link_flag_if_available
#:
#:    Conditionally register a linker flag if the toolchain accepts it.
#:
#:    Checks support for a given linker flag and, if available, forwards to
#:    ``einsums_add_link_flag``. This avoids injecting unsupported options that would
#:    break the build on certain platforms or toolchains.
#:
#:    **Signature**
#:    ``einsums_add_link_flag_if_available(<flag> [NAME <symbol>] [PUBLIC] [TARGETS <types...>])``
#:
#:    **Arguments**
#:    - ``<flag>`` *(required)*:
#:      The linker flag to probe, e.g. ``-Wl,--as-needed`` or ``-Wl,-z,now``.
#:
#:    - ``NAME <symbol>`` *(optional)*:
#:      Override the auto‑generated cache symbol used for the probe. If omitted, a sanitized
#:      name is derived from the flag (spaces removed; leading ``-`` dropped; ``=``, ``-``, ``,`` → ``_``;
#:      ``+`` → ``X``; upper‑cased).
#:
#:    - ``PUBLIC`` *(optional switch)*:
#:      If set, attach the flag to ``einsums_public_flags``; otherwise to ``einsums_private_flags``.
#:
#:    - ``TARGETS <types...>`` *(optional)*:
#:      Restrict application to consuming target types (see ``TYPE``).
#:
#:    **Behavior**
#:    - Uses ``check_cxx_compiler_flag`` to attempt a compile‑time probe with the given
#:      flag (stored in a cache variable named ``WITH_LINKER_FLAG_<NAME>``).
#:    - On success, calls ``einsums_add_link_flag`` with the original arguments
#:      (propagating ``PUBLIC`` and ``TARGETS``).
#:    - On failure, prints an informational message (``Linker "<flag>" not available.``) and makes no changes.
#:
#:    **Examples**
#:    Prefer but don’t require section‑gc on ELF linkers:
#:    .. code-block:: cmake
#:
#:       einsums_add_link_flag_if_available(-Wl,--gc-sections TARGETS EXECUTABLE SHARED_LIBRARY)
#:
#:    Publicly apply ``-Wl,-z,now`` if supported, with a stable probe name:
#:    .. code-block:: cmake
#:
#:       einsums_add_link_flag_if_available(-Wl,-z,now NAME z_now PUBLIC)
#:
#:    **Notes**
#:    - The probe relies on the C++ compiler driver to accept/forward the flag to the linker.
#:      Some link‑only flags may require different detection logic on certain toolchains.
#:
#:    **See also**
#:    - ``einsums_add_link_flag``
#:    - ``check_cxx_compiler_flag``
#:    - ``target_link_options``
macro(einsums_add_link_flag_if_available FLAG)
  set(options PUBLIC)
  set(one_value_args NAME)
  set(multi_value_args TARGETS)
  cmake_parse_arguments(
    EINSUMS_ADD_LINK_FLAG_IA "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN}
  )

  set(_public)
  if(EINSUMS_ADD_LINK_FLAG_IA_PUBLIC)
    set(_public PUBLIC)
  endif()

  if(EINSUMS_ADD_LINK_FLAG_IA_NAME)
    string(TOUPPER ${EINSUMS_ADD_LINK_FLAG_IA_NAME} _name)
  else()
    string(TOUPPER ${FLAG} _name)
  endif()

  string(REGEX REPLACE " " "" _name ${_name})
  string(REGEX REPLACE "^-+" "" _name ${_name})
  string(REGEX REPLACE "[=\\-]" "_" _name ${_name})
  string(REGEX REPLACE "," "_" _name ${_name})
  string(REGEX REPLACE "\\+" "X" _name ${_name})

  check_cxx_compiler_flag("${FLAG}" WITH_LINKER_FLAG_${_name})
  if(WITH_LINKER_FLAG_${_name})
    einsums_add_link_flag(${FLAG} TARGETS ${EINSUMS_ADD_LINK_FLAG_IA_TARGETS} ${_public})
  else()
    einsums_info("Linker \"${FLAG}\" not available.")
  endif()

endmacro()
