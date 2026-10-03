#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

#[=======================================================================[.rst:
waggle_instrument
-----------------

Link a target against Waggle and choose which of its zones it records.

.. code-block:: cmake

  waggle_instrument(<target> [DETAIL] [DISABLE])

``DETAIL``
  Also record ``WAGGLE_ZONE_DETAIL`` zones, the ones a library keeps for profiling its own
  internals.

``DISABLE``
  Expand every instrumentation macro in the target's sources to nothing. The target still shares
  the process's one profiler, so the zones of other libraries are unaffected.

Both are per target and apply to the target's own sources; a library whose public headers use the
macros sets ``WAGGLE_DETAIL`` or ``WAGGLE_DISABLE`` from a header of its own instead, so that its
users' translation units agree with it.
#]=======================================================================]
function(waggle_instrument target)
  cmake_parse_arguments(_waggle "DETAIL;DISABLE" "" "" ${ARGN})
  if(_waggle_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR "waggle_instrument: unknown arguments: ${_waggle_UNPARSED_ARGUMENTS}")
  endif()
  target_link_libraries(${target} PUBLIC Waggle::waggle)
  if(_waggle_DETAIL)
    target_compile_definitions(${target} PRIVATE WAGGLE_DETAIL)
  endif()
  if(_waggle_DISABLE)
    target_compile_definitions(${target} PRIVATE WAGGLE_DISABLE)
  endif()
endfunction()
