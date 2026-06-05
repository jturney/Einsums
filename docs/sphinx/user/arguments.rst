..
    ----------------------------------------------------------------------------------------------
     Copyright (c) The Einsums Developers. All rights reserved.
     Licensed under the MIT License. See LICENSE.txt in the project root for license information.
    ----------------------------------------------------------------------------------------------

.. _arguments:

######################
Command Line Arguments
######################

There are several command line arguments that can control the behavior of Einsums. This is the documentation
for these arguments.

===============
Basic Arguments
===============

.. option:: --einsums:log:level <level>

    Set the level to see in the logger. Lower values provide more information. By default, it is set to 
    3 for the release build and 2 for the debug build.

    * 0: Tracing messages. Very verbose.
    * 1: Debugging messages.
    * 2: Information messages
    * 3: Warnings.
    * 4: Errors.
    * 5: Critical errors.

    .. versionadded:: 1.0.0
    .. versionchanged:: 1.1.0
        This option now also sets the HIP log level if Einsums was built with GPU support.
    .. versionchanged:: 2.0.0
        This option no longer sets the HIP log level. Use the :code:`AMD_LOG_LEVEL` environment variable.
        This option's name has also been changed.

.. option:: --einsums:log:destination [cerr | cout]

    Set whether the logger will log to standard output or standard error.

    .. versionadded:: 1.0.0
    .. versionchanged:: 2.0.0
        This option's name has been changed.

.. option:: --einsums:profile:no-report

    Tells Einsums not to output the profiling information.

    .. versionadded:: 1.0.0
    .. versionchanged:: 2.0.0
        This option's name has been changed.

.. option:: --einsums:profile:filename

    The name of the file for the profiler output.

    .. versionadded:: 1.0.0
    .. versionchanged:: 2.0.0
        This option's name has been changed.

.. option:: --einsums:buffer-size

    The amount of memory Einsums is allowed to use. It takes a string containing a number and units.
    The units are either bytes or words. See :cpp:func:`einsums::string_util::memory_string` for more information.

    .. versionadded:: 1.0.0

.. option:: --einsums:gpu-buffer-size

    The amount of memory Einsums is allowed to use on the GPU. It takes a string containing a number and units.
    The units are either bytes or words. See :cpp:func:`einsums::string_util::memory_string` for more information.

    .. versionadded:: 1.0.0


==================
Advanced Arguments
==================

.. option:: --einsums:debug:no-install-signal-handlers

    Tells Einsums not to install its custom signal handlers.

    .. versionadded:: 1.0.0
    .. versionchanged:: 2.0.0
        This option's name has been changed.

.. option:: --einsums:debug:no-attach-debugger

    Tells Einsums not to allow users the ability to attach a debugger when an error is detected.

    .. versionadded:: 1.0.0
    .. versionchanged:: 2.0.0
        This option's name has been changed.

.. option:: --einsums:debug:no-diagnostics-on-terminate

    When present, Einsums won't print extra diagnostics on termination.

    .. versionadded:: 1.0.0
    .. versionchanged:: 2.0.0
        This option's name has been changed.

.. option:: --einsums:log:format

    A format string used for the logger output.

    .. versionadded:: 1.0.0
    .. versionchanged:: 2.0.0
        This option's name has been changed.

.. option:: --einsums:profiler:no-append

    If present, the profiling information will be appended to the profiling file. Otherwise, the profiling
    file will be overwritten.

    .. versionadded:: 1.0.0
    .. versionchanged:: 2.0.0
        This option's name has been changed.

==========================
ComputeGraph Pass Options
==========================

.. option:: --einsums:pass:disable <PASSES>

    Comma-separated list of optimization pass names to skip when ``PassManager::run()``
    is called. Use this to A/B test whether a specific pass helps or hurts performance.

    Example: ``--einsums:pass:disable CSE,Reorder``

    .. versionadded:: 2.0.0

.. option:: --einsums:pass:analyze

    Run all passes in analysis-only mode. Each pass executes normally and reports what
    it found, but the graph is restored to its original state afterward. No modifications
    persist.

    .. versionadded:: 2.0.0

.. option:: --einsums:pass:verbose

    Log node count and wall-clock time before and after each optimization pass, even for
    passes that don't modify the graph.

    .. versionadded:: 2.0.0