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

Run any Einsums program with ``--help`` to get the same list from the program itself, including the default
in effect for your build and the environment variable each option reads.

=================
How Options Reach
=================

Every option below resolves through four sources. Each is free to overwrite the one before it:

.. code-block:: text

   1. the default compiled into the option
   2. a config file
   3. an environment variable
   4. the command line

Environment variables
=====================

Every option is readable from the environment under a name derived from its own: upper-case the long name
and turn each run of separators into a single underscore. So :option:`--einsums:log:level` reads
``EINSUMS_LOG_LEVEL`` and :option:`--einsums:profile:wait-for-viewer` reads
``EINSUMS_PROFILE_WAIT_FOR_VIEWER``.

This is what lets an inherited variable configure a job without editing its command line, while the command
line still wins when it says so. An unset **or empty** variable counts as not specified, so ``FOO=`` leaves
the default in place.

.. versionadded:: 2.0.0

Flags and their negations
=========================

Every flag is declared once, in the positive, and Einsums generates the ``no-`` spelling for it. So
:option:`--einsums:profile:report` is turned off with ``--einsums:profile:no-report``, and a flag that is
off by default is turned on by naming it.

``--help`` lists the positive spelling and names the negation beside it rather than listing both.

.. versionadded:: 2.0.0
    Flags used to be declared in whichever polarity was not the default, which is why several of the
    options below changed name in this release. The old ``no-`` spellings all still parse.

===============
Basic Arguments
===============

.. option:: --einsums:buffer-size <size>

    The amount of memory Einsums is allowed to use. It takes a string containing a number and units.
    The units are either bytes or words. See :cpp:func:`einsums::string_util::memory_string` for more information.

    Defaults to ``4MB``.

    .. versionadded:: 1.0.0

.. option:: --einsums:work-buffer-size <size>

    The largest buffer size to use for buffered contractions. This should be much smaller than
    :option:`--einsums:buffer-size`, whose value has to cover roughly three buffers per thread along with
    everything else the program needs. Setting it to zero lets Einsums decide.

    .. versionadded:: 1.0.0

.. option:: --einsums:row-major

    Construct tensors in row-major order rather than column-major.

    Off by default: Einsums tensors are column-major at construction, which is what the BLAS and LAPACK
    backends expect.

    .. versionadded:: 2.0.0
        The setting was read from the tensor constructors in earlier releases but no option ever set it.

=======
Logging
=======

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

.. option:: --einsums:log:format <format>

    A format string used for the logger output.

    .. versionadded:: 1.0.0
    .. versionchanged:: 2.0.0
        This option's name has been changed.

=========
Profiling
=========

.. option:: --einsums:profile:disable

    Do not record profiling zones or annotations at all. This is a large speedup for workloads made of
    many small operations, where the bookkeeping can cost more than the work being measured.

    .. versionadded:: 2.0.0

.. option:: --einsums:profile:report

    Write the profiling report when the program exits. On by default; turn it off with
    ``--einsums:profile:no-report``.

    .. versionadded:: 1.0.0
    .. versionchanged:: 2.0.0
        This option's name has been changed, and again later in the 2.0.0 cycle to its positive spelling.

.. option:: --einsums:profile:filename <filename>

    The name of the file for the profiler output. Defaults to ``profile.txt``.

    .. versionadded:: 1.0.0
    .. versionchanged:: 2.0.0
        This option's name has been changed.

.. option:: --einsums:profile:append

    Append the profiling information to the profiling file rather than truncating it. On by default;
    turn it off with ``--einsums:profile:no-append``.

    .. versionadded:: 1.0.0
    .. versionchanged:: 2.0.0
        This option's name has been changed to its positive spelling. It was previously documented as
        ``--einsums:profiler:no-append``, which never parsed.

.. option:: --einsums:profile:detailed

    Print a detailed profile report rather than the summary.

    .. versionadded:: 2.0.0

.. option:: --einsums:profile:save <filename>

    Save the profile session as JSON for the imgui viewer. Empty, the default, means do not save one.

    .. versionadded:: 2.0.0

.. option:: --einsums:profile:port <PORT>

    The port the profile server listens on. Defaults to ``19216``.

    .. versionadded:: 2.0.0

.. option:: --einsums:profile:wait-for-viewer

    Block at startup until a profiler viewer connects, so that a short run cannot finish before the
    viewer has attached.

    .. versionadded:: 2.0.0

=========================
ComputeGraph Pass Options
=========================

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

.. option:: --einsums:graph:profile-groups

    Break a grouped batched GEMM into one profiler zone per shape class.

    This deliberately changes how the contraction runs: it takes the slower unfused form, because a
    per-group breakdown of one parallel loop is not obtainable any other way. Read it for where the
    arithmetic is, not for what the node costs.

    .. versionadded:: 2.0.0

.. option:: --einsums:graph:verify-levels

    Before each level-scheduled replay, check that no execution level holds two nodes touching
    overlapping storage.

    On by default in debug builds. This option is how a release build turns it on, which is what a
    nondeterministic result wants. It costs a second pass over every operand and can report a conflict
    it cannot disprove.

    .. versionadded:: 2.0.0

===
GPU
===

.. option:: --einsums:gpu:disable

    Keep every node on the host, making the GPU placement pass a no-op.

    .. versionadded:: 2.0.0

====
HPTT
====

.. option:: --einsums:hptt:selection-method <METHOD>

    The plan selection method used by the tensor transpose backend: one of ``estimate``, ``measure``,
    ``patient``, or ``crazy``. Defaults to ``estimate``. The later methods spend more time choosing a
    plan in exchange for a faster transpose, which pays off when one plan is reused many times.

    .. versionadded:: 2.0.0

==============
Tensor Storage
==============

.. option:: --einsums:scratch-dir <DIR>

    The scratch directory for Einsums tensor files. Defaults to the system temporary directory.

    .. versionadded:: 1.0.0

.. option:: --einsums:hdf5-file-name <filename>

    The name of the HDF5 scratch file. The default carries the process id, so concurrent runs do not
    collide.

    .. versionadded:: 1.0.0

.. option:: --einsums:delete-hdf5-files

    Clean up the HDF5 scratch file on exit. On by default; keep the file for inspection with
    ``--einsums:no-delete-hdf5-files``.

    .. versionadded:: 1.0.0
    .. versionchanged:: 2.0.0
        This option's name has been changed to its positive spelling.

==================
Advanced Arguments
==================

.. option:: --einsums:debug:install-signal-handlers

    Install Einsums' custom signal handlers. On by default; turn them off with
    ``--einsums:debug:no-install-signal-handlers``, which is what a host process that installs its own
    handlers wants.

    .. versionadded:: 1.0.0
    .. versionchanged:: 2.0.0
        This option's name has been changed, and again later in the 2.0.0 cycle to its positive spelling.

.. option:: --einsums:debug:attach-debugger

    Offer the chance to attach a debugger when an error is detected. On by default; turn it off with
    ``--einsums:debug:no-attach-debugger``, which is what a test harness wants so that a failing test
    cannot hang waiting for a human.

    .. versionadded:: 1.0.0
    .. versionchanged:: 2.0.0
        This option's name has been changed, and again later in the 2.0.0 cycle to its positive spelling.

.. option:: --einsums:debug:diagnostics-on-terminate

    Print extra diagnostics on termination. On by default; turn it off with
    ``--einsums:debug:no-diagnostics-on-terminate``.

    .. versionadded:: 1.0.0
    .. versionchanged:: 2.0.0
        This option's name has been changed, and again later in the 2.0.0 cycle to its positive spelling.
