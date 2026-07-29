..
    Copyright (c) The Einsums Developers. All rights reserved.
    Licensed under the MIT License. See LICENSE.txt in the project root for license information.

.. _modules_Einsums_Profile:

=========
Profiling
=========

This module contains symbols for profiling Einsums.

See the :ref:`API reference <modules_Einsums_Profile_api>` of this module for more
details.

--------------
Public Symbols
--------------

- :cpp:func:`~einsums::profile::print_report` - print the compact (or detailed) profiling report to standard output.
- :cpp:func:`~einsums::profile::export_json` - write the aggregated profile to a JSON file.
- :cpp:func:`~einsums::profile::flush` - drain the per-thread ring buffers so that recent events are visible in reports.

-----------------------------
Building without the profiler
-----------------------------

Configuring with ``-DEINSUMS_WITH_PROFILER=OFF`` compiles the recording machinery out of the library.
Instrumented code does not have to change: ``LabeledSection`` and its siblings expand to nothing, and the
functions above remain callable and do nothing, so a translation unit or a Python script that profiles still
builds and still runs.
Ask :cpp:func:`~einsums::profile::available` which kind of build you are on; the readers all report empty
rather than fail.

What does not survive is the profiler's machinery.
The ring buffers, the aggregating consumer and the TCP server that the viewer connects to are absent
entirely, so the few call sites that reach for one of those must guard themselves with
``#if defined(EINSUMS_HAVE_PROFILER)``.
