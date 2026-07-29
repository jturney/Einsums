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
