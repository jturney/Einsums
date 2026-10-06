..
    Copyright (c) The Einsums Developers. All rights reserved.
    Licensed under the MIT License. See LICENSE.txt in the project root for license information.

.. _modules_Einsums_Profile:

=========
Profiling
=========

Einsums profiles itself with `Waggle <https://github.com/Einsums/Waggle>`_, the profiler it shares with the other libraries in a process, so the zones Einsums opens and the zones your own code opens land in one tree.
This module is Einsums' side of that: the header that includes Waggle with Einsums' configuration, the ``--einsums:profile:*`` options, and ``einsums.profile`` in Python.

See the :ref:`API reference <modules_Einsums_Profile_api>` of this module for more details.

--------------
From C++
--------------

Include ``<Einsums/Profile/Profile.hpp>`` rather than Waggle's header, so that Einsums' configuration reaches the zone macros.
Then ``WAGGLE_ZONE``, ``WAGGLE_ANNOTATE`` and the rest of Waggle's C++ API are available, and the zones written inside ``namespace einsums`` belong to the ``einsums`` domain.
``waggle::flush()``, ``waggle::print_report()`` and ``waggle::export_json()`` read what has been recorded.

--------------
From Python
--------------

``einsums.profile`` is Waggle's Python API under the names Einsums has always used:
``section`` (a zone around a ``with`` block), the ``profile`` decorator, ``annotate``, ``flush``, ``print_report`` and ``export_json``.
For a zone entered often, ``waggle.Zone`` registers its site once and is the cheaper choice.

-----------------------------
Building without the profiler
-----------------------------

Configuring with ``-DEINSUMS_WITH_PROFILER=OFF`` takes Einsums' own zones out of the library.
Instrumented code does not have to change: ``WAGGLE_ZONE`` and its siblings expand to nothing in every translation unit that includes Einsums' headers, your own included, and ``einsums.profile`` stays callable and records nothing, so a script that profiles still runs.
Ask ``einsums.profile.available()`` which kind of build you are on; the readers all report empty rather than fail.

Waggle itself is still built and linked, since other libraries in the process may use it.
