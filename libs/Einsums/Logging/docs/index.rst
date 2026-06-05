..
    Copyright (c) The Einsums Developers. All rights reserved.
    Licensed under the MIT License. See LICENSE.txt in the project root for license information.

.. _modules_Einsums_Logging:

===============
Einsums Logging
===============

This module contains several macros for setting up logs, as well as providing log levels. It is considered
an internal module. No symbols would be considered useful to users.

See the :ref:`API reference <modules_Einsums_Logging_api>` of this module for more
details.

Public Symbols
--------------

There are some useful symbols that this module provides — the logging macros. They are
documented in full in the :ref:`API reference <modules_Einsums_Logging_api>`:

- :c:macro:`EINSUMS_LOG_TRACE` — intensive-debugging traces (compile-time disablable in Release).
- :c:macro:`EINSUMS_LOG_DEBUG` — maintainer debugging messages (compile-time disablable in Release).
- :c:macro:`EINSUMS_LOG_INFO` — informational messages (environment, configuration, …).
- :c:macro:`EINSUMS_LOG_WARN` — a recoverable issue occurred.
- :c:macro:`EINSUMS_LOG_ERROR` — an issue left the program in an unstable state.
- :c:macro:`EINSUMS_LOG_CRITICAL` — an unrecoverable issue; the program usually aborts.