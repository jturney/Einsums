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

There are some useful symbols that this module provides. They are the logging symbols.

.. c:macro:: EINSUMS_LOG_TRACE(...)

    Logs a trace message. These are for intensive debugging. They are disabled at compile time by configuring in the Release
    configuration.

    :param ...: The format string and arguments to use for the message.

    .. versionadded:: 1.0.0

.. c:macro:: EINSUMS_LOG_DEBUG(...)

    Logs a debug message. They are disabled at compile time by configuring in the Release
    configuration.

    :param ...: The format string and arguments to use for the message.

    .. versionadded:: 1.0.0

.. c:macro:: EINSUMS_LOG_INFO(...)

    Logs an informational message. These messages give information about the program, such as environment and configuration variables.

    :param ...: The format string and arguments to use for the message.

    .. versionadded:: 1.0.0

.. c:macro:: EINSUMS_LOG_WARN(...)

    Logs a warning message. These messages indicate that a recoverable issue occurred.

    :param ...: The format string and arguments to use for the message.

    .. versionadded:: 1.0.0

.. c:macro:: EINSUMS_LOG_ERROR(...)

    Logs an error message. These messages indicate that an issue occurred that left the program in an unstable state. These are often accompanied by
    an exception.

    :param ...: The format string and arguments to use for the message.

    .. versionadded:: 1.0.0

.. c:macro:: EINSUMS_LOG_CRITICAL(...)

    Logs a critical error message. These messages indicate that an unrecoverable issue occurred. Usually, the program will abort after logging this.

    :param ...: The format string and arguments to use for the message.

    .. versionadded:: 1.0.0