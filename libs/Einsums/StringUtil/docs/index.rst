..
    Copyright (c) The Einsums Developers. All rights reserved.
    Licensed under the MIT License. See LICENSE.txt in the project root for license information.

.. _modules_Einsums_StringUtil:

==========
StringUtil
==========

This module contains functions for modifying strings.

See the :ref:`API reference <modules_Einsums_StringUtil_api>` of this module for more
details.

Public API
----------

Some functions may be useful to users.

- :cpp:func:`~einsums::string_util::ltrim` - trims whitespace from the beginning of the string in place.
- :cpp:func:`~einsums::string_util::rtrim` - trims whitespace from the end of the string in place.
- :cpp:func:`~einsums::string_util::trim` - trims whitespace from the beginning and end of the string in place.
- :cpp:func:`~einsums::string_util::ltrim_copy` - like ``ltrim``, but modifies and returns a copy of the input string.
- :cpp:func:`~einsums::string_util::rtrim_copy` - like ``rtrim``, but modifies and returns a copy of the input string.
- :cpp:func:`~einsums::string_util::trim_copy` - like ``trim``, but modifies and returns a copy of the input string.
