..
    Copyright (c) The Einsums Developers. All rights reserved.
    Licensed under the MIT License. See LICENSE.txt in the project root for license information.

.. _modules_Einsums_Iterator:

========
Iterator
========

This module defines symbols for iterating over things at compile time.

See the :ref:`API reference <modules_Einsums_Iterator_api>` of this module for more
details.

Public API
----------

A few of the symbols may be useful for those using Einsums.

- :cpp:func:`~einsums::for_sequence` - a compile-time for loop that calls a functor for each element of a sequence.