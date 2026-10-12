..
    Copyright (c) The Einsums Developers. All rights reserved.
    Licensed under the MIT License. See LICENSE.txt in the project root for license information.

.. _modules_Einsums_Json:

============
Einsums Json
============

A strict JSON document model, used for every file Einsums writes and reads back: the graph IR and
the hardware profile.

A reader that silently skips unknown fields loads a half-understood file as if it were fully
understood, and the dropped part surfaces later as a wrong number. So every object records which
keys a reader consumed, and a loader refuses a document with an unconsumed key, naming the key and
its path. Integers are a distinct type from doubles, object keys keep their insertion order, and
doubles round-trip bit-exactly.

The module depends only on ``Config``, ``CXX23`` and ``Errors``, so it can be read from low in the
module stack.

See the :ref:`API reference <modules_Einsums_Json_api>` of this module for more
details.
