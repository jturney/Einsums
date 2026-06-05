..
    Copyright (c) The Einsums Developers. All rights reserved.
    Licensed under the MIT License. See LICENSE.txt in the project root for license information.

.. _modules_Einsums_BufferAllocator:

=======================
BufferAllocator
=======================

This module contains an allocator that keeps track of how much memory has been allocated for buffers.
The allocator is controlled by global configuration options.

See the :ref:`API reference <modules_Einsums_BufferAllocator_api>` of this module for more
details.

.. cpp:namespace:: einsums

Interacting with the Allocator
------------------------------

To modify the allocator's maximum size, set the string option `"buffer-size"` in the :cpp:class:`GlobalConfigMap` to
a memory string. A memory string is a number, possibly a decimal, followed by an optional prefix and unit.
The understood prefixes are `k`, `M`, `G`, and `T`, all case insensitive. The units can either be `B`, `W`, or `o`,
also case insensitive. The `B` and `o` units represent bytes, with the `o` provided for people who use the 
octet convention. The `W` unit represents words and are the size of `size_t` on the user's system. On 64-bit
systems, one word is generally eight bytes. The setting may be set in the program using the global option mentioned
before or may be set on the command line using the `--einsums:buffer-size` option. The total amount of memory
is shared among all processes, not on a per-process basis.

The :cpp:class:`BufferAllocator` is a normal allocator. As such, it can be used in C++ containers and can be interacted
with in the same way as other C++ allocators. Users must be careful when resizing allocations, though, as in order to
resize an allocation, the full size of the final allocation will be requested before the original amount is freed, which
can cause an out of memory exception. There are a few extra methods provided as well, though these are considered to be
best for internal use.