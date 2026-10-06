..
    Copyright (c) The Einsums Developers. All rights reserved.
    Licensed under the MIT License. See LICENSE.txt in the project root for license information.

.. _modules_Einsums_Print:

=====
Print
=====

This module contains :code:`einsums::println` and :code:`einsums::fprintln`, thin wrappers around fmt that print a
formatted line, as well as a few special symbols for other tasks. The overloads that print tensors live with the tensor
types, in the TensorImpl and Tensor modules.

See the :ref:`API reference <modules_Einsums_Print_api>` of this module for more
details.

--------------
Public Symbols
--------------

There are a few symbols that may be useful to users.

- :cpp:class:`~einsums::print::ordinal` - a wrapper for a value that allows formatting an integer as an ordinal, such as 1st, 2nd, etc.