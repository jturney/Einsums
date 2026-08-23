..
    Copyright (c) The Einsums Developers. All rights reserved.
    Licensed under the MIT License. See LICENSE.txt in the project root for license information.

.. _modules_Einsums_Errors:

======
Errors
======

This module contains symbols used for error handling. This includes exception classes, and macros for throwing
exceptions with location information. 

See the :ref:`API reference <modules_Einsums_Errors_api>` of this module for more
details.

Error Reference
---------------

If all goes well, Einsums should never throw an error. Of course, this is wishful thinking. A reference as to
what each kind of error means is provided for users who wish to debug their code, or at least be able to handle
the errors as they come up. For the most part, the kinds of errors thrown by a function should be documented, though
this is still a work in progress.

- :cpp:class:`~einsums::CodedError` - wraps another exception class with an integer error code so that several errors of the same class can be told apart.
- :cpp:class:`~einsums::DimensionError` - the dimensions of some tensor arguments are not compatible with the given operation.
- :cpp:class:`~einsums::TensorCompatError` - two or more tensors are not compatible with each other for the requested operation.
- :cpp:class:`~einsums::NumArgumentError` - a function taking a variable number of arguments did not receive the right number of arguments.
- :cpp:class:`~einsums::NotEnoughArgs` - a function taking a variable number of arguments did not receive enough arguments.
- :cpp:class:`~einsums::TooManyArgs` - a function taking a variable number of arguments received too many arguments.
- :cpp:class:`~einsums::AccessDenied` - an operation was stopped due to access restrictions, such as writing read-only data.
- :cpp:class:`~einsums::TodoError` - the requested code path is not yet finished.
- :cpp:class:`~einsums::NotImplemented` - the requested action is not implemented.
- :cpp:class:`~einsums::BadLogic` - equivalent to :code:`std::logic_error`, but catchable without matching every exception derived from it.
- :cpp:class:`~einsums::UninitializedError` - the code is handling uninitialized data, usually because Einsums was not initialized.
- :cpp:class:`~einsums::SystemError` - a system call or system utility failed.
- :cpp:class:`~einsums::EnumError` - an invalid enum value was passed to a function.
- :cpp:class:`~einsums::RankError` - a tensor argument had an invalid or incompatible rank.
- :cpp:class:`~einsums::ComplexConversionError` - an attempt was made to convert a complex number to a real number.