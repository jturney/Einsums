..
    Copyright (c) The Einsums Developers. All rights reserved.
    Licensed under the MIT License. See LICENSE.txt in the project root for license information.

.. _modules_Einsums_Assertion:

=========
Assertion
=========

This module contains several macros for handling assertions.

See the :ref:`API reference <modules_Einsums_Assertion_api>` of this module for more
details.

Here is a list of public utilities defined in this module. See the
:ref:`API reference <modules_Einsums_Assertion_api>` for full signatures and details.

- :c:macro:`EINSUMS_ASSERT` asserts an expression in debug builds, with no custom message.
- :c:macro:`EINSUMS_ASSERT_MSG` asserts an expression with a custom failure message.

-------------
Example Usage
-------------

As an example, suppose we want to make sure that the result of an operation is correct. We can use the assert macros to
make sure that it is correct.

.. code:: C++

    Tensor<1, double> vec = create_random_tensor("vector", 10);
    double out = dot(vec, vec);

    // The dot product of a vector with itself can not be negative.
    // If it is, we are in an invalid state, so we should probably exit.
    EINSUMS_ASSERT_MSG(out >= 0, "The dot product can not be negative! What happened!?");

The :c:macro:`EINSUMS_ASSERT` macro is similar, but it doesn't take any debug info. 

.. code:: C++

    // No debug info this time.
    EINSUMS_ASSERT(out >= 0);
