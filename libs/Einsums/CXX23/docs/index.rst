..
    ----------------------------------------------------------------------------------------------
     Copyright (c) The Einsums Developers. All rights reserved.
     Licensed under the MIT License. See LICENSE.txt in the project root for license information.
    ----------------------------------------------------------------------------------------------

.. _modules_Einsums_CXX23:

#####
CXX23
#####

The ``CXX23`` module is a small collection of C++23 polyfills for compilers
or standard library versions where a feature isn't yet available.

Einsums builds as C++20 by default (``EINSUMS_WITH_CXX_STANDARD``), and
even a C++23 build can meet a standard library that lacks a feature. Examples
include older libstdc++, Apple libc++ on certain SDK levels, and Intel oneAPI.
``CXX23`` provides drop-in replacements behind feature-test-macro guards so
the rest of the codebase can use the C++23 spelling unconditionally.

What's polyfilled
=================

The headers in this module follow a uniform pattern: include the standard
header if the feature-test macro indicates the feature is present;
otherwise expose a compatible implementation in the ``einsums`` namespace.

Currently provided:

- ``Einsums/CXX23/Expected.hpp`` provides ``einsums::expected<T, E>`` and
  ``einsums::unexpected<E>``, guarded on ``__cpp_lib_expected``. The header
  includes ``<version>`` itself, so every translation unit makes the same
  choice whatever it included first.

Usage
=====

.. code-block:: cpp

    #include <Einsums/CXX23/Expected.hpp>

    using einsums::expected;
    using einsums::unexpected;

    expected<int, std::string> parse(std::string_view s) {
        if (s.empty()) {
            return unexpected(std::string("empty input"));
        }
        return std::stoi(std::string{s});
    }

When Einsums is built as C++23 against a library that has
``std::expected``, ``einsums::expected`` becomes an alias for it and existing
call sites need no change.

See the :ref:`API reference <modules_Einsums_CXX23_api>` of this module for
the polyfill surface.
