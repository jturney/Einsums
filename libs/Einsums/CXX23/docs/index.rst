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

Einsums targets the C++23 baseline, but some toolchains (older libstdc++,
Apple libc++ on certain SDK levels, intel oneAPI) lag on individual
library features. ``CXX23`` provides drop-in replacements behind
feature-test-macro guards so the rest of the codebase can use the C++23
spelling unconditionally.

What's polyfilled
=================

The headers in this module follow a uniform pattern: include the standard
header if the feature-test macro indicates the feature is present;
otherwise expose a compatible implementation in the ``einsums::cxx23``
namespace.

Currently provided:

- ``Einsums/CXX23/Expected.hpp`` — ``std::expected<T, E>`` polyfill,
  guarded on ``__cpp_lib_expected``. Required to be included as the very
  first standard header so the macro is set consistently across
  translation units; mismatches between TUs that did vs didn't see the
  polyfill cause ODR violations at link time.

Usage
=====

.. code-block:: cpp

    #include <Einsums/CXX23/Expected.hpp>

    using einsums::cxx23::expected;
    using einsums::cxx23::unexpected;

    expected<int, std::string> parse(std::string_view s) {
        if (s.empty()) {
            return unexpected("empty input");
        }
        return std::stoi(std::string{s});
    }

When toolchain support catches up, the polyfill becomes a no-op
``using std::expected;`` alias and existing call sites need no change.

See the :ref:`API reference <modules_Einsums_CXX23_api>` of this module for
the polyfill surface.
