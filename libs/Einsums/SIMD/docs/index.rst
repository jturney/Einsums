..
    ----------------------------------------------------------------------------------------------
     Copyright (c) The Einsums Developers. All rights reserved.
     Licensed under the MIT License. See LICENSE.txt in the project root for license information.
    ----------------------------------------------------------------------------------------------

.. _modules_Einsums_SIMD:

****
SIMD
****

Einsums' SIMD code is `Stripes <https://github.com/Einsums/Stripes>`_, a
separate library that began as this module: portable vectors (``Vec<T>``,
masks, gathers, shuffles, reductions, ``exp``, ``erf``) from SSE2 through
AVX-512 and NEON, and the runtime dispatch that compiles a kernel once per
instruction-set rung (up to the SME rung on aarch64) and runs the best copy
the machine supports. Its documentation is in its repository. Einsums fetches a pinned commit of it
(``cmake/Einsums_SetupStripes.cmake``), or uses an installed Stripes 1.x, and
HPTT and PackedGemm build their per-rung kernels with its
``stripes_add_dispatch_sources()``.

This module is what is Einsums' own about it: the
:option:`--einsums:simd:arch` option, which caps the rung dispatched kernels
run at, and its environment variable ``EINSUMS_SIMD_ARCH``. When the runtime
starts it hands the option to Stripes (``stripes::set_arch_override()``) and
routes Stripes' messages to the Einsums log. Left empty, Stripes' own
``STRIPES_ARCH`` environment variable decides, and otherwise the highest rung
the CPU and operating system support.

The build options ``EINSUMS_WITH_SIMD_DISPATCH``, ``EINSUMS_SIMD_NATIVE_ARCH``
and ``EINSUMS_SIMD_TARGET_CPU`` set Stripes' ``STRIPES_WITH_DISPATCH``,
``STRIPES_NATIVE_ARCH`` and ``STRIPES_TARGET_CPU``.

See the :ref:`API reference <modules_Einsums_SIMD_api>` of this module for more details.
