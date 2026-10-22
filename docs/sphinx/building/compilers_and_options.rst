..
    ----------------------------------------------------------------------------------------------
     Copyright (c) The Einsums Developers. All rights reserved.
     Licensed under the MIT License. See LICENSE.txt in the project root for license information.
    ----------------------------------------------------------------------------------------------

Compiler selection and customizing a build
******************************************

Selecting a specific compiler
=============================

CMake supports the standard environment variable ``CXX`` to select a specific C++ compiler.
This environment variable is documented in the `CMake docs
<https://cmake.org/cmake/help/latest/envvar/CC.html>`__.

Note that environment variables only get applied from a clean build, because they affect
the configuration stage. An incremental rebuild does not react to changes in environment
variables.

CMake also supports passing the ``-DCMAKE_CXX_COMPILER=`` `variable
<https://cmake.org/cmake/help/latest/variable/CMAKE_LANG_COMPILER.html>`__
to the command-line call to ``cmake``.

Selecting build type
====================

CMake natively supports four build types: ``Debug``, ``Release``, ``MinSizeRel``, and ``RelWithDebInfo``.

1. ``Release``: high optimization level, no debug info, code or asserts.

2. ``Debug``: No optimization, asserts enabled, [custom debug (output) code enabled],
   debug info included in executable (so you can step through the code with a
   debugger and have address to source-file:line-number translation).

3. ``RelWithDebInfo``: optimized, *with* debug info, but no debug (output) code or asserts.

4. ``MinSizeRel``: same as Release but optimizing for size rather than speed.

Sanitizers are not build types. Enable them with ``-DEINSUMS_WITH_SANITIZERS=<list>``, a
comma-separated list of ``address``, ``leak``, ``memory``, ``thread``, and ``undefined`` that is
passed to ``-fsanitize=``, for example ``-DEINSUMS_WITH_SANITIZERS=address,undefined``. It works
with any ``CMAKE_BUILD_TYPE``.

1. ``address``: Address Sanitizer (aka ASan) is a memory error detector for C/C++. It finds:

    * Use after free (dangling pointer dereference)

    * Heap buffer overflow

    * Stack buffer overflow

    * Global buffer overflow

    * Use after return

    * Use after scope

    * Initialization order bugs

    * Memory leaks

2. ``memory``: Memory Sanitizer (aka MSan) is a detector of uninitialized memory reads in C/C++ programs.
   Uninitialized values occur when stack- or heap-allocated memory is read before it is written.
   MSan detects cases where such values affect program execution.
   MSan is bit-exact: it can track uninitialized bits in a bitfield.
   It will tolerate copying of uninitialized memory, and also simple logic and arithmetic operations with it.

   In general, MSan silently tracks the spread of uninitialized data in memory, and reports a warning when a
   code branch is taken (or not taken) depending on an uninitialized value.

   MSan implements a subset of functionality found in Valgrind (Memcheck tool). It is significantly faster
   than Memcheck.

3. ``undefined``: Undefined Behavior Sanitizer (UBSan) is a fast undefined behavior detector. UBSan modifies
   the program at compile-time to catch various kinds of undefined behavior during program execution, for
   example:

   * Array subscript out of bounds, where the bounds can be statically determined

   * Bitwise shifts that are out of bounds for their data type

   * Dereferencing misaligned or null pointers

   * Signed integer overflow

   * Conversion to, from, or between floating-point types which would overflow the destination

See :ref:`the page on CMake variables <cmake_variables>` for a list of options that can be passed to
CMake when configuring.

Building for GPU
================

Einsums has two GPU backends: CUDA, and Apple's Metal Performance Shaders (MPS), which is
on by default on macOS. When neither is enabled, the GPU module still builds against a
*mock* backend that runs every operation on the CPU, so GPU-facing code compiles and tests
everywhere.

Building for CUDA
-----------------

Add the CUDA dependencies to your conda environment with ``--gpu cuda``, then configure
with ``-DEINSUMS_WITH_CUDA=ON``:

.. code:: bash

   python3 devtools/conda-envs/merge_yml.py --output=einsums.yml --gpu cuda gcc openblas
   conda env create -f einsums.yml
   conda activate einsums-dev

   cmake -S . -B build -GNinja -DEINSUMS_WITH_CUDA=ON

The environment supplies the CUDA toolkit, so no separate system install is needed;
``find_package(CUDAToolkit)`` picks it up from ``$CONDA_PREFIX``.

.. important::

   The host compiler must be **gcc or clang**. ``nvcc`` rejects Intel's ``icpx``
   outright ("unsupported Intel ICX compiler"), so an ``intel`` environment cannot
   build the CUDA backend. Note that conda's ``cuda-nvcc`` activation script derives
   ``-ccbin`` from ``$CXX``, so an intel environment silently produces a broken
   ``NVCC_PREPEND_FLAGS``.

Target architectures are controlled by ``EINSUMS_WITH_CUDA_ARCHITECTURES``, which feeds
``CMAKE_CUDA_ARCHITECTURES``. It defaults to ``native`` when a GPU is visible at
configure time and ``all-major`` otherwise, so CI and cross-builds produce a portable
fat binary while a developer machine builds only what it can run.

.. note::

   The CUDA backend is a work in progress. ``gemm``, the strided-batched ``gemm``
   variants, streams, events and device memory management are implemented against
   cuBLAS and the CUDA runtime. Several entry points - ``gemv``, complex ``gemm``,
   the reduced-precision kernels, and all of ``gpu::solver`` - are not yet
   implemented and raise an error rather than returning a wrong answer.

Testing the discrete-device paths without a GPU
-----------------------------------------------

``-DEINSUMS_WITH_GPU_MOCK_DISCRETE=ON`` makes the mock backend behave like a
discrete card rather than a unified-memory one. It requires no GPU and no CUDA
toolkit.

This exists because the ordinary mock cannot fail in the ways that matter. It
services every ``gpu::blas`` call with CPU BLAS over ``malloc``'d memory, so it
is numerically correct however broken the real CUDA path is, and
``gpu::has_unified_memory`` being true compiles out the whole host-to-device
transfer path in the ComputeGraph executor. Under mock-discrete:

* ``has_unified_memory`` is false, so shadow allocation, uploads, copy-backs and
  tensor pointer swapping all execute;
* each device allocation is a guard-paged ``mmap`` held at ``PROT_NONE``, so a
  host dereference of a device pointer faults at the offending instruction
  instead of quietly succeeding;
* the entry points that raise on CUDA raise here too, so the mock is a contract
  test for the vendor backends rather than a passthrough that always agrees.

The backend's own kernels still reach device memory - cuBLAS reads it as a
matter of course - through ``gpu::MockDeviceKernelScope``. Code outside such a
scope does not, which is what catches a host kernel running on operands that
are still swapped to device shadows.

It is POSIX-only (``mmap``/``mprotect``) and mutually exclusive with a real
backend. CI runs it as the ``mock-discrete`` variant; locally,
``./devtools/docker/run-ci-leg.sh mock-discrete``.
