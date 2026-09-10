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

Einsums provides additional build types: ``ASAN``, ``MSAN``, and ``UBSAN``. These are not used very often and currently are not
guaranteed to work.

5. ``ASAN``: Address Sanitizer (aka ASan) is a memory error detector for C/C++. It finds:

    * Use after free (dangling pointer dereference)

    * Heap buffer overflow

    * Stack buffer overflow

    * Global buffer overflow

    * Use after return

    * Use after scope

    * Initialization order bugs

    * Memory leaks

6. ``MSAN``: Memory Sanitizer (aka MSan) is a detector of uninitialized memory reads in C/C++ programs.
   Uninitialized values occur when stack- or heap-allocated memory is read before it is written.
   MSan detects cases where such values affect program execution.
   MSan is bit-exact: it can track uninitialized bits in a bitfield.
   It will tolerate copying of uninitialized memory, and also simple logic and arithmetic operations with it.

   In general, MSan silently tracks the spread of uninitialized data in memory, and reports a warning when a
   code branch is taken (or not taken) depending on an uninitialized value.

   MSan implements a subset of functionality found in Valgrind (Memcheck tool). It is significantly faster
   than Memcheck.

7. ``UBSAN``: Undefined Behavior Sanitizer (UBSan) is a fast undefined behavior detector. UBSan modifies
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

Einsums has two GPU backends, CUDA and HIP. They are mutually exclusive: turning both
``EINSUMS_WITH_CUDA`` and ``EINSUMS_WITH_HIP`` on is a configure error. When neither is
enabled, the GPU module still builds against a *mock* backend that runs every operation
on the CPU, so GPU-facing code compiles and tests everywhere.

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

Building for HIP
----------------

Set ``-DEINSUMS_WITH_HIP=ON`` and specify a HIP-compatible compiler; AMD's fork of Clang
that ships with HIP is a good choice. For CMake to find the appropriate libraries,
``-DCMAKE_HIP_COMPILER_ROCM_ROOT`` needs to be set to the root directory of the ROCm
installation. On Linux this is often ``/opt/rocm``. Since that flag has a long name,
Einsums provides an alias: ``-DHIP_ROCM_ROOT``. You may set either one and the other will
be populated. If you get configuration errors about being unable to find certain
HIP/ROCm libraries, set the variables CMake is asking for; the CMake files can be found
under ``${HIP_ROCM_ROOT}/lib/cmake``.

HIP can also target NVIDIA hardware, in which case CUDA is required as well. That is
still the ``EINSUMS_WITH_HIP`` path, not ``EINSUMS_WITH_CUDA``.