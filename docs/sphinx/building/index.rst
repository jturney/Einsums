..
    ----------------------------------------------------------------------------------------------
     Copyright (c) The Einsums Developers. All rights reserved.
     Licensed under the MIT License. See LICENSE.txt in the project root for license information.
    ----------------------------------------------------------------------------------------------

.. _building-from-source:

Building from source
====================

.. note::

   If you are only trying to install Einsums, we recommend using binaries.
   See :ref:`Installation Instructions <installing>` for details on that.

Building Einsums from source requires setting up system-level dependencies
(compilers, BLAS/LAPACK libraries, etc.) first, and then invoking a build. The
build may be done in order to install Einsums for local usage, develop Einsums
itself, or build redistributable binary packages. Any it may be desired to
customize aspects of how the build is done. This guide will cover all these
aspects. In addition, it provides background information on how the Einsums build
works.

.. _system-level:

System-level dependencies
-------------------------

Einsums is a C++ compiled library, which means you need compilers and some
other system-level dependencies to build it on your system.

.. note::

  If you are using Conda, you can skip the steps in this section - with the
  exception of installing the Apple Developer Tools for macOS. All other
  dependencies will be installed automatically. Generate an environment file
  with the merge script and create the environment from it:

  .. code:: bash

    python3 devtools/conda-envs/merge_yml.py --output=einsums.yml
    conda env create -f einsums.yml
    conda activate einsums-dev

  The merge script picks a sensible toolchain and BLAS for your platform, but
  both can be overridden. The compiler options are ``default`` (gcc on Linux,
  clang on macOS/Windows), ``gcc``, ``clang``, and ``intel``. The BLAS options
  are ``openblas``, ``mkl``, and ``accelerate`` (macOS). If you plan on building
  the docs, add the ``--docs`` flag. For example:

  .. code:: bash

    python3 devtools/conda-envs/merge_yml.py --output=einsums.yml [--docs] <compiler> <blas>
    conda env create -f einsums.yml
    conda activate einsums-dev

  If you don't have a conda installation yet, we recommend using
  Condaforge_; any conda flavor will work though.

.. tab-set::

  .. tab-item:: General
    :sync: general

    You will need:

    * C++ compiler with C++20 support (GCC, LLVM/Clang, or Intel).

    * BLAS and LAPACK libraries. `OpenBLAS <https://github.com/OpenMathLib/OpenBLAS/>`__
      is the Einsums default; other variants include Apple Accelerate,
      `MKL <https://software.intel.com/en-us/intel-mkl>`__,
      `ATLAS <https://math-atlas.sourceforce.net/>`__ and
      `Netlib <https://www.netlib.org/lapack/index.hmtl>`__ ( or "Reference")
      BLAS and LAPACK.

    * CMake

    * Ninja for building. Since cpptrace 1.0, Unix Makefiles are no longer supported.

    The following are also required, but will be downloaded if not given:

    * fmtlib >= 11

    * Catch2 >= 3

    * gabime/spdlog >= 1

    * HDF5 for disk operations.

    * ZLIB headers

    Recommended

    * MIMalloc for faster memory allocation.

    Optional:

    * jeremy-rifkin/cpptrace to create backtraces when errors occur.

    * For the Fourier Transform abilities, you will need either `FFTW3 <https://www.fftw.org>`__
      or MKL.

    * HIP for GPU support. If using an Nvidia platform, CUDA is also required.

    * pybind11 for the Python extension module.

    * NumPy and SciPy for the Python tests.

    * LibreTT for GPU tensor transpositions. This may be superceded by hipTensor in the future when it becomes more mature.


  .. tab-item:: Linux
    :sync: linux

    Optional:

    * cpptrace for C++ backtraces.

  .. tab-item:: macOS
    :sync: macos

    Install Apple Developer Tools. An easy way to do this is to
    `open a terminal window <https://blog.teamtreehouse.com/introduction-to-the-mac-os-x-command-line>`_,
    enter the command::

        xcode-select --install

    and follow the prompts. Apple Developer Tools includes Git, the Clang C/C++
    compilers, and other development utilities that may be required.

  .. tab-item:: Windows
    :sync: windows

    Windows is not supported at this time.

Building Einsums from source
----------------------------

If you want to build from source in order to work on Einsums itself, first clone
the Einsums repository.::

    git clone https://github.com/Einsums/Einsums.git
    cd Einsums

Then you will want to do the following:

1. Create a dedicated development environment (conda environment),
2. Install all needed dependencies (*build*, and also *test*, and *doc*
   dependencies).
3. Build Einsums.

To create an ``einsums-dev`` development environment with every required and
optional dependency installed, except for HIP, perform the operations in the previous section.

To build Einsums in an activated development environment, run::

    mkdir build
    cd build
    cmake ..
    make

This will build Einsums inside the ``build`` directory. You can then run tests
(``ctest`` and ``pytest``), or take other development steps like build the html documentation
or running benchmarks.

Customizing builds
------------------

.. toctree::
   :maxdepth: 1

   compilers_and_options

.. _Condaforge: https://github.com/conda-forge/miniforge#condaforge
