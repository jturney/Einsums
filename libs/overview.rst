..
    ----------------------------------------------------------------------------------------------
     Copyright (c) The Einsums Developers. All rights reserved.
     Licensed under the MIT License. See LICENSE.txt in the project root for license information.
    ----------------------------------------------------------------------------------------------

.. _modules_overview:

===============
Library Modules
===============

|einsums| is organized into modules with clear dependencies and no cycles.
As an end-user, the use of these modules is completely transparent. If
you use e.g. ``add_einsums_executable`` to create a target in your project you will
automatically get all modules as dependencies. Currently these are nothing more than an
internal grouping and do not affect usage. They cannot be consumed individually
at the moment.

Each module page below describes what the module does and links to the
generated API reference for its public headers.

.. toctree::
    :maxdepth: 1

    /libs/Einsums/Assertion/docs/index.rst
    /libs/Einsums/BLAS/docs/index.rst
    /libs/Einsums/BLASBase/docs/index.rst
    /libs/Einsums/BLASVendor/docs/index.rst
    /libs/Einsums/BlockManager/docs/index.rst
    /libs/Einsums/BufferAllocator/docs/index.rst
    /libs/Einsums/CXX23/docs/index.rst
    /libs/Einsums/Comm/docs/index.rst
    /libs/Einsums/CommandLine/docs/index.rst
    /libs/Einsums/ComputeGraph/docs/index.rst
    /libs/Einsums/ComputeGraphTypes/docs/index.rst
    /libs/Einsums/Concepts/docs/index.rst
    /libs/Einsums/Config/docs/index.rst
    /libs/Einsums/Debugging/docs/index.rst
    /libs/Einsums/Errors/docs/index.rst
    /libs/Einsums/FFT/docs/index.rst
    /libs/Einsums/GPU/docs/index.rst
    /libs/Einsums/HPTT/docs/index.rst
    /libs/Einsums/Hardware/docs/index.rst
    /libs/Einsums/Iterator/docs/index.rst
    /libs/Einsums/LinearAlgebra/docs/index.rst
    /libs/Einsums/Logging/docs/index.rst
    /libs/Einsums/PackedGemm/docs/index.rst
    /libs/Einsums/Preprocessor/docs/index.rst
    /libs/Einsums/Print/docs/index.rst
    /libs/Einsums/Profile/docs/index.rst
    /libs/Einsums/Python/docs/index.rst
    /libs/Einsums/PythonDemo/docs/index.rst
    /libs/Einsums/Runtime/docs/index.rst
    /libs/Einsums/RuntimeConfiguration/docs/index.rst
    /libs/Einsums/SIMD/docs/index.rst
    /libs/Einsums/StringUtil/docs/index.rst
    /libs/Einsums/TaskPool/docs/index.rst
    /libs/Einsums/Tensor/docs/index.rst
    /libs/Einsums/TensorAlgebra/docs/index.rst
    /libs/Einsums/TensorBase/docs/index.rst
    /libs/Einsums/TensorIO/docs/index.rst
    /libs/Einsums/TensorImpl/docs/index.rst
    /libs/Einsums/TensorUtilities/docs/index.rst
    /libs/Einsums/TypeSupport/docs/index.rst
    /libs/Einsums/Utilities/docs/index.rst
    /libs/Einsums/Version/docs/index.rst
