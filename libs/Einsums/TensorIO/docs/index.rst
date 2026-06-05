..
    ----------------------------------------------------------------------------------------------
     Copyright (c) The Einsums Developers. All rights reserved.
     Licensed under the MIT License. See LICENSE.txt in the project root for license information.
    ----------------------------------------------------------------------------------------------

.. _modules_Einsums_TensorIO:

########
TensorIO
########

The ``TensorIO`` module provides on-disk tensor storage via HDF5 plus
graph-aware slab I/O primitives that integrate with the
:ref:`ComputeGraph <modules_Einsums_ComputeGraph>` for streaming workloads
that don't fit in memory.

DiskTensor
==========

``DiskTensor<T, Rank>`` is a tensor whose data lives in an HDF5 dataset
rather than RAM. The path string maps to an HDF5 dataset path; the file
is created on first write, persisted across program runs, and lazily read
when accessed.

.. code-block:: cpp

    #include <Einsums/TensorIO/DiskTensor.hpp>

    using namespace einsums;

    DiskTensor<double, 2> A{"/integrals/oovv", 100, 100};

    // Write a row (or any sub-region)
    auto in_memory_row = create_random_tensor("row", 100);
    A(0, All) = in_memory_row;

    // Read it back
    auto row_back = A(0, All);  // triggers HDF5 read

Slab IO
=======

For loops that walk a large tensor block by block, ``tensor_io::Slab``
captures a window onto a ``DiskTensor`` and exposes it via two graph-aware
wrappers:

- ``read_slice_etn`` — schedule a read of one slab into an in-memory
  destination as a ComputeGraph node.
- ``write_slice_etn`` — schedule a write of an in-memory source into one
  slab as a ComputeGraph node.

The slab is captured by reference, so the SAME slab object can be advanced
between graph executions to walk an arbitrarily large tensor with a small
working set.

.. code-block:: cpp

    #include <Einsums/TensorIO/Slab.hpp>

    using namespace einsums::tensor_io;

    DiskTensor<double, 3> big{"/integrals/full", 1000, 1000, 1000};

    Slab slab{big};
    slab.set_window({0, 0, 0}, {100, 1000, 1000});

    // Build a graph that reads, processes, writes — once.
    Graph g;
    {
        Capture cap{g};
        auto src = read_slice_etn(g, slab, ...);
        // ... use src in subsequent graph nodes ...
        write_slice_etn(g, slab, processed);
    }

    // Walk the tensor by advancing the slab and re-executing.
    for (size_t z = 0; z < 1000; z += 100) {
        slab.set_window({z, 0, 0}, {100, 1000, 1000});
        g.execute();
    }

Python surface
==============

``read_slice`` and ``write_slice`` are exposed in Python as
``einsums.io.read_slice`` and ``einsums.io.write_slice``. The same
slab-by-reference pattern applies — the Python ``Slab`` object can be
advanced between captures.

HDF5 file lifecycle
===================

By default Einsums creates a per-process HDF5 file named
``einsums.<pid>.h5`` and cleans it up on shutdown. Override the path with
``--einsums:hdf5-file-name <path>`` and disable cleanup with
``--einsums:no-delete-hdf5-files`` to persist data across runs.

See the :ref:`API reference <modules_Einsums_TensorIO_api>` of this module
for ``DiskTensor``, ``Slab``, and the graph wrappers.
