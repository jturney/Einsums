..
    ----------------------------------------------------------------------------------------------
     Copyright (c) The Einsums Developers. All rights reserved.
     Licensed under the MIT License. See LICENSE.txt in the project root for license information.
    ----------------------------------------------------------------------------------------------

.. _modules_Einsums_TensorIO:

########
TensorIO
########

The ``TensorIO`` module stores tensors on disk in a native binary format
(``.etn``) and adds graph-aware slab I/O primitives that integrate with the
:ref:`ComputeGraph <modules_Einsums_ComputeGraph>`. Together they let you
stream a tensor too large for memory through a computation block by block.

TensorFile
==========

``TensorFile`` is the serial entry point: it reads and writes whole tensors,
or hyperslabs of them, in the ``.etn`` format. A file holds many named
tensors. New tensors append to the end without rewriting existing data.

.. code-block:: cpp

    #include <Einsums/TensorIO/TensorFile.hpp>

    using namespace einsums::tensor_io;

    // Write tensors to a file.
    TensorFile out("integrals.etn", TensorFile::Mode::Write);
    out.write("A", A);
    out.write("B", B);

    // Read one back.
    TensorFile in("integrals.etn", TensorFile::Mode::Read);
    in.read("A", A);

    // Read a sub-region without loading the whole tensor.
    in.read_slice("A", slice, {{3, 8}, {0, 10}});

Slab I/O
========

For a loop that walks a large stored tensor block by block, ``tensor_io::Slab``
names a hyperslab and the two graph-aware wrappers schedule I/O against it:

- ``read_slice_etn`` records a ComputeGraph node that reads the slab into an
  in-memory destination.
- ``write_slice_etn`` records a ComputeGraph node that writes an in-memory
  source into the slab.

A ``Slab`` holds ``ranges``, one half-open ``{start, end}`` pair per
dimension. The wrappers capture the ``Slab`` by reference, so the same graph
node can drive a loop: mutate ``slab.ranges`` between executions to walk an
arbitrarily large tensor with a small working set.

.. code-block:: cpp

    #include <Einsums/TensorIO/GraphIO.hpp>

    using namespace einsums::tensor_io;
    namespace cg = einsums::compute_graph;

    // "full" already exists in the file; "block" is sized to one slab.
    Slab slab{{ {0, 100}, {0, 1000}, {0, 1000} }};
    auto block = create_zero_tensor<double>("block", 100, 1000, 1000);

    // Build a graph that reads one slab, processes it, and writes it back.
    cg::Graph g("slab_walk");
    {
        cg::CaptureGuard guard(g);
        read_slice_etn("full.etn", "T", slab, &block);
        // ... use block in subsequent graph nodes ...
        write_slice_etn("full.etn", "T", slab, &block);
    }

    // Walk the tensor by advancing the slab and re-executing the same graph.
    for (size_t z = 0; z < 1000; z += 100) {
        slab.ranges[0] = {z, z + 100};
        g.execute();
    }

The target entry must already exist before ``write_slice_etn`` runs, for
example from a prior ``write_etn`` or ``TensorFile::reserve``. The in-memory
tensor must match the slab shape; the bound ``TensorFile`` methods throw at
execute time if the dimensions disagree.

Python surface
==============

The graph wrappers are exposed in Python under ``einsums.io``. Each C++
``read_slice_etn`` / ``write_slice_etn`` is bound as ``einsums.io.read_slice``
/ ``einsums.io.write_slice``, and ``read_etn`` / ``write_etn`` as
``einsums.io.read`` / ``einsums.io.write``. The slab-by-reference pattern
carries over: a Python ``Slab`` object can be advanced between graph
executions the same way.

See the :ref:`API reference <modules_Einsums_TensorIO_api>` for ``TensorFile``,
``Slab``, the graph wrappers, distributed I/O, and checkpointing.
