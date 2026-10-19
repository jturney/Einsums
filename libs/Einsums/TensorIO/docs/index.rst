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
tensors. Writing a name the file already holds replaces that entry. Changes
reach the file in commits, when the handle closes or is flushed: new elements
and a new entry table go where the last commit does not reach, and a final
header write switches to them, so a program killed at any point leaves a file
that opens with its last commit's entries. Storing a tensor again at every
iteration alternates between two regions of the file, so the file does not
grow.

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

Copying, renaming, swapping and removing entries
================================================

These functions change which entry a name reaches, fill one entry from
another, or tidy a file, without the caller holding any entry in memory. Each
commits before it returns:

- ``copy_entry(src_path, src_entry, dst_path, dst_entry)`` copies an entry into
  another, in the same file or another, creating the destination with the
  source's extents, element type and stored order when it does not exist. The
  elements pass through a buffer of at most ``--einsums:work-buffer-size``.
- ``rename_entry(path, old_name, new_name)`` renames an entry. It refuses a
  name the file already holds.
- ``swap_entries(path, first, second)`` exchanges the names of two entries.
- ``remove_entry(path, name)`` removes an entry. Its bytes become space later
  entries reuse.
- ``restage_entry(path, name, order)`` stores an entry with its axes in another
  order, moving its elements through the same bounded buffer. The entry holds
  its old layout or its new one whenever the program stops, and the file needs
  room for a second copy while it runs.
- ``compact_file(path)`` rewrites the file without the space no entry uses,
  into a new file that then replaces it. It needs room on disk for the
  entries' bytes while it runs.

A rename, a swap or a removal rewrites only the file's entry table, so it costs
the same whatever the entries hold.

Each runs at once outside a capture. Inside one it records a node that runs at
every replay, after every read and write of the file that comes before it in
the graph and before every one after it, including the loads and stores of
tensors backed by the file.

Updating an entry
=================

A tensor too large for memory that a graph both reads and updates, the
amplitudes of an iterative method for instance, is declared with an update
backing. Each replay starts with the tensor holding what the entry holds. The
graph reads and writes it as any tensor, its writes go to the scratch entry
``<entry>.next``, and when the replay ends that entry replaces the entry in one
commit (``TensorFile::replace``). A crash before then leaves the entry holding
the previous replay's value. The update backings a graph has on one file are
published by one commit together, so the amplitudes of an iteration, T1 and T2
for instance, are either all the previous replay's or all the new ones.

.. code-block:: cpp

    #include <Einsums/TensorIO/Backing.hpp>

    tensor_io::EtnFile const amps("amps.etn");
    auto &t2 = graph.declare_runtime_tensor<double>("t2", amps.entry("t2", false, {}, true));
    auto &r2 = graph.declare_runtime_tensor<double>("r2", {o, o, v, v}, /*intermediate=*/true);
    {
        cg::CaptureGuard guard(graph);
        // ... the residual r2 from t2 ...
        cg::direct_division(1.0, r2, D, 0.0, &t2);  // the new t2
    }
    for (int iter = 0; iter < max_iter; ++iter) {
        graph.execute();  // "t2" holds the newest amplitudes after every replay
    }

Statement order is what decides which value a read sees, as it is for a tensor
in memory: the reads before the first write of the tensor see the value the
replay started from, and ``SlabStreaming`` may read those from the entry a
block at a time; the reads after it see the new value, in memory. The tensor
is written in memory and stored whole after its last writer.

Python surface
==============

The graph wrappers are exposed in Python under ``einsums.io``. Each C++
``read_slice_etn`` / ``write_slice_etn`` is bound as ``einsums.io.read_slice``
/ ``einsums.io.write_slice``, and ``read_etn`` / ``write_etn`` as
``einsums.io.read`` / ``einsums.io.write``, and ``copy_entry``,
``rename_entry``, ``swap_entries``, ``remove_entry``, ``restage_entry`` and
``compact_file`` as ``einsums.io.copy``, ``einsums.io.rename``,
``einsums.io.swap``, ``einsums.io.remove``, ``einsums.io.restage`` and
``einsums.io.compact``. ``tensor_io::EtnFile`` is ``einsums.io.EtnFile``, and
``etn_backing`` is ``einsums.io.etn``, whose ``update=True`` gives an update
backing. The slab-by-reference pattern
carries over: a Python ``Slab`` object can be advanced between graph
executions the same way.

See the :ref:`API reference <modules_Einsums_TensorIO_api>` for ``TensorFile``,
``Slab``, the graph wrappers, distributed I/O, and checkpointing.
