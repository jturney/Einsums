:orphan:

.. Copyright (c) The Einsums Developers. All rights reserved.
   Licensed under the MIT License. See LICENSE.txt in the project root for license information.

==================
Tensor I/O (.etn)
==================

The ``TensorIO`` module provides high-performance tensor file I/O in a native
binary format, ``.etn``. It replaces HDF5 for performance-critical workloads
and pulls in no external dependencies.

Features
=========

- **Binary format**: 64-byte header, 160-byte entries, 64-byte aligned data.
- **Fast**: raw ``pwrite``/``pread`` with no metadata overhead, near memory-bandwidth throughput.
- **Column-major on disk**: data is always stored column-major, whatever the tensor's layout in memory.
  Contiguous column-major tensors are transferred directly; row-major tensors (``--einsums:row-major``)
  and other strided layouts are reordered through a 1 MiB staging buffer on the way through.
- **All datatypes**: float32, float64, complex64, complex128, int32, int64, uint32, uint64.
- **Ranks 0-8**: scalars through 8-index tensors.
- **Type-checked**: reads and slice writes never convert element types. A dtype or rank mismatch
  throws ``std::invalid_argument``, and a slab past the stored extents throws ``std::out_of_range``.
- **Slice reads**: read a hyperslab without loading the full tensor.
- **Multi-tensor files**: store many tensors in one file. Writing a name the file holds replaces its entry.
- **Crash-safe**: a process killed at any point, or with ``--einsums:io:sync`` on a power loss, leaves
  the file as its last commit left it.
- **Entry operations**: copy an entry through a bounded buffer, rename or swap entries by rewriting
  only the entry table, remove or restage an entry, or compact the file, at once or as nodes of a graph.
- **Distributed I/O**: MPI-coordinated writes to a single file using ``MPI_Exscan``.
- **ComputeGraph integration**: ``read_etn``/``write_etn`` create DiskRead/DiskWrite nodes.
- **Checkpointing**: save and restore Graph or Workspace state with one call.

File Format
============

.. code-block:: text

   [FileHeader]     (64 bytes at offset 0)
   [Data region 0]  (raw bytes, 64-byte aligned)
   [Data region 1]
   ...
   [TensorEntry 0]  (160 bytes each, the entry table)
   [TensorEntry 1]
   ...

The header names the entry table and each entry names its data region, so
regions and the table can sit anywhere after the header, with gaps between
them that later regions reuse. Files written through the distributed path are
standard binary, so serial programs can read them.

Changes reach the file in commits. A handle's writes, rewrites and reserved
entries go into space the file's last commit does not use, and when the handle
closes or is flushed, or an entry operation finishes, it writes a new table
the same way and then rewrites the 64-byte header to name it. Until that one
write the file reads as the last commit left it, so a process killed at any
point leaves a file that opens, holding either the old entries or the new
ones. Rewriting an entry therefore writes its new elements beside the old
ones, which the next rewrite reuses: a file holds at most two copies of an
entry rewritten again and again. With ``--einsums:io:sync`` on, the default,
the file is synced before and after the header is rewritten, which carries the
same guarantee across a power loss.

The header records a checksum of itself and of the table, and a count of the
commits made to the file. Opening a file checks both checksums and that every
region lies inside the file without sharing a byte with another, and refuses a
file that fails, in every mode and without writing to it. A handle refuses to
commit, or to write into the file at all, once another handle has committed to
it since this one read it.

Each entry's data is dense and column-major. By default its axes are stored in
the order they are declared; an entry can instead be stored with its axes in
another order, which makes a different set of boxes contiguous runs of the
file. The entry records that order, and every read and write takes and returns
the declared order whatever it is, so a program never states it to read the
data back:

.. code-block:: cpp

   // Declared [Q, a, b], stored [a, b, Q]: a block of Q is one contiguous run.
   out.reserve<double>("Bvv", {naux, nvir, nvir}, {1, 2, 0});

Every commit writes format version 3, which adds the checksums and the commit
count. Versions 1 and 2 still open, checked by their structure alone, and
become version 3 at their first commit.

TensorFile
===========

Basic read/write for serial programs:

.. code-block:: cpp

   #include <Einsums/TensorIO/TensorFile.hpp>
   using namespace einsums::tensor_io;

   // Write tensors
   TensorFile out("data.etn", TensorFile::Mode::Write);
   out.write("A", A);
   out.write("B", B);

   // Read back
   TensorFile in("data.etn", TensorFile::Mode::Read);
   in.read("A", A);

   // Slice read (only loads the requested range)
   in.read_slice("A", slice, {{3, 8}, {0, 10}});

   // Query
   in.contains("A");       // true
   in.dims("A");           // {10, 8}
   in.dtype("A");          // DType::Float64
   in.tensor_names();      // {"A", "B"}

   // Append in ReadWrite mode
   TensorFile rw("data.etn", TensorFile::Mode::ReadWrite);
   rw.write("C", C);  // Adds C beside A and B
   rw.write("A", A2); // Replaces A once the handle commits; until then A holds its old elements
   rw.flush();        // Commit, and sync to disk

   // Entries; each commits before it returns
   rw.rename("C", "D");              // only the entry table changes
   rw.swap("A", "B");                // "A" now reaches what "B" did, and the other way round
   rw.replace("B", "A");             // "A" now reaches what "B" did, and "B" is gone
   rw.copy(rw, "A", "A_saved");      // through a bounded buffer
   rw.restage("A_saved", {1, 0});    // store its axes in another order, moving its elements
   rw.remove("D");                   // its bytes become space later entries reuse
   rw.unused_bytes();                // what compact() would give back
   rw.compact();                     // rewrite the file without its gaps

DistributedTensorFile
======================

MPI-parallel I/O where all ranks write to one file:

.. code-block:: cpp

   #include <Einsums/TensorIO/DistributedTensorFile.hpp>
   using namespace einsums::tensor_io;

   // All ranks collectively write
   DistributedTensorFile out("checkpoint.etn", DistributedTensorFile::Mode::Write);
   out.write("global", tensor);           // Rank 0 writes (replicated)
   out.write_local("partition", local);   // Each rank writes its part
   out.close();

   // All ranks collectively read
   DistributedTensorFile in("checkpoint.etn", DistributedTensorFile::Mode::Read);
   in.read("global", tensor);             // All ranks read same data
   in.read_local("partition", local);     // Each rank reads its part

File operations go through POSIX ``pwrite``/``pread``, which behaves reliably
on all platforms. MPI ``Exscan`` and ``Gather`` coordinate the per-rank
offsets. There is no dependency on MPI-IO.

Checkpointing
==============

Save/restore ComputeGraph or Workspace state:

.. code-block:: cpp

   #include <Einsums/TensorIO/Checkpoint.hpp>
   namespace ckpt = einsums::tensor_io::checkpoint;

   // Graph
   ckpt::save("scf_iter_5.etn", graph);
   ckpt::save("scf_iter_5.etn", graph, {"Fock", "Density"});  // subset
   ckpt::restore("scf_iter_5.etn", graph);

   // Workspace
   ckpt::save("workspace.etn", workspace);
   ckpt::restore("workspace.etn", workspace);

   // Distributed
   ckpt::save_distributed("checkpoint.etn", graph);
   ckpt::restore_distributed("checkpoint.etn", graph);

ComputeGraph Integration
=========================

Record I/O operations as graph nodes for scheduling and async overlap:

.. code-block:: cpp

   #include <Einsums/TensorIO/GraphIO.hpp>
   using namespace einsums::tensor_io;

   {
       cg::CaptureGuard guard(graph);
       read_etn("integrals.etn", "ERI", &eri);    // DiskRead node
       cg::einsum("ik;kj->ij", &C, eri, B);
       write_etn("result.etn", "C", &C);           // DiskWrite node
       checkpoint_etn("iter_5.etn", graph);         // Save all tensors
   }

   // IOPrefetch pass moves reads early for async overlap
   auto pm = cg::PassManager::create_default();
   graph.apply(pm);

   cg::DataflowExecutor df;
   graph.execute(df);  // I/O overlaps with compute
