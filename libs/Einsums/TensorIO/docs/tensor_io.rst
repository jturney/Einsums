:orphan:

.. Copyright (c) The Einsums Developers. All rights reserved.
   Licensed under the MIT License. See LICENSE.txt in the project root for license information.

==================
Tensor I/O (.etn)
==================

The ``TensorIO`` module provides high-performance tensor file I/O in a native
binary format (``.etn``). It replaces HDF5 for performance-critical workloads
with zero external dependencies.

Features
=========

- **Binary format**: 64-byte header, 160-byte entries, 64-byte aligned data
- **Fast**: Raw ``pwrite``/``pread`` — no metadata overhead, near-memory-bandwidth throughput
- **All datatypes**: float32, float64, complex64, complex128, int32, int64, uint32, uint64
- **Ranks 1-8**: Vectors through 8-index tensors
- **Slice reads**: Read a hyperslab without loading the full tensor
- **Multi-tensor files**: Store many tensors in one file
- **Distributed I/O**: MPI-coordinated writes to a single file using ``MPI_Exscan``
- **ComputeGraph integration**: ``read_etn``/``write_etn`` create DiskRead/DiskWrite nodes
- **Checkpointing**: Save/restore Graph or Workspace state with one call

File Format
============

.. code-block:: text

   [FileHeader]     (64 bytes at offset 0)
   [Data region 0]  (raw bytes, 64-byte aligned)
   [Data region 1]
   ...
   [TensorEntry 0]  (160 bytes each, at end of file)
   [TensorEntry 1]
   ...

Entry table at the end enables appending tensors without shifting existing data.
Files created with MPI-IO are standard binary — readable by serial programs.

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
   rw.write("C", C);  // Appends without overwriting A, B

DistributedTensorFile
======================

MPI-parallel I/O — all ranks write to one file:

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

Uses POSIX ``pwrite``/``pread`` for file operations (reliable on all platforms)
with MPI ``Exscan`` and ``Gather`` for offset coordination. No MPI-IO dependency.

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
