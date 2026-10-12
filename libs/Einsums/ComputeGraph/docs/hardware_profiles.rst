.. Copyright (c) The Einsums Developers. All rights reserved.
   Licensed under the MIT License. See LICENSE.txt in the project root for license information.

.. _computegraph_hardware_profiles:

==================
Hardware Profiles
==================

Passes that must choose between alternatives - ``ContractionPlanning`` and
``GPUPlacement`` among them - estimate execution time through a ``CostModel``.
That model is only as good as the machine measurements behind it, and this page
covers where those measurements come from: the built-in table, and the
hardware profile the ``einsums-tune`` tool measures.

Profile Structure
==================

A ``CostModel`` holds two ``DeviceProfile`` entries (CPU and GPU). An abridged
view (the full struct is in ``CostModel.hpp``):

.. code-block:: cpp

   struct DeviceProfile {
       std::string name;              // "Apple M4 Pro", "NVIDIA A100"
       DeviceType  device_type;       // CPU or GPU
       double peak_gflops_fp64;       // Peak FP64 throughput
       double peak_gflops_fp32;       // Peak FP32 throughput
       double mem_bandwidth_gbps;     // Main memory bandwidth
       double device_bandwidth_gbps;  // GPU device memory bandwidth
       double pcie_bandwidth_gbps;    // Host↔Device transfer bandwidth
       // Network (for distributed):
       double inter_node_bandwidth_gbps;
       double inter_node_latency_us;
       // GEMM efficiency table:
       std::vector<GemmEfficiencyPoint> gemm_efficiency;
       // Cache hierarchy:
       std::vector<CacheLevel> caches;
   };

The GEMM efficiency table maps matrix shapes to measured GFLOPS, enabling
shape-dependent cost estimation (small GEMMs are much slower per FLOP than
large ones).

Auto-Detection
===============

``CostModel::detect_default()`` matches the current hardware against
a built-in database of 25+ device profiles:

**CPU profiles**: Apple M1/M2/M3/M4 (Pro/Max), Intel Skylake/Ice Lake/Sapphire
Rapids, AMD EPYC Rome/Milan/Genoa; an unmatched CPU falls back to a single
Generic CPU profile

**GPU profiles**: NVIDIA V100/A100/H100/RTX 3090/RTX 4090, AMD MI250X/MI300X,
Apple MPS

Detection uses:

- macOS: ``sysctlbyname("machdep.cpu.brand_string")``
- other x86-64: CPUID brand string
- aarch64 Linux: implementer and part codes from ``/proc/cpuinfo``
- GPU: ``gpu::device_name()`` (cudaGetDeviceProperties / MTLDevice.name)

The best match is selected by longest-substring-wins on the brand string.
Cache sizes are detected at runtime and fill in any profile that carries none.

Cost Estimation
================

.. code-block:: cpp

   auto profile = CostModel::detect_default();

   // GEMM time with shape-dependent efficiency
   double us = profile.estimate_gemm_time_us(256, 128, 512, Target::CPU);

   // Total time including memory traffic (roofline model)
   double total = profile.estimate_total_gemm_time_us(M, N, K, sizeof(double), Target::GPU);

   // Host↔Device transfer
   double xfer = profile.estimate_transfer_time_us(bytes);

   // Network communication
   double ar = profile.estimate_allreduce_time_us(bytes, num_ranks);

Measuring a Machine
====================

The ``einsums-tune`` tool, installed with Einsums, measures the machine it runs
on and writes its hardware profile:

.. code-block:: bash

   einsums-tune                          # this machine's profile, in the cache directory
   einsums-tune --output profile.json    # somewhere else, to share or install for a site

It sweeps DGEMM over a log-spaced range of sizes (default 16 to 2048, set by
``--min-size``/``--max-size``) and measures copy and per-cache-level bandwidth,
permute throughput, BLAS call overhead, per-kernel thread-efficiency curves
(skipped with ``--no-thread-sweep``) and the OpenMP region cost at every team
size. It starts from the built-in table rather than from an earlier profile, so
nothing it does not measure is inherited from one.

Everything goes into one file, whose ``cost_model`` section is this page's
model and whose other sections belong to the Hardware module; see
:ref:`the hardware profile <modules_Einsums_Hardware>` for its layout and for
when a profile applies. ``CostModel::detect_default()`` uses the profile in
effect, so every cost-model pass in a default pass manager prices against it
without any code naming a path. To use a particular file:

.. code-block:: bash

   ./my_program --einsums:hardware:profile profile.json
   EINSUMS_HARDWARE_PROFILE=profile.json ./my_program

A profile that does not apply, or will not load, is a warning and a fall back to
the built-in table rather than a failure: the profile shapes which optimization
is chosen, never whether the answer is right.

A model can also be handed to a pass directly. ``CostModel::save_json`` and
``CostModel::load_json`` write and read a model on its own:

.. code-block:: cpp

   auto profile = CostModel::load_json("model.json");  // expected<CostModel, GraphError>
   if (profile) {
       pm.add<cg::passes::ContractionPlanning>(*profile);
   }

Shared Profile in create_default()
====================================

``PassManager::create_default()`` detects hardware once and shares that one
profile across every cost-model pass:

.. code-block:: cpp

   // Internally:
   auto cost_model = CostModel::detect_default();
   pm.add<passes::TiledExpansion>(4096, -1.0, Densify::Auto, FuseTiles::Auto, cost_model);
   pm.add<passes::DistributiveFactoring>(cost_model);
   pm.add<passes::LayoutAssignment>(cost_model);
   pm.add<passes::ContractionPlanning>(cost_model);
   pm.add<passes::GEMMBatching>(cost_model);
   pm.add<passes::StreamContractionFusion>(cost_model);
   pm.add<passes::GPUPlacement>(cost_model);   // GPU builds only

Sharing matters beyond saving a detection: these passes make interlocking
decisions, and a densify-or-not choice priced against one profile while the
batching gate is priced against another can leave the graph in a shape neither
pass would have picked on its own.
