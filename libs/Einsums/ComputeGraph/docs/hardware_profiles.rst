.. Copyright (c) The Einsums Developers. All rights reserved.
   Licensed under the MIT License. See LICENSE.txt in the project root for license information.

.. _computegraph_hardware_profiles:

==================
Hardware Profiles
==================

Passes that must choose between alternatives - ``ContractionPlanning`` and
``GPUPlacement`` among them - estimate execution time through a ``CostModel``.
That model is only as good as the machine measurements behind it, and this page
covers where those measurements come from: the built-in table, the calibration
tool, and the :option:`--einsums:hardware:profile` override.

Profile Structure
==================

A ``CostModel`` holds two ``DeviceProfile`` entries (CPU and GPU):

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
Rapids, AMD EPYC Rome/Milan/Genoa, Generic x86-64, Generic ARM

**GPU profiles**: NVIDIA V100/A100/H100/RTX 3090/RTX 4090, AMD MI250X/MI300X,
Apple MPS

Detection uses:

- macOS: ``sysctlbyname("machdep.cpu.brand_string")``
- x86 Linux: CPUID brand string
- GPU: ``gpu::device_name()`` (cudaGetDeviceProperties / MTLDevice.name)

The best match is selected by longest-substring-wins on the brand string.

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

Calibration Tool
=================

The ``calibrate_hardware`` tool measures real performance on the current machine:

.. code-block:: bash

   ./calibrate_hardware --output my_hardware.json

It sweeps DGEMM across matrix sizes (16 to 2048), measures memory bandwidth,
and BLAS kernel overhead. Output is a JSON file loadable by:

.. code-block:: cpp

   auto profile = CostModel::load_json("my_hardware.json");
   pm.add<cg::passes::ContractionPlanning>(profile);

Handing the same file to :option:`--einsums:hardware:profile` puts it behind
``CostModel::detect_default()`` instead, so every cost-model pass in a default
pass manager prices against it without any code naming the path:

.. code-block:: bash

   ./my_program --einsums:hardware:profile my_hardware.json
   EINSUMS_HARDWARE_PROFILE=my_hardware.json ./my_program

A file that will not load is a warning and a fall back to the built-in table
rather than a failure: the profile shapes which optimization is chosen, never
whether the answer is right.

Shared Profile in create_default()
====================================

``PassManager::create_default()`` detects hardware once and shares that one
profile across every cost-model pass:

.. code-block:: cpp

   // Internally:
   auto cost_model = CostModel::detect_default();
   pm.add<passes::TiledExpansion>(4096, -1.0, Densify::Auto, FuseTiles::Auto, cost_model);
   pm.add<passes::ContractionPlanning>(cost_model);
   pm.add<passes::GEMMBatching>(cost_model);
   pm.add<passes::StreamContractionFusion>(cost_model);
   pm.add<passes::GPUPlacement>(cost_model);   // GPU builds only

Sharing matters beyond saving a detection: these passes make interlocking
decisions, and a densify-or-not choice priced against one profile while the
batching gate is priced against another can leave the graph in a shape neither
pass would have picked on its own.
