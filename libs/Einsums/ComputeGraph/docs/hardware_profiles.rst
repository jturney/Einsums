.. Copyright (c) The Einsums Developers. All rights reserved.
   Licensed under the MIT License. See LICENSE.txt in the project root for license information.

==================
Hardware Profiles
==================

The ``HardwareProfile`` system provides architecture-specific performance data
for cost-model-based optimization. The ``ContractionPlanning`` and ``GPUPlacement``
passes use this data to estimate execution time and make informed decisions.

Profile Structure
==================

A ``HardwareProfile`` contains two ``DeviceProfile`` entries (CPU and GPU):

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

``HardwareProfile::detect_default()`` matches the current hardware against
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

   auto profile = HardwareProfile::detect_default();

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

   auto profile = HardwareProfile::load_json("my_hardware.json");
   pm.add<cg::passes::ContractionPlanning>(profile);

Shared Profile in create_default()
====================================

``PassManager::create_default()`` detects hardware once and shares the profile
across ``GPUPlacement`` and ``ContractionPlanning``:

.. code-block:: cpp

   // Internally:
   auto profile = HardwareProfile::detect_default();
   pm.add<passes::GPUPlacement>(profile);
   pm.add<passes::ContractionPlanning>(profile);

Both passes use the same cost model for consistent decisions.
