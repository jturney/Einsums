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
hardware profile that ``einsums tune`` measures.

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
       // GEMM efficiency table, per engine:
       std::vector<GemmEfficiencyPoint> gemm_efficiency;
       // Extra cost per element by contiguous run length:
       std::vector<RunCostPoint> scatter_cost;  // writing C through the scatter
       std::vector<RunCostPoint> pack_cost;     // gathering an operand into its panel
       // Cache hierarchy:
       std::vector<CacheLevel> caches;
   };

The GEMM efficiency table maps matrix shapes to measured GFLOPS, each point tagged with the engine it was measured on: ``routed``, whatever the string einsum hands a flat contraction to, or ``packed``, PackedGemm's tile kernel.
Rates between points are interpolated in log space: trilinearly over a grid of shapes, or by size along a table of cubes, which knows nothing of shape.

A GEMM's M, N and K are not all a contraction costs.
A contraction whose C interleaves its operands' indices is built in a scratch block and scattered into place, at a cost per element set by how long C's contiguous runs are, and an operand split by a stride break is gathered into its panel the same way.
``estimate_contraction_time_us`` prices a contraction from ``packed_gemm::describe_packed_contraction``, the plan PackedGemm would run: a call per split-K share, the arithmetic at the engine's rate but never faster than the operands and C can stream, the scatter's and the packers' extra costs at their run lengths, and split-K's buffers.
AxisTiling prices each tile of a loop this way, so a small block pays for its short runs and its repeated packing rather than only its arithmetic.

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

``einsums tune`` measures the machine it runs on and writes its hardware profile.
The same code runs as ``python -m einsums.tune`` and as the ``einsums-tune``
executable, which an installation without Python has:

.. code-block:: bash

   einsums tune                          # this machine's profile, in the cache directory
   einsums tune --output profile.json    # somewhere else, to share or install for a site
   einsums tune --show                   # the profile in effect, and why one is not used
   einsums tune --check                  # exit 0 only when a current profile applies
   einsums tune --import profile.json    # adopt one measured on another machine of a kind
   einsums tune --export                 # the path of the profile in effect

An imported file is filed under the machine it describes, so on a cluster whose
home directories are shared, a login node can import the compute nodes'
profile and only the compute nodes use it.

It measures GEMM rates on a grid of shapes, every M, N and K from a log-spaced range (default 16 to 2048 in 8 steps, set by ``--min-size``, ``--max-size`` and ``--num-points``) up to 2\ :sup:`30` multiply-adds, once through the string einsum and once through PackedGemm pinned to its tile kernel.
Each extent is moved just below its power of two (508 rather than 512): a power-of-two leading dimension makes the packers' reads collide in the cache, and PackedGemm runs 30 to 40% slower there than at the extents around it.
It measures the scatter's extra cost per element by timing the CCSD ladder's tile ``a,b,e,f <- Q,a,e ; Q,b,f`` against the same contraction writing ``a,e,b,f`` in place, at runs from 4 to 1024, and the packers' the same way with an operand split at runs from 2 to 32.
It also measures copy and per-cache-level bandwidth, permute throughput, the string einsum's call overhead, per-kernel thread-efficiency curves (skipped with ``--no-thread-sweep``) and the OpenMP region cost at every team size.
It starts from the built-in table rather than from an earlier profile, so nothing it does not measure is inherited from one.
A profile written before the scatter and pack costs were measured keeps its own rates and takes the built-in table's curves, with a warning to measure again.

Then it measures the thresholds that decide between two routes and were set on
one machine: the outer-product floor, the outer product's GEMM-to-ger limit,
the stream minimum, the stream's elements per thread, and the parallel
break-even rate. Each runs the decision's own shape through ``einsum`` with the
threshold forced to either extreme, checks that the two sides took different
routes, and records where one overtakes the other; a probe that cannot separate
its routes says so and writes nothing. ``--quick`` skips this part.

Everything goes into one file, whose ``cost_model`` section is this page's
model, whose ``options`` section holds tuned option values, applied below the
environment and the command line so anything a user sets still wins, and whose
other sections belong to the Hardware module; see
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
