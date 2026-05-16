.. Copyright (c) The Einsums Developers. All rights reserved.
   Licensed under the MIT License. See LICENSE.txt in the project root for license information.

===================
ComputeGraph Module
===================

The ComputeGraph module provides a deferred-execution computation graph for Einsums,
inspired by CUDA Graphs and PyTorch FX. It enables capturing, optimizing, and replaying
sequences of tensor operations with transparent GPU offloading and distributed computing.

.. toctree::
   :maxdepth: 2

   getting_started
   string_einsum
   operations
   pipeline
   control_flow
   executor
   workspace
   distributed
   hardware_profiles
   optimization_passes
   gemm_batching
   symmetry
   tensor_lifetime
   views
   rebind
   api_reference
