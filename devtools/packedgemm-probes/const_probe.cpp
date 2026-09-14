//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/PackedGemm/MicroKernel.hpp>
#include <Einsums/PackedGemm/Packing.hpp>
#include <Einsums/Runtime.hpp>

#include <cstdio>
int main(int argc, char **argv) {
    einsums::initialize(argc, argv);
    auto const &c = einsums::packed_gemm::cpu_config();
    std::printf("cpu_config: MR=%d NR=%d L1=%lld L2=%lld L3=%lld min_parallel_flops=%lld\n", c.MR, c.NR, (long long)c.l1_cache_size,
                (long long)c.l2_cache_size, (long long)c.l3_cache_size, (long long)c.min_parallel_flops);
    for (int es : {4, 8}) {
        auto b = einsums::packed_gemm::compute_blocking(es);
        std::printf("blocking(elem=%d): MC=%lld NC=%lld KC=%lld NR=%lld\n", es, (long long)b.MC, (long long)b.NC, (long long)b.KC,
                    (long long)b.NR);
    }
    auto sf = einsums::packed_gemm::micro_kernel_shape<float>();
    auto sd = einsums::packed_gemm::micro_kernel_shape<double>();
    std::printf("shape<float>: mr=%d nr=%d kc=%lld fast_scatter=%d block_gemm=%d\n", sf.mr, sf.nr, (long long)sf.kc, sf.fast_scatter,
                sf.block_gemm);
    std::printf("shape<double>: mr=%d nr=%d kc=%lld fast_scatter=%d block_gemm=%d\n", sd.mr, sd.nr, (long long)sd.kc, sd.fast_scatter,
                sd.block_gemm);
    einsums::finalize();
}
