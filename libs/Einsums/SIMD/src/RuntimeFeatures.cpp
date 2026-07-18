//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Logging.hpp>
#include <Einsums/SIMD/RuntimeFeatures.hpp>

#include <cstdlib>
#include <string>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#    define EINSUMS_SIMD_DETECT_X86 1
#    if defined(_MSC_VER)
#        include <intrin.h>
#    else
#        include <cpuid.h>
#    endif
#elif defined(__aarch64__) || defined(_M_ARM64)
#    define EINSUMS_SIMD_DETECT_AARCH64 1
#    if defined(__APPLE__)
#        include <sys/sysctl.h>
#    elif defined(__linux__)
#        include <sys/auxv.h>
#    endif
#endif

namespace einsums::simd {

namespace {

#if defined(EINSUMS_SIMD_DETECT_X86)

struct CpuidRegs {
    std::uint32_t eax = 0, ebx = 0, ecx = 0, edx = 0;
};

CpuidRegs cpuid(std::uint32_t leaf, std::uint32_t subleaf) {
    CpuidRegs r;
#    if defined(_MSC_VER)
    int out[4];
    __cpuidex(out, static_cast<int>(leaf), static_cast<int>(subleaf));
    r.eax = static_cast<std::uint32_t>(out[0]);
    r.ebx = static_cast<std::uint32_t>(out[1]);
    r.ecx = static_cast<std::uint32_t>(out[2]);
    r.edx = static_cast<std::uint32_t>(out[3]);
#    else
    __get_cpuid_count(leaf, subleaf, &r.eax, &r.ebx, &r.ecx, &r.edx);
#    endif
    return r;
}

// XGETBV reads the extended control register holding the OS-enabled state
// mask. The compiler intrinsic requires compiling the TU with XSAVE support
// (-mxsave), which would defeat the whole point of keeping this file at the
// baseline ISA, so use inline assembly on non-MSVC compilers. Only call
// when CPUID reports OSXSAVE, otherwise the instruction faults.
std::uint64_t xgetbv0() {
#    if defined(_MSC_VER)
    return _xgetbv(0);
#    else
    std::uint32_t eax = 0, edx = 0;
    __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
    return (static_cast<std::uint64_t>(edx) << 32) | eax;
#    endif
}

bool bit(std::uint32_t reg, int idx) {
    return (reg >> idx) & 1U;
}

CpuFeatures detect() {
    CpuFeatures f;

    CpuidRegs const id0 = cpuid(0, 0);
    if (id0.eax < 1) {
        return f;
    }

    CpuidRegs const id1 = cpuid(1, 0);
    f.sse2              = bit(id1.edx, 26);
    f.sse3              = bit(id1.ecx, 0);
    f.ssse3             = bit(id1.ecx, 9);
    f.sse41             = bit(id1.ecx, 19);
    f.sse42             = bit(id1.ecx, 20);
    f.popcnt            = bit(id1.ecx, 23);
    f.cmpxchg16b        = bit(id1.ecx, 13);
    f.movbe             = bit(id1.ecx, 22);
    f.f16c              = bit(id1.ecx, 29);
    f.osxsave           = bit(id1.ecx, 27);

    // OS state gates. XCR0 bit 1 = XMM, bit 2 = YMM, bits 5-7 = opmask +
    // upper halves of ZMM0-15 + ZMM16-31. AVX-family instructions fault
    // unless the OS saves/restores the corresponding state on context
    // switch, so features below are reported as usable only behind these
    // gates (see CpuFeatures docs).
    if (f.osxsave) {
        std::uint64_t const xcr0 = xgetbv0();
        f.os_avx                 = (xcr0 & 0x06) == 0x06;
        f.os_avx512              = f.os_avx && (xcr0 & 0xE0) == 0xE0;
    }

    f.avx = f.os_avx && bit(id1.ecx, 28);
    f.fma = f.os_avx && bit(id1.ecx, 12);

    if (id0.eax >= 7) {
        CpuidRegs const id7 = cpuid(7, 0);
        f.bmi1              = bit(id7.ebx, 3);
        f.bmi2              = bit(id7.ebx, 8);
        f.avx2              = f.os_avx && bit(id7.ebx, 5);
        f.avx512f           = f.os_avx512 && bit(id7.ebx, 16);
        f.avx512dq          = f.os_avx512 && bit(id7.ebx, 17);
        f.avx512cd          = f.os_avx512 && bit(id7.ebx, 28);
        f.avx512bw          = f.os_avx512 && bit(id7.ebx, 30);
        f.avx512vl          = f.os_avx512 && bit(id7.ebx, 31);
        f.avx512_vnni       = f.os_avx512 && bit(id7.ecx, 11);
        f.avx512_fp16       = f.os_avx512 && bit(id7.edx, 23);

        if (id7.eax >= 1) {
            CpuidRegs const id7_1 = cpuid(7, 1);
            f.avx_vnni            = f.os_avx && bit(id7_1.eax, 4);
            f.avx512_bf16         = f.os_avx512 && bit(id7_1.eax, 5);
        }
    }

    CpuidRegs const ext0 = cpuid(0x80000000U, 0);
    if (ext0.eax >= 0x80000001U) {
        CpuidRegs const ext1 = cpuid(0x80000001U, 0);
        f.lahf_sahf          = bit(ext1.ecx, 0);
        f.lzcnt              = bit(ext1.ecx, 5);
    }

    return f;
}

#elif defined(EINSUMS_SIMD_DETECT_AARCH64)

#    if defined(__APPLE__)

bool sysctl_flag(char const *name) {
    int    value = 0;
    size_t size  = sizeof(value);
    if (sysctlbyname(name, &value, &size, nullptr, 0) != 0) {
        return false;
    }
    return value != 0;
}

CpuFeatures detect() {
    CpuFeatures f;
    f.neon         = true;
    f.neon_fp16    = sysctl_flag("hw.optional.arm.FEAT_FP16");
    f.neon_bf16    = sysctl_flag("hw.optional.arm.FEAT_BF16");
    f.neon_i8mm    = sysctl_flag("hw.optional.arm.FEAT_I8MM");
    f.neon_dotprod = sysctl_flag("hw.optional.arm.FEAT_DotProd");
    f.sme          = sysctl_flag("hw.optional.arm.FEAT_SME");
    f.sme2         = sysctl_flag("hw.optional.arm.FEAT_SME2");
    f.sme_f64f64   = sysctl_flag("hw.optional.arm.FEAT_SME_F64F64");
    return f;
}

#    elif defined(__linux__)

CpuFeatures detect() {
    CpuFeatures f;
    f.neon = true;

    unsigned long const hwcap  = getauxval(AT_HWCAP);
#        if defined(HWCAP_ASIMDHP)
    f.neon_fp16                = (hwcap & HWCAP_ASIMDHP) != 0;
#        endif
#        if defined(HWCAP_ASIMDDP)
    f.neon_dotprod             = (hwcap & HWCAP_ASIMDDP) != 0;
#        endif

#        if defined(AT_HWCAP2)
    unsigned long const hwcap2 = getauxval(AT_HWCAP2);
#            if defined(HWCAP2_BF16)
    f.neon_bf16                = (hwcap2 & HWCAP2_BF16) != 0;
#            endif
#            if defined(HWCAP2_I8MM)
    f.neon_i8mm                = (hwcap2 & HWCAP2_I8MM) != 0;
#            endif
#            if defined(HWCAP2_SME)
    f.sme                      = (hwcap2 & HWCAP2_SME) != 0;
#            endif
#            if defined(HWCAP2_SME2)
    f.sme2                     = (hwcap2 & HWCAP2_SME2) != 0;
#            endif
#            if defined(HWCAP2_SME_F64F64)
    f.sme_f64f64               = (hwcap2 & HWCAP2_SME_F64F64) != 0;
#            endif
#        endif

    return f;
}

#    else

// aarch64 on an OS without a feature-query interface we support: NEON is
// architecturally guaranteed, the optional features stay off.
CpuFeatures detect() {
    CpuFeatures f;
    f.neon = true;
    return f;
}

#    endif

#else

// Neither x86-64 nor aarch64: no runtime SIMD features; callers run the
// baseline (scalar) path.
CpuFeatures detect() {
    return {};
}

#endif

} // namespace

CpuFeatures const &cpu_features() {
    static CpuFeatures const features = detect();
    return features;
}

char const *to_string(InstructionSet set) {
    switch (set) {
    case InstructionSet::Baseline:
        return "baseline";
    case InstructionSet::V2:
        return "x86-64-v2";
    case InstructionSet::V3:
        return "x86-64-v3";
    case InstructionSet::V4:
        return "x86-64-v4";
    case InstructionSet::Sme:
        return "sme";
    }
    return "baseline";
}

std::optional<InstructionSet> parse_instruction_set(std::string_view name) {
    std::string lowered(name);
    for (char &c : lowered) {
        if (c >= 'A' && c <= 'Z') {
            c = static_cast<char>(c - 'A' + 'a');
        }
    }

    if (lowered == "baseline" || lowered == "sse2") {
        return InstructionSet::Baseline;
    }
    if (lowered == "v2" || lowered == "x86-64-v2" || lowered == "sse4.2") {
        return InstructionSet::V2;
    }
    if (lowered == "v3" || lowered == "x86-64-v3" || lowered == "avx2") {
        return InstructionSet::V3;
    }
    if (lowered == "v4" || lowered == "x86-64-v4" || lowered == "avx512") {
        return InstructionSet::V4;
    }
    if (lowered == "sme" || lowered == "sme2") {
        return InstructionSet::Sme;
    }
    return std::nullopt;
}

InstructionSet highest_supported(CpuFeatures const &f) {
    // aarch64 ladder: the sme rung requires SME2 with FP64 outer products,
    // since the rung's TUs are compiled with +sme2+sme-f64f64 and may emit
    // any of it anywhere.
    if (f.sme && f.sme2 && f.sme_f64f64) {
        return InstructionSet::Sme;
    }

    // Full psABI gates: every extension of a level must be present for the
    // level to qualify, because a compiler told -march=x86-64-v3 may emit
    // any of them anywhere in the TU.
    bool const v2 = f.sse3 && f.ssse3 && f.sse41 && f.sse42 && f.popcnt && f.cmpxchg16b && f.lahf_sahf;
    bool const v3 = v2 && f.avx && f.avx2 && f.fma && f.bmi1 && f.bmi2 && f.f16c && f.lzcnt && f.movbe && f.os_avx;
    bool const v4 = v3 && f.avx512f && f.avx512bw && f.avx512cd && f.avx512dq && f.avx512vl && f.os_avx512;

    if (v4) {
        return InstructionSet::V4;
    }
    if (v3) {
        return InstructionSet::V3;
    }
    if (v2) {
        return InstructionSet::V2;
    }
    return InstructionSet::Baseline;
}

InstructionSet resolve_arch(CpuFeatures const &features, std::optional<std::string_view> override_name) {
    InstructionSet const ceiling = highest_supported(features);

    if (!override_name.has_value()) {
        return ceiling;
    }

    auto const requested = parse_instruction_set(*override_name);
    if (!requested.has_value()) {
        EINSUMS_LOG_WARN("EINSUMS_SIMD_ARCH=\"{}\" is not a recognized instruction-set name; ignoring the override. "
                         "Accepted: baseline, v2, v3, v4 (aliases: sse2, sse4.2, avx2, avx512).",
                         *override_name);
        return ceiling;
    }

    if (*requested > ceiling) {
        EINSUMS_LOG_WARN("EINSUMS_SIMD_ARCH requests {} but this CPU/OS only supports {}; clamping to {}.", to_string(*requested),
                         to_string(ceiling), to_string(ceiling));
        return ceiling;
    }

    return *requested;
}

InstructionSet selected_arch() {
    static InstructionSet const selected = [] {
        std::optional<std::string_view> override_name;
        if (char const *env = std::getenv("EINSUMS_SIMD_ARCH"); env != nullptr && *env != '\0') {
            override_name = env;
        }
        InstructionSet const arch = resolve_arch(cpu_features(), override_name);
        EINSUMS_LOG_DEBUG("SIMD dispatch rung: {}", to_string(arch));
        return arch;
    }();
    return selected;
}

} // namespace einsums::simd
