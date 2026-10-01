//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Config/Namespace.hpp>

#include <cstdint>
#include <optional>
#include <span>
#include <string_view>

EINSUMS_NAMESPACE_BEGIN(simd)

/**
 * @brief The instruction-set family a CpuFeatures describes.
 *
 * Each family has its own dispatch ladder (see preference_order()), so a rung
 * of one family is never considered on a machine of another.
 *
 * @versionadded{2.0.0}
 */
enum class Architecture : std::uint8_t {
    Other   = 0, ///< Neither x86 nor aarch64: only the Baseline rung exists.
    X86     = 1, ///< x86 / x86-64: the psABI ladder Baseline, V2, V3, V4.
    Aarch64 = 2, ///< aarch64: Baseline (NEON) and the optional Sme rung.
};

/**
 * @brief CPU features detected at runtime.
 *
 * This is the runtime counterpart of the compile-time `has_*` constants in
 * Platform.hpp: the same feature vocabulary, but describing the machine the
 * process is actually running on rather than the ISA baseline the translation
 * unit was compiled for. Field names mirror the `has_*` constants.
 *
 * All x86 vector-extension fields report *usability*, not just CPU support:
 * a feature is only reported when the CPU advertises it via CPUID *and* the
 * operating system has enabled the corresponding register state (OSXSAVE +
 * XCR0 bits, checked with `xgetbv`). A CPU with AVX2 silicon under an OS
 * that never enabled YMM state reports `avx2 == false`, because issuing an
 * AVX2 instruction there faults. This is the gate most hand-rolled
 * detectors forget.
 *
 * On non-x86, non-aarch64 targets every field is false; such targets run the
 * baseline (scalar) code path.
 *
 * @versionadded{2.0.0}
 */
struct CpuFeatures {
    /// The family of the machine; detect() sets it, and it picks the ladder the other fields are read against.
    Architecture arch = Architecture::Other;

    // ---- x86 ----
    bool sse2  = false; ///< SSE2 (part of the x86-64 baseline; true on every x86-64 CPU).
    bool sse3  = false; ///< SSE3.
    bool ssse3 = false; ///< Supplemental SSE3.
    bool sse41 = false; ///< SSE4.1.
    bool sse42 = false; ///< SSE4.2.

    bool popcnt     = false; ///< POPCNT instruction.
    bool cmpxchg16b = false; ///< CMPXCHG16B instruction.
    bool lahf_sahf  = false; ///< LAHF/SAHF in 64-bit mode.
    bool bmi1       = false; ///< Bit-manipulation instructions 1.
    bool bmi2       = false; ///< Bit-manipulation instructions 2.
    bool f16c       = false; ///< FP16 <-> FP32 conversion instructions.
    bool lzcnt      = false; ///< LZCNT instruction.
    bool movbe      = false; ///< MOVBE instruction.

    bool avx  = false; ///< AVX, gated on OS YMM state (see class docs).
    bool avx2 = false; ///< AVX2, gated on OS YMM state.
    bool fma  = false; ///< FMA3, gated on OS YMM state.

    bool avx512f  = false; ///< AVX-512 Foundation, gated on OS ZMM state.
    bool avx512vl = false; ///< AVX-512 Vector Length extensions, gated on OS ZMM state.
    bool avx512bw = false; ///< AVX-512 Byte/Word, gated on OS ZMM state.
    bool avx512dq = false; ///< AVX-512 Doubleword/Quadword, gated on OS ZMM state.
    bool avx512cd = false; ///< AVX-512 Conflict Detection, gated on OS ZMM state.

    bool avx512_fp16 = false; ///< AVX-512 FP16 arithmetic, gated on OS ZMM state.
    bool avx512_bf16 = false; ///< AVX-512 BF16 arithmetic, gated on OS ZMM state.
    bool avx_vnni    = false; ///< 256-bit VNNI (Alder Lake+), gated on OS YMM state.
    bool avx512_vnni = false; ///< 512-bit VNNI, gated on OS ZMM state.

    bool osxsave   = false; ///< OS advertises XSAVE/XRSTOR support (CPUID.1:ECX.OSXSAVE).
    bool os_avx    = false; ///< OS enabled XMM+YMM state in XCR0; prerequisite for all AVX-family reports.
    bool os_avx512 = false; ///< OS enabled opmask+ZMM state in XCR0; prerequisite for all AVX-512 reports.

    // ---- ARM ----
    bool neon         = false; ///< Advanced SIMD (baseline on aarch64; true on every aarch64 CPU).
    bool neon_fp16    = false; ///< FEAT_FP16: native half-precision vector arithmetic.
    bool neon_bf16    = false; ///< FEAT_BF16: bfloat16 vector instructions.
    bool neon_i8mm    = false; ///< FEAT_I8MM: int8 matrix-multiply instructions.
    bool neon_dotprod = false; ///< FEAT_DotProd: vdotq int8 dot product.

    bool sve  = false; ///< FEAT_SVE: non-streaming Scalable Vector Extension.
    bool sve2 = false; ///< FEAT_SVE2: non-streaming SVE2.

    bool sme        = false; ///< FEAT_SME: Scalable Matrix Extension (streaming SVE + ZA tiles).
    bool sme2       = false; ///< FEAT_SME2: SME2 (multi-vector, required by the sme rung).
    bool sme_f64f64 = false; ///< FEAT_SME_F64F64: FP64 outer-product FMOPA into ZA64 tiles.
};

/**
 * @brief Detect the features of the CPU this process is running on.
 *
 * The detection runs once (thread-safe, on first call) and the result is
 * cached for the lifetime of the process.
 *
 * @return A reference to the cached feature set.
 *
 * @versionadded{2.0.0}
 */
EINSUMS_EXPORT CpuFeatures const &cpu_features();

/**
 * @brief The rungs of the runtime dispatch ladder.
 *
 * The x86 rungs follow the psABI micro-architecture levels
 * (`x86-64-v2/-v3/-v4`), which are the industry-standard grouping of ISA
 * extensions and map one-to-one onto compiler flags (`-march=x86-64-v3`,
 * MSVC `/arch:AVX2`, ...):
 *
 * - `Baseline`: what the toolchain's default target provides. SSE2 on
 *   x86-64, NEON on aarch64. Always runnable, by construction.
 * - `V2` (x86-64-v2): SSE3/SSSE3/SSE4.1/SSE4.2, POPCNT, CMPXCHG16B
 *   (Nehalem 2008 / Jaguar 2013 and newer).
 * - `V3` (x86-64-v3): adds AVX/AVX2, FMA, BMI1/BMI2, F16C, LZCNT, MOVBE
 *   (Haswell 2013 / Excavator 2015 and newer).
 * - `V4` (x86-64-v4): adds AVX-512 F/BW/CD/DQ/VL
 *   (Skylake-X 2017 / Zen 4 2022 and newer).
 *
 * On aarch64 `Baseline` is NEON, and `Sme` is the one optional rung: SME2
 * with FP64 outer products (FEAT_SME2 + FEAT_SME_F64F64, Apple M4 and
 * newer). Runtime rungs for further aarch64 features (FEAT_FP16,
 * FEAT_BF16) are future extensions of this enum.
 *
 * The enumerator values are identifiers, not a ranking. Whether a machine
 * can run a rung is supports(), and the order in which rungs are preferred
 * is preference_order(), which is per architecture: aarch64 features do not
 * nest the way the x86 psABI levels do (Apple M4 has SME without SVE), so
 * no single ordinal can say which rung "implies" another.
 *
 * @versionadded{2.0.0}
 */
enum class InstructionSet : std::uint8_t {
    Baseline = 0, ///< Toolchain default target: SSE2 on x86-64, NEON on aarch64.
    V2       = 1, ///< x86-64-v2 (SSE4.2 era).
    V3       = 2, ///< x86-64-v3 (AVX2 + FMA era).
    V4       = 3, ///< x86-64-v4 (AVX-512 era).
    Sme      = 4, ///< aarch64 SME2 + FP64 FMOPA (Apple M4 era).
};

/**
 * @brief Human-readable name of a rung: "baseline", "x86-64-v2", ...
 *
 * @param[in] set The rung to name.
 *
 * @return A static string; never nullptr.
 *
 * @versionadded{2.0.0}
 */
EINSUMS_EXPORT char const *to_string(InstructionSet set);

/**
 * @brief Width in bits of the vector register a rung's kernels are written for.
 *
 * Baseline and V2 are SSE2/SSE4.2 (128), V3 is AVX2 (256), V4 is AVX-512 (512).
 * The aarch64 rungs report NEON's 128: the SME rung's ZA tiles are a matrix
 * unit with its own geometry (see PackedGemm's SME kernel), not a wider
 * vector, and its NEON-side code is still 128-bit.
 *
 * This is the one place the rung-to-width mapping lives; hardware::cpu_info()
 * derives `simd_width_f64` from it so that blocking built from that field
 * agrees with the kernel ladder's choice.
 *
 * @param[in] set The rung.
 *
 * @return 128, 256 or 512.
 *
 * @versionadded{2.0.0}
 */
EINSUMS_EXPORT int vector_bits(InstructionSet set);

/**
 * @brief Parse a rung name, as accepted by the `--einsums:simd:arch`
 *        option.
 *
 * Accepted spellings (case-insensitive): `baseline`, `v2`, `v3`, `v4`,
 * `x86-64-v2/-v3/-v4`, `sme`, and the colloquial aliases `sse2` (baseline),
 * `sse4.2` (v2), `avx2` (v3), `avx512` (v4), `sme2` (sme).
 *
 * @param[in] name The spelling to parse.
 *
 * @return The rung, or std::nullopt if the spelling is not recognized.
 *
 * @versionadded{2.0.0}
 */
EINSUMS_EXPORT std::optional<InstructionSet> parse_instruction_set(std::string_view name);

/**
 * @brief The rungs of one architecture's ladder, most preferred first.
 *
 * x86 is `{V4, V3, V2, Baseline}` and aarch64 is `{Sme, Baseline}`; every
 * other architecture has only `{Baseline}`. Every list ends in Baseline.
 * Dispatch walks this list, never the enumerator values.
 *
 * @param[in] arch The architecture whose ladder to return.
 *
 * @return A view of a static array; valid for the lifetime of the process.
 *
 * @versionadded{2.0.0}
 */
EINSUMS_EXPORT std::span<InstructionSet const> preference_order(Architecture arch);

/**
 * @brief Whether a machine with @p features can execute code compiled for @p set.
 *
 * Pure function of the feature set. A rung of another architecture is never
 * supported. The x86 levels apply the full psABI gate (every listed
 * extension, including the OS-state gates folded into CpuFeatures). The
 * `Sme` rung needs SME2 with FP64 outer products, and also whatever else the
 * compiler enables for the rung's translation units: a compiler that turns
 * on non-streaming SVE or SVE2 with `+sme2` (GCC before 15) may emit it
 * anywhere in those TUs, so for such a build the rung needs SVE or SVE2 too.
 *
 * @param[in] features The feature set to test.
 * @param[in] set The rung.
 *
 * @return True when every feature the rung's code may use is present.
 *
 * @versionadded{2.0.0}
 */
EINSUMS_EXPORT bool supports(CpuFeatures const &features, InstructionSet set);

/**
 * @brief The most preferred rung this CPU can execute.
 *
 * The first entry of `preference_order(features.arch)` that supports()
 * accepts. Useful directly in tests; most callers want selected_arch()
 * instead.
 *
 * @param[in] features The feature set to classify.
 *
 * @return The best supported rung; Baseline when nothing else qualifies.
 *
 * @versionadded{2.0.0}
 */
EINSUMS_EXPORT InstructionSet highest_supported(CpuFeatures const &features);

/**
 * @brief Resolve the rung to dispatch to, given a feature set and an
 *        optional override spelling.
 *
 * The override (normally the `--einsums:simd:arch` option) can
 * only choose a rung the machine supports. A supported rung is used as
 * given. A rung of this architecture that the machine cannot run is
 * replaced, with a logged warning, by the next supported rung after it in
 * preference_order(), so asking for `v4` on an AVX2 machine gives `v3`. A
 * rung of another architecture, or an unparseable spelling, is ignored with
 * a logged warning. This is the pure, deterministic core of selected_arch(),
 * separated so tests can drive it with synthetic feature sets and override
 * strings.
 *
 * @param[in] features The detected (or synthetic) feature set.
 * @param[in] override_name Optional rung spelling; pass std::nullopt for "no override".
 *
 * @return The rung to dispatch to.
 *
 * @versionadded{2.0.0}
 */
EINSUMS_EXPORT InstructionSet resolve_arch(CpuFeatures const &features, std::optional<std::string_view> override_name);

/**
 * @brief The rung the current process dispatches to.
 *
 * Equivalent to `resolve_arch(cpu_features(), <--einsums:simd:arch>)`,
 * computed once (thread-safe) and cached for the lifetime of the process.
 * Because the result is cached, the option must be parsed before the first
 * call anywhere in the process, on the command line or in the environment;
 * changing it afterwards has no effect. Test code that needs different rungs
 * should call resolve_arch() directly instead of setting the option.
 *
 * @return The cached rung.
 *
 * @versionadded{2.0.0}
 */
EINSUMS_EXPORT InstructionSet selected_arch();

namespace dispatch_detail {
/// Position of @p set in the argument list select() takes.
constexpr int ladder_slot(InstructionSet set) noexcept {
    switch (set) {
    case InstructionSet::V2:
        return 1;
    case InstructionSet::V3:
        return 2;
    case InstructionSet::V4:
        return 3;
    case InstructionSet::Sme:
        return 4;
    case InstructionSet::Baseline:
        break;
    }
    return 0;
}
} // namespace dispatch_detail

/**
 * @brief select() against an explicit feature set and starting rung.
 *
 * Walks `preference_order(features.arch)` from @p start onward and returns
 * the first entry that was built (not nullptr) and that supports() accepts.
 * The support test matters on aarch64, where a rung later in the list is not
 * implied by an earlier one. select() calls this with cpu_features() and
 * selected_arch(); tests call it directly with synthetic machines.
 *
 * @param[in] features The machine to dispatch for.
 * @param[in] start The rung to start from, normally selected_arch().
 * @param[in] baseline Entry point for the Baseline rung; must not be nullptr.
 * @param[in] v2 Entry point for the V2 rung, or nullptr if not built.
 * @param[in] v3 Entry point for the V3 rung, or nullptr if not built.
 * @param[in] v4 Entry point for the V4 rung, or nullptr if not built.
 * @param[in] sme Entry point for the Sme rung, or nullptr if not built.
 *
 * @return The entry point to call; never nullptr.
 *
 * @versionadded{2.0.0}
 */
template <typename F>
F select_for(CpuFeatures const &features, InstructionSet start, F baseline, F v2 = nullptr, F v3 = nullptr, F v4 = nullptr,
             F sme = nullptr) {
    F const slots[] = {baseline, v2, v3, v4, sme};
    bool    reached = false;
    for (InstructionSet const rung : preference_order(features.arch)) {
        reached = reached || rung == start;
        if (!reached) {
            continue;
        }
        F const entry = slots[dispatch_detail::ladder_slot(rung)];
        if (entry != nullptr && supports(features, rung)) {
            return entry;
        }
    }
    return baseline;
}

/**
 * @brief Pick the best available entry point for the selected rung.
 *
 * Generic dispatch helper for modules that compile a kernel once per rung:
 * pass one entry point per rung (nullptr for rungs the module does not
 * build) and get back the entry for the most preferred built rung at or
 * after selected_arch() in preference_order(). Falls through nullptr
 * entries, so a module may build any subset of rungs; `baseline` must
 * always be provided.
 *
 * @code
 * using KernelFn = void (*)(float const *, float *, std::size_t);
 * static KernelFn const kernel = einsums::simd::select<KernelFn>(
 *     &arch_baseline::kernel, &arch_v2::kernel, &arch_v3::kernel, &arch_v4::kernel);
 * @endcode
 *
 * @param[in] baseline Entry point for the Baseline rung; must not be nullptr.
 * @param[in] v2 Entry point for the V2 rung, or nullptr if not built.
 * @param[in] v3 Entry point for the V3 rung, or nullptr if not built.
 * @param[in] v4 Entry point for the V4 rung, or nullptr if not built.
 * @param[in] sme Entry point for the Sme rung, or nullptr if not built.
 *
 * @return The entry point to call; never nullptr.
 *
 * @versionadded{2.0.0}
 */
template <typename F>
F select(F baseline, F v2 = nullptr, F v3 = nullptr, F v4 = nullptr, F sme = nullptr) {
    return select_for<F>(cpu_features(), selected_arch(), baseline, v2, v3, v4, sme);
}

EINSUMS_NAMESPACE_END(simd)
