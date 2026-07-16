//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <cstdint>
#include <optional>
#include <string_view>

namespace einsums::simd {

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
 * @versionadded{2.1.0}
 */
struct CpuFeatures {
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
};

/**
 * @brief Detect the features of the CPU this process is running on.
 *
 * The detection runs once (thread-safe, on first call) and the result is
 * cached for the lifetime of the process.
 *
 * @return A reference to the cached feature set.
 *
 * @versionadded{2.1.0}
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
 * On aarch64 only `Baseline` exists today; NEON *is* the baseline there.
 * Runtime rungs for optional aarch64 features (FEAT_FP16, FEAT_BF16) are
 * future extensions of this enum. Rungs are ordered: a larger enumerator
 * value strictly implies every feature of the smaller ones, so ordinary
 * comparison operators express "at least" relationships.
 *
 * @versionadded{2.1.0}
 */
enum class InstructionSet : std::uint8_t {
    Baseline = 0, ///< Toolchain default target: SSE2 on x86-64, NEON on aarch64.
    V2       = 1, ///< x86-64-v2 (SSE4.2 era).
    V3       = 2, ///< x86-64-v3 (AVX2 + FMA era).
    V4       = 3, ///< x86-64-v4 (AVX-512 era).
};

/**
 * @brief Human-readable name of a rung: "baseline", "x86-64-v2", ...
 *
 * @param[in] set The rung to name.
 *
 * @return A static string; never nullptr.
 *
 * @versionadded{2.1.0}
 */
EINSUMS_EXPORT char const *to_string(InstructionSet set);

/**
 * @brief Parse a rung name, as accepted by the `EINSUMS_SIMD_ARCH`
 *        environment variable.
 *
 * Accepted spellings (case-insensitive): `baseline`, `v2`, `v3`, `v4`,
 * `x86-64-v2/-v3/-v4`, and the colloquial aliases `sse2` (baseline),
 * `sse4.2` (v2), `avx2` (v3), `avx512` (v4).
 *
 * @param[in] name The spelling to parse.
 *
 * @return The rung, or std::nullopt if the spelling is not recognized.
 *
 * @versionadded{2.1.0}
 */
EINSUMS_EXPORT std::optional<InstructionSet> parse_instruction_set(std::string_view name);

/**
 * @brief The highest rung this CPU can execute.
 *
 * Pure function of the feature set: applies the full psABI gate for each
 * level (all listed extensions must be present, including the OS-state
 * gates folded into CpuFeatures). Useful directly in tests; most callers
 * want selected_arch() instead.
 *
 * @param[in] features The feature set to classify.
 *
 * @return The highest rung whose complete feature list is present.
 *
 * @versionadded{2.1.0}
 */
EINSUMS_EXPORT InstructionSet highest_supported(CpuFeatures const &features);

/**
 * @brief Resolve the rung to dispatch to, given a feature set and an
 *        optional override spelling.
 *
 * The override (normally the `EINSUMS_SIMD_ARCH` environment variable) can
 * only lower the rung: an override above what the hardware supports is
 * clamped to the hardware's ceiling with a logged warning, and an
 * unparseable override is ignored with a logged warning. This is the pure,
 * deterministic core of selected_arch(), separated so tests can drive it
 * with synthetic feature sets and override strings.
 *
 * @param[in] features The detected (or synthetic) feature set.
 * @param[in] override_name Optional rung spelling; pass std::nullopt for "no override".
 *
 * @return The rung to dispatch to.
 *
 * @versionadded{2.1.0}
 */
EINSUMS_EXPORT InstructionSet resolve_arch(CpuFeatures const &features, std::optional<std::string_view> override_name);

/**
 * @brief The rung the current process dispatches to.
 *
 * Equivalent to `resolve_arch(cpu_features(), <EINSUMS_SIMD_ARCH env var>)`,
 * computed once (thread-safe) and cached for the lifetime of the process.
 * Because the result is cached, the environment variable must be set before
 * the first call anywhere in the process; changing it afterwards has no
 * effect. Test code that needs different rungs should call resolve_arch()
 * directly instead of mutating the environment.
 *
 * @return The cached rung.
 *
 * @versionadded{2.1.0}
 */
EINSUMS_EXPORT InstructionSet selected_arch();

/**
 * @brief Pick the best available entry point for the selected rung.
 *
 * Generic dispatch helper for modules that compile a kernel once per rung:
 * pass one entry point per rung (nullptr for rungs the module does not
 * build) and get back the entry for the highest built rung at or below
 * selected_arch(). Falls down the ladder through nullptr entries, so a
 * module may build any subset of rungs; `baseline` must always be provided.
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
 *
 * @return The entry point to call; never nullptr.
 *
 * @versionadded{2.1.0}
 */
template <typename F>
F select(F baseline, F v2 = nullptr, F v3 = nullptr, F v4 = nullptr) {
    F const ladder[] = {baseline, v2, v3, v4};
    for (int rung = static_cast<int>(selected_arch()); rung > 0; --rung) {
        if (ladder[rung] != nullptr) {
            return ladder[rung];
        }
    }
    return baseline;
}

} // namespace einsums::simd
