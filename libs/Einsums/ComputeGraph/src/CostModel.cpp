//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/CostModel.hpp>
#include <Einsums/ComputeGraph/Detail/Json.hpp>
#include <Einsums/ComputeGraph/Options.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Errors.hpp>
#include <Einsums/GPU/Platform.hpp>
#include <Einsums/GPU/Runtime.hpp>
#include <Einsums/Hardware/CpuInfo.hpp>
#include <Einsums/Logging.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <map>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#ifdef __APPLE__
#    include <sys/sysctl.h>
#endif

EINSUMS_NAMESPACE_BEGIN(compute_graph)

// ═══════════════════════════════════════════════════════════════════════════════
// DeviceProfileDB: runtime detection
// ═══════════════════════════════════════════════════════════════════════════════

std::string DeviceProfileDB::detect_cpu_brand() {
#ifdef __APPLE__
    char   buf[256] = {}; // NOLINT
    size_t len      = sizeof(buf);
    if (sysctlbyname("machdep.cpu.brand_string", buf, &len, nullptr, 0) == 0) {
        return {buf};
    }
#elif defined(__x86_64__) || defined(_M_X64)
    // CPUID brand string: functions 0x80000002 - 0x80000004
    char     brand[49] = {};
    unsigned regs[4];
    for (int i = 0; i < 3; i++) {
        __asm__ __volatile__("cpuid" : "=a"(regs[0]), "=b"(regs[1]), "=c"(regs[2]), "=d"(regs[3]) : "a"(0x80000002 + i), "c"(0));
        std::memcpy(brand + i * 16, regs, 16);
    }
    return std::string(brand);
#elif defined(__linux__)
    // Non-x86 Linux, in practice aarch64: there is no CPUID, and an ARM
    // /proc/cpuinfo carries no "model name" line - the CPU is identified by
    // numeric implementer and part codes instead. Returning "Unknown CPU" here
    // left every aarch64 Linux run without a device profile, matching nothing
    // in the database and silently costing against a generic CPU.
    //
    // A hypervisor commonly masks the part number (Docker on Apple silicon
    // reports implementer 0x61 with part 0x000), so the implementer alone has
    // to be enough to name something useful.
    {
        auto value_of = [](std::string const &l) -> std::string {
            auto const colon = l.find(':');
            if (colon == std::string::npos) {
                return {};
            }
            auto const v = l.substr(colon + 1);
            auto const b = v.find_first_not_of(" \t");
            auto const e = v.find_last_not_of(" \t\r\n");
            return b == std::string::npos ? std::string{} : v.substr(b, e - b + 1);
        };

        std::ifstream cpuinfo("/proc/cpuinfo");
        std::string   line;
        std::string   implementer;
        std::string   part;
        while (std::getline(cpuinfo, line)) {
            // Some ARM boards, and every x86 kernel, do publish a name. Prefer
            // it whenever it is present.
            if (line.rfind("model name", 0) == 0 || line.rfind("Model", 0) == 0) {
                if (auto const v = value_of(line); !v.empty()) {
                    return v;
                }
            } else if (line.rfind("CPU implementer", 0) == 0) {
                implementer = value_of(line);
            } else if (line.rfind("CPU part", 0) == 0) {
                part = value_of(line);
            }
        }

        if (!implementer.empty()) {
            struct CodeName {
                char const *code;
                char const *name;
            };
            // Implementer codes are assigned by ARM. Anything unlisted still
            // reports its raw code rather than falling through to "Unknown CPU".
            static constexpr CodeName kImplementers[] = {
                {"0x41", "ARM"},    {"0x42", "Broadcom"}, {"0x43", "Cavium"},   {"0x46", "Fujitsu"}, {"0x48", "HiSilicon"},
                {"0x4e", "NVIDIA"}, {"0x50", "APM"},      {"0x51", "Qualcomm"}, {"0x53", "Samsung"}, {"0x56", "Marvell"},
                {"0x61", "Apple"},  {"0x69", "Intel"},    {"0x70", "Phytium"},  {"0xc0", "Ampere"}};
            // ARM's own cores, where the part number names a specific design.
            static constexpr CodeName kArmParts[] = {{"0xd03", "Cortex-A53"},  {"0xd05", "Cortex-A55"},  {"0xd07", "Cortex-A57"},
                                                     {"0xd08", "Cortex-A72"},  {"0xd09", "Cortex-A73"},  {"0xd0a", "Cortex-A75"},
                                                     {"0xd0b", "Cortex-A76"},  {"0xd0c", "Neoverse-N1"}, {"0xd40", "Neoverse-V1"},
                                                     {"0xd49", "Neoverse-N2"}, {"0xd4f", "Neoverse-V2"}};

            char const *vendor = nullptr;
            for (auto const &entry : kImplementers) {
                if (implementer == entry.code) {
                    vendor = entry.name;
                    break;
                }
            }
            if (vendor != nullptr && implementer == std::string("0x41")) {
                for (auto const &entry : kArmParts) {
                    if (part == entry.code) {
                        return fmt::format("{} {}", vendor, entry.name);
                    }
                }
            }
            if (vendor != nullptr) {
                return fmt::format("{} aarch64 (part {})", vendor, part.empty() ? "unknown" : part);
            }
            return fmt::format("aarch64 (implementer {}, part {})", implementer, part.empty() ? "unknown" : part);
        }
    }
#endif
    return "Unknown CPU";
}

std::string DeviceProfileDB::detect_gpu_name() {
    return gpu::device_name();
}

namespace {

/// The CPU data-cache hierarchy, in L1 -> L3 order, taken from the one detector
/// (Einsums_Hardware). This used to run its own sysctl/sysfs query, which is how
/// it and PackedGemm ended up disagreeing about which L2 to report on Apple
/// Silicon. Bandwidth and latency stay unmeasured (zero) - consumers key off
/// size_bytes.
std::vector<CacheLevel> detect_cpu_caches() {
    auto const             &cache = einsums::hardware::cpu_info().cache;
    std::vector<CacheLevel> levels;
    for (std::int64_t const bytes : {cache.l1, cache.l2, cache.l3}) {
        if (bytes > 0) {
            levels.push_back({.size_bytes = static_cast<size_t>(bytes)});
        }
    }
    return levels;
}

} // namespace

std::string DeviceProfileDB::normalize(std::string const &s) {
    std::string result;
    result.reserve(s.size());
    for (char const c : s) {
        if (std::isspace(static_cast<unsigned char>(c)))
            continue;
        result += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return result;
}

DeviceProfile const *DeviceProfileDB::find_best_match(std::string const &brand, DeviceType type) const {
    std::string const    norm_brand = normalize(brand);
    DeviceProfile const *best       = nullptr;
    size_t               best_len   = 0;

    for (auto const &p : _profiles) {
        if (p.device_type != type)
            continue;
        for (auto const &pattern : p.match_patterns) {
            std::string const norm_pat = normalize(pattern);
            if (norm_brand.find(norm_pat) != std::string::npos && norm_pat.size() > best_len) {
                best     = &p;
                best_len = norm_pat.size();
            }
        }
    }
    return best;
}

DeviceProfile const &DeviceProfileDB::match_cpu() const {
    std::string const brand = detect_cpu_brand();
    auto             *match = find_best_match(brand, DeviceType::CPU);
    return match ? *match : _fallback_cpu;
}

DeviceProfile const &DeviceProfileDB::match_gpu() const {
    std::string const name = detect_gpu_name();
    if (name.empty())
        return _fallback_gpu;
    auto *match = find_best_match(name, DeviceType::GPU);
    return match ? *match : _fallback_gpu;
}

CostModel DeviceProfileDB::build_cost_model() const {
    CostModel p;
    p.cpu    = match_cpu();
    p.gpu    = match_gpu();
    p.source = "database";
    return p;
}

void DeviceProfileDB::upsert(DeviceProfile profile) {
    for (auto &p : _profiles) {
        if (p.brand_family == profile.brand_family && p.device_type == profile.device_type) {
            p = std::move(profile);
            return;
        }
    }
    _profiles.push_back(std::move(profile));
}

// ═══════════════════════════════════════════════════════════════════════════════
// Built-in default profiles
// ═══════════════════════════════════════════════════════════════════════════════

DeviceProfileDB DeviceProfileDB::load_defaults() {
    DeviceProfileDB db;

    // ── Fallbacks ──────────────────────────────────────────────────────────
    db._fallback_cpu.name               = "Generic CPU";
    db._fallback_cpu.device_type        = DeviceType::CPU;
    db._fallback_cpu.brand_family       = "generic_cpu";
    db._fallback_cpu.peak_gflops_fp64   = 50.0;
    db._fallback_cpu.peak_gflops_fp32   = 100.0;
    db._fallback_cpu.mem_bandwidth_gbps = 40.0;
    db._fallback_cpu.gemm_efficiency    = {{.M = 16, .N = 16, .K = 16, .gflops = 5.0},
                                           {.M = 64, .N = 64, .K = 64, .gflops = 30.0},
                                           {.M = 256, .N = 256, .K = 256, .gflops = 45.0},
                                           {.M = 1024, .N = 1024, .K = 1024, .gflops = 49.0}};

    db._fallback_gpu.name         = "";
    db._fallback_gpu.device_type  = DeviceType::GPU;
    db._fallback_gpu.brand_family = "none";

    // Shared CPU DeviceProfile builder: the vendor tables below differ only in
    // the launch/alloc overheads and the gemm-efficiency curve, so everything
    // else lives here once.
    auto cpu_base = [](std::string name, std::string family, std::vector<std::string> patterns, double fp64, double bw, double launch_us,
                       double alloc_us, std::vector<GemmEfficiencyPoint> gemm) {
        DeviceProfile p;
        p.name                      = std::move(name);
        p.device_type               = DeviceType::CPU;
        p.brand_family              = std::move(family);
        p.match_patterns            = std::move(patterns);
        p.source                    = "default";
        p.peak_gflops_fp64          = fp64;
        p.peak_gflops_fp32          = fp64 * 2.0;
        p.mem_bandwidth_gbps        = bw;
        p.kernel_launch_overhead_us = launch_us;
        p.alloc_overhead_us         = alloc_us;
        p.gemm_efficiency           = std::move(gemm);
        return p;
    };

    // ── Apple Silicon ──────────────────────────────────────────────────────
    auto apple_base = [&](std::string name, std::string family, std::vector<std::string> patterns, double fp64, double bw) {
        return cpu_base(std::move(name), std::move(family), std::move(patterns), fp64, bw, 0.3, 1.0,
                        {{.M = 16, .N = 16, .K = 16, .gflops = fp64 * 0.08},
                         {.M = 32, .N = 32, .K = 32, .gflops = fp64 * 0.25},
                         {.M = 64, .N = 64, .K = 64, .gflops = fp64 * 0.55},
                         {.M = 128, .N = 128, .K = 128, .gflops = fp64 * 0.80},
                         {.M = 256, .N = 256, .K = 256, .gflops = fp64 * 0.92},
                         {.M = 512, .N = 512, .K = 512, .gflops = fp64 * 0.97},
                         {.M = 1024, .N = 1024, .K = 1024, .gflops = fp64 * 0.99}});
    };

    db._profiles.push_back(apple_base("Apple M1", "apple_m1", {"Apple M1"}, 60.0, 68.0));
    db._profiles.push_back(apple_base("Apple M1 Pro", "apple_m1_pro", {"Apple M1 Pro"}, 75.0, 200.0));
    db._profiles.push_back(apple_base("Apple M1 Max", "apple_m1_max", {"Apple M1 Max"}, 75.0, 400.0));
    db._profiles.push_back(apple_base("Apple M2", "apple_m2", {"Apple M2"}, 80.0, 100.0));
    db._profiles.push_back(apple_base("Apple M2 Pro", "apple_m2_pro", {"Apple M2 Pro"}, 90.0, 200.0));
    db._profiles.push_back(apple_base("Apple M2 Max", "apple_m2_max", {"Apple M2 Max"}, 90.0, 400.0));
    db._profiles.push_back(apple_base("Apple M3", "apple_m3", {"Apple M3"}, 85.0, 100.0));
    db._profiles.push_back(apple_base("Apple M3 Pro", "apple_m3_pro", {"Apple M3 Pro"}, 100.0, 150.0));
    db._profiles.push_back(apple_base("Apple M3 Max", "apple_m3_max", {"Apple M3 Max"}, 100.0, 400.0));
    db._profiles.push_back(apple_base("Apple M4", "apple_m4", {"Apple M4"}, 100.0, 120.0));
    db._profiles.push_back(apple_base("Apple M4 Pro", "apple_m4_pro", {"Apple M4 Pro"}, 120.0, 273.0));
    db._profiles.push_back(apple_base("Apple M4 Max", "apple_m4_max", {"Apple M4 Max"}, 120.0, 546.0));

    // ── Intel / AMD ──────────────────────────────────────────────────────────
    auto intel_base = [&](std::string name, std::string family, std::vector<std::string> patterns, double fp64, double bw) {
        return cpu_base(std::move(name), std::move(family), std::move(patterns), fp64, bw, 0.5, 2.0,
                        {{.M = 16, .N = 16, .K = 16, .gflops = fp64 * 0.10},
                         {.M = 64, .N = 64, .K = 64, .gflops = fp64 * 0.60},
                         {.M = 256, .N = 256, .K = 256, .gflops = fp64 * 0.90},
                         {.M = 1024, .N = 1024, .K = 1024, .gflops = fp64 * 0.98}});
    };

    db._profiles.push_back(intel_base("Intel Skylake", "intel_skylake", {"Skylake", "i7-6", "i9-6", "E5-26"}, 50.0, 40.0));
    db._profiles.push_back(intel_base("Intel Ice Lake", "intel_icelake", {"Ice Lake", "i7-10", "i9-10"}, 80.0, 50.0));
    db._profiles.push_back(intel_base("Intel Sapphire Rapids", "intel_spr", {"Sapphire Rapids", "w9-3", "w7-3"}, 120.0, 80.0));

    // ── AMD ────────────────────────────────────────────────────────────────
    db._profiles.push_back(intel_base("AMD EPYC Rome", "amd_rome", {"Rome", "EPYC 7"}, 60.0, 50.0));
    db._profiles.push_back(intel_base("AMD EPYC Milan", "amd_milan", {"Milan", "EPYC 73"}, 80.0, 80.0));
    db._profiles.push_back(intel_base("AMD EPYC Genoa", "amd_genoa", {"Genoa", "EPYC 9"}, 120.0, 115.0));

    // ── NVIDIA GPUs ────────────────────────────────────────────────────────
    auto nvidia_gpu = [](std::string name, std::string family, std::vector<std::string> patterns, double fp64, double fp32, double dev_bw,
                         double pcie_bw) {
        DeviceProfile p;
        p.name                      = std::move(name);
        p.device_type               = DeviceType::GPU;
        p.brand_family              = std::move(family);
        p.match_patterns            = std::move(patterns);
        p.source                    = "default";
        p.peak_gflops_fp64          = fp64;
        p.peak_gflops_fp32          = fp32;
        p.device_bandwidth_gbps     = dev_bw;
        p.pcie_bandwidth_gbps       = pcie_bw;
        p.gpu_launch_latency_us     = 5.0;
        p.kernel_launch_overhead_us = 1.0;
        p.alloc_overhead_us         = 5.0;
        p.gemm_efficiency           = {{.M = 16, .N = 16, .K = 16, .gflops = fp64 * 0.01},
                                       {.M = 64, .N = 64, .K = 64, .gflops = fp64 * 0.20},
                                       {.M = 256, .N = 256, .K = 256, .gflops = fp64 * 0.70},
                                       {.M = 1024, .N = 1024, .K = 1024, .gflops = fp64 * 0.96},
                                       {.M = 4096, .N = 4096, .K = 4096, .gflops = fp64 * 0.99}};
        return p;
    };

    db._profiles.push_back(nvidia_gpu("NVIDIA V100", "nvidia_v100", {"V100"}, 7800.0, 15700.0, 900.0, 12.0));
    db._profiles.push_back(nvidia_gpu("NVIDIA A100", "nvidia_a100", {"A100"}, 9700.0, 19500.0, 2039.0, 25.0));
    db._profiles.push_back(nvidia_gpu("NVIDIA H100", "nvidia_h100", {"H100"}, 30000.0, 60000.0, 3350.0, 50.0));
    db._profiles.push_back(nvidia_gpu("NVIDIA RTX 3090", "nvidia_rtx3090", {"RTX 3090", "3090"}, 556.0, 35600.0, 936.0, 12.0));
    db._profiles.push_back(nvidia_gpu("NVIDIA RTX 4090", "nvidia_rtx4090", {"RTX 4090", "4090"}, 1290.0, 82600.0, 1008.0, 12.0));

    // ── AMD GPUs ───────────────────────────────────────────────────────────
    db._profiles.push_back(nvidia_gpu("AMD MI250X", "amd_mi250x", {"MI250", "MI250X"}, 47900.0, 47900.0, 3277.0, 25.0));
    db._profiles.push_back(nvidia_gpu("AMD MI300X", "amd_mi300x", {"MI300", "MI300X"}, 81900.0, 163800.0, 5300.0, 50.0));

    // ── Apple MPS (GPU profile for MPS backend) ────────────────────────────
    {
        DeviceProfile p;
        p.name                  = "Apple MPS (Metal)";
        p.device_type           = DeviceType::GPU;
        p.brand_family          = "apple_mps";
        p.match_patterns        = {"Apple"};
        p.source                = "default";
        p.peak_gflops_fp64      = 0.0; // MPS doesn't support FP64 GEMM
        p.peak_gflops_fp32      = 200.0;
        p.device_bandwidth_gbps = 200.0; // Unified memory
        p.pcie_bandwidth_gbps   = 200.0; // No PCIe, unified
        p.gpu_launch_latency_us = 5.0;
        p.gemm_efficiency       = {{.M = 16, .N = 16, .K = 16, .gflops = 10.0},
                                   {.M = 64, .N = 64, .K = 64, .gflops = 80.0},
                                   {.M = 256, .N = 256, .K = 256, .gflops = 160.0},
                                   {.M = 1024, .N = 1024, .K = 1024, .gflops = 195.0}};
        db._profiles.push_back(p);
    }

    return db;
}

// ═══════════════════════════════════════════════════════════════════════════════
// CostModel factory
// ═══════════════════════════════════════════════════════════════════════════════

CostModel CostModel::detect_default() {
    // A calibrated profile (written by the calibrate_hardware tool) takes
    // precedence over the built-in table: point --einsums:hardware:profile at
    // its JSON and every profile consumer (ContractionPlanning's chain DP,
    // GPUPlacement, GEMMBatching's profitability gate) uses the measured
    // efficiency curve and bandwidths instead of generic estimates. A
    // missing or unreadable file falls back to the table with a warning
    // rather than failing: the profile shapes optimization choices, never
    // correctness.
    // Cache sizes are pure topology, so runtime detection beats any table or
    // calibration file that omits them; a profile that DOES carry cache data
    // (a calibrated JSON) keeps its own numbers.
    auto const with_detected_caches = [](CostModel model) {
        if (model.cpu.caches.empty()) {
            model.cpu.caches = detect_cpu_caches();
        }
        return model;
    };

    if (auto const profile_path = config::get(option::HardwareProfile); !profile_path.empty()) {
        auto loaded = load_json(profile_path);
        if (loaded) {
            EINSUMS_LOG_INFO("CostModel: using calibrated profile from --einsums:hardware:profile={}", profile_path);
            return with_detected_caches(*loaded);
        }
        EINSUMS_LOG_WARN("CostModel: --einsums:hardware:profile={} could not be loaded ({}); falling back to the built-in table",
                         profile_path, loaded.error().message);
    }

    auto db = DeviceProfileDB::load_defaults();
    return with_detected_caches(db.build_cost_model());
}

// ═══════════════════════════════════════════════════════════════════════════════
// JSON I/O
// ═══════════════════════════════════════════════════════════════════════════════

namespace {

/// The number at @p key, or @p fallback when the key is missing or is not a number.
double number_or(json::Object const &obj, std::string_view key, double fallback) {
    auto const *value = obj.peek(key);
    return (value != nullptr && value->is_number()) ? value->as_double() : fallback;
}

/// The string at @p key, or "" when the key is missing or is not a string.
std::string string_or(json::Object const &obj, std::string_view key) {
    auto const *value = obj.peek(key);
    return (value != nullptr && value->is_string()) ? value->as_string() : std::string{};
}

/// Call @p on_object once per object element of the array at @p key.
///
/// A key that is absent, or holds something other than an array, contributes
/// nothing; a non-object element is skipped. That leniency is deliberate and is
/// the one place this file departs from the strict, consumed-key policy the rest
/// of the graph IR follows: a profile is a machine measurement someone may have
/// written by hand or with an older build, and a row this build cannot price is
/// dropped rather than turned into a load failure.
void for_each_object(json::Object const &obj, std::string_view key, auto const &on_object) {
    auto const *value = obj.peek(key);
    if (value == nullptr || !value->is_array()) {
        return;
    }
    for (auto const &item : value->as_array()) {
        if (item.is_object()) {
            on_object(item.as_object());
        }
    }
}

json::Object write_profile(DeviceProfile const &p) {
    json::Object obj;
    obj.set("name", p.name);
    obj.set("device_type", p.device_type == DeviceType::GPU ? "gpu" : "cpu");
    obj.set("brand_family", p.brand_family);

    json::Array patterns;
    for (auto const &pattern : p.match_patterns) {
        patterns.emplace_back(pattern);
    }
    obj.set("match_patterns", json::Value{std::move(patterns)});

    obj.set("peak_gflops_fp64", p.peak_gflops_fp64);
    obj.set("peak_gflops_fp32", p.peak_gflops_fp32);
    obj.set("mem_bandwidth_gbps", p.mem_bandwidth_gbps);
    obj.set("kernel_launch_overhead_us", p.kernel_launch_overhead_us);
    obj.set("alloc_overhead_us", p.alloc_overhead_us);
    obj.set("device_bandwidth_gbps", p.device_bandwidth_gbps);
    obj.set("pcie_bandwidth_gbps", p.pcie_bandwidth_gbps);
    obj.set("gpu_launch_latency_us", p.gpu_launch_latency_us);

    obj.set("inter_node_bandwidth_gbps", p.inter_node_bandwidth_gbps);
    obj.set("inter_node_latency_us", p.inter_node_latency_us);
    obj.set("nccl_bandwidth_gbps", p.nccl_bandwidth_gbps);

    json::Array caches;
    for (auto const &level : p.caches) {
        json::Object entry;
        entry.set("size_bytes", level.size_bytes);
        entry.set("bandwidth_gbps", level.bandwidth_gbps);
        entry.set("latency_ns", level.latency_ns);
        caches.emplace_back(std::move(entry));
    }
    obj.set("caches", json::Value{std::move(caches)});

    json::Array gemm;
    for (auto const &pt : p.gemm_efficiency) {
        json::Object entry;
        entry.set("M", pt.M);
        entry.set("N", pt.N);
        entry.set("K", pt.K);
        entry.set("gflops", pt.gflops);
        gemm.emplace_back(std::move(entry));
    }
    obj.set("gemm_efficiency", json::Value{std::move(gemm)});

    json::Array permute;
    for (auto const &pt : p.permute_efficiency) {
        json::Object entry;
        entry.set("bytes", pt.bytes);
        entry.set("rank", pt.rank);
        entry.set("gbps", pt.gbps);
        permute.emplace_back(std::move(entry));
    }
    obj.set("permute_efficiency", json::Value{std::move(permute)});

    obj.set("max_threads", static_cast<std::int64_t>(p.max_threads));

    // One row per (family, size class, width) rather than one object per curve
    // with a nested widths array. The shape is kept for the sake of profile
    // files already on disk, and it reads no worse: a row names everything it
    // needs, so a reader that does not know a family drops that row alone.
    json::Array curves;
    for (auto const &curve : p.thread_efficiency) {
        if (!curve.valid()) {
            continue;
        }
        for (size_t i = 0; i < curve.widths.size(); i++) {
            json::Object entry;
            entry.set("family", to_string(curve.family));
            entry.set("size_class", to_string(curve.size_class));
            entry.set("width", static_cast<std::int64_t>(curve.widths[i]));
            entry.set("speedup", curve.speedup[i]);
            curves.emplace_back(std::move(entry));
        }
    }
    obj.set("thread_efficiency", json::Value{std::move(curves)});

    return obj;
}

DeviceProfile read_profile(json::Object const &obj) {
    DeviceProfile p;

    p.name                      = string_or(obj, "name");
    p.brand_family              = string_or(obj, "brand_family");
    p.device_type               = (string_or(obj, "device_type") == "gpu") ? DeviceType::GPU : DeviceType::CPU;
    p.peak_gflops_fp64          = number_or(obj, "peak_gflops_fp64", 50.0);
    p.peak_gflops_fp32          = number_or(obj, "peak_gflops_fp32", 100.0);
    p.mem_bandwidth_gbps        = number_or(obj, "mem_bandwidth_gbps", 40.0);
    p.kernel_launch_overhead_us = number_or(obj, "kernel_launch_overhead_us", 0.5);
    p.alloc_overhead_us         = number_or(obj, "alloc_overhead_us", 2.0);
    p.device_bandwidth_gbps     = number_or(obj, "device_bandwidth_gbps", 0.0);
    p.pcie_bandwidth_gbps       = number_or(obj, "pcie_bandwidth_gbps", 0.0);
    p.gpu_launch_latency_us     = number_or(obj, "gpu_launch_latency_us", 0.0);

    // The three network figures used to be dropped on the floor: the writer
    // never emitted them, so a calibrated profile carrying a measured fabric
    // reloaded with the struct defaults and every collective was priced wrong.
    p.inter_node_bandwidth_gbps = number_or(obj, "inter_node_bandwidth_gbps", 0.0);
    p.inter_node_latency_us     = number_or(obj, "inter_node_latency_us", 1.0);
    p.nccl_bandwidth_gbps       = number_or(obj, "nccl_bandwidth_gbps", 0.0);

    if (auto const *patterns = obj.peek("match_patterns"); patterns != nullptr && patterns->is_array()) {
        for (auto const &item : patterns->as_array()) {
            if (item.is_string()) {
                p.match_patterns.push_back(item.as_string());
            }
        }
    }

    for_each_object(obj, "gemm_efficiency", [&](json::Object const &entry) {
        GemmEfficiencyPoint pt;
        pt.M      = static_cast<size_t>(number_or(entry, "M", 0.0));
        pt.N      = static_cast<size_t>(number_or(entry, "N", 0.0));
        pt.K      = static_cast<size_t>(number_or(entry, "K", 0.0));
        pt.gflops = number_or(entry, "gflops", 0.0);
        if (pt.M > 0 && pt.gflops > 0) {
            p.gemm_efficiency.push_back(pt);
        }
    });

    for_each_object(obj, "permute_efficiency", [&](json::Object const &entry) {
        PermuteEfficiencyPoint pt;
        pt.bytes = static_cast<size_t>(number_or(entry, "bytes", 0.0));
        pt.rank  = static_cast<size_t>(number_or(entry, "rank", 0.0));
        pt.gbps  = number_or(entry, "gbps", 0.0);
        if (pt.bytes > 0 && pt.gbps > 0) {
            p.permute_efficiency.push_back(pt);
        }
    });

    // The cache array was written but never read back, so a calibrated profile
    // silently lost its cache sizes on reload and fell back to whatever the
    // detector reported for the machine doing the loading.
    for_each_object(obj, "caches", [&](json::Object const &entry) {
        CacheLevel level;
        level.size_bytes     = static_cast<size_t>(number_or(entry, "size_bytes", 0.0));
        level.bandwidth_gbps = number_or(entry, "bandwidth_gbps", 0.0);
        level.latency_ns     = number_or(entry, "latency_ns", 0.0);
        if (level.size_bytes > 0) {
            p.caches.push_back(level);
        }
    });

    p.max_threads = static_cast<unsigned>(number_or(obj, "max_threads", 0.0));

    // Flat (family, size class, width, speedup) rows regrouped into curves.
    // Rows naming a family or size class this build does not know are dropped:
    // pricing them as something else would be worse than pricing them from the
    // default model.
    std::map<std::pair<std::uint8_t, std::uint8_t>, EfficiencyCurve> curves;
    for_each_object(obj, "thread_efficiency", [&](json::Object const &entry) {
        KernelFamily family{};
        SizeClass    size_class{};
        if (!kernel_family_from_string(string_or(entry, "family"), family) ||
            !size_class_from_string(string_or(entry, "size_class"), size_class)) {
            return;
        }
        auto const width = static_cast<std::uint32_t>(number_or(entry, "width", 0.0));
        if (width == 0) {
            return;
        }
        auto &curve      = curves[{static_cast<std::uint8_t>(family), static_cast<std::uint8_t>(size_class)}];
        curve.family     = family;
        curve.size_class = size_class;
        curve.widths.push_back(width);
        curve.speedup.push_back(number_or(entry, "speedup", 0.0));
    });

    for (auto &[key, curve] : curves) {
        // Rows may arrive in any order; the query interpolates and so needs
        // ascending rungs.
        std::vector<size_t> order(curve.widths.size());
        for (size_t idx = 0; idx < order.size(); idx++) {
            order[idx] = idx;
        }
        std::ranges::sort(order, [&](size_t a, size_t b) { return curve.widths[a] < curve.widths[b]; });

        EfficiencyCurve sorted;
        sorted.family     = curve.family;
        sorted.size_class = curve.size_class;
        for (size_t const idx : order) {
            sorted.widths.push_back(curve.widths[idx]);
            sorted.speedup.push_back(curve.speedup[idx]);
        }
        if (sorted.valid()) {
            p.thread_efficiency.push_back(std::move(sorted));
        }
    }

    return p;
}

/// Write @p document to @p path, laid out one key per line.
expected<void, GraphError> write_document(std::string const &path, json::Value const &document, std::string_view what) {
    std::ofstream f(path);
    if (!f) {
        return unexpected(GraphError::io(fmt::format("{}: cannot open '{}'", what, path)));
    }
    f << json::emit(document, json::EmitOptions{.style = json::EmitStyle::Pretty});
    if (!f) {
        return unexpected(GraphError::io(fmt::format("{}: could not write '{}'", what, path)));
    }
    return {};
}

/// Read @p path and parse it as a JSON object.
expected<json::Value, GraphError> read_document(std::string const &path, std::string_view what) {
    std::ifstream f(path);
    if (!f) {
        return unexpected(GraphError::io(fmt::format("{}: cannot open '{}'", what, path)));
    }
    std::string const content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

    auto document = json::parse(content);
    if (!document) {
        return unexpected(GraphError::parse(fmt::format("{}: '{}' is not valid JSON: {}", what, path, document.error().to_string())));
    }
    if (!document->is_object()) {
        return unexpected(GraphError::parse(
            fmt::format("{}: '{}' holds a {} where an object was expected", what, path, json::Value::type_name(document->type()))));
    }
    return std::move(*document);
}

} // namespace

expected<void, GraphError> CostModel::save_json(std::string const &path) const {
    json::Object root;
    root.set("source", source);
    root.set("cpu", write_profile(cpu));
    root.set("gpu", write_profile(gpu));
    return write_document(path, json::Value{std::move(root)}, "CostModel::save_json");
}

expected<CostModel, GraphError> CostModel::load_json(std::string const &path) {
    auto document = read_document(path, "CostModel::load_json");
    if (!document) {
        return unexpected(document.error());
    }
    auto const &root = document->as_object();

    CostModel p;
    // Only when the file names one: an absent key leaves the struct default
    // ("default") rather than blanking it.
    if (auto const *named = root.peek("source"); named != nullptr && named->is_string()) {
        p.source = named->as_string();
    }
    if (auto const *cpu_obj = root.peek("cpu"); cpu_obj != nullptr && cpu_obj->is_object()) {
        p.cpu = read_profile(cpu_obj->as_object());
    }
    if (auto const *gpu_obj = root.peek("gpu"); gpu_obj != nullptr && gpu_obj->is_object()) {
        p.gpu = read_profile(gpu_obj->as_object());
    }
    return p;
}

expected<void, GraphError> DeviceProfileDB::save_json(std::string const &path) const {
    json::Array profiles;
    for (auto const &profile : _profiles) {
        profiles.emplace_back(write_profile(profile));
    }

    json::Object root;
    root.set("version", std::int64_t{1});
    root.set("profiles", json::Value{std::move(profiles)});
    return write_document(path, json::Value{std::move(root)}, "DeviceProfileDB::save_json");
}

expected<DeviceProfileDB, GraphError> DeviceProfileDB::load_json(std::string const &path) {
    auto document = read_document(path, "DeviceProfileDB::load_json");
    if (!document) {
        return unexpected(document.error());
    }

    DeviceProfileDB db = load_defaults(); // Start with defaults, overlay from file

    for_each_object(document->as_object(), "profiles", [&](json::Object const &entry) {
        auto p = read_profile(entry);
        if (!p.name.empty()) {
            p.source = "database";
            db.upsert(std::move(p));
        }
    });

    return db;
}

EINSUMS_NAMESPACE_END(compute_graph)
