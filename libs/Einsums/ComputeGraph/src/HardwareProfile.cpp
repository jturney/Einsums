//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/HardwareProfile.hpp>
#include <Einsums/Errors.hpp>
#include <Einsums/GPU/Platform.hpp>
#include <Einsums/GPU/Runtime.hpp>
#include <Einsums/Logging.hpp>

#include <fmt/format.h>

#include <cctype>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>

#ifdef __APPLE__
#    include <sys/sysctl.h>
#endif

namespace einsums::compute_graph {

// ═══════════════════════════════════════════════════════════════════════════════
// HardwareProfileDB: runtime detection
// ═══════════════════════════════════════════════════════════════════════════════

std::string HardwareProfileDB::detect_cpu_brand() {
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
#endif
    return "Unknown CPU";
}

std::string HardwareProfileDB::detect_gpu_name() {
    return gpu::device_name();
}

std::string HardwareProfileDB::normalize(std::string const &s) {
    std::string result;
    result.reserve(s.size());
    for (char const c : s) {
        if (std::isspace(static_cast<unsigned char>(c)))
            continue;
        result += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return result;
}

DeviceProfile const *HardwareProfileDB::find_best_match(std::string const &brand, DeviceType type) const {
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

DeviceProfile const &HardwareProfileDB::match_cpu() const {
    std::string const brand = detect_cpu_brand();
    auto             *match = find_best_match(brand, DeviceType::CPU);
    return match ? *match : _fallback_cpu;
}

DeviceProfile const &HardwareProfileDB::match_gpu() const {
    std::string const name = detect_gpu_name();
    if (name.empty())
        return _fallback_gpu;
    auto *match = find_best_match(name, DeviceType::GPU);
    return match ? *match : _fallback_gpu;
}

HardwareProfile HardwareProfileDB::build_profile() const {
    HardwareProfile p;
    p.cpu    = match_cpu();
    p.gpu    = match_gpu();
    p.source = "database";
    return p;
}

void HardwareProfileDB::upsert(DeviceProfile profile) {
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

HardwareProfileDB HardwareProfileDB::load_defaults() {
    HardwareProfileDB db;

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

    // ── Apple Silicon ──────────────────────────────────────────────────────
    auto apple_base = [](std::string name, std::string family, std::vector<std::string> patterns, double fp64, double bw) {
        DeviceProfile p;
        p.name                      = std::move(name);
        p.device_type               = DeviceType::CPU;
        p.brand_family              = std::move(family);
        p.match_patterns            = std::move(patterns);
        p.source                    = "default";
        p.peak_gflops_fp64          = fp64;
        p.peak_gflops_fp32          = fp64 * 2.0;
        p.mem_bandwidth_gbps        = bw;
        p.kernel_launch_overhead_us = 0.3;
        p.alloc_overhead_us         = 1.0;
        p.gemm_efficiency = {{.M = 16, .N = 16, .K = 16, .gflops = fp64 * 0.08},      {.M = 32, .N = 32, .K = 32, .gflops = fp64 * 0.25},
                             {.M = 64, .N = 64, .K = 64, .gflops = fp64 * 0.55},      {.M = 128, .N = 128, .K = 128, .gflops = fp64 * 0.80},
                             {.M = 256, .N = 256, .K = 256, .gflops = fp64 * 0.92},   {.M = 512, .N = 512, .K = 512, .gflops = fp64 * 0.97},
                             {.M = 1024, .N = 1024, .K = 1024, .gflops = fp64 * 0.99}};
        return p;
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

    // ── Intel ──────────────────────────────────────────────────────────────
    auto intel_base = [](std::string name, std::string family, std::vector<std::string> patterns, double fp64, double bw) {
        DeviceProfile p;
        p.name                      = std::move(name);
        p.device_type               = DeviceType::CPU;
        p.brand_family              = std::move(family);
        p.match_patterns            = std::move(patterns);
        p.source                    = "default";
        p.peak_gflops_fp64          = fp64;
        p.peak_gflops_fp32          = fp64 * 2.0;
        p.mem_bandwidth_gbps        = bw;
        p.kernel_launch_overhead_us = 0.5;
        p.alloc_overhead_us         = 2.0;
        p.gemm_efficiency           = {{.M = 16, .N = 16, .K = 16, .gflops = fp64 * 0.10},
                                       {.M = 64, .N = 64, .K = 64, .gflops = fp64 * 0.60},
                                       {.M = 256, .N = 256, .K = 256, .gflops = fp64 * 0.90},
                                       {.M = 1024, .N = 1024, .K = 1024, .gflops = fp64 * 0.98}};
        return p;
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
// HardwareProfile factory
// ═══════════════════════════════════════════════════════════════════════════════

HardwareProfile HardwareProfile::detect_default() {
    // A calibrated profile (written by the calibrate_hardware tool) takes
    // precedence over the built-in table: point EINSUMS_HARDWARE_PROFILE at
    // its JSON and every profile consumer (ContractionPlanning's chain DP,
    // GPUPlacement, GEMMBatching's profitability gate) uses the measured
    // efficiency curve and bandwidths instead of generic estimates. A
    // missing or unreadable file falls back to the table with a warning
    // rather than failing: the profile shapes optimization choices, never
    // correctness.
    if (char const *env_path = std::getenv("EINSUMS_HARDWARE_PROFILE"); env_path != nullptr && *env_path != '\0') {
        auto loaded = load_json(env_path);
        if (loaded) {
            EINSUMS_LOG_INFO("HardwareProfile: using calibrated profile from EINSUMS_HARDWARE_PROFILE={}", env_path);
            return *loaded;
        }
        EINSUMS_LOG_WARN("HardwareProfile: EINSUMS_HARDWARE_PROFILE={} could not be loaded ({}); falling back to the built-in table",
                         env_path, loaded.error().message);
    }

    auto db = HardwareProfileDB::load_defaults();
    return db.build_profile();
}

// ═══════════════════════════════════════════════════════════════════════════════
// JSON I/O
// ═══════════════════════════════════════════════════════════════════════════════

namespace {

void write_device_profile_json(std::ostream &f, DeviceProfile const &p, std::string const &indent) {
    f << indent << "{\n";
    f << indent << fmt::format("  \"name\": \"{}\",\n", p.name);
    f << indent << fmt::format("  \"device_type\": \"{}\",\n", p.device_type == DeviceType::GPU ? "gpu" : "cpu");
    f << indent << fmt::format("  \"brand_family\": \"{}\",\n", p.brand_family);

    f << indent << "  \"match_patterns\": [";
    for (size_t i = 0; i < p.match_patterns.size(); i++) {
        if (i > 0)
            f << ", ";
        f << fmt::format("\"{}\"", p.match_patterns[i]);
    }
    f << "],\n";

    f << indent << fmt::format("  \"peak_gflops_fp64\": {:.1f},\n", p.peak_gflops_fp64);
    f << indent << fmt::format("  \"peak_gflops_fp32\": {:.1f},\n", p.peak_gflops_fp32);
    f << indent << fmt::format("  \"mem_bandwidth_gbps\": {:.1f},\n", p.mem_bandwidth_gbps);
    f << indent << fmt::format("  \"kernel_launch_overhead_us\": {:.2f},\n", p.kernel_launch_overhead_us);
    f << indent << fmt::format("  \"alloc_overhead_us\": {:.2f},\n", p.alloc_overhead_us);
    f << indent << fmt::format("  \"device_bandwidth_gbps\": {:.1f},\n", p.device_bandwidth_gbps);
    f << indent << fmt::format("  \"pcie_bandwidth_gbps\": {:.1f},\n", p.pcie_bandwidth_gbps);
    f << indent << fmt::format("  \"gpu_launch_latency_us\": {:.2f},\n", p.gpu_launch_latency_us);

    f << indent << "  \"caches\": [";
    for (size_t i = 0; i < p.caches.size(); i++) {
        if (i > 0)
            f << ",";
        f << fmt::format("\n{}    {{\"size_bytes\": {}, \"bandwidth_gbps\": {:.1f}, \"latency_ns\": {:.1f}}}", indent,
                         p.caches[i].size_bytes, p.caches[i].bandwidth_gbps, p.caches[i].latency_ns);
    }
    f << "\n" << indent << "  ],\n";

    f << indent << "  \"gemm_efficiency\": [";
    for (size_t i = 0; i < p.gemm_efficiency.size(); i++) {
        if (i > 0)
            f << ",";
        auto const &pt = p.gemm_efficiency[i];
        f << fmt::format("\n{}    {{\"M\": {}, \"N\": {}, \"K\": {}, \"gflops\": {:.1f}}}", indent, pt.M, pt.N, pt.K, pt.gflops);
    }
    f << "\n" << indent << "  ]\n";
    f << indent << "}";
}

DeviceProfile parse_device_profile_json(std::string const &obj) {
    DeviceProfile p;

    auto extract_string = [&](std::string const &key) -> std::string {
        auto pos = obj.find("\"" + key + "\"");
        if (pos == std::string::npos)
            return "";
        pos = obj.find('\"', pos + key.size() + 2);
        if (pos == std::string::npos)
            return "";
        pos++;
        auto end = obj.find('\"', pos);
        return obj.substr(pos, end - pos);
    };

    auto extract_double = [&](std::string const &key, double def) -> double {
        auto pos = obj.find("\"" + key + "\"");
        if (pos == std::string::npos)
            return def;
        pos = obj.find(':', pos);
        if (pos == std::string::npos)
            return def;
        try {
            return std::stod(obj.substr(pos + 1));
        } catch (...) {
            return def;
        }
    };

    p.name                      = extract_string("name");
    p.brand_family              = extract_string("brand_family");
    p.device_type               = (extract_string("device_type") == "gpu") ? DeviceType::GPU : DeviceType::CPU;
    p.peak_gflops_fp64          = extract_double("peak_gflops_fp64", 50.0);
    p.peak_gflops_fp32          = extract_double("peak_gflops_fp32", 100.0);
    p.mem_bandwidth_gbps        = extract_double("mem_bandwidth_gbps", 40.0);
    p.kernel_launch_overhead_us = extract_double("kernel_launch_overhead_us", 0.5);
    p.alloc_overhead_us         = extract_double("alloc_overhead_us", 2.0);
    p.device_bandwidth_gbps     = extract_double("device_bandwidth_gbps", 0.0);
    p.pcie_bandwidth_gbps       = extract_double("pcie_bandwidth_gbps", 0.0);
    p.gpu_launch_latency_us     = extract_double("gpu_launch_latency_us", 0.0);

    // Parse match_patterns
    auto mp_pos = obj.find("\"match_patterns\"");
    if (mp_pos != std::string::npos) {
        auto arr_start = obj.find('[', mp_pos);
        auto arr_end   = obj.find(']', arr_start);
        if (arr_start != std::string::npos && arr_end != std::string::npos) {
            std::string const arr    = obj.substr(arr_start + 1, arr_end - arr_start - 1);
            size_t            search = 0;
            while (true) {
                auto qs = arr.find('\"', search);
                if (qs == std::string::npos)
                    break;
                auto qe = arr.find('\"', qs + 1);
                if (qe == std::string::npos)
                    break;
                p.match_patterns.push_back(arr.substr(qs + 1, qe - qs - 1));
                search = qe + 1;
            }
        }
    }

    // Parse gemm_efficiency
    auto ge_pos = obj.find("\"gemm_efficiency\"");
    if (ge_pos != std::string::npos) {
        auto arr_start = obj.find('[', ge_pos);
        auto arr_end   = obj.find(']', arr_start);
        if (arr_start != std::string::npos && arr_end != std::string::npos) {
            std::string const arr    = obj.substr(arr_start, arr_end - arr_start + 1);
            size_t            search = 0;
            while (true) {
                auto os = arr.find('{', search);
                if (os == std::string::npos)
                    break;
                auto oe = arr.find('}', os);
                if (oe == std::string::npos)
                    break;
                std::string sub = arr.substr(os, oe - os + 1);

                auto get_f = [&](std::string const &k) -> double {
                    auto fp = sub.find("\"" + k + "\"");
                    if (fp == std::string::npos)
                        return 0;
                    fp = sub.find(':', fp);
                    try {
                        return std::stod(sub.substr(fp + 1));
                    } catch (...) {
                        return 0;
                    }
                };

                GemmEfficiencyPoint pt;
                pt.M      = static_cast<size_t>(get_f("M"));
                pt.N      = static_cast<size_t>(get_f("N"));
                pt.K      = static_cast<size_t>(get_f("K"));
                pt.gflops = get_f("gflops");
                if (pt.M > 0 && pt.gflops > 0)
                    p.gemm_efficiency.push_back(pt);
                search = oe + 1;
            }
        }
    }

    return p;
}

} // namespace

expected<void, GraphError> HardwareProfile::save_json(std::string const &path) const {
    std::ofstream f(path);
    if (!f) {
        return unexpected(GraphError::io(fmt::format("HardwareProfile::save_json: cannot open '{}'", path)));
    }

    f << "{\n";
    f << fmt::format("  \"source\": \"{}\",\n", source);
    f << "  \"cpu\": ";
    write_device_profile_json(f, cpu, "  ");
    f << ",\n";
    f << "  \"gpu\": ";
    write_device_profile_json(f, gpu, "  ");
    f << "\n}\n";
    return {};
}

expected<HardwareProfile, GraphError> HardwareProfile::load_json(std::string const &path) {
    std::ifstream f(path);
    if (!f) {
        return unexpected(GraphError::io(fmt::format("HardwareProfile::load_json: cannot open '{}'", path)));
    }
    std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

    HardwareProfile p;

    // Extract source
    auto src_pos = content.find("\"source\"");
    if (src_pos != std::string::npos) {
        auto q1 = content.find('\"', src_pos + 8);
        if (q1 != std::string::npos) {
            q1++;
            auto q2  = content.find('\"', q1);
            p.source = content.substr(q1, q2 - q1);
        }
    }

    // Extract cpu and gpu sub-objects
    auto extract_sub = [&](std::string const &key) -> std::string {
        auto pos = content.find("\"" + key + "\"");
        if (pos == std::string::npos)
            return "";
        auto obj_start = content.find('{', pos);
        if (obj_start == std::string::npos)
            return "";
        // Find matching closing brace (handle nested braces)
        int  depth = 1;
        auto idx   = obj_start + 1;
        while (idx < content.size() && depth > 0) {
            if (content[idx] == '{')
                depth++;
            else if (content[idx] == '}')
                depth--;
            idx++;
        }
        return content.substr(obj_start, idx - obj_start);
    };

    std::string const cpu_json = extract_sub("cpu");
    std::string const gpu_json = extract_sub("gpu");

    if (!cpu_json.empty())
        p.cpu = parse_device_profile_json(cpu_json);
    if (!gpu_json.empty())
        p.gpu = parse_device_profile_json(gpu_json);

    return p;
}

expected<void, GraphError> HardwareProfileDB::save_json(std::string const &path) const {
    std::ofstream f(path);
    if (!f) {
        return unexpected(GraphError::io(fmt::format("HardwareProfileDB::save_json: cannot open '{}'", path)));
    }

    f << "{\n";
    f << "  \"version\": 1,\n";
    f << "  \"profiles\": [\n";
    for (size_t i = 0; i < _profiles.size(); i++) {
        if (i > 0)
            f << ",\n";
        write_device_profile_json(f, _profiles[i], "    ");
    }
    f << "\n  ]\n}\n";
    return {};
}

expected<HardwareProfileDB, GraphError> HardwareProfileDB::load_json(std::string const &path) {
    std::ifstream f(path);
    if (!f) {
        return unexpected(GraphError::io(fmt::format("HardwareProfileDB::load_json: cannot open '{}'", path)));
    }
    std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

    HardwareProfileDB db = load_defaults(); // Start with defaults, overlay from file

    // Find "profiles" array
    auto arr_pos = content.find("\"profiles\"");
    if (arr_pos == std::string::npos)
        return db;

    auto arr_start = content.find('[', arr_pos);
    if (arr_start == std::string::npos)
        return db;

    // Extract each { ... } object in the array
    size_t search = arr_start + 1;
    while (true) {
        auto os = content.find('{', search);
        if (os == std::string::npos)
            break;
        // Find matching }
        int    depth = 1;
        size_t idx   = os + 1;
        while (idx < content.size() && depth > 0) {
            if (content[idx] == '{')
                depth++;
            else if (content[idx] == '}')
                depth--;
            idx++;
        }
        if (depth != 0)
            break;

        std::string const obj = content.substr(os, idx - os);
        auto              p   = parse_device_profile_json(obj);
        if (!p.name.empty()) {
            p.source = "database";
            db.upsert(std::move(p));
        }
        search = idx;
    }

    return db;
}

} // namespace einsums::compute_graph
