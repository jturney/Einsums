//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Hardware/Machine.hpp>

#include <fmt/format.h>

#include <Stripes/RuntimeFeatures.hpp>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <set>
#include <string>
#include <thread>
#include <utility>

#include "Detect.hpp"

#if defined(__APPLE__)
#    include <sys/sysctl.h>
#    include <sys/types.h>
#elif defined(_WIN32)
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#else
#    include <unistd.h>
#endif

EINSUMS_NAMESPACE_BEGIN(hardware)

namespace {

#if defined(__linux__) && !(defined(__x86_64__) || defined(_M_X64))
/// The value after the colon of a /proc/cpuinfo line, trimmed.
std::string cpuinfo_value(std::string const &line) {
    auto const colon = line.find(':');
    if (colon == std::string::npos) {
        return {};
    }
    auto const value = line.substr(colon + 1);
    auto const begin = value.find_first_not_of(" \t");
    auto const end   = value.find_last_not_of(" \t\r\n");
    return begin == std::string::npos ? std::string{} : value.substr(begin, end - begin + 1);
}
#endif

/// Logical processors: every hardware thread the OS schedules on.
int detect_logical_cores() {
    unsigned const count = std::thread::hardware_concurrency();
    return count == 0 ? 1 : static_cast<int>(count);
}

/// Physical cores, or zero when the platform does not say.
int detect_physical_cores() {
#if defined(__APPLE__)
    int    value = 0;
    size_t len   = sizeof(value);
    if (sysctlbyname("hw.physicalcpu", &value, &len, nullptr, 0) == 0 && value > 0) {
        return value;
    }
    return 0;
#elif defined(_WIN32)
    DWORD length = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &length);
    if (length == 0) {
        return 0;
    }
    std::string buffer(length, '\0');
    auto       *info = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *>(buffer.data());
    if (GetLogicalProcessorInformationEx(RelationProcessorCore, info, &length) == 0) {
        return 0;
    }
    int   cores = 0;
    DWORD seen  = 0;
    while (seen < length) {
        auto const *entry = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX const *>(buffer.data() + seen);
        ++cores;
        seen += entry->Size;
    }
    return cores;
#elif defined(__linux__)
    // One core per distinct (package, core) pair. An aarch64 kernel publishes neither, and then
    // the logical count is the honest answer.
    std::ifstream                                 cpuinfo("/proc/cpuinfo");
    std::string                                   line;
    std::string                                   package;
    std::set<std::pair<std::string, std::string>> cores;
    while (std::getline(cpuinfo, line)) {
        if (line.rfind("physical id", 0) == 0) {
            package = line.substr(line.find(':') + 1);
        } else if (line.rfind("core id", 0) == 0) {
            cores.emplace(package, line.substr(line.find(':') + 1));
        }
    }
    return static_cast<int>(cores.size());
#else
    return 0;
#endif
}

/// A 64-bit FNV-1a hash: small, stable across builds and platforms, and enough to tell machine
/// kinds apart. Not a security property.
std::uint64_t fnv1a(std::string_view text) {
    std::uint64_t hash = 0xcbf29ce484222325ULL;
    for (char const c : text) {
        hash ^= static_cast<unsigned char>(c);
        hash *= 0x100000001b3ULL;
    }
    return hash;
}

/// The identity's fields as a JSON object, without the fingerprint: the domain the fingerprint
/// hashes. Key order is part of it, so it is fixed here.
json::Object identity_fields(MachineIdentity const &identity) {
    json::Object caches;
    caches.set("l1", identity.caches.l1);
    caches.set("l2", identity.caches.l2);
    caches.set("l3", identity.caches.l3);

    json::Object fields;
    fields.set("cpu", identity.cpu);
    fields.set("physical_cores", identity.physical_cores);
    fields.set("logical_cores", identity.logical_cores);
    fields.set("caches", std::move(caches));
    fields.set("memory_gib", identity.memory_gib);
    fields.set("simd", identity.simd);
    return fields;
}

} // namespace

std::string cpu_brand() {
#if defined(__APPLE__)
    char   buffer[256] = {}; // NOLINT
    size_t len         = sizeof(buffer);
    if (sysctlbyname("machdep.cpu.brand_string", buffer, &len, nullptr, 0) == 0) {
        return {buffer};
    }
#elif defined(__x86_64__) || defined(_M_X64)
    // CPUID brand string: functions 0x80000002 to 0x80000004.
    char     brand[49] = {};
    unsigned regs[4];
    for (int i = 0; i < 3; i++) {
        __asm__ __volatile__("cpuid" : "=a"(regs[0]), "=b"(regs[1]), "=c"(regs[2]), "=d"(regs[3]) : "a"(0x80000002 + i), "c"(0));
        std::memcpy(brand + i * 16, regs, 16);
    }
    std::string name(brand);
    auto const  end = name.find_last_not_of(" \t");
    return end == std::string::npos ? std::string{"Unknown CPU"} : name.substr(0, end + 1);
#elif defined(__linux__)
    // Non-x86 Linux, in practice aarch64: there is no CPUID, and an ARM /proc/cpuinfo usually
    // carries no "model name" line, identifying the CPU by numeric implementer and part codes
    // instead. A hypervisor commonly masks the part number (Docker on Apple silicon reports
    // implementer 0x61 with part 0x000), so the implementer alone has to be enough to name
    // something useful.
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string   line;
    std::string   implementer;
    std::string   part;
    while (std::getline(cpuinfo, line)) {
        // Some ARM boards, and every x86 kernel, do publish a name. Prefer it whenever present.
        if (line.rfind("model name", 0) == 0 || line.rfind("Model", 0) == 0) {
            if (auto const value = cpuinfo_value(line); !value.empty()) {
                return value;
            }
        } else if (line.rfind("CPU implementer", 0) == 0) {
            implementer = cpuinfo_value(line);
        } else if (line.rfind("CPU part", 0) == 0) {
            part = cpuinfo_value(line);
        }
    }

    if (!implementer.empty()) {
        struct CodeName {
            char const *code;
            char const *name;
        };
        // Implementer codes are assigned by ARM. Anything unlisted still reports its raw code
        // rather than falling through to "Unknown CPU".
        static constexpr CodeName kImplementers[] = {{"0x41", "ARM"},       {"0x42", "Broadcom"}, {"0x43", "Cavium"}, {"0x46", "Fujitsu"},
                                                     {"0x48", "HiSilicon"}, {"0x4e", "NVIDIA"},   {"0x50", "APM"},    {"0x51", "Qualcomm"},
                                                     {"0x53", "Samsung"},   {"0x56", "Marvell"},  {"0x61", "Apple"},  {"0x69", "Intel"},
                                                     {"0x70", "Phytium"},   {"0xc0", "Ampere"}};
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
#endif
    return "Unknown CPU";
}

std::int64_t physical_memory_bytes() {
#if defined(__APPLE__)
    std::int64_t physical = 0;
    size_t       len      = sizeof(physical);
    if (sysctlbyname("hw.memsize", &physical, &len, nullptr, 0) == 0 && physical > 0) {
        return physical;
    }
    return 0;
#elif defined(_WIN32)
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status) != 0) {
        return static_cast<std::int64_t>(status.ullTotalPhys);
    }
    return 0;
#else
    long const pages = sysconf(_SC_PHYS_PAGES);
    long const page  = sysconf(_SC_PAGE_SIZE);
    if (pages > 0 && page > 0) {
        return static_cast<std::int64_t>(pages) * static_cast<std::int64_t>(page);
    }
    return 0;
#endif
}

MachineIdentity const &machine_identity() {
    static MachineIdentity const identity = [] {
        MachineIdentity id;
        id.cpu            = cpu_brand();
        id.logical_cores  = detect_logical_cores();
        int const cores   = detect_physical_cores();
        id.physical_cores = cores > 0 ? cores : id.logical_cores;
        id.caches         = detail::detected_cache_sizes();
        std::int64_t const gib{std::int64_t{1} << 30};
        id.memory_gib = (physical_memory_bytes() + gib / 2) / gib;
        id.simd       = stripes::to_string(stripes::selected_arch());
        return id;
    }();
    return identity;
}

std::string fingerprint(MachineIdentity const &identity) {
    return fmt::format("{:016x}", fnv1a(json::emit(json::Value{identity_fields(identity)})));
}

std::string machine_fingerprint() {
    return fingerprint(machine_identity());
}

json::Object machine_section(MachineIdentity const &identity) {
    json::Object section = identity_fields(identity);
    section.set("fingerprint", fingerprint(identity));
    return section;
}

std::optional<MachineIdentity> read_machine_section(json::Object const &section, std::string &error) {
    auto const string_at = [&](std::string_view key, std::string &out) {
        auto const *value = section.take(key);
        if (value == nullptr || !value->is_string()) {
            error = fmt::format("machine.{} is missing or is not a string", key);
            return false;
        }
        out = value->as_string();
        return true;
    };
    auto const int_at = [&](json::Object const &object, std::string_view path, std::string_view key, std::int64_t &out) {
        auto const *value = object.take(key);
        if (value == nullptr || !value->is_int()) {
            error = fmt::format("{}.{} is missing or is not an integer", path, key);
            return false;
        }
        out = value->as_int();
        return true;
    };

    MachineIdentity id;
    std::int64_t    physical = 0;
    std::int64_t    logical  = 0;
    if (!string_at("cpu", id.cpu) || !int_at(section, "machine", "physical_cores", physical) ||
        !int_at(section, "machine", "logical_cores", logical) || !int_at(section, "machine", "memory_gib", id.memory_gib) ||
        !string_at("simd", id.simd)) {
        return std::nullopt;
    }
    id.physical_cores = static_cast<int>(physical);
    id.logical_cores  = static_cast<int>(logical);

    auto const *caches = section.take("caches");
    if (caches == nullptr || !caches->is_object()) {
        error = "machine.caches is missing or is not an object";
        return std::nullopt;
    }
    auto const &levels = caches->as_object();
    if (!int_at(levels, "machine.caches", "l1", id.caches.l1) || !int_at(levels, "machine.caches", "l2", id.caches.l2) ||
        !int_at(levels, "machine.caches", "l3", id.caches.l3)) {
        return std::nullopt;
    }
    return id;
}

EINSUMS_NAMESPACE_END(hardware)
