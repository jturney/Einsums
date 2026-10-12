//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/**
 * @file Machine.hpp
 * @brief What machine this is, as a hardware profile names it.
 *
 * A profile measured on one machine is valid on another only when the two are the same kind of
 * machine, so the identity is the hardware a measurement depends on and nothing else: the CPU,
 * its core counts, its caches, its memory and the SIMD rung the kernels dispatch to. The host
 * name is deliberately absent. On a cluster with shared home directories every node of one type
 * shares one profile, and a login node with different hardware has a different identity, so its
 * profile is never chosen on a compute node.
 */

#include <Einsums/Config/ExportDefinitions.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Hardware/CpuInfo.hpp>
#include <Einsums/Json/Json.hpp>
#include <Einsums/Python/Annotations.hpp>

#include <cstdint>
#include <optional>
#include <string>

EINSUMS_NAMESPACE_BEGIN(hardware)

/**
 * @brief The hardware a measurement depends on.
 *
 * Every field is detected, never taken from an option, so pinning a cache size for an A/B run
 * does not make the machine look like a different one.
 *
 * @versionadded{2.0.0}
 */
struct MachineIdentity {
    /// The CPU brand string, such as "Apple M4 Pro".
    std::string cpu;
    /// Physical cores; the logical count when the platform does not say.
    int physical_cores{1};
    /// Logical processors, hardware threads included.
    int logical_cores{1};
    /// Data-cache sizes in bytes, detected rather than overridden.
    CacheSizes caches;
    /// Physical memory in GiB, rounded to the nearest. Identical nodes can report a few MiB apart
    /// once the kernel has reserved its share, and an exact byte count would tell them apart.
    std::int64_t memory_gib{0};
    /// The SIMD rung the kernels dispatch to, such as "x86-64-v3" or "sme".
    std::string simd;

    /// @brief Field-by-field equality.
    bool operator==(MachineIdentity const &) const = default;
};

/// @brief The CPU brand string: sysctl on macOS, CPUID on x86-64, /proc/cpuinfo on other Linux.
/// @return The brand, or "Unknown CPU" when the platform will not say.
/// @versionadded{2.0.0}
APIARY_EXPOSE APIARY_MODULE("hardware") EINSUMS_EXPORT std::string cpu_brand();

/// @brief Physical memory in bytes.
/// @return The total, or zero when the platform will not say.
/// @versionadded{2.0.0}
APIARY_EXPOSE APIARY_MODULE("hardware") EINSUMS_EXPORT std::int64_t physical_memory_bytes();

/// @brief This machine's identity, detected once.
/// @return The identity.
/// @versionadded{2.0.0}
EINSUMS_EXPORT MachineIdentity const &machine_identity();

/**
 * @brief The identity as the @c machine section of a profile writes it, fingerprint included.
 * @param[in] identity The identity to write.
 * @return The section.
 * @versionadded{2.0.0}
 */
EINSUMS_EXPORT json::Object machine_section(MachineIdentity const &identity);

/**
 * @brief Read an identity back from a @c machine section.
 *
 * Every key the section must have is consumed, and the fingerprint is left for the caller, which
 * compares it with the one it recomputes.
 *
 * @param[in] section The section.
 * @param[out] error Why the section is unreadable, when it is.
 * @return The identity, or nothing when a field is missing or of the wrong type.
 * @versionadded{2.0.0}
 */
EINSUMS_EXPORT std::optional<MachineIdentity> read_machine_section(json::Object const &section, std::string &error);

/**
 * @brief A short, stable name for an identity: 16 hexadecimal digits.
 *
 * A hash of the identity's canonical JSON, so two machines share a fingerprint exactly when every
 * field agrees, and the same machine has the same one in every process and every build.
 *
 * @param[in] identity The identity.
 * @return The fingerprint.
 * @versionadded{2.0.0}
 */
EINSUMS_EXPORT std::string fingerprint(MachineIdentity const &identity);

/// @brief This machine's fingerprint, the name its profile is filed under.
/// @return @ref fingerprint of @ref machine_identity.
/// @versionadded{2.0.0}
APIARY_EXPOSE APIARY_MODULE("hardware") EINSUMS_EXPORT std::string machine_fingerprint();

EINSUMS_NAMESPACE_END(hardware)
