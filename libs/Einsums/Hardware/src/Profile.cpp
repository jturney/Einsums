//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Config/Version.hpp>
#include <Einsums/Hardware/Options.hpp>
#include <Einsums/Hardware/Profile.hpp>
#include <Einsums/Logging.hpp>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <mutex>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "Detect.hpp"

#ifdef _OPENMP
#    include <omp.h>
#endif

#ifndef EINSUMS_PROFILE_SITE_DIR
#    define EINSUMS_PROFILE_SITE_DIR ""
#endif

EINSUMS_NAMESPACE_BEGIN(hardware)

namespace {

/// The top-level keys a profile may hold. Anything else makes the file invalid.
constexpr std::array<std::string_view, 6> kKnownSections{"einsums_profile",    "machine",   "software", "measured",
                                                         "omp_region_cost_ns", "cost_model"};

std::atomic<std::string (*)()> blas_provider{nullptr};

/// "2.0.3" into its three numbers.
std::optional<std::array<int, 3>> parse_version(std::string_view text) {
    std::array<int, 3> parts{};
    char const        *at  = text.data();
    char const        *end = text.data() + text.size();
    for (std::size_t i = 0; i < parts.size(); ++i) {
        auto const [next, ec] = std::from_chars(at, end, parts[i]);
        if (ec != std::errc{}) {
            return std::nullopt;
        }
        at = next;
        if (i + 1 < parts.size()) {
            if (at == end || *at != '.') {
                return std::nullopt;
            }
            ++at;
        }
    }
    if (at != end) {
        return std::nullopt;
    }
    return parts;
}

/// The fields of two identities that differ, by name, for a reason a user can act on.
std::string machine_difference(MachineIdentity const &file, MachineIdentity const &here) {
    std::vector<std::string> fields;
    auto const               note = [&fields](std::string_view name, auto const &a, auto const &b) {
        if (a != b) {
            fields.push_back(fmt::format("{} {} here, {} in the file", name, b, a));
        }
    };
    note("cpu", file.cpu, here.cpu);
    note("physical cores", file.physical_cores, here.physical_cores);
    note("logical cores", file.logical_cores, here.logical_cores);
    note("L1", file.caches.l1, here.caches.l1);
    note("L2", file.caches.l2, here.caches.l2);
    note("L3", file.caches.l3, here.caches.l3);
    note("memory GiB", file.memory_gib, here.memory_gib);
    note("simd", file.simd, here.simd);
    return fmt::format("{}", fmt::join(fields, "; "));
}

/// The profile in effect and the option values it was resolved under.
struct Active {
    std::string   key;
    ProfileStatus status;
    json::Value   document;
};

std::mutex                    active_mutex;
std::shared_ptr<Active const> active;

/// The site directory: --einsums:hardware:site-profiles, else the installation's.
std::filesystem::path site_directory() {
    if (auto dir = config::get(option::HardwareSiteProfiles); !dir.empty()) {
        return std::filesystem::path(dir);
    }
    return std::filesystem::path(EINSUMS_PROFILE_SITE_DIR);
}

std::string file_name() {
    return machine_fingerprint() + ".json";
}

/// Read and parse @p path, or say why not.
std::optional<json::Value> read_document(std::filesystem::path const &path, std::string &error) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        error = "cannot be opened";
        return std::nullopt;
    }
    std::string const text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    auto              document = json::parse(text);
    if (!document) {
        error = fmt::format("is not valid JSON: {}", document.error().to_string());
        return std::nullopt;
    }
    return std::move(*document);
}

/// Evaluate the file at @p path, if there is one.
std::optional<std::pair<ProfileStatus, json::Value>> consider(std::filesystem::path const &path, bool named) {
    std::error_code ec;
    if (path.empty() || !std::filesystem::exists(path, ec)) {
        if (named) {
            return std::pair{
                ProfileStatus{.state = ProfileState::Invalid, .path = path.string(), .reason = "the named file does not exist"},
                json::Value{}};
        }
        return std::nullopt;
    }
    std::string error;
    auto        document = read_document(path, error);
    if (!document) {
        return std::pair{ProfileStatus{.state = ProfileState::Invalid, .path = path.string(), .reason = "the file " + error},
                         json::Value{}};
    }
    auto status = evaluate_profile(*document, path.string(), machine_identity(), software_identity(), named);
    return std::pair{std::move(status), std::move(*document)};
}

/// Find the profile that applies, keeping the first reason one did not.
std::shared_ptr<Active const> resolve(std::string key) {
    auto result = std::make_shared<Active>();
    result->key = std::move(key);

    std::vector<std::pair<std::filesystem::path, bool>> candidates;
    if (auto const named = config::get(option::HardwareProfile); !named.empty()) {
        candidates.emplace_back(std::filesystem::path(named), true);
    } else {
        for (auto const &place : profile_search_path()) {
            candidates.emplace_back(std::filesystem::path(place), false);
        }
    }

    std::optional<ProfileStatus> first_refusal;
    for (auto const &[path, named] : candidates) {
        auto found = consider(path, named);
        if (!found) {
            continue;
        }
        if (found->first.in_use()) {
            result->status   = std::move(found->first);
            result->document = std::move(found->second);
            first_refusal.reset();
            break;
        }
        if (!first_refusal) {
            first_refusal = std::move(found->first);
        }
    }
    if (first_refusal) {
        result->status = std::move(*first_refusal);
    }

    switch (result->status.state) {
    case ProfileState::Loaded:
        EINSUMS_LOG_INFO("hardware profile: using {}", result->status.path);
        break;
    case ProfileState::LoadedForeign:
        EINSUMS_LOG_WARN("hardware profile: using {}, which was measured on another machine ({})", result->status.path,
                         result->status.reason);
        break;
    case ProfileState::None:
        EINSUMS_LOG_INFO("hardware profile: none for this machine; run einsums-tune to measure one");
        break;
    default:
        EINSUMS_LOG_WARN("hardware profile: not using {}: {}; run einsums-tune to measure a current one", result->status.path,
                         result->status.reason);
        break;
    }
    return result;
}

/// The profile in effect, resolved again only when an option that locates it has changed.
std::shared_ptr<Active const> current() {
    std::string                       key = fmt::format("{}\n{}\n{}", config::get(option::HardwareProfile), config::get(option::CacheDir),
                                                        config::get(option::HardwareSiteProfiles));
    std::lock_guard<std::mutex> const guard(active_mutex);
    if (!active || active->key != key) {
        active = resolve(std::move(key));
    }
    return active;
}

} // namespace

void set_blas_identity_provider(std::string (*provider)()) {
    blas_provider.store(provider);
}

SoftwareIdentity const &software_identity() {
    static SoftwareIdentity const identity = [] {
        SoftwareIdentity id;
        id.major            = EINSUMS_VERSION_MAJOR;
        id.minor            = EINSUMS_VERSION_MINOR;
        id.patch            = EINSUMS_VERSION_PATCH;
        auto const provider = blas_provider.load();
        id.blas             = provider != nullptr ? provider() : std::string{"unidentified"};
        return id;
    }();
    return identity;
}

json::Object software_section(SoftwareIdentity const &identity) {
    json::Object section;
    section.set("einsums", fmt::format("{}.{}.{}", identity.major, identity.minor, identity.patch));
    section.set("blas", identity.blas);
    return section;
}

std::string_view to_string(ProfileState state) noexcept {
    switch (state) {
    case ProfileState::None:
        return "none";
    case ProfileState::Loaded:
        return "loaded";
    case ProfileState::LoadedForeign:
        return "loaded-foreign";
    case ProfileState::Foreign:
        return "foreign";
    case ProfileState::Stale:
        return "stale";
    case ProfileState::Invalid:
        return "invalid";
    }
    return "invalid";
}

ProfileStatus evaluate_profile(json::Value const &document, std::string const &path, MachineIdentity const &machine,
                               SoftwareIdentity const &software, bool named) {
    auto const invalid = [&path](std::string reason) {
        return ProfileStatus{.state = ProfileState::Invalid, .path = path, .reason = std::move(reason)};
    };
    if (!document.is_object()) {
        return invalid(fmt::format("the file holds a {} where an object was expected", document.type_name()));
    }
    auto const &root = document.as_object();
    for (auto const &key : root.keys()) {
        if (std::ranges::find(kKnownSections, key) == kKnownSections.end()) {
            return invalid(fmt::format("unknown section '{}'", key));
        }
    }

    auto const *format = root.peek("einsums_profile");
    if (format == nullptr || !format->is_int()) {
        return invalid("einsums_profile is missing or is not an integer");
    }
    if (format->as_int() != profile_format) {
        return invalid(fmt::format("format {} where this build reads {}", format->as_int(), profile_format));
    }

    // The machine: its fields, then the stored fingerprint against the fields, so a hand edit
    // that changed one without the other is caught rather than trusted.
    auto const *machine_value = root.peek("machine");
    if (machine_value == nullptr || !machine_value->is_object()) {
        return invalid("the machine section is missing or is not an object");
    }
    json::Object const &machine_object = machine_value->as_object();
    std::string         error;
    auto const          file_machine = read_machine_section(machine_object, error);
    if (!file_machine) {
        return invalid(error);
    }
    auto const *stored = machine_object.take("fingerprint");
    if (stored == nullptr || !stored->is_string()) {
        return invalid("machine.fingerprint is missing or is not a string");
    }
    if (auto const left = machine_object.unconsumed_keys(); !left.empty()) {
        return invalid(fmt::format("unknown key 'machine.{}'", left.front()));
    }
    if (stored->as_string() != fingerprint(*file_machine)) {
        return invalid("machine.fingerprint does not match the machine it describes; the file was edited");
    }

    // The software.
    auto const *software_value = root.peek("software");
    if (software_value == nullptr || !software_value->is_object()) {
        return invalid("the software section is missing or is not an object");
    }
    json::Object const &software_object = software_value->as_object();
    auto const         *einsums         = software_object.take("einsums");
    auto const         *blas            = software_object.take("blas");
    if (einsums == nullptr || !einsums->is_string() || blas == nullptr || !blas->is_string()) {
        return invalid("software.einsums or software.blas is missing or is not a string");
    }
    if (auto const left = software_object.unconsumed_keys(); !left.empty()) {
        return invalid(fmt::format("unknown key 'software.{}'", left.front()));
    }
    auto const version = parse_version(einsums->as_string());
    if (!version) {
        return invalid(fmt::format("software.einsums '{}' is not a version", einsums->as_string()));
    }

    // When. Only the date so far; a section is read strictly like every other.
    if (auto const *measured = root.peek("measured"); measured != nullptr) {
        if (!measured->is_object()) {
            return invalid("the measured section is not an object");
        }
        auto const &measured_object = measured->as_object();
        if (auto const *date = measured_object.take("date"); date != nullptr && !date->is_string()) {
            return invalid("measured.date is not a string");
        }
        if (auto const left = measured_object.unconsumed_keys(); !left.empty()) {
            return invalid(fmt::format("unknown key 'measured.{}'", left.front()));
        }
    }

    // Kernels change between minor versions; a patch release keeps them.
    if ((*version)[0] != software.major || (*version)[1] != software.minor) {
        return ProfileStatus{.state  = ProfileState::Stale,
                             .path   = path,
                             .reason = fmt::format("measured with Einsums {}, and this is {}.{}.{}", einsums->as_string(), software.major,
                                                   software.minor, software.patch)};
    }
    if (blas->as_string() != software.blas) {
        return ProfileStatus{.state = ProfileState::Stale,
                             .path  = path,
                             .reason =
                                 fmt::format("measured with BLAS '{}', and this process uses '{}'", blas->as_string(), software.blas)};
    }

    if (*file_machine != machine) {
        std::string const difference = machine_difference(*file_machine, machine);
        return ProfileStatus{.state = named ? ProfileState::LoadedForeign : ProfileState::Foreign, .path = path, .reason = difference};
    }
    return ProfileStatus{.state = ProfileState::Loaded, .path = path, .reason = {}};
}

ProfileStatus profile_status() {
    return current()->status;
}

std::string profile_state() {
    return std::string(to_string(current()->status.state));
}

std::string profile_path() {
    return current()->status.path;
}

std::string profile_reason() {
    return current()->status.reason;
}

std::optional<json::Value> profile_section(std::string_view name) {
    auto const profile = current();
    if (!profile->status.in_use()) {
        return std::nullopt;
    }
    auto const *section = profile->document.as_object().peek(name);
    if (section == nullptr) {
        return std::nullopt;
    }
    return *section;
}

std::string default_profile_path() {
    auto const dir = detail::cache_directory();
    if (dir.empty()) {
        return {};
    }
    return (dir / "profiles" / file_name()).string();
}

std::vector<std::string> profile_search_path() {
    std::vector<std::string> places;
    if (auto path = default_profile_path(); !path.empty()) {
        places.push_back(std::move(path));
    }
    if (auto const site = site_directory(); !site.empty()) {
        places.push_back((site / file_name()).string());
    }
    return places;
}

json::Object profile_header() {
    json::Object header;
    header.set("einsums_profile", profile_format);
    header.set("machine", machine_section(machine_identity()));
    header.set("software", software_section(software_identity()));
    return header;
}

std::string profile_header_json() {
    return json::emit(json::Value{profile_header()});
}

json::Object measure_region_cost_section() {
#ifdef _OPENMP
    int const max_threads = omp_get_max_threads();
#else
    int const max_threads = 1;
#endif
    json::Object section;
    for (int t = 1; t <= max_threads; ++t) {
#ifdef _OPENMP
        omp_set_num_threads(t);
#endif
        section.set(std::to_string(t), detail::measure_omp_region_cost_ns());
    }
#ifdef _OPENMP
    omp_set_num_threads(max_threads);
#endif
    return section;
}

bool write_profile(std::string const &path, json::Value const &document, std::string &error) {
    std::filesystem::path const target(path);
    if (target.empty()) {
        error = "no path given and no usable cache directory";
        return false;
    }
    std::error_code ec;
    if (target.has_parent_path()) {
        std::filesystem::create_directories(target.parent_path(), ec);
        if (ec) {
            error = fmt::format("could not create {}: {}", target.parent_path().string(), ec.message());
            return false;
        }
    }
    std::filesystem::path const tmp = target.string() + ".tmp";
    {
        std::ofstream out(tmp, std::ios::trunc | std::ios::binary);
        if (!out) {
            error = fmt::format("could not open {} for writing", tmp.string());
            return false;
        }
        out << json::emit(document, json::EmitOptions{.style = json::EmitStyle::Pretty}) << '\n';
        if (!out) {
            error = fmt::format("could not write {}", tmp.string());
            return false;
        }
    }
    std::filesystem::rename(tmp, target, ec);
    if (ec) {
        std::filesystem::remove(tmp, ec);
        error = fmt::format("could not rename into {}", target.string());
        return false;
    }
    return true;
}

EINSUMS_NAMESPACE_END(hardware)
