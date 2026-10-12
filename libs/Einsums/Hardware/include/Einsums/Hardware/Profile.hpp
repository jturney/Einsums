//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/**
 * @file Profile.hpp
 * @brief The hardware profile: one file per machine holding everything a tune measured.
 *
 * @par The file
 * One JSON object. Its header says what it describes: @c einsums_profile (the format),
 * @c machine (see Machine.hpp), @c software (the Einsums version and the BLAS) and @c measured
 * (when). Each remaining section belongs to the module that reads it: @c omp_region_cost_ns to
 * this one, @c cost_model to ComputeGraph's cost model, and @c options to the option parser, as a
 * layer below the environment and the command line. An unknown top-level key makes the whole
 * file invalid, naming the key, for the reason the graph IR refuses one: a reader that skips what
 * it does not understand loads a half-understood file as if it were whole.
 *
 * @par Which file applies
 * The file named by @c --einsums:hardware:profile, used as given except that a different
 * software makes it stale; otherwise the first applicable file among
 * @c <cache-dir>/profiles/<fingerprint>.json and the same name under the site directory. A file
 * applies when its machine is this machine, its BLAS is the one in use, and its Einsums version
 * differs from this build's at most in the patch number. A file that does not apply is skipped,
 * and the first such reason is kept for @ref profile_status, so a user who tuned and sees no
 * effect can be told why.
 *
 * @par When it is read
 * Once per process, on first use, and again only if one of the options that locate it changes,
 * which tests rely on. Nothing in the library writes a profile; the @c einsums-tune tool does.
 */

#include <Einsums/Config/ExportDefinitions.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Hardware/Machine.hpp>
#include <Einsums/Json/Json.hpp>
#include <Einsums/Python/Annotations.hpp>

#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(hardware)

/// The profile format this build reads and writes.
inline constexpr std::int64_t profile_format = 1;

/**
 * @brief The software a measurement depends on: this build's version and the BLAS in use.
 * @versionadded{2.0.0}
 */
struct SoftwareIdentity {
    int         major{0}; ///< Einsums major version.
    int         minor{0}; ///< Einsums minor version.
    int         patch{0}; ///< Einsums patch version.
    std::string blas;     ///< The BLAS library providing dgemm, with its version where it reports one.

    /// @brief Field-by-field equality.
    bool operator==(SoftwareIdentity const &) const = default;
};

/**
 * @brief Register the function that names the BLAS in use.
 *
 * The BLAS sits above this module, so it registers itself at start-up rather than this module
 * reaching up for it. Until it does, the BLAS is named "unidentified".
 *
 * @param[in] provider Returns the name; called once, on first use of @ref software_identity.
 * @versionadded{2.0.0}
 */
EINSUMS_EXPORT void set_blas_identity_provider(std::string (*provider)());

/// @brief This build's software identity.
/// @return The identity.
/// @versionadded{2.0.0}
EINSUMS_EXPORT SoftwareIdentity const &software_identity();

/// @brief The identity as the @c software section of a profile writes it.
/// @param[in] identity The identity.
/// @return The section.
/// @versionadded{2.0.0}
EINSUMS_EXPORT json::Object software_section(SoftwareIdentity const &identity);

/**
 * @brief What became of a profile file.
 * @versionadded{2.0.0}
 */
enum class ProfileState : std::uint8_t {
    None,          ///< No file was found; built-in values are used.
    Loaded,        ///< A file applies and is in use.
    LoadedForeign, ///< A named file from another machine is in use, as the user asked.
    Foreign,       ///< A file was found but describes another machine, so it is not used.
    Stale,         ///< A file was found but another Einsums version or BLAS measured it, so it is not used.
    Invalid,       ///< A file was found but cannot be read, so it is not used.
};

/// @brief The state's name: "none", "loaded", "loaded-foreign", "foreign", "stale" or "invalid".
/// @param[in] state The state.
/// @return The name.
/// @versionadded{2.0.0}
EINSUMS_EXPORT std::string_view to_string(ProfileState state) noexcept;

/**
 * @brief A profile file's verdict: its state, its path and, unless loaded cleanly, why.
 * @versionadded{2.0.0}
 */
struct ProfileStatus {
    ProfileState state{ProfileState::None}; ///< What became of the file.
    std::string  path;                      ///< The file the verdict is about; empty for @ref ProfileState::None.
    std::string  reason;                    ///< Why it is not plainly loaded; empty when it is.

    /// @brief Whether the file's sections are in use.
    [[nodiscard]] bool in_use() const noexcept { return state == ProfileState::Loaded || state == ProfileState::LoadedForeign; }
};

/**
 * @brief Decide whether a profile may set an option.
 *
 * Returns an empty string when it may, and otherwise why not. The option parser registers one at
 * start-up, since the registry of options lives above this module; until it does, a profile may
 * set no option.
 *
 * @versionadded{2.0.0}
 */
using ProfileOptionCheck = std::string (*)(std::string_view name);

/// @brief Register the check @ref evaluate_profile applies to the @c options section.
/// @param[in] check The check.
/// @versionadded{2.0.0}
EINSUMS_EXPORT void set_profile_option_check(ProfileOptionCheck check);

/**
 * @brief Decide what a profile document means here.
 *
 * Pure, so every rule can be tested without a file: the format, the known top-level keys, the
 * machine (whose stored fingerprint must also match its fields), the software, and the
 * @c options section, whose values must be scalars and whose names @p option_check must accept.
 *
 * @param[in] document The parsed file.
 * @param[in] path Where it came from, for the status.
 * @param[in] machine This machine.
 * @param[in] software This build and BLAS.
 * @param[in] named Whether the user named the file, which makes another machine a warning
 *                  rather than a refusal.
 * @param[in] option_check Whether the @c options section may set each option; null refuses all.
 * @return The verdict.
 * @versionadded{2.0.0}
 */
EINSUMS_EXPORT ProfileStatus evaluate_profile(json::Value const &document, std::string const &path, MachineIdentity const &machine,
                                              SoftwareIdentity const &software, bool named, ProfileOptionCheck option_check);

/// @brief The verdict on the profile in effect.
/// @return The status, resolved on first use.
/// @versionadded{2.0.0}
EINSUMS_EXPORT ProfileStatus profile_status();

/// @brief The state of the profile in effect, by name (see @ref to_string).
/// @return The name.
/// @versionadded{2.0.0}
APIARY_EXPOSE APIARY_MODULE("hardware") EINSUMS_EXPORT std::string profile_state();

/// @brief The file the profile in effect came from, or empty when none was found.
/// @return The path.
/// @versionadded{2.0.0}
APIARY_EXPOSE APIARY_MODULE("hardware") EINSUMS_EXPORT std::string profile_path();

/// @brief Why the profile in effect is not plainly loaded, or empty when it is.
/// @return The reason.
/// @versionadded{2.0.0}
APIARY_EXPOSE APIARY_MODULE("hardware") EINSUMS_EXPORT std::string profile_reason();

/**
 * @brief A section of the profile in effect.
 *
 * @param[in] name The top-level key, such as "cost_model".
 * @return A copy of the section, or nothing when no profile is in use or it has no such section.
 * @versionadded{2.0.0}
 */
EINSUMS_EXPORT std::optional<json::Value> profile_section(std::string_view name);

/**
 * @brief The @c options section of the profile in effect, as the option parser's config layer.
 *
 * Each value spelled as the command line would spell it, keyed by the option's long name.
 *
 * @return The options, or an empty map when no profile is in use or it sets none.
 * @versionadded{2.0.0}
 */
EINSUMS_EXPORT std::map<std::string, std::string, std::less<>> profile_options();

/**
 * @brief Adopt a profile from another machine of a kind, or from a site.
 *
 * Checks the file against this machine and build and files it in the cache directory under its
 * own machine's fingerprint, so one from another kind of machine is kept for that kind and never
 * used here. A stale or invalid file is refused.
 *
 * @param[in] path The file.
 * @param[out] message What happened, for the user.
 * @return Whether the file was filed.
 * @versionadded{2.0.0}
 */
EINSUMS_EXPORT bool import_profile(std::string const &path, std::string &message);

/// @brief Where a tune writes this machine's profile: @c <cache-dir>/profiles/<fingerprint>.json.
/// @return The path, or empty when there is no cache directory.
/// @versionadded{2.0.0}
APIARY_EXPOSE APIARY_MODULE("hardware") EINSUMS_EXPORT std::string default_profile_path();

/// @brief Every place a profile is looked for, in order, after a named file.
/// @return The paths, existing or not.
/// @versionadded{2.0.0}
APIARY_EXPOSE APIARY_MODULE("hardware") EINSUMS_EXPORT std::vector<std::string> profile_search_path();

/**
 * @brief The header a profile for this machine and build starts with.
 *
 * @c einsums_profile, @c machine and @c software, in that order; a tune adds @c measured and its
 * sections after them.
 *
 * @return The header.
 * @versionadded{2.0.0}
 */
EINSUMS_EXPORT json::Object profile_header();

/// @brief @ref profile_header as JSON text, for tests and scripts that assemble a profile.
/// @return The canonical JSON.
/// @versionadded{2.0.0}
APIARY_EXPOSE APIARY_MODULE("hardware") EINSUMS_EXPORT std::string profile_header_json();

/**
 * @brief Measure the OpenMP region cost at every team size, as the @c omp_region_cost_ns section.
 *
 * Team sizes 1 to @c omp_get_max_threads(), a few milliseconds each; the team size is restored.
 *
 * @return The section, keyed by team size.
 * @versionadded{2.0.0}
 */
EINSUMS_EXPORT json::Object measure_region_cost_section();

/**
 * @brief Write a profile document, atomically.
 *
 * Written to a temporary beside @p path and renamed into place, so a process reading the profile
 * never sees half of one. Parent directories are created.
 *
 * @param[in] path Destination.
 * @param[in] document The profile.
 * @param[out] error Why it was not written, when it was not.
 * @return Whether the file was written.
 * @versionadded{2.0.0}
 */
EINSUMS_EXPORT bool write_profile(std::string const &path, json::Value const &document, std::string &error);

EINSUMS_NAMESPACE_END(hardware)
