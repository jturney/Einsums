//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Config/Namespace.hpp>
#include <Einsums/HPTT/HPTTTypes.hpp>
#include <Einsums/Options/Get.hpp>

/*
 * HPTT's plan-selection option, declared where the plans are built.
 */

EINSUMS_NAMESPACE_BEGIN(option)

/// How hard HPTT looks for a good transpose plan: estimate, measure, patient,
/// or crazy, in increasing order of search effort.
inline constinit cl::ConfigOption<std::string> HpttSelectionMethod = cl::config_opt<std::string>(
    "einsums:hptt:selection-method", "HPTT plan selection method (estimate, measure, patient, crazy)", "HPTT", "estimate", "METHOD");

EINSUMS_NAMESPACE_END(option)

EINSUMS_NAMESPACE_BEGIN()

/// @brief How hard HPTT should search for a plan. Reads
///        @ref option::HpttSelectionMethod and maps it to HPTT's enum.
///
/// Defined in the library rather than read from the descriptor at the call
/// site. A ConfigOption caches the address of its registry entry inside itself,
/// filled in when its module registers it, and a translation unit that compiles
/// these headers gets its OWN copy of the descriptor, whose entry is never
/// filled - so an inline read there quietly returns the default however the
/// option was set.
EINSUMS_EXPORT hptt::SelectionMethod hptt_plan_selection();

EINSUMS_NAMESPACE_END()

EINSUMS_NAMESPACE_BEGIN()

/**
 * @brief Give HPTT's option its command-line presence. Idempotent.
 */
EINSUMS_EXPORT int register_Einsums_HPTT_options();

namespace detail {
[[maybe_unused]] static int const register_options_Einsums_HPTT = register_Einsums_HPTT_options();
}

EINSUMS_NAMESPACE_END()
