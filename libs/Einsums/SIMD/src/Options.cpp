//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Logging.hpp>
#include <Einsums/Options/Declare.hpp>
#include <Einsums/SIMD/Options.hpp>

#include <Stripes/RuntimeFeatures.hpp>
#include <string>
#include <string_view>

EINSUMS_NAMESPACE_BEGIN()

int register_Einsums_SIMD_options() {
    cl::register_option(option::SimdArch);
    return 0;
}

namespace {
void log_stripes_message(stripes::MessageLevel level, std::string_view message) {
    if (level == stripes::MessageLevel::Warning) {
        EINSUMS_LOG_WARN("{}", message);
    } else {
        EINSUMS_LOG_DEBUG("SIMD {}", message);
    }
}
} // namespace

void apply_SIMD_options() {
    stripes::set_message_handler(&log_stripes_message);
    std::string const requested = config::get(option::SimdArch);
    if (requested.empty()) {
        return; // Stripes' own STRIPES_ARCH, or the best rung, decides
    }
    if (!stripes::set_arch_override(requested)) {
        EINSUMS_LOG_WARN("--einsums:simd:arch=\"{}\" came after the SIMD dispatch rung was chosen ({}), so it has no effect.", requested,
                         stripes::to_string(stripes::selected_arch()));
    }
}

EINSUMS_NAMESPACE_END()
