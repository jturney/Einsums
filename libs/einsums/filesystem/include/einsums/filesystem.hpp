//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <filesystem>
#include <string>
#include <system_error>

namespace einsums::detail::filesystem {

EINSUMS_EXPORT std::filesystem::path initial_path();
EINSUMS_EXPORT std::string basename(std::filesystem::path const &p);
EINSUMS_EXPORT std::filesystem::path canonical(std::filesystem::path const &p, std::filesystem::path const &base);
EINSUMS_EXPORT std::filesystem::path canonical(std::filesystem::path const &p, std::filesystem::path const &base, std::error_code &ec);

} // namespace einsums::detail::filesystem