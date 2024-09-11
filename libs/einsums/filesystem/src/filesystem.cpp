//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <einsums/config.hpp>

#include <einsums/filesystem.hpp>

#include <filesystem>
#include <string>
#include <system_error>

namespace einsums::detail::filesystem {

std::filesystem::path initial_path() {
    static std::filesystem::path ip = std::filesystem::current_path();
    return ip;
}

std::string basename(std::filesystem::path const &p) {
    return p.stem().string();
}

std::filesystem::path canonical(std::filesystem::path const &p, std::filesystem::path const &base) {
    if (p.is_relative()) {
        return canonical(base / p);
    } else {
        return canonical(p);
    }
}

std::filesystem::path canonical(std::filesystem::path const &p, std::filesystem::path const &base, std::error_code &ec) {
    if (p.is_relative()) {
        return canonical(base / p, ec);
    } else {
        return canonical(p, ec);
    }
}

} // namespace einsums::detail::filesystem