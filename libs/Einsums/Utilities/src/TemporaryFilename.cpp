//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Utilities/TemporaryFilename.hpp>

#include <random>

namespace einsums {

std::filesystem::path make_temp_path() {
    static thread_local std::mt19937_64     rng(std::random_device{}());
    std::uniform_int_distribution<uint64_t> dist;

    auto name = std::to_string(dist(rng)) + ".tmp";
    return std::filesystem::temp_directory_path() / name;
}

} // namespace einsums