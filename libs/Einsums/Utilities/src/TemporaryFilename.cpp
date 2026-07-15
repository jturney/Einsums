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

    // Re-draw on the (statistically negligible, but possible) collision with
    // an existing file, e.g. leftovers from a crashed prior run. This is
    // advisory only - a check-then-use race remains by nature - so callers
    // that create the file should still open it exclusively.
    auto const dir = std::filesystem::temp_directory_path();
    for (;;) {
        auto candidate = dir / (std::to_string(dist(rng)) + ".tmp");
        if (!std::filesystem::exists(candidate)) {
            return candidate;
        }
    }
}

} // namespace einsums