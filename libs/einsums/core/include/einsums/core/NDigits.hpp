//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

namespace einsums {

template <typename T>
inline auto ndigits(T number) -> int {
    int digits{0};
    if (number < 0)
        digits = 1; // Remove this line if '-' counts as a digit
    while (number) {
        number /= 10;
        digits++;
    }
    return digits;
}

} // namespace einsums