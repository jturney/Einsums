//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file ClampAndSanitize.cpp
/// @brief Branch-free conditionals: comparisons make masks, select applies them.
///
/// A SIMD loop cannot branch per lane, so a conditional becomes a mask. cmp_lt and its siblings
/// return a Mask<float>, one flag per lane, set where the comparison holds; select(mask, a, b) then
/// takes a in the set lanes and b in the rest. Masks combine with &, |, ^ and !, any(mask) or
/// all(mask) turn one into a single bool, which is how a loop can skip work, or stop, when no lane
/// needs it, and count(mask) says how many lanes are set.
///
/// Here: replace every NaN with zero, clamp to [lo, hi], and count how many values were changed.
/// A NaN is the one value not equal to itself, so cmp_ne(v, v) finds them.

#include <Einsums/Runtime.hpp>
#include <Einsums/SIMD/Operations.hpp>
#include <Einsums/SIMD/Partial.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

using namespace einsums::simd;

namespace {

/// Clamps @p x to [lo, hi] with NaNs replaced by zero, and returns how many elements changed.
std::size_t sanitize(std::size_t n, float *x, float lo, float hi) {
    constexpr std::size_t L   = lanes<float>;
    Vec<float> const      vlo = broadcast(lo), vhi = broadcast(hi), zero = broadcast(0.0f);
    std::size_t           changed = 0;

    for (std::size_t i = 0; i < n; i += L) {
        std::size_t const width = std::min(L, n - i);
        Vec<float> const  v     = loadu_partial(x + i, width);

        Mask<float> const is_nan    = cmp_ne(v, v);
        Mask<float> const too_low   = cmp_lt(v, vlo);
        Mask<float> const too_high  = cmp_gt(v, vhi);
        Mask<float> const needs_fix = is_nan | too_low | too_high;
        if (!any(needs_fix)) {
            continue; // the common case: nothing in this Vec to change, so skip the store
        }

        // min and max return their second argument for a NaN, so clamp first and put the zeros in
        // afterwards rather than relying on how a NaN passes through them.
        Vec<float> const clamped = min(max(v, vlo), vhi);
        storeu_partial(x + i, select(is_nan, zero, clamped), width);

        // Count the set lanes that are real elements, not the zeroed padding of a partial load.
        changed += static_cast<std::size_t>(count(needs_fix & first_n<float>(width)));
    }
    return changed;
}

} // namespace

int einsums_main() {
    float const        nan = std::numeric_limits<float>::quiet_NaN();
    std::vector<float> x   = {0.5f, -3.0f, nan, 2.0f, 1.0f, 7.5f, -0.25f, nan, 0.0f, 1.5f, -1.0f, 4.0f, 0.75f};
    std::vector<float> expect(x.size());
    std::size_t        expect_changed = 0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        expect[i] = std::isnan(x[i]) ? 0.0f : std::fmin(std::fmax(x[i], -1.0f), 2.0f);
        expect_changed += std::isnan(x[i]) || x[i] < -1.0f || x[i] > 2.0f;
    }

    std::size_t const changed  = sanitize(x.size(), x.data(), -1.0f, 2.0f);
    int               failures = changed != expect_changed;
    std::cout << "changed " << changed << " of " << x.size() << ":";
    for (std::size_t i = 0; i < x.size(); ++i) {
        std::cout << " " << x[i];
        failures += x[i] != expect[i];
    }
    std::cout << "\n";
    return failures ? 1 : 0;
}

int main(int argc, char **argv) {
    return einsums::start(einsums_main, argc, argv);
}
