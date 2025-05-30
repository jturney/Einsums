//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Assert.hpp>
#include <Einsums/Print.hpp>
#include <Einsums/TypeSupport/Unused.hpp>

// We're going to initially support knowing how many threads to use at compile time
#if defined(EINSUMS_HAVE_MORE_THAN_64_THREADS) || (defined(EINSUMS_HAVE_MAX_CPU_COUNT) && EINSUMS_HAVE_MAX_CPU_COUNT > 64)
#    include <bitset>
#endif

namespace einsums::topology::detail {

#if !defined(EINSUMS_HAVE_MORE_THAN_64_THREADS) || (defined(EINSUMS_HAVE_MAX_CPU_COUNT) && EINSUMS_HAVE_MAX_CPU_COUNT <= 64)

using MaskType     = std::uint64_t;
using MaskCRefType = std::uint64_t;

inline std::uint64_t bits(std::size_t idx) {
    EINSUMS_ASSERT(idx < CHAR_BIT * sizeof(MaskType));
    return static_cast<std::uint64_t>(1) << idx;
}

inline bool any(MaskCRefType mask) {
    return mask != 0;
}

inline MaskType not_(MaskCRefType mask) {
    return ~mask;
}

inline bool test(MaskCRefType mask, std::size_t idx) {
    EINSUMS_ASSERT(idx < CHAR_BIT * sizeof(MaskType));
    return (bits(idx) & mask) != 0;
}

inline void set(MaskType &mask, std::size_t idx) {
    EINSUMS_ASSERT(idx < CHAR_BIT * sizeof(MaskType));
    mask |= bits(idx);
}

inline void unset(MaskType &mask, std::size_t idx) {
    EINSUMS_ASSERT(idx < CHAR_BIT * sizeof(MaskType));
    mask &= not_(bits(idx));
}

inline std::size_t mask_size(MaskCRefType /*mask*/) {
    return CHAR_BIT * sizeof(MaskType);
}

inline void resize(MaskType & /*mask*/, std::size_t s) {
    EINSUMS_ASSERT(s <= CHAR_BIT * sizeof(MaskType));
    EINSUMS_UNUSED(s);
}

inline std::size_t find_first(MaskCRefType mask) {
    if (mask) {
        std::size_t c = 0; // Will count mask's trailing zero bits.

        // Set mask's trailing 0s to 1s and zero rest.
        mask = (mask ^ (mask - 1)) >> 1;
        for (/**/; mask; ++c)
            mask >>= 1;

        return c;
    }
    return ~std::size_t(0);
}

inline bool equal(MaskCRefType lhs, MaskCRefType rhs, std::size_t = 0) {
    return lhs == rhs;
}

// return true if at least one of the masks has a bit set
inline bool bit_or(MaskCRefType lhs, MaskCRefType rhs, std::size_t = 0) {
    return (lhs | rhs) != 0;
}

// return true if at least one bit is set in both masks
inline bool bit_and(MaskCRefType lhs, MaskCRefType rhs, std::size_t = 0) {
    return (lhs & rhs) != 0;
}

// returns the number of bits set
// taken from https://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetKernighan
inline std::size_t count(MaskCRefType mask) {
    std::size_t c; // c accumulates the total bits set in v
    for (c = 0; mask; c++) {
        mask &= mask - 1; // clear the least significant bit set
    }
    return c;
}

inline void reset(MaskType &mask) {
    mask = 0ull;
}

#else

using MaskType     = std::bitset<EINSUMS_HAVE_MAX_CPU_COUNT>;
using MaskCRefType = std::bitset<EINSUMS_HAVE_MAX_CPU_COUNT> const &;

inline bool any(MaskCRefType mask) {
    return mask.any();
}

inline MaskType not_(MaskCRefType mask) {
    return ~mask;
}

inline bool test(MaskCRefType mask, std::size_t idx) {
    return mask.test(idx);
}

inline void set(MaskType &mask, std::size_t idx) {
    mask.set(idx);
}

inline void unset(MaskType &mask, std::size_t idx) {
    mask.set(idx, 0);
}

inline std::size_t mask_size(MaskCRefType mask) {
    return mask.size();
}

inline void resize(MaskType &mask, std::size_t s) {
    EINSUMS_ASSERT(s <= mask.size());
    EINSUMS_UNUSED(mask);
    EINSUMS_UNUSED(s);
}

inline std::size_t find_first(MaskCRefType mask) {
    if (mask.any()) {
        for (std::size_t i = 0; i != EINSUMS_HAVE_MAX_CPU_COUNT; ++i) {
            if (mask[i])
                return i;
        }
    }
    return ~std::size_t(0);
}

inline bool equal(MaskCRefType lhs, MaskCRefType rhs, std::size_t = 0) {
    return lhs == rhs;
}

// return true if at least one of the masks has a bit set
inline bool bit_or(MaskCRefType lhs, MaskCRefType rhs, std::size_t = 0) {
    return (lhs | rhs).any();
}

// return true if at least one bit is set in both masks
inline bool bit_and(MaskCRefType lhs, MaskCRefType rhs, std::size_t = 0) {
    return (lhs & rhs).any();
}

// returns the number of bits set
inline std::size_t count(MaskCRefType mask) {
    return mask.count();
}

inline void reset(MaskType &mask) {
    mask.reset();
}

#endif

EINSUMS_EXPORT std::string to_string(MaskCRefType);

} // namespace einsums::topology::detail