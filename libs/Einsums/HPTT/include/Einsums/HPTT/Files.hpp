//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config/ExportDefinitions.hpp>
#include <Einsums/Config/Namespace.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <ranges>
#include <stdexcept>
#include <utility>
EINSUMS_NAMESPACE_BEGIN(hptt)

constexpr char endian_char() {
    if constexpr (std::endian::native == std::endian::big) {
        return 'B';
    } else if constexpr (std::endian::native == std::endian::little) {
        return 'L';
    } else {
        throw std::runtime_error("Mixed endian systems are not supported.");
    }
}

template <std::integral T>
constexpr T byteswap(T value) noexcept {
    auto bytes = std::bit_cast<std::array<uint8_t, sizeof(T)>>(value);
    std::ranges::reverse(bytes);
    return std::bit_cast<T>(bytes);
}

template <>
constexpr uint8_t byteswap<uint8_t>(uint8_t value) noexcept {
    return value;
}

template <>
constexpr uint16_t byteswap<uint16_t>(uint16_t value) noexcept {
    union {
        uint16_t whole;
        uint8_t  bytes[2]; // NOLINT
    } convert;

    convert.whole = value;
    std::swap(convert.bytes[0], convert.bytes[1]);
    return convert.whole;
}

template <>
constexpr uint32_t byteswap<uint32_t>(uint32_t value) noexcept {
    union {
        uint32_t whole;
        uint8_t  bytes[4]; // NOLINT
    } convert;

    convert.whole = value;
    std::swap(convert.bytes[0], convert.bytes[3]);
    std::swap(convert.bytes[1], convert.bytes[2]);
    return convert.whole;
}

template <>
constexpr int32_t byteswap<int32_t>(int32_t value) noexcept {
    union {
        int32_t whole;
        uint8_t bytes[4]; // NOLINT
    } convert;

    convert.whole = value;
    std::swap(convert.bytes[0], convert.bytes[3]);
    std::swap(convert.bytes[1], convert.bytes[2]);
    return convert.whole;
}

struct NodeConstants {
    ptrdiff_t start, end, inc, offDiffAB;
    size_t    lda, ldb;
    uint16_t  indexA, indexB, has_next;
    uint16_t  pad;
};

struct TransposeConstants {
    int32_t dim;
    int32_t numThreads;
    size_t  innerStrideA;
    size_t  innerStrideB;
    int32_t selectedParallelStrategy;
    int32_t selectedLoopOrderId;
    int32_t conjA;
    int32_t pad;
};

/**
 * File header specification for transpose files.
 *
 * ``version[2]`` is the format version (plan_file_format) and ``version[3]`` the writer's
 * endianness ('B' or 'L'); the other two bytes are zero.
 */
struct FileHeader {
    char magic[4];   // NOLINT
    char version[4]; // NOLINT

    uint32_t checksum;
};

/**
 * Format version written to ``FileHeader::version[2]``.
 *
 * Version 0 files carry no PlanTarget. They are refused on read, because nothing in them says
 * what vector width the plan's loop increments were built for.
 */
inline constexpr char plan_file_format = 1;

/**
 * The machine geometry a plan was built for; follows the FileHeader from format version 1 on.
 *
 * A plan stores loop increments equal to the macro-kernel block, which is four vectors of the
 * writing translation unit's register width. A reader compiled for a different width, or for a
 * different element size, steps by one block while the plan steps by another, and silently
 * skips or overruns elements. So the reader refuses a plan whose vector_bits or element_size
 * differs from its own. The rung is recorded for the error message only: two rungs with the
 * same vector width (baseline and x86-64-v2) build identical plans.
 */
struct PlanTarget {
    uint16_t vector_bits;  ///< Register width, in bits, of the rung that wrote the plan.
    uint8_t  element_size; ///< sizeof the element type the plan was built for.
    uint8_t  rung;         ///< stripes::InstructionSet of the writing rung.
    uint32_t pad;
};

EINSUMS_EXPORT void setup_file(std::FILE *fp);

EINSUMS_EXPORT uint32_t compute_checksum(std::FILE *fp);

/**
 * Check a transpose file's header and checksum.
 *
 * @return 0 when the file is intact; 1 or 2 when it cannot be read; 3 for a missing magic;
 *         4 for a different endianness; 5 for a checksum mismatch; 6 for a format version this
 *         build does not read.
 */
EINSUMS_EXPORT int verify_file(std::FILE *fp);

EINSUMS_NAMESPACE_END(hptt)