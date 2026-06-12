//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/// @file Format.hpp
/// @brief Binary format definitions for .etn tensor files.
///
/// The .etn format stores tensors as raw binary data with a compact header:
///
/// ```
/// [FileHeader]     (64 bytes at offset 0)
/// [Data region 0]  (raw bytes, 64-byte aligned)
/// [Data region 1]
/// ...
/// [Data region N-1]
/// [TensorEntry 0]  (160 bytes each, at end of file)
/// [TensorEntry 1]
/// ...
/// [TensorEntry N-1]
/// ```
///
/// Putting the entry table at the end allows appending without shifting
/// existing data. Files written through the distributed path are standard
/// binary, so they can be read back without MPI.

#include <algorithm>
#include <complex>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>

namespace einsums::tensor_io {

/// Magic bytes identifying an .etn file.
inline constexpr char ETN_MAGIC[8] = {'E', 'I', 'N', 'S', 'U', 'M', 'S', '\0'}; // NOLINT(modernize-avoid-c-arrays)

/// Current format version.
inline constexpr uint32_t ETN_VERSION = 1;

/// Alignment for data regions (64 bytes for SIMD/cache-line compatibility).
inline constexpr size_t ETN_DATA_ALIGNMENT = 64;

/// Element type identifiers.
enum class DType : uint8_t {
    Float32    = 0,
    Float64    = 1,
    Complex64  = 2, ///< std::complex<float>
    Complex128 = 3, ///< std::complex<double>
    Int32      = 4,
    Int64      = 5,
    UInt32     = 6,
    UInt64     = 7,
};

/// File header (64 bytes, at offset 0).
struct FileHeader {
    char     magic[8];           ///< "EINSUMS\0"  // NOLINT(modernize-avoid-c-arrays)
    uint32_t version;            ///< Format version (ETN_VERSION)
    uint32_t num_tensors;        ///< Number of TensorEntry records
    uint64_t entry_table_offset; ///< Byte offset to first TensorEntry (end of data)
    uint64_t data_offset;        ///< Byte offset to first data region (typically 64)
    uint64_t total_size;         ///< Total file size in bytes
    uint32_t flags;              ///< Bit flags: 0x1 = has distributed tensors
    uint32_t num_ranks;          ///< MPI world_size at write time (0 = serial)
    char     reserved[16];       ///< Reserved for future use  // NOLINT(modernize-avoid-c-arrays)

    /// Initialize with default values.
    void init() {
        std::memcpy(magic, ETN_MAGIC, 8);
        version            = ETN_VERSION;
        num_tensors        = 0;
        entry_table_offset = 0;
        data_offset        = ETN_DATA_ALIGNMENT; // First data region after aligned header
        total_size         = 0;
        flags              = 0;
        num_ranks          = 0;
        std::memset(reserved, 0, sizeof(reserved));
    }

    /// Validate magic and version.
    [[nodiscard]] bool is_valid() const { return std::memcmp(magic, ETN_MAGIC, 8) == 0 && version == ETN_VERSION; }
};

static_assert(sizeof(FileHeader) == 64, "FileHeader must be exactly 64 bytes");

/// Maximum tensor name length (including null terminator).
inline constexpr size_t ETN_MAX_NAME = 64;

/// Maximum supported tensor rank.
inline constexpr size_t ETN_MAX_RANK = 8;

/// Sentinel value for owning_rank meaning "all ranks / serial".
inline constexpr uint32_t ETN_ALL_RANKS = UINT32_MAX;

/// Per-tensor metadata entry (160 bytes).
struct TensorEntry {
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    char    name[ETN_MAX_NAME]; ///< Null-terminated tensor name
    uint8_t dtype;              ///< DType enum value
    uint8_t rank;               ///< Number of dimensions (1-8)
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    uint8_t reserved1[6]; ///< Padding
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    uint64_t dims[ETN_MAX_RANK]; ///< Dimension sizes (unused dims = 0)
    uint64_t data_offset;        ///< Byte offset from file start to data
    uint64_t data_size;          ///< Size of data region in bytes
    uint32_t owning_rank;        ///< ETN_ALL_RANKS for serial/replicated
    uint32_t reserved2;          ///< Padding

    /// Initialize to zeros.
    void init() {
        std::memset(this, 0, sizeof(*this));
        owning_rank = ETN_ALL_RANKS;
    }

    /// Set the tensor name (truncates if too long).
    void set_name(std::string_view n) {
        size_t const len = std::min(n.size(), ETN_MAX_NAME - 1);
        std::memcpy(name, n.data(), len);
        name[len] = '\0';
    }

    /// Get the tensor name as a string.
    [[nodiscard]] std::string get_name() const { return {name}; }
};

static_assert(sizeof(TensorEntry) == 160, "TensorEntry must be exactly 160 bytes");

/// Round up to alignment boundary.
[[nodiscard]] inline constexpr uint64_t align_up(uint64_t offset, uint64_t alignment) {
    return (offset + alignment - 1) & ~(alignment - 1);
}

/// Get the element size for a DType.
[[nodiscard]] inline constexpr size_t dtype_size(DType dt) {
    switch (dt) {
    case DType::Float32:
        return 4;
    case DType::Float64:
    case DType::Complex64:
        return 8;
    case DType::Complex128:
        return 16;
    case DType::Int32:
        return 4;
    case DType::Int64:
        return 8;
    case DType::UInt32:
        return 4;
    case DType::UInt64:
        return 8;
    }
    return 0;
}

/// Map C++ type to DType.
template <typename T>
constexpr DType dtype_for();
template <>
inline constexpr DType dtype_for<float>() {
    return DType::Float32;
}
template <>
inline constexpr DType dtype_for<double>() {
    return DType::Float64;
}
template <>
inline constexpr DType dtype_for<std::complex<float>>() {
    return DType::Complex64;
}
template <>
inline constexpr DType dtype_for<std::complex<double>>() {
    return DType::Complex128;
}
template <>
inline constexpr DType dtype_for<int32_t>() {
    return DType::Int32;
}
template <>
inline constexpr DType dtype_for<int64_t>() {
    return DType::Int64;
}
template <>
inline constexpr DType dtype_for<uint32_t>() {
    return DType::UInt32;
}
template <>
inline constexpr DType dtype_for<uint64_t>() {
    return DType::UInt64;
}

} // namespace einsums::tensor_io
