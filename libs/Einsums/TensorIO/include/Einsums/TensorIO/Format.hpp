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
/// The entry table is last so appending moves no data. Distributed files read back without MPI.
///
/// Data is always column-major, whatever the tensor's layout in memory; there is no layout flag,
/// and row-major tensors are reordered on the way through.

#include <Einsums/Config/ExportDefinitions.hpp>
#include <Einsums/Config/Namespace.hpp>

#include <algorithm>
#include <complex>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(tensor_io)

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

/// Get the name of a DType, spelled as numpy spells it ("float32", "complex128", ...).
/// A value outside the enum, which only a damaged file can hold, is named "unknown".
[[nodiscard]] inline constexpr std::string_view dtype_name(DType dt) {
    switch (dt) {
    case DType::Float32:
        return "float32";
    case DType::Float64:
        return "float64";
    case DType::Complex64:
        return "complex64";
    case DType::Complex128:
        return "complex128";
    case DType::Int32:
        return "int32";
    case DType::Int64:
        return "int64";
    case DType::UInt32:
        return "uint32";
    case DType::UInt64:
        return "uint64";
    }
    return "unknown";
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

namespace detail {

/// Check that a stored entry can be copied to or from a destination of element type @p want.
///
/// Called before every read and slice write moves a byte. There is no type conversion: reading
/// float32 into float64 throws.
///
/// @param path Path of the file holding the entry, for the message.
/// @param operation Qualified name of the calling method ("TensorFile::read"), which leads the message.
/// @param entry The stored entry.
/// @param want Element type of the destination (or, for a slice write, the source).
/// @param want_rank Rank of the destination when it is fixed at compile time; empty when the
///        destination takes the stored rank.
/// @throws std::invalid_argument if the stored dtype or rank differs from the requested one.
/// @throws std::runtime_error if the entry's rank, dtype, or data size is inconsistent with its dims.
EINSUMS_EXPORT void check_entry_type(std::string_view path, std::string_view operation, TensorEntry const &entry, DType want,
                                     std::optional<size_t> want_rank);

/// Check that @p ranges names a hyperslab inside the stored entry: one half-open range per stored
/// dimension, each with start <= end <= dim.
///
/// @throws std::invalid_argument if the number of ranges differs from the stored rank or a range is reversed.
/// @throws std::out_of_range if a range runs past the stored dimension.
EINSUMS_EXPORT void check_slab_ranges(std::string_view path, std::string_view operation, TensorEntry const &entry,
                                      std::vector<std::pair<size_t, size_t>> const &ranges);

/// Dims and element strides of a tensor in memory, as the element transfers below need them.
struct MemoryLayout {
    std::vector<size_t> dims;
    std::vector<size_t> strides;          ///< In elements.
    size_t              elem_size{0};     ///< In bytes.
    bool                file_order{true}; ///< Memory already holds the elements in the file's column-major order.
};

/// Whether memory with these dims and strides holds its elements densely in column-major order,
/// which is the order of an .etn data region. Dimensions of extent one place no constraint.
[[nodiscard]] EINSUMS_EXPORT bool is_file_order(std::vector<size_t> const &dims, std::vector<size_t> const &strides);

/// Describe a dense tensor's memory for the transfers below.
template <typename T, typename TensorType>
[[nodiscard]] MemoryLayout memory_layout(TensorType const &tensor, size_t rank) {
    MemoryLayout layout;
    layout.elem_size = sizeof(T);
    layout.dims.resize(rank);
    layout.strides.resize(rank);
    for (size_t d = 0; d < rank; ++d) {
        layout.dims[d]    = tensor.dim(static_cast<int>(d));
        layout.strides[d] = tensor.stride(static_cast<int>(d));
    }
    layout.file_order = is_file_order(layout.dims, layout.strides);
    return layout;
}

/// Copy @p count elements, starting at column-major linear index @p first, out of a strided
/// tensor into @p dst, which receives them in column-major order.
EINSUMS_EXPORT void gather_file_order(char *dst, char const *tensor, MemoryLayout const &layout, size_t first, size_t count);

/// Copy @p count elements held in column-major order in @p src into a strided tensor, starting at
/// column-major linear index @p first.
EINSUMS_EXPORT void scatter_file_order(char *tensor, char const *src, MemoryLayout const &layout, size_t first, size_t count);

/// Bytes staged at a time when a transfer has to reorder elements.
inline constexpr size_t REORDER_CHUNK_BYTES = size_t{1} << 20;

/// Read the elements with column-major linear indices [first, first + count) of a tensor from
/// the file bytes starting at @p file_offset, which hold those elements in column-major order.
/// @p read_at is the file's (offset, destination, bytes) reader.
template <typename ReadAt>
void read_elements(ReadAt &&read_at, uint64_t file_offset, char *tensor, MemoryLayout const &layout, size_t first, size_t count) {
    size_t const es = layout.elem_size;
    if (layout.file_order) {
        read_at(file_offset, tensor + first * es, count * es);
        return;
    }
    size_t const      chunk = std::max<size_t>(1, REORDER_CHUNK_BYTES / es);
    std::vector<char> staging(std::min(count, chunk) * es);
    for (size_t done = 0; done < count;) {
        size_t const n = std::min(chunk, count - done);
        read_at(file_offset + done * es, staging.data(), n * es);
        scatter_file_order(tensor, staging.data(), layout, first + done, n);
        done += n;
    }
}

/// Write the elements with column-major linear indices [first, first + count) of a tensor to the
/// file bytes starting at @p file_offset, in column-major order. @p write_at is the file's
/// (offset, source, bytes) writer.
template <typename WriteAt>
void write_elements(WriteAt &&write_at, uint64_t file_offset, char const *tensor, MemoryLayout const &layout, size_t first, size_t count) {
    size_t const es = layout.elem_size;
    if (layout.file_order) {
        write_at(file_offset, tensor + first * es, count * es);
        return;
    }
    size_t const      chunk = std::max<size_t>(1, REORDER_CHUNK_BYTES / es);
    std::vector<char> staging(std::min(count, chunk) * es);
    for (size_t done = 0; done < count;) {
        size_t const n = std::min(chunk, count - done);
        gather_file_order(staging.data(), tensor, layout, first + done, n);
        write_at(file_offset + done * es, staging.data(), n * es);
        done += n;
    }
}

} // namespace detail

EINSUMS_NAMESPACE_END(tensor_io)
