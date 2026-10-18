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

/// Newest format version this library reads and writes. Version 2 adds an entry stored in an axis
/// order other than the one it is declared in (@ref TensorEntry::stored_order). A file is written at
/// version 2 only when it holds such an entry; one that holds none stays at
/// @ref ETN_VERSION_DECLARED_ORDER, which every release reads.
inline constexpr uint32_t ETN_VERSION = 2;

/// The version of a file whose every entry is stored in its declared axis order, and the oldest this
/// library reads. A release that reads only this version refuses a file with a permuted entry rather
/// than reading its elements in the wrong order.
inline constexpr uint32_t ETN_VERSION_DECLARED_ORDER = 1;

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
        version            = ETN_VERSION_DECLARED_ORDER;
        num_tensors        = 0;
        entry_table_offset = 0;
        data_offset        = ETN_DATA_ALIGNMENT; // First data region after aligned header
        total_size         = 0;
        flags              = 0;
        num_ranks          = 0;
        std::memset(reserved, 0, sizeof(reserved));
    }

    /// Validate magic and version.
    [[nodiscard]] bool is_valid() const {
        return std::memcmp(magic, ETN_MAGIC, 8) == 0 && version >= ETN_VERSION_DECLARED_ORDER && version <= ETN_VERSION;
    }
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
    /// The stored order, three bits per stored axis naming the declared axis held there; all zero
    /// for the declared order. Read and written through @ref stored_order and @ref set_stored_order.
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    uint8_t order[3];
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    uint8_t reserved1[3]; ///< Padding
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

    /// Whether the entry is stored in an axis order other than its declared one.
    [[nodiscard]] bool permuted() const { return (order[0] | order[1] | order[2]) != 0; }

    /// The declared axis held at each stored position: element @c s names the declared axis stored
    /// @c s-th, so the declared order is ``0, 1, ..., rank - 1``. Throws when the bytes do not name a
    /// permutation of the entry's axes, which only a damaged file holds.
    [[nodiscard]] EINSUMS_EXPORT std::vector<size_t> stored_order() const;

    /// Record @p stored, a permutation of the entry's axes in the form @ref stored_order returns.
    /// The declared order is recorded as all zero. Throws when @p stored is not a permutation of
    /// ``0, ..., rank - 1``.
    EINSUMS_EXPORT void set_stored_order(std::vector<size_t> const &stored);

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

/// Refuse a tensor of rank above @ref ETN_MAX_RANK before an entry is filled from it: the entry
/// stores that many extents, and a further one would be written over the fields after them.
EINSUMS_EXPORT void check_writable_rank(std::string_view path, std::string_view operation, std::string_view name, size_t rank);

/// Refuse a name an entry cannot hold: an empty one, or one longer than @ref ETN_MAX_NAME less its
/// terminator, which the entry would cut short so that a later lookup by the full name misses.
EINSUMS_EXPORT void check_entry_name(std::string_view path, std::string_view operation, std::string_view name);

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

/// Call @p visit(file_offset, dst_offset, bytes) for each contiguous run of a hyperslab of a
/// column-major tensor, in the order the slab holds its elements densely in column-major order.
///
/// A run is as long as the slab's shape allows: every leading axis the slab covers whole, and the
/// first axis it does not, read as one range, since the file holds those elements back to back and
/// so does the slab. A block of the last axis is one run, and a slab is never split more finely
/// than its first partial axis requires. The caller checks that the sizes match the rank.
template <typename T, typename Visit>
void walk_slab(uint64_t entry_data_offset, std::vector<size_t> const &entry_dims, std::vector<std::pair<size_t, size_t>> const &ranges,
               Visit &&visit) {
    size_t const rank = entry_dims.size();
    if (rank == 0) {
        // A scalar entry has exactly one slab, the empty one, and it holds the scalar.
        visit(entry_data_offset, std::size_t{0}, sizeof(T));
        return;
    }
    // A slab empty along any axis holds no element. The walk below visits its first run before it
    // asks whether there is a next one, so without this it would read a whole inner run into a
    // destination sized for nothing, from an offset that may lie past the entry.
    if (std::ranges::any_of(ranges, [](auto const &range) { return range.first == range.second; })) {
        return;
    }

    // Column-major strides for the stored tensor: stride[0]=1, stride[d]=prod dims[0..d-1].
    std::vector<size_t> strides(rank);
    strides[0] = 1;
    for (size_t d = 1; d < rank; ++d) {
        strides[d] = strides[d - 1] * entry_dims[d - 1];
    }

    // The run: the leading axes the slab covers whole, then the first one it covers in part.
    size_t inner = 0;
    while (inner + 1 < rank && ranges[inner].first == 0 && ranges[inner].second == entry_dims[inner]) {
        ++inner;
    }
    size_t const run_bytes = strides[inner] * (ranges[inner].second - ranges[inner].first) * sizeof(T);

    // The axes after the run, walked as an odometer.
    std::vector<size_t> idx(rank);
    for (size_t d = 0; d < rank; ++d) {
        idx[d] = ranges[d].first;
    }
    auto advance_outer = [&]() -> bool {
        for (size_t d = inner + 1; d < rank; ++d) {
            idx[d]++;
            if (idx[d] < ranges[d].second) {
                return true;
            }
            idx[d] = ranges[d].first;
        }
        return false;
    };

    size_t dst_offset = 0;
    do {
        uint64_t file_offset = entry_data_offset;
        for (size_t d = inner; d < rank; ++d) {
            file_offset += idx[d] * strides[d] * sizeof(T);
        }
        visit(file_offset, dst_offset, run_bytes);
        dst_offset += run_bytes;
    } while (advance_outer());
}

/// The memory of a dense column-major tensor of extents @p dims and element size @p elem_size.
[[nodiscard]] EINSUMS_EXPORT MemoryLayout dense_layout(std::vector<size_t> const &dims, size_t elem_size);

/// A box of an entry and the memory that holds it, both in the entry's STORED axis order, which is
/// the order @ref walk_slab and the element transfers below walk them in.
struct StoredBox {
    std::vector<size_t>                    dims;   ///< The entry's extents.
    std::vector<std::pair<size_t, size_t>> ranges; ///< The box.
    MemoryLayout                           layout; ///< The memory holding the box.
};

/// The box @p ranges of @p entry, held in memory described by @p memory, both given in the entry's
/// declared axis order, restated in its stored order. An entry stored in its declared order is
/// returned as given.
[[nodiscard]] EINSUMS_EXPORT StoredBox stored_box(TensorEntry const &entry, std::vector<std::pair<size_t, size_t>> const &ranges,
                                                  MemoryLayout const &memory);

/// Every element of @p entry, as a box in its declared axis order.
[[nodiscard]] EINSUMS_EXPORT std::vector<std::pair<size_t, size_t>> whole_box(TensorEntry const &entry);

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
