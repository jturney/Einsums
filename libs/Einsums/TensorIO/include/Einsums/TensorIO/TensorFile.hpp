//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/// @file TensorFile.hpp
/// @brief High-performance tensor I/O in the .etn binary format.
///
/// @code
/// using namespace einsums::tensor_io;
///
/// // Write
/// TensorFile out("integrals.etn", TensorFile::Mode::Write);
/// out.write("ERI", eri);
/// out.write("C_mo", C);
///
/// // Read
/// TensorFile in("integrals.etn", TensorFile::Mode::Read);
/// in.read("ERI", eri);
///
/// // Slice read (only loads the requested range)
/// in.read_slice("ERI", eri_slice, {{0,10}, {0,10}, {0,10}, {0,10}});
/// @endcode

#include <Einsums/Config.hpp>

#include <Einsums/Python/Annotations.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorIO/Format.hpp>

#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace einsums {
namespace APIARY_MODULE("io") tensor_io {

class EINSUMS_EXPORT APIARY_EXPOSE APIARY_NOCOPY APIARY_NOMOVE TensorFile {
  public:
    enum class APIARY_EXPOSE Mode : std::uint8_t { Read, Write, ReadWrite };

    /// Open or create a tensor file.
    /// @param path File path (creates parent directories if needed).
    /// @param mode Read, Write, or ReadWrite.
    APIARY_EXPOSE explicit TensorFile(std::string path, Mode mode = Mode::ReadWrite);

    /// Flushes pending writes and closes the file.
    ~TensorFile();

    // Non-copyable, movable.
    TensorFile(TensorFile const &)            = delete;
    TensorFile &operator=(TensorFile const &) = delete;
    TensorFile(TensorFile &&)                 = default;
    TensorFile &operator=(TensorFile &&)      = default;

    // ── Write ────────────────────────────────────────────────────────────

    /// Write a full tensor to the file.
    template <typename T, size_t Rank>
    void write(std::string_view name, Tensor<T, Rank> const &tensor);

    /// Write this rank's local partition of a distributed tensor.
    template <typename T, size_t Rank>
    void write_local(std::string_view name, Tensor<T, Rank> const &tensor, int rank, int num_ranks);

    /// Write a slice (hyperslab) of an existing tensor entry. The entry
    /// must already exist (e.g. from a previous full @ref write or a
    /// pre-allocated entry). The user tensor's shape must match the
    /// slab implied by @p ranges.
    /// @param ranges Per-dimension [start, end) ranges into the stored tensor.
    template <typename T, size_t Rank>
    void write_slice(std::string_view name, Tensor<T, Rank> const &tensor, std::array<std::pair<size_t, size_t>, Rank> const &ranges);

    /// Reserve space for a tensor that will be filled in via @ref
    /// write_slice. Stores the entry header (name, dtype, dims) and
    /// allocates the data region but writes no data. Use this when you
    /// plan to fill the tensor block-by-block from a graph or loop.
    template <typename T>
    void reserve(std::string_view name, std::vector<size_t> const &dims);

    // ── Read ─────────────────────────────────────────────────────────────

    /// Read a full tensor. Resizes the tensor if dimensions don't match.
    template <typename T, size_t Rank>
    void read(std::string_view name, Tensor<T, Rank> &tensor);

    /// Read a slice (hyperslab) of a tensor.
    /// @param ranges Per-dimension [start, end) ranges.
    template <typename T, size_t Rank>
    void read_slice(std::string_view name, Tensor<T, Rank> &tensor, std::array<std::pair<size_t, size_t>, Rank> const &ranges);

    /// Read this rank's local partition of a distributed tensor.
    template <typename T, size_t Rank>
    void read_local(std::string_view name, Tensor<T, Rank> &tensor, int rank);

    // ── RuntimeTensor overloads ──────────────────────────────────────────
    //
    // RuntimeTensor variants of read / write / read_slice / write_slice.
    // These are what the Python bindings call into — RuntimeTensor is
    // the runtime-rank type bound to numpy via the buffer protocol.

    /// Read a full tensor into a RuntimeTensor. Resizes the tensor.
    template <typename T, typename Alloc>
    APIARY_EXPOSE APIARY_INSTANTIATE_MEMBER_AS("read", T = float, Alloc = std::allocator<float>)
        APIARY_INSTANTIATE_MEMBER_AS("read", T = double, Alloc = std::allocator<double>)
            APIARY_INSTANTIATE_MEMBER_AS("read", T = std::complex<float>, Alloc = std::allocator<std::complex<float>>)
                APIARY_INSTANTIATE_MEMBER_AS("read", T = std::complex<double>, Alloc = std::allocator<std::complex<double>>) void read(
                    std::string_view name, GeneralRuntimeTensor<T, Alloc> &tensor);

    /// Write a full RuntimeTensor as a new entry.
    template <typename T, typename Alloc>
    APIARY_EXPOSE APIARY_INSTANTIATE_MEMBER_AS("write", T = float, Alloc = std::allocator<float>)
        APIARY_INSTANTIATE_MEMBER_AS("write", T = double, Alloc = std::allocator<double>)
            APIARY_INSTANTIATE_MEMBER_AS("write", T = std::complex<float>, Alloc = std::allocator<std::complex<float>>)
                APIARY_INSTANTIATE_MEMBER_AS("write", T = std::complex<double>, Alloc = std::allocator<std::complex<double>>) void write(
                    std::string_view name, GeneralRuntimeTensor<T, Alloc> const &tensor);

    /// Read a slice into a pre-sized RuntimeTensor. The tensor's dims
    /// must equal the slab (ranges[d].second - ranges[d].first).
    template <typename T, typename Alloc>
    APIARY_EXPOSE APIARY_INSTANTIATE_MEMBER_AS("read_slice", T = float, Alloc = std::allocator<float>)
        APIARY_INSTANTIATE_MEMBER_AS("read_slice", T = double, Alloc = std::allocator<double>)
            APIARY_INSTANTIATE_MEMBER_AS("read_slice", T = std::complex<float>, Alloc = std::allocator<std::complex<float>>)
                APIARY_INSTANTIATE_MEMBER_AS(
                    "read_slice", T = std::complex<double>,
                    Alloc = std::allocator<std::complex<double>>) void read_slice(std::string_view                              name,
                                                                                  GeneralRuntimeTensor<T, Alloc>               &tensor,
                                                                                  std::vector<std::pair<size_t, size_t>> const &ranges);

    /// Write a slice (hyperslab) of an existing entry from a RuntimeTensor.
    template <typename T, typename Alloc>
    APIARY_EXPOSE APIARY_INSTANTIATE_MEMBER_AS("write_slice", T = float, Alloc = std::allocator<float>)
        APIARY_INSTANTIATE_MEMBER_AS("write_slice", T = double, Alloc = std::allocator<double>)
            APIARY_INSTANTIATE_MEMBER_AS("write_slice", T = std::complex<float>, Alloc = std::allocator<std::complex<float>>)
                APIARY_INSTANTIATE_MEMBER_AS(
                    "write_slice", T = std::complex<double>,
                    Alloc = std::allocator<std::complex<double>>) void write_slice(std::string_view                              name,
                                                                                   GeneralRuntimeTensor<T, Alloc> const         &tensor,
                                                                                   std::vector<std::pair<size_t, size_t>> const &ranges);

    // ── Query ────────────────────────────────────────────────────────────

    /// Check if a tensor exists in the file.
    APIARY_EXPOSE [[nodiscard]] bool contains(std::string_view name) const;

    /// Get the stored dimensions of a tensor.
    APIARY_EXPOSE [[nodiscard]] std::vector<size_t> dims(std::string_view name) const;

    /// Get the stored dtype of a tensor (currently not exposed to Python:
    /// DType enum lives in Format.hpp and isn't bound). Use ``dims`` and
    /// ``contains`` from Python; per-dtype dispatch happens via the
    /// ``read``/``write`` ``dtype=`` kwarg.
    [[nodiscard]] DType dtype(std::string_view name) const;

    /// List all tensor names in the file.
    APIARY_EXPOSE [[nodiscard]] std::vector<std::string> tensor_names() const;

    /// Number of tensors in the file.
    APIARY_EXPOSE [[nodiscard]] size_t num_tensors() const { return _entries.size(); }

    /// Flush pending writes to disk.
    APIARY_EXPOSE void flush();

    /// Get the file path.
    APIARY_EXPOSE [[nodiscard]] std::string const &path() const { return _path; }

  private:
    std::string                             _path;
    Mode                                    _mode;
    int                                     _fd{-1}; ///< POSIX file descriptor
    FileHeader                              _header;
    std::vector<TensorEntry>                _entries;
    std::unordered_map<std::string, size_t> _name_to_index;    ///< name → entry index
    uint64_t                                _next_data_offset; ///< Next available offset for data

    /// Find entry by name (throws if not found).
    [[nodiscard]] TensorEntry const &find_entry(std::string_view name) const;

    /// Write raw bytes at a specific offset.
    void write_at(uint64_t offset, void const *data, size_t bytes);

    /// Read raw bytes from a specific offset.
    void read_at(uint64_t offset, void *data, size_t bytes) const;

    /// Write the header and entry table to disk.
    void write_metadata();

    /// Read the header and entry table from disk.
    void read_metadata();

    /// Add a new entry and return its index.
    size_t add_entry(TensorEntry entry);
};

// ═══════════════════════════════════════════════════════════════════════════════
// Slab walker — shared between read_slice and write_slice (static + runtime).
// ═══════════════════════════════════════════════════════════════════════════════

namespace detail {

/// Walk every contiguous innermost run inside a hyperslab of a
/// column-major tensor and invoke @p visit(file_offset, dst_offset,
/// inner_bytes) for each. Lets read_slice and write_slice share the
/// same offset-arithmetic without code duplication.
///
/// `entry_dims` and `ranges` are runtime-rank (vectors). The caller is
/// responsible for ensuring sizes agree with the actual tensor rank.
template <typename T, typename Visit>
void walk_slab(uint64_t entry_data_offset, std::vector<size_t> const &entry_dims, std::vector<std::pair<size_t, size_t>> const &ranges,
               Visit &&visit) {
    size_t const rank = entry_dims.size();
    if (rank == 0) {
        // Scalar entry — emit a single zero-byte run.
        visit(entry_data_offset, std::size_t{0}, std::size_t{0});
        return;
    }

    // Column-major strides for the stored tensor: stride[0]=1, stride[d]=prod dims[0..d-1].
    std::vector<size_t> strides(rank);
    strides[0] = 1;
    for (size_t d = 1; d < rank; ++d) {
        strides[d] = strides[d - 1] * entry_dims[d - 1];
    }

    size_t const inner_count = ranges[0].second - ranges[0].first;
    size_t const inner_bytes = inner_count * sizeof(T);

    if (rank == 1) {
        uint64_t const src_offset = entry_data_offset + ranges[0].first * sizeof(T);
        visit(src_offset, std::size_t{0}, inner_bytes);
        return;
    }

    std::vector<size_t> idx(rank);
    for (size_t d = 0; d < rank; ++d) {
        idx[d] = ranges[d].first;
    }

    auto advance_outer = [&]() -> bool {
        for (size_t d = 1; d < rank; ++d) {
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
        for (size_t d = 1; d < rank; ++d) {
            file_offset += idx[d] * strides[d] * sizeof(T);
        }
        file_offset += ranges[0].first * sizeof(T);
        visit(file_offset, dst_offset, inner_bytes);
        dst_offset += inner_bytes;
    } while (advance_outer());
}

} // namespace detail

// ═══════════════════════════════════════════════════════════════════════════════
// Template implementations
// ═══════════════════════════════════════════════════════════════════════════════

template <typename T, size_t Rank>
void TensorFile::write(std::string_view name, Tensor<T, Rank> const &tensor) {
    TensorEntry entry;
    entry.init();
    entry.set_name(name);
    entry.dtype = static_cast<uint8_t>(dtype_for<T>());
    entry.rank  = static_cast<uint8_t>(Rank);
    for (size_t d = 0; d < Rank; d++)
        entry.dims[d] = tensor.dim(d);

    size_t data_bytes = tensor.size() * sizeof(T);
    entry.data_offset = align_up(_next_data_offset, ETN_DATA_ALIGNMENT);
    entry.data_size   = data_bytes;
    entry.owning_rank = ETN_ALL_RANKS;

    write_at(entry.data_offset, tensor.data(), data_bytes);
    _next_data_offset = entry.data_offset + data_bytes;

    add_entry(entry);
}

template <typename T, size_t Rank>
void TensorFile::write_local(std::string_view name, Tensor<T, Rank> const &tensor, int rank, int num_ranks) {
    TensorEntry entry;
    entry.init();
    entry.set_name(name);
    entry.dtype = static_cast<uint8_t>(dtype_for<T>());
    entry.rank  = static_cast<uint8_t>(Rank);
    for (size_t d = 0; d < Rank; d++)
        entry.dims[d] = tensor.dim(d);

    size_t data_bytes = tensor.size() * sizeof(T);
    entry.data_offset = align_up(_next_data_offset, ETN_DATA_ALIGNMENT);
    entry.data_size   = data_bytes;
    entry.owning_rank = static_cast<uint32_t>(rank);

    write_at(entry.data_offset, tensor.data(), data_bytes);
    _next_data_offset = entry.data_offset + data_bytes;

    add_entry(entry);

    // Update header flags
    _header.flags |= 0x1; // has distributed tensors
    _header.num_ranks = std::max(_header.num_ranks, static_cast<uint32_t>(num_ranks));
}

template <typename T, size_t Rank>
void TensorFile::read(std::string_view name, Tensor<T, Rank> &tensor) {
    auto const &entry = find_entry(name);

    // Resize tensor if needed
    Dim<Rank> dims;
    for (size_t d = 0; d < Rank; d++)
        dims[d] = entry.dims[d];
    tensor.resize(dims);

    read_at(entry.data_offset, tensor.data(), entry.data_size);
}

template <typename T, size_t Rank>
void TensorFile::read_slice(std::string_view name, Tensor<T, Rank> &tensor, std::array<std::pair<size_t, size_t>, Rank> const &ranges) {
    auto const &entry = find_entry(name);

    Dim<Rank> slice_dims;
    for (size_t d = 0; d < Rank; ++d) {
        slice_dims[d] = ranges[d].second - ranges[d].first;
    }
    tensor.resize(slice_dims);

    std::vector<size_t> entry_dims(Rank);
    for (size_t d = 0; d < Rank; ++d) {
        entry_dims[d] = entry.dims[d];
    }
    std::vector<std::pair<size_t, size_t>> rng(ranges.begin(), ranges.end());

    char *dst = reinterpret_cast<char *>(tensor.data());
    detail::walk_slab<T>(entry.data_offset, entry_dims, rng,
                         [&](uint64_t file_off, size_t dst_off, size_t bytes) { read_at(file_off, dst + dst_off, bytes); });
}

template <typename T, size_t Rank>
void TensorFile::write_slice(std::string_view name, Tensor<T, Rank> const &tensor,
                             std::array<std::pair<size_t, size_t>, Rank> const &ranges) {
    auto const &entry = find_entry(name);

    // Validate user tensor matches the requested slab shape.
    for (size_t d = 0; d < Rank; ++d) {
        size_t const want = ranges[d].second - ranges[d].first;
        if (tensor.dim(d) != want) {
            throw std::runtime_error(fmt::format("TensorFile::write_slice: tensor dim {} = {} but slab range is [{}, {}) (size {})", d,
                                                 tensor.dim(d), ranges[d].first, ranges[d].second, want));
        }
    }

    std::vector<size_t> entry_dims(Rank);
    for (size_t d = 0; d < Rank; ++d) {
        entry_dims[d] = entry.dims[d];
    }
    std::vector<std::pair<size_t, size_t>> rng(ranges.begin(), ranges.end());

    char const *src = reinterpret_cast<char const *>(tensor.data());
    detail::walk_slab<T>(entry.data_offset, entry_dims, rng,
                         [&](uint64_t file_off, size_t src_off, size_t bytes) { write_at(file_off, src + src_off, bytes); });
}

template <typename T>
void TensorFile::reserve(std::string_view name, std::vector<size_t> const &dims) {
    if (dims.size() > 8) {
        throw std::runtime_error(fmt::format("TensorFile::reserve: rank {} exceeds maximum supported rank 8", dims.size()));
    }

    TensorEntry entry;
    entry.init();
    entry.set_name(name);
    entry.dtype = static_cast<uint8_t>(dtype_for<T>());
    entry.rank  = static_cast<uint8_t>(dims.size());
    for (size_t d = 0; d < dims.size(); ++d) {
        entry.dims[d] = dims[d];
    }

    size_t total = 1;
    for (size_t const d : dims) {
        total *= d;
    }
    size_t const data_bytes = total * sizeof(T);

    entry.data_offset = align_up(_next_data_offset, ETN_DATA_ALIGNMENT);
    entry.data_size   = data_bytes;
    entry.owning_rank = ETN_ALL_RANKS;
    _next_data_offset = entry.data_offset + data_bytes;

    add_entry(entry);
}

// ── RuntimeTensor overloads ──────────────────────────────────────────

template <typename T, typename Alloc>
void TensorFile::read(std::string_view name, GeneralRuntimeTensor<T, Alloc> &tensor) {
    auto const &entry = find_entry(name);

    std::vector<size_t> dims(entry.rank);
    for (size_t d = 0; d < entry.rank; ++d) {
        dims[d] = entry.dims[d];
    }
    tensor.resize(dims);

    read_at(entry.data_offset, tensor.data(), entry.data_size);
}

template <typename T, typename Alloc>
void TensorFile::write(std::string_view name, GeneralRuntimeTensor<T, Alloc> const &tensor) {
    TensorEntry entry;
    entry.init();
    entry.set_name(name);
    entry.dtype = static_cast<uint8_t>(dtype_for<T>());
    entry.rank  = static_cast<uint8_t>(tensor.rank());
    for (size_t d = 0; d < tensor.rank(); ++d) {
        entry.dims[d] = tensor.dim(d);
    }

    size_t const data_bytes = tensor.size() * sizeof(T);
    entry.data_offset       = align_up(_next_data_offset, ETN_DATA_ALIGNMENT);
    entry.data_size         = data_bytes;
    entry.owning_rank       = ETN_ALL_RANKS;

    write_at(entry.data_offset, tensor.data(), data_bytes);
    _next_data_offset = entry.data_offset + data_bytes;

    add_entry(entry);
}

template <typename T, typename Alloc>
void TensorFile::read_slice(std::string_view name, GeneralRuntimeTensor<T, Alloc> &tensor,
                            std::vector<std::pair<size_t, size_t>> const &ranges) {
    auto const &entry = find_entry(name);
    if (ranges.size() != entry.rank) {
        throw std::runtime_error(fmt::format("TensorFile::read_slice: ranges size {} != entry rank {}", ranges.size(), entry.rank));
    }

    std::vector<size_t> slab_dims(ranges.size());
    for (size_t d = 0; d < ranges.size(); ++d) {
        slab_dims[d] = ranges[d].second - ranges[d].first;
    }
    tensor.resize(slab_dims);

    std::vector<size_t> entry_dims(entry.rank);
    for (size_t d = 0; d < entry.rank; ++d) {
        entry_dims[d] = entry.dims[d];
    }

    char *dst = reinterpret_cast<char *>(tensor.data());
    detail::walk_slab<T>(entry.data_offset, entry_dims, ranges,
                         [&](uint64_t file_off, size_t dst_off, size_t bytes) { read_at(file_off, dst + dst_off, bytes); });
}

template <typename T, typename Alloc>
void TensorFile::write_slice(std::string_view name, GeneralRuntimeTensor<T, Alloc> const &tensor,
                             std::vector<std::pair<size_t, size_t>> const &ranges) {
    auto const &entry = find_entry(name);
    if (ranges.size() != entry.rank) {
        throw std::runtime_error(fmt::format("TensorFile::write_slice: ranges size {} != entry rank {}", ranges.size(), entry.rank));
    }
    if (tensor.rank() != entry.rank) {
        throw std::runtime_error(fmt::format("TensorFile::write_slice: tensor rank {} != entry rank {}", tensor.rank(), entry.rank));
    }
    for (size_t d = 0; d < ranges.size(); ++d) {
        size_t const want = ranges[d].second - ranges[d].first;
        if (tensor.dim(d) != want) {
            throw std::runtime_error(fmt::format("TensorFile::write_slice: tensor dim {} = {} but slab range is [{}, {}) (size {})", d,
                                                 tensor.dim(d), ranges[d].first, ranges[d].second, want));
        }
    }

    std::vector<size_t> entry_dims(entry.rank);
    for (size_t d = 0; d < entry.rank; ++d) {
        entry_dims[d] = entry.dims[d];
    }

    char const *src = reinterpret_cast<char const *>(tensor.data());
    detail::walk_slab<T>(entry.data_offset, entry_dims, ranges,
                         [&](uint64_t file_off, size_t src_off, size_t bytes) { write_at(file_off, src + src_off, bytes); });
}

template <typename T, size_t Rank>
void TensorFile::read_local(std::string_view name, Tensor<T, Rank> &tensor, int rank) {
    // Find the entry for this specific rank
    std::string const name_str(name);
    for (auto const &entry : _entries) {
        if (entry.get_name() == name_str && std::cmp_equal(entry.owning_rank, rank)) {
            Dim<Rank> dims;
            for (size_t d = 0; d < Rank; d++)
                dims[d] = entry.dims[d];
            tensor.resize(dims);
            read_at(entry.data_offset, tensor.data(), entry.data_size);
            return;
        }
    }
    throw std::runtime_error(fmt::format("TensorFile: no entry for '{}' with rank {}", name, rank));
}

} // namespace APIARY_MODULE("io")tensor_io
} // namespace einsums
