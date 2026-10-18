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

#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/Python/Annotations.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorIO/Format.hpp>

#include <algorithm>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

EINSUMS_NAMESPACE_BEGIN()
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

    /// Write a hyperslab of an existing entry; @p tensor must have the slab's shape.
    /// @param ranges Per-dimension [start, end) ranges into the stored tensor.
    template <typename T, size_t Rank>
    void write_slice(std::string_view name, Tensor<T, Rank> const &tensor, std::array<std::pair<size_t, size_t>, Rank> const &ranges);

    /// Create an entry and its data region without writing data, to fill with @ref write_slice.
    /// @param[in] name The entry's name.
    /// @param[in] dims Its extents, in its declared axis order.
    /// @param[in] stored_order The declared axis to store at each position (see
    ///            @ref TensorEntry::stored_order); empty, or the declared order, stores it as declared.
    ///            Every read and write of the entry takes and returns its declared order whichever
    ///            order it is stored in; the stored order only decides which boxes of it are long
    ///            contiguous runs of the file.
    template <typename T>
    void reserve(std::string_view name, std::vector<size_t> const &dims, std::vector<size_t> const &stored_order = {});

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

    // ── RuntimeTensor overloads, which the Python bindings call ─────────

    /// Read a full tensor into a RuntimeTensor. Resizes the tensor.
    template <typename T, typename Alloc>
    APIARY_EXPOSE APIARY_INSTANTIATE_MEMBER_AS("read", T = float, Alloc = std::allocator<float>)
    APIARY_INSTANTIATE_MEMBER_AS("read", T = double, Alloc = std::allocator<double>)
    APIARY_INSTANTIATE_MEMBER_AS("read", T = std::complex<float>, Alloc = std::allocator<std::complex<float>>)
    APIARY_INSTANTIATE_MEMBER_AS("read", T = std::complex<double>, Alloc = std::allocator<std::complex<double>>) void
    read(std::string_view name, GeneralRuntimeTensor<T, Alloc> &tensor);

    /// Write a full RuntimeTensor as a new entry.
    template <typename T, typename Alloc>
    APIARY_EXPOSE APIARY_INSTANTIATE_MEMBER_AS("write", T = float, Alloc = std::allocator<float>)
    APIARY_INSTANTIATE_MEMBER_AS("write", T = double, Alloc = std::allocator<double>)
    APIARY_INSTANTIATE_MEMBER_AS("write", T = std::complex<float>, Alloc = std::allocator<std::complex<float>>)
    APIARY_INSTANTIATE_MEMBER_AS("write", T = std::complex<double>, Alloc = std::allocator<std::complex<double>>) void
    write(std::string_view name, GeneralRuntimeTensor<T, Alloc> const &tensor);

    /// Read a slice into a pre-sized RuntimeTensor. The tensor's dims
    /// must equal the slab (ranges[d].second - ranges[d].first).
    template <typename T, typename Alloc>
    APIARY_EXPOSE APIARY_INSTANTIATE_MEMBER_AS("read_slice", T = float, Alloc = std::allocator<float>)
    APIARY_INSTANTIATE_MEMBER_AS("read_slice", T = double, Alloc = std::allocator<double>)
    APIARY_INSTANTIATE_MEMBER_AS("read_slice", T = std::complex<float>, Alloc = std::allocator<std::complex<float>>)
    APIARY_INSTANTIATE_MEMBER_AS("read_slice", T = std::complex<double>, Alloc = std::allocator<std::complex<double>>) void
    read_slice(std::string_view name, GeneralRuntimeTensor<T, Alloc> &tensor, std::vector<std::pair<size_t, size_t>> const &ranges);

    /// Write a slice (hyperslab) of an existing entry from a RuntimeTensor.
    template <typename T, typename Alloc>
    APIARY_EXPOSE APIARY_INSTANTIATE_MEMBER_AS("write_slice", T = float, Alloc = std::allocator<float>)
    APIARY_INSTANTIATE_MEMBER_AS("write_slice", T = double, Alloc = std::allocator<double>)
    APIARY_INSTANTIATE_MEMBER_AS("write_slice", T = std::complex<float>, Alloc = std::allocator<std::complex<float>>)
    APIARY_INSTANTIATE_MEMBER_AS("write_slice", T = std::complex<double>, Alloc = std::allocator<std::complex<double>>) void
    write_slice(std::string_view name, GeneralRuntimeTensor<T, Alloc> const &tensor, std::vector<std::pair<size_t, size_t>> const &ranges);

    /// Read a slab into raw memory, whatever the entry's element type.
    ///
    /// @p destination receives the slab's elements densely and in column-major order over the
    /// slab's own extents, which is the file's order, so no element is converted or moved: each
    /// contiguous run of the slab is one read. It must hold the slab's element count times the
    /// entry's element size. An empty range on any axis reads nothing.
    /// @param name The entry.
    /// @param destination Where the elements go.
    /// @param ranges Per-dimension [start, end) ranges into the stored tensor.
    void read_slab(std::string_view name, void *destination, std::vector<std::pair<size_t, size_t>> const &ranges) const;

    /// Write a slab from raw memory into an existing entry, whatever its element type.
    ///
    /// The mirror of @ref read_slab: @p source holds the slab's elements densely and in
    /// column-major order over the slab's own extents, and each contiguous run is one write.
    /// @param name The entry, which must exist (see @ref reserve).
    /// @param source Where the elements come from.
    /// @param ranges Per-dimension [start, end) ranges into the stored tensor.
    void write_slab(std::string_view name, void const *source, std::vector<std::pair<size_t, size_t>> const &ranges);

    // ── Entries ──────────────────────────────────────────────────────────

    /**
     * @brief Give the entry @p old_name the name @p new_name.
     *
     * Only the entry table changes, so a rename costs the same whatever the entry holds. The table
     * is written before this returns. Renaming an entry to its own name does nothing.
     *
     * Another TensorFile open on the same path in a writable mode writes the table it read when it
     * closes, over this one, so rename through the handle that is open, or close it first.
     * @throws std::runtime_error when there is no entry @p old_name, or the file is open for reading.
     * @throws std::invalid_argument when an entry is already named @p new_name, or no entry can hold
     *         that name.
     */
    APIARY_EXPOSE void rename(std::string_view old_name, std::string_view new_name);

    /**
     * @brief Exchange the names of the entries @p first and @p second.
     *
     * Only the entry table changes, as for @ref rename. Afterwards @p first names what @p second
     * held and the other way round, which is the step a program that writes the next value of a
     * tensor beside the current one takes to make the next value current. Swapping an entry with
     * itself does nothing.
     * @throws std::runtime_error when either entry is missing, or the file is open for reading.
     */
    APIARY_EXPOSE void swap(std::string_view first, std::string_view second);

    /**
     * @brief Copy the entry @p source_name of @p source into the entry @p name of this file.
     *
     * The entry @p name is created with the source's extents, element type and stored order when
     * it does not exist. When it does, it must have the same extents and element type, and keeps
     * its own stored order. The elements pass through a buffer of at most @p buffer_bytes, so
     * neither entry is ever held whole: entries stored in the same order are copied as runs of the
     * file, and otherwise a box at a time, in runs of the source's order. @p source may be this
     * file, and copying an entry of this file onto itself does nothing.
     * @param source The file holding the entry to copy.
     * @param source_name The entry to copy.
     * @param name The entry to copy it into.
     * @param buffer_bytes The most the copy holds in memory at once; zero takes
     *        ``--einsums:work-buffer-size``. A buffer smaller than one element holds one element.
     * @throws std::runtime_error when the source entry is missing or damaged, or this file is open for reading.
     * @throws std::invalid_argument when the entry @p name exists with other extents or another
     *         element type, or no entry can hold the name.
     */
    void copy(TensorFile const &source, std::string_view source_name, std::string_view name, size_t buffer_bytes = 0);

    // ── Query ────────────────────────────────────────────────────────────

    /// Check if a tensor exists in the file.
    APIARY_EXPOSE [[nodiscard]] bool contains(std::string_view name) const;

    /// Get the stored dimensions of a tensor.
    APIARY_EXPOSE [[nodiscard]] std::vector<size_t> dims(std::string_view name) const;

    /// The stored dtype. Not bound to Python, where ``read``/``write`` take ``dtype=``.
    [[nodiscard]] DType dtype(std::string_view name) const;

    /// The order @p name is stored in: the declared axis held at each stored position, which is
    /// ``0, 1, ..., rank - 1`` for an entry stored as declared. See @ref TensorEntry::stored_order.
    [[nodiscard]] std::vector<size_t> stored_order(std::string_view name) const;

    /// Store @p name with its axes in @p order from now on, recording it in the entry table.
    ///
    /// Only the record changes: the elements already in the data region are not moved, so this is
    /// for an entry nothing has been written to yet, as a program that reserved it and then chose
    /// how to lay it out has. Throws when @p order is not a permutation of the entry's axes.
    void set_stored_order(std::string_view name, std::vector<size_t> const &order);

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

    /// The (offset, destination, bytes) reader the element transfers in Format.hpp take.
    [[nodiscard]] auto reader() const {
        return [this](uint64_t offset, void *data, size_t bytes) { read_at(offset, data, bytes); };
    }

    /// The (offset, source, bytes) writer the element transfers in Format.hpp take.
    [[nodiscard]] auto writer() {
        return [this](uint64_t offset, void const *data, size_t bytes) { write_at(offset, data, bytes); };
    }

    /// Read the box @p ranges of @p entry into @p tensor, already sized to it. The box is in the
    /// entry's declared axis order, as @p tensor is, whatever order the entry is stored in.
    template <typename T, typename TensorType>
    void read_box(TensorEntry const &entry, TensorType &tensor, size_t rank, std::vector<std::pair<size_t, size_t>> const &ranges) const {
        auto const box = detail::stored_box(entry, ranges, detail::memory_layout<T>(tensor, rank));
        char      *dst = reinterpret_cast<char *>(tensor.data());
        detail::walk_slab<T>(entry.data_offset, box.dims, box.ranges, [&](uint64_t file_off, size_t dst_off, size_t bytes) {
            detail::read_elements(reader(), file_off, dst, box.layout, dst_off / sizeof(T), bytes / sizeof(T));
        });
    }

    /// Write @p tensor, sized to it, to the box @p ranges of @p entry; the mirror of @ref read_box.
    template <typename T, typename TensorType>
    void write_box(TensorEntry const &entry, TensorType const &tensor, size_t rank, std::vector<std::pair<size_t, size_t>> const &ranges) {
        auto const  box = detail::stored_box(entry, ranges, detail::memory_layout<T>(tensor, rank));
        char const *src = reinterpret_cast<char const *>(tensor.data());
        detail::walk_slab<T>(entry.data_offset, box.dims, box.ranges, [&](uint64_t file_off, size_t src_off, size_t bytes) {
            detail::write_elements(writer(), file_off, src, box.layout, src_off / sizeof(T), bytes / sizeof(T));
        });
    }

    /// Write the header and entry table to disk.
    void write_metadata();

    /// Read the header and entry table from disk.
    void read_metadata();

    /// Add a new entry and return its index.
    size_t add_entry(TensorEntry entry);

    /// Map each entry's name to its record, the last of the name when several ranks hold one.
    void index_entries();

    /// Refuse to change a file open for reading, before anything changes.
    void require_writable(std::string_view operation) const;

    /**
     * @brief The entry a write of @p name fills, with its name, type, extents and data region set.
     *
     * An entry of that name and owning rank already in the file is replaced, never kept beside the
     * new one: when it holds the same type and extents its data region and stored order are reused,
     * so rewriting an entry does not grow the file, and otherwise its record is given a new region
     * at the end of the data. A new entry is stored in its declared order.
     * @throws std::invalid_argument when @p name is empty or longer than an entry holds.
     */
    TensorEntry &place_entry(std::string_view operation, std::string_view name, DType dtype, std::vector<size_t> const &dims,
                             uint32_t owning_rank);
};

// ═══════════════════════════════════════════════════════════════════════════════
// Template implementations
// ═══════════════════════════════════════════════════════════════════════════════

template <typename T, size_t Rank>
void TensorFile::write(std::string_view name, Tensor<T, Rank> const &tensor) {
    static_assert(Rank <= ETN_MAX_RANK, "an .etn entry holds at most ETN_MAX_RANK extents");
    std::vector<size_t> dims(Rank);
    for (size_t d = 0; d < Rank; d++)
        dims[d] = tensor.dim(d);
    auto const &entry = place_entry("TensorFile::write", name, dtype_for<T>(), dims, ETN_ALL_RANKS);
    write_box<T>(entry, tensor, Rank, detail::whole_box(entry));
}

template <typename T, size_t Rank>
void TensorFile::write_local(std::string_view name, Tensor<T, Rank> const &tensor, int rank, int num_ranks) {
    static_assert(Rank <= ETN_MAX_RANK, "an .etn entry holds at most ETN_MAX_RANK extents");
    std::vector<size_t> dims(Rank);
    for (size_t d = 0; d < Rank; d++)
        dims[d] = tensor.dim(d);
    auto const &entry = place_entry("TensorFile::write_local", name, dtype_for<T>(), dims, static_cast<uint32_t>(rank));
    write_box<T>(entry, tensor, Rank, detail::whole_box(entry));

    // Update header flags
    _header.flags |= 0x1; // has distributed tensors
    _header.num_ranks = std::max(_header.num_ranks, static_cast<uint32_t>(num_ranks));
}

template <typename T, size_t Rank>
void TensorFile::read(std::string_view name, Tensor<T, Rank> &tensor) {
    auto const &entry = find_entry(name);
    detail::check_entry_type(_path, "TensorFile::read", entry, dtype_for<T>(), Rank);

    // Resize tensor if needed
    Dim<Rank> dims;
    for (size_t d = 0; d < Rank; d++)
        dims[d] = entry.dims[d];
    tensor.resize(dims);

    read_box<T>(entry, tensor, Rank, detail::whole_box(entry));
}

template <typename T, size_t Rank>
void TensorFile::read_slice(std::string_view name, Tensor<T, Rank> &tensor, std::array<std::pair<size_t, size_t>, Rank> const &ranges) {
    auto const                                  &entry = find_entry(name);
    std::vector<std::pair<size_t, size_t>> const rng(ranges.begin(), ranges.end());
    detail::check_entry_type(_path, "TensorFile::read_slice", entry, dtype_for<T>(), Rank);
    detail::check_slab_ranges(_path, "TensorFile::read_slice", entry, rng);

    Dim<Rank> slice_dims;
    for (size_t d = 0; d < Rank; ++d) {
        slice_dims[d] = ranges[d].second - ranges[d].first;
    }
    tensor.resize(slice_dims);
    read_box<T>(entry, tensor, Rank, rng);
}

template <typename T, size_t Rank>
void TensorFile::write_slice(std::string_view name, Tensor<T, Rank> const &tensor,
                             std::array<std::pair<size_t, size_t>, Rank> const &ranges) {
    auto const                                  &entry = find_entry(name);
    std::vector<std::pair<size_t, size_t>> const rng(ranges.begin(), ranges.end());
    detail::check_entry_type(_path, "TensorFile::write_slice", entry, dtype_for<T>(), Rank);
    detail::check_slab_ranges(_path, "TensorFile::write_slice", entry, rng);

    // Validate user tensor matches the requested slab shape.
    for (size_t d = 0; d < Rank; ++d) {
        size_t const want = ranges[d].second - ranges[d].first;
        if (tensor.dim(d) != want) {
            throw std::runtime_error(fmt::format("TensorFile::write_slice: tensor dim {} = {} but slab range is [{}, {}) (size {})", d,
                                                 tensor.dim(d), ranges[d].first, ranges[d].second, want));
        }
    }

    write_box<T>(entry, tensor, Rank, rng);
}

template <typename T>
void TensorFile::reserve(std::string_view name, std::vector<size_t> const &dims, std::vector<size_t> const &stored_order) {
    detail::check_writable_rank(_path, "TensorFile::reserve", name, dims.size());
    if (!stored_order.empty() && stored_order.size() != dims.size()) {
        // Checked before the entry is placed, so a refused order leaves the file as it was.
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "TensorFile::reserve: a stored order for '{}' names {} axes, not its {}", name,
                                stored_order.size(), dims.size());
    }

    // A reserved entry holds nothing yet, so a reused one takes the order asked for.
    auto &entry = place_entry("TensorFile::reserve", name, dtype_for<T>(), dims, ETN_ALL_RANKS);
    if (stored_order.empty()) {
        std::vector<size_t> declared(dims.size());
        for (size_t d = 0; d < declared.size(); ++d) {
            declared[d] = d;
        }
        entry.set_stored_order(declared);
    } else {
        entry.set_stored_order(stored_order);
    }
}

// ── RuntimeTensor overloads ──────────────────────────────────────────

template <typename T, typename Alloc>
void TensorFile::read(std::string_view name, GeneralRuntimeTensor<T, Alloc> &tensor) {
    auto const &entry = find_entry(name);
    detail::check_entry_type(_path, "TensorFile::read", entry, dtype_for<T>(), std::nullopt);

    std::vector<size_t> dims(entry.rank);
    for (size_t d = 0; d < entry.rank; ++d) {
        dims[d] = entry.dims[d];
    }
    tensor.resize(dims);

    read_box<T>(entry, tensor, tensor.rank(), detail::whole_box(entry));
}

template <typename T, typename Alloc>
void TensorFile::write(std::string_view name, GeneralRuntimeTensor<T, Alloc> const &tensor) {
    detail::check_writable_rank(_path, "TensorFile::write", name, tensor.rank());
    std::vector<size_t> dims(tensor.rank());
    for (size_t d = 0; d < tensor.rank(); ++d) {
        dims[d] = tensor.dim(d);
    }
    auto const &entry = place_entry("TensorFile::write", name, dtype_for<T>(), dims, ETN_ALL_RANKS);
    write_box<T>(entry, tensor, tensor.rank(), detail::whole_box(entry));
}

template <typename T, typename Alloc>
void TensorFile::read_slice(std::string_view name, GeneralRuntimeTensor<T, Alloc> &tensor,
                            std::vector<std::pair<size_t, size_t>> const &ranges) {
    auto const &entry = find_entry(name);
    detail::check_entry_type(_path, "TensorFile::read_slice", entry, dtype_for<T>(), std::nullopt);
    detail::check_slab_ranges(_path, "TensorFile::read_slice", entry, ranges);

    std::vector<size_t> slab_dims(ranges.size());
    for (size_t d = 0; d < ranges.size(); ++d) {
        slab_dims[d] = ranges[d].second - ranges[d].first;
    }
    tensor.resize(slab_dims);
    read_box<T>(entry, tensor, tensor.rank(), ranges);
}

template <typename T, typename Alloc>
void TensorFile::write_slice(std::string_view name, GeneralRuntimeTensor<T, Alloc> const &tensor,
                             std::vector<std::pair<size_t, size_t>> const &ranges) {
    auto const &entry = find_entry(name);
    detail::check_entry_type(_path, "TensorFile::write_slice", entry, dtype_for<T>(), std::nullopt);
    detail::check_slab_ranges(_path, "TensorFile::write_slice", entry, ranges);
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

    write_box<T>(entry, tensor, tensor.rank(), ranges);
}

template <typename T, size_t Rank>
void TensorFile::read_local(std::string_view name, Tensor<T, Rank> &tensor, int rank) {
    // Find the entry for this specific rank
    std::string const name_str(name);
    for (auto const &entry : _entries) {
        if (entry.get_name() == name_str && std::cmp_equal(entry.owning_rank, rank)) {
            detail::check_entry_type(_path, "TensorFile::read_local", entry, dtype_for<T>(), Rank);
            Dim<Rank> dims;
            for (size_t d = 0; d < Rank; d++)
                dims[d] = entry.dims[d];
            tensor.resize(dims);
            read_box<T>(entry, tensor, Rank, detail::whole_box(entry));
            return;
        }
    }
    throw std::runtime_error(fmt::format("TensorFile: no entry for '{}' with rank {}", name, rank));
}

} // namespace tensor_io
EINSUMS_NAMESPACE_END()
