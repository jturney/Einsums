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
#include <functional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

EINSUMS_NAMESPACE_BEGIN()
namespace APIARY_MODULE("io") tensor_io {

/**
 * @brief A handle on an ``.etn`` file.
 *
 * Writing, rewriting and reserving entries change the handle's own table until it commits, which
 * @ref close, @ref flush and the destructor do, and each entry operation (@ref rename, @ref swap,
 * @ref copy, @ref remove, @ref restage, @ref compact) does before it returns. A commit puts new data
 * and a new table into space the file's last commit does not use and then rewrites the header, so a
 * process that stops at any point leaves the file as its last commit left it. With
 * ``--einsums:io:sync`` on, the default, a power loss does too. A handle refuses to commit over a
 * file another handle has committed to since this one last read it.
 */
class EINSUMS_EXPORT APIARY_EXPOSE APIARY_NOCOPY APIARY_NOMOVE TensorFile {
  public:
    enum class APIARY_EXPOSE Mode : std::uint8_t { Read, Write, ReadWrite };

    /// Open or create a tensor file.
    /// @param path File path (creates parent directories if needed).
    /// @param mode Read, Write, or ReadWrite.
    APIARY_EXPOSE explicit TensorFile(std::string path, Mode mode = Mode::ReadWrite);

    /// Commits and closes the file. A commit that fails here is logged and dropped, leaving the
    /// file as its last commit left it; @ref close reports it instead.
    ~TensorFile();

    // Non-copyable, movable: a moved-from handle holds no file.
    TensorFile(TensorFile const &)            = delete;
    TensorFile &operator=(TensorFile const &) = delete;
    TensorFile(TensorFile &&other) noexcept;
    TensorFile &operator=(TensorFile &&other) noexcept;

    // ── Write ────────────────────────────────────────────────────────────

    /// Write a full tensor to the file.
    template <typename T, size_t Rank>
    void write(std::string_view name, Tensor<T, Rank> const &tensor);

    /// Write this rank's local partition of a distributed tensor.
    template <typename T, size_t Rank>
    void write_local(std::string_view name, Tensor<T, Rank> const &tensor, int rank, int num_ranks);

    /// Write a hyperslab of an existing entry; @p tensor must have the slab's shape.
    /// @param name The entry, which must exist (see @ref reserve).
    /// @param tensor The slab's elements.
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
    /// @param name The entry.
    /// @param tensor Where the slab goes, already of its shape.
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
     * Only the entry table changes, so a rename costs the same whatever the entry holds. It is
     * committed before this returns. Renaming an entry to its own name does nothing.
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
     * @brief Make the entry @p source the entry @p target, dropping what @p target held.
     *
     * Only the entry table changes, as for @ref rename, and in one commit: a crash leaves either
     * both entries as they were or @p target holding what @p source held and no entry @p source.
     * The space @p target's old elements took is free from that commit on. This is the step that
     * makes a value written beside the current one current, leaving no second entry behind.
     * Replacing an entry with itself does nothing.
     * @throws std::runtime_error when there is no entry @p source, or the file is open for reading.
     * @throws std::invalid_argument when no entry can hold the name @p target.
     * @versionadded{2.0.0}
     */
    APIARY_EXPOSE void replace(std::string_view source, std::string_view target);

    /**
     * @brief Copy the entry @p source_name of @p source into the entry @p name of this file.
     *
     * The entry @p name is created with the source's extents, element type and stored order when
     * it does not exist. When it does, it must have the same extents and element type, keeps its own
     * stored order, and holds its old elements until the copy is committed. The elements pass through a buffer of at most @p buffer_bytes,
     * so neither entry is ever held whole: entries stored in the same order are copied as runs of the file, and otherwise a box at a time,
     * in runs of the source's order. @p source may be this file, and copying an entry of this file onto itself does nothing.
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

    /**
     * @brief Remove the entry @p name, for every rank that holds one, and commit.
     *
     * Its data region becomes space later entries reuse; @ref compact gives it back to the disk.
     * @throws std::runtime_error when there is no entry @p name, or the file is open for reading.
     */
    APIARY_EXPOSE void remove(std::string_view name);

    /**
     * @brief Store the entry @p name with its axes in @p order, moving its elements, and commit.
     *
     * The elements are copied into a new region in the new order through a buffer of at most
     * @p buffer_bytes, and one commit then points the entry at it, so the entry holds its old layout
     * or its new one whenever the program stops. The file needs room for a second copy of the entry
     * while this runs. Restaging to the order the entry already has does nothing.
     * @param name The entry.
     * @param order The declared axis to store at each position, as @ref stored_order returns it.
     * @param buffer_bytes As for @ref copy.
     * @throws std::runtime_error when there is no entry @p name, or the file is open for reading.
     * @throws std::invalid_argument when @p order is not a permutation of the entry's axes.
     */
    APIARY_EXPOSE void restage(std::string_view name, std::vector<size_t> const &order, size_t buffer_bytes = 0);

    /**
     * @brief Rewrite the file without the space no entry uses.
     *
     * Every entry is copied, in its stored order and through a buffer of at most @p buffer_bytes,
     * into a new file beside this one, which then replaces it. The file is this one or the
     * compacted one whenever the program stops, and the disk needs room for the entries' bytes
     * while this runs. Another handle open on the file refuses to commit afterwards; backings,
     * which open the file for each slab, see the compacted file at their next slab.
     * @param buffer_bytes As for @ref copy.
     * @throws std::runtime_error when the file is open for reading.
     */
    APIARY_EXPOSE void compact(size_t buffer_bytes = 0);

    // ── Query ────────────────────────────────────────────────────────────

    /// Check if a tensor exists in the file.
    APIARY_EXPOSE [[nodiscard]] bool contains(std::string_view name) const;

    /// Get the stored dimensions of a tensor.
    APIARY_EXPOSE [[nodiscard]] std::vector<size_t> dims(std::string_view name) const;

    /// The stored dtype. Not bound to Python, where ``read``/``write`` take ``dtype=``.
    [[nodiscard]] DType dtype(std::string_view name) const;

    /// The order @p name is stored in: the declared axis held at each stored position, which is
    /// ``0, 1, ..., rank - 1`` for an entry stored as declared. See @ref TensorEntry::stored_order.
    APIARY_EXPOSE [[nodiscard]] std::vector<size_t> stored_order(std::string_view name) const;

    /// Store @p name with its axes in @p order from now on, recording it in the entry table.
    ///
    /// Only the record changes: the elements already in the data region are not moved, so this is
    /// for an entry nothing has been written to yet, as a program that reserved it and then chose
    /// how to lay it out has. Throws when @p order is not a permutation of the entry's axes.
    void set_stored_order(std::string_view name, std::vector<size_t> const &order);

    /// The bytes of the file, as of the last commit, that hold neither the header, an entry's
    /// elements nor the entry table: what @ref compact would give back.
    APIARY_EXPOSE [[nodiscard]] size_t unused_bytes() const;

    /// List all tensor names in the file.
    APIARY_EXPOSE [[nodiscard]] std::vector<std::string> tensor_names() const;

    /// Number of tensors in the file.
    APIARY_EXPOSE [[nodiscard]] size_t num_tensors() const { return _entries.size(); }

    /// Commit, and sync the file to disk whatever ``--einsums:io:sync`` says.
    APIARY_EXPOSE void flush();

    /// Commit and close the file. Unlike the destructor, throws when the commit fails.
    APIARY_EXPOSE void close();

    /// Get the file path.
    APIARY_EXPOSE [[nodiscard]] std::string const &path() const { return _path; }

  private:
    /// A byte range of the file, [first, second).
    using Extent = std::pair<uint64_t, uint64_t>;

    std::string                             _path;
    Mode                                    _mode;
    int                                     _fd{-1}; ///< POSIX file descriptor
    FileHeader                              _header;
    std::vector<TensorEntry>                _entries;
    std::unordered_map<std::string, size_t> _name_to_index; ///< name → entry index
    bool                                    _dirty{false};  ///< The table differs from the last commit.
    bool                                    _has_header{false};
    std::vector<Extent>                     _committed; ///< The table and regions the header on disk reaches.
    std::pair<uint64_t, uint64_t>           _identity{0, 0};

    /// Open @p _path in @p _mode and read or write its header.
    void open();

    /// Write this handle's table and point the header at it, unless nothing changed. Syncs before
    /// and after the header when @p sync or ``--einsums:io:sync`` asks.
    void commit(bool sync);

    /// Sync the file to disk.
    void sync_to_disk();

    /// Refuse to change the file when another handle has committed to it, or replaced it, since
    /// this one last read it: space this one thinks free may hold what the other committed.
    void require_current() const;

    /// Where @p bytes bytes can go: the first gap neither the last commit nor this handle's table
    /// uses that holds them, or the end of the data.
    [[nodiscard]] uint64_t allocate(uint64_t bytes) const;

    /// The regions this handle's table names, and the last commit's table and regions, merged.
    [[nodiscard]] std::vector<Extent> used_extents() const;

    /// Read or write the box @p ranges of @p entry as raw elements, densely in its declared order.
    void read_entry_slab(TensorEntry const &entry, void *destination, std::vector<std::pair<size_t, size_t>> const &ranges) const;
    void write_entry_slab(TensorEntry const &entry, void const *source, std::vector<std::pair<size_t, size_t>> const &ranges);

    /// Copy the elements of @p from in @p source into the region of @p to here, which has its
    /// extents and type, through a buffer of at most @p buffer_bytes.
    void copy_elements(TensorFile const &source, TensorEntry const &from, TensorEntry const &to, size_t buffer_bytes);

    /// This handle's table, to put back when a change fails partway.
    struct TableState {
        std::vector<TensorEntry> entries;
        bool                     dirty;
    };
    [[nodiscard]] TableState table_state() const { return {.entries = _entries, .dirty = _dirty}; }
    void                     restore(TableState state);

    /// Make @p change to the table and commit it, or, when either fails, put the table back.
    void commit_change(std::function<void()> const &change);

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

    /// Read the header and entry table from disk.
    void read_metadata();

    /// Add a new entry and return its index.
    size_t add_entry(TensorEntry entry);

    /// Map each entry's name to its record, the last of the name when several ranks hold one.
    void index_entries();

    /// Refuse to change a file open for reading, before anything changes.
    void require_writable(std::string_view operation) const;

    /**
     * @brief The entry a write of @p name fills, with its name, type, extents and a new data region.
     *
     * An entry of that name and owning rank already in the file is replaced, never kept beside the
     * new one. Its region is not written over: the new elements go to a region of their own, and the
     * old one is reused only after the commit that drops it, so a crash leaves the old elements. A
     * rewrite of the same type and extents keeps the entry's stored order; a new entry is stored in
     * its declared order.
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
    auto saved = table_state();
    try {
        auto const &entry = place_entry("TensorFile::write", name, dtype_for<T>(), dims, ETN_ALL_RANKS);
        write_box<T>(entry, tensor, Rank, detail::whole_box(entry));
    } catch (...) {
        restore(std::move(saved));
        throw;
    }
}

template <typename T, size_t Rank>
void TensorFile::write_local(std::string_view name, Tensor<T, Rank> const &tensor, int rank, int num_ranks) {
    static_assert(Rank <= ETN_MAX_RANK, "an .etn entry holds at most ETN_MAX_RANK extents");
    std::vector<size_t> dims(Rank);
    for (size_t d = 0; d < Rank; d++)
        dims[d] = tensor.dim(d);
    auto saved = table_state();
    try {
        auto const &entry = place_entry("TensorFile::write_local", name, dtype_for<T>(), dims, static_cast<uint32_t>(rank));
        write_box<T>(entry, tensor, Rank, detail::whole_box(entry));
    } catch (...) {
        restore(std::move(saved));
        throw;
    }

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

    require_current();
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

    // A reserved entry holds nothing yet, so a replaced one takes the order asked for.
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
    auto saved = table_state();
    try {
        auto const &entry = place_entry("TensorFile::write", name, dtype_for<T>(), dims, ETN_ALL_RANKS);
        write_box<T>(entry, tensor, tensor.rank(), detail::whole_box(entry));
    } catch (...) {
        restore(std::move(saved));
        throw;
    }
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

    require_current();
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
