//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/BufferAllocator/ModuleVars.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/TensorIO/TensorFile.hpp>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstddef>
#include <cstring>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "PosixFileCompat.hpp"

EINSUMS_NAMESPACE_BEGIN(tensor_io)

namespace detail {

void check_entry_type(std::string_view path, std::string_view operation, TensorEntry const &entry, DType want,
                      std::optional<size_t> want_rank) {
    std::string const name   = entry.get_name();
    auto const        stored = static_cast<DType>(entry.dtype);

    if (entry.rank > ETN_MAX_RANK) {
        EINSUMS_THROW_EXCEPTION(std::runtime_error, "{}: entry '{}' in '{}' is damaged: stored rank {} exceeds the maximum {}", operation,
                                name, path, entry.rank, ETN_MAX_RANK);
    }
    if (dtype_size(stored) == 0) {
        EINSUMS_THROW_EXCEPTION(std::runtime_error, "{}: entry '{}' in '{}' is damaged: unknown dtype code {}", operation, name, path,
                                entry.dtype);
    }
    if (stored != want) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                "{}: entry '{}' in '{}' holds {} data, but the tensor holds {}; element types are never converted",
                                operation, name, path, dtype_name(stored), dtype_name(want));
    }
    if (want_rank.has_value() && *want_rank != entry.rank) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "{}: entry '{}' in '{}' has rank {}, but the tensor has rank {}", operation, name,
                                path, entry.rank, *want_rank);
    }

    // The copy sizes the destination from dims and moves data_size bytes, so the two must agree.
    uint64_t elements = 1;
    for (size_t d = 0; d < entry.rank; ++d) {
        if (entry.dims[d] != 0 && elements > std::numeric_limits<uint64_t>::max() / entry.dims[d]) {
            EINSUMS_THROW_EXCEPTION(std::runtime_error, "{}: entry '{}' in '{}' is damaged: its dims overflow", operation, name, path);
        }
        elements *= entry.dims[d];
    }
    uint64_t const elem_bytes = dtype_size(stored);
    if (elements > std::numeric_limits<uint64_t>::max() / elem_bytes || entry.data_size != elements * elem_bytes) {
        EINSUMS_THROW_EXCEPTION(std::runtime_error,
                                "{}: entry '{}' in '{}' is damaged: {} {} elements need {} bytes, but the entry records {}", operation,
                                name, path, elements, dtype_name(stored), elements * elem_bytes, entry.data_size);
    }
}

void check_writable_rank(std::string_view path, std::string_view operation, std::string_view name, size_t rank) {
    if (rank > ETN_MAX_RANK) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "{}: '{}' in '{}' has rank {}, and an .etn entry holds at most {} extents",
                                operation, name, path, rank, ETN_MAX_RANK);
    }
}

void check_entry_name(std::string_view path, std::string_view operation, std::string_view name) {
    if (name.empty()) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "{}: an entry of '{}' needs a name", operation, path);
    }
    if (name.size() >= ETN_MAX_NAME || name.find('\0') != std::string_view::npos) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                "{}: '{}' cannot name an entry of '{}': an entry holds a name of at most {} bytes, with no NUL byte",
                                operation, name, path, ETN_MAX_NAME - 1);
    }
}

void check_slab_ranges(std::string_view path, std::string_view operation, TensorEntry const &entry,
                       std::vector<std::pair<size_t, size_t>> const &ranges) {
    std::string const name = entry.get_name();
    if (ranges.size() != entry.rank) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "{}: {} ranges given for entry '{}' in '{}', which has rank {}", operation,
                                ranges.size(), name, path, entry.rank);
    }
    for (size_t d = 0; d < ranges.size(); ++d) {
        auto const [first, last] = ranges[d];
        if (first > last) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument, "{}: range [{}, {}) for dim {} of entry '{}' in '{}' is reversed", operation,
                                    first, last, d, name, path);
        }
        if (last > entry.dims[d]) {
            EINSUMS_THROW_EXCEPTION(std::out_of_range, "{}: range [{}, {}) for dim {} of entry '{}' in '{}' runs past the stored extent {}",
                                    operation, first, last, d, name, path, entry.dims[d]);
        }
    }
}

MemoryLayout dense_layout(std::vector<size_t> const &dims, size_t elem_size) {
    MemoryLayout layout;
    layout.dims      = dims;
    layout.elem_size = elem_size;
    layout.strides.resize(dims.size());
    size_t stride = 1;
    for (size_t d = 0; d < dims.size(); ++d) {
        layout.strides[d] = stride;
        stride *= dims[d];
    }
    layout.file_order = true;
    return layout;
}

StoredBox stored_box(TensorEntry const &entry, std::vector<std::pair<size_t, size_t>> const &ranges, MemoryLayout const &memory) {
    StoredBox box;
    if (!entry.permuted()) {
        box.dims.assign(entry.dims, entry.dims + entry.rank);
        box.ranges = ranges;
        box.layout = memory;
        return box;
    }
    std::vector<size_t> const order = entry.stored_order();
    box.layout.elem_size            = memory.elem_size;
    for (size_t const declared : order) {
        box.dims.push_back(entry.dims[declared]);
        box.ranges.push_back(ranges[declared]);
        box.layout.dims.push_back(memory.dims[declared]);
        box.layout.strides.push_back(memory.strides[declared]);
    }
    box.layout.file_order = is_file_order(box.layout.dims, box.layout.strides);
    return box;
}

std::vector<std::pair<size_t, size_t>> whole_box(TensorEntry const &entry) {
    std::vector<std::pair<size_t, size_t>> ranges;
    for (size_t d = 0; d < entry.rank; ++d) {
        ranges.emplace_back(0, entry.dims[d]);
    }
    return ranges;
}

bool is_file_order(std::vector<size_t> const &dims, std::vector<size_t> const &strides) {
    size_t expected = 1;
    for (size_t d = 0; d < dims.size(); ++d) {
        if (dims[d] > 1 && strides[d] != expected) {
            return false;
        }
        expected *= dims[d];
    }
    return true;
}

namespace {

/// Walk @p count elements in column-major order from linear index @p first, calling
/// @p visit(memory_offset_in_elements) for each.
template <typename Visit>
void walk_file_order(MemoryLayout const &layout, size_t first, size_t count, Visit &&visit) {
    if (count == 0) {
        return;
    }
    size_t const        rank = layout.dims.size();
    std::vector<size_t> index(rank);
    size_t              offset    = 0;
    size_t              remaining = first;
    for (size_t d = 0; d < rank; ++d) {
        index[d] = remaining % layout.dims[d];
        remaining /= layout.dims[d];
        offset += index[d] * layout.strides[d];
    }
    for (size_t n = 0; n < count; ++n) {
        visit(offset);
        for (size_t d = 0; d < rank; ++d) {
            if (++index[d] < layout.dims[d]) {
                offset += layout.strides[d];
                break;
            }
            offset -= (layout.dims[d] - 1) * layout.strides[d];
            index[d] = 0;
        }
    }
}

} // namespace

void gather_file_order(char *dst, char const *tensor, MemoryLayout const &layout, size_t first, size_t count) {
    size_t const es = layout.elem_size;
    walk_file_order(layout, first, count, [&](size_t offset) {
        std::memcpy(dst, tensor + offset * es, es);
        dst += es;
    });
}

void scatter_file_order(char *tensor, char const *src, MemoryLayout const &layout, size_t first, size_t count) {
    size_t const es = layout.elem_size;
    walk_file_order(layout, first, count, [&](size_t offset) {
        std::memcpy(tensor + offset * es, src, es);
        src += es;
    });
}

} // namespace detail

TensorFile::TensorFile(std::string path, Mode mode) : _path(std::move(path)), _mode(mode) {
    detail::OpenMode open_mode = detail::OpenMode::ReadWrite;
    switch (_mode) {
    case Mode::Read:
        open_mode = detail::OpenMode::Read;
        break;
    case Mode::Write:
        open_mode = detail::OpenMode::WriteTruncate;
        break;
    case Mode::ReadWrite:
        open_mode = detail::OpenMode::ReadWrite;
        break;
    }

    _fd = detail::open_file(_path, open_mode);
    if (_fd < 0) {
        EINSUMS_THROW_EXCEPTION(std::runtime_error, "TensorFile: cannot open '{}': {}", _path, std::strerror(errno));
    }

    if (_mode == Mode::Write) {
        // New file: write initial header
        _header.init();
        _next_data_offset = _header.data_offset;
        write_metadata();
    } else {
        // Existing file: read header and entries
        struct stat st;
        if (::fstat(_fd, &st) == 0 && st.st_size > 0) {
            read_metadata();
        } else {
            // Empty file or new file in ReadWrite mode
            _header.init();
            _next_data_offset = _header.data_offset;
            if (_mode == Mode::ReadWrite) {
                write_metadata();
            }
        }
    }
}

// NOLINTNEXTLINE(bugprone-exception-escape)
TensorFile::~TensorFile() {
    if (_fd >= 0) {
        if (_mode != Mode::Read) {
            write_metadata(); // Flush entry table
        }
        detail::close_file(_fd);
        _fd = -1;
    }
}

namespace {
/// The most one pread or pwrite is asked to move. macOS rejects a request over INT_MAX bytes with
/// EINVAL, and the Windows shim narrows the count to a 32-bit DWORD, which turns a request of a
/// multiple of 4 GiB into a read of nothing; a merged slab run can be that long. The loops below
/// move the rest.
constexpr size_t kMaxIoBytes = size_t{1} << 30;
} // namespace

void TensorFile::write_at(uint64_t offset, void const *data, size_t bytes) {
    if (bytes == 0)
        return;
    auto  *ptr   = static_cast<char const *>(data);
    size_t total = 0;
    while (total < bytes) {
        detail::io_result_t const written =
            detail::pwrite_file(_fd, ptr + total, std::min(bytes - total, kMaxIoBytes), static_cast<detail::file_offset_t>(offset + total));
        if (written < 0) {
            if (errno == EINTR)
                continue;
            EINSUMS_THROW_EXCEPTION(std::runtime_error, "TensorFile: pwrite failed at offset {}: {}", offset, std::strerror(errno));
        }
        total += static_cast<size_t>(written);
    }
}

void TensorFile::read_at(uint64_t offset, void *data, size_t bytes) const {
    if (bytes == 0)
        return;
    auto  *ptr   = static_cast<char *>(data);
    size_t total = 0;
    while (total < bytes) {
        detail::io_result_t const nread =
            detail::pread_file(_fd, ptr + total, std::min(bytes - total, kMaxIoBytes), static_cast<detail::file_offset_t>(offset + total));
        if (nread < 0) {
            if (errno == EINTR)
                continue;
            EINSUMS_THROW_EXCEPTION(std::runtime_error, "TensorFile: pread failed at offset {}: {}", offset, std::strerror(errno));
        }
        if (nread == 0) {
            EINSUMS_THROW_EXCEPTION(std::runtime_error, "TensorFile: unexpected EOF at offset {} (read {}/{})", offset + total, total,
                                    bytes);
        }
        total += static_cast<size_t>(nread);
    }
}

std::vector<size_t> TensorEntry::stored_order() const {
    std::vector<size_t> out(rank);
    if (!permuted()) {
        for (size_t s = 0; s < rank; ++s) {
            out[s] = s;
        }
        return out;
    }
    uint32_t const bits =
        static_cast<uint32_t>(order[0]) | (static_cast<uint32_t>(order[1]) << 8U) | (static_cast<uint32_t>(order[2]) << 16U);
    std::vector<bool> seen(rank, false);
    for (size_t s = 0; s < rank; ++s) {
        size_t const declared = (bits >> (3U * s)) & 0x7U;
        if (declared >= rank || seen[declared]) {
            EINSUMS_THROW_EXCEPTION(std::runtime_error,
                                    "TensorFile: entry '{}' records a stored order that is not a permutation of its {} axes", get_name(),
                                    rank);
        }
        seen[declared] = true;
        out[s]         = declared;
    }
    return out;
}

void TensorEntry::set_stored_order(std::vector<size_t> const &stored) {
    std::vector<bool> seen(rank, false);
    bool              identity = true;
    uint32_t          bits     = 0;
    if (stored.size() != rank) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "TensorFile: a stored order for entry '{}' names {} axes, not its {}", get_name(),
                                stored.size(), rank);
    }
    for (size_t s = 0; s < stored.size(); ++s) {
        if (stored[s] >= rank || seen[stored[s]]) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                    "TensorFile: the stored order given for entry '{}' is not a permutation of its {} axes", get_name(),
                                    rank);
        }
        seen[stored[s]] = true;
        identity        = identity && stored[s] == s;
        bits |= static_cast<uint32_t>(stored[s]) << (3U * s);
    }
    if (identity) {
        bits = 0;
    }
    order[0] = static_cast<uint8_t>(bits & 0xFFU);
    order[1] = static_cast<uint8_t>((bits >> 8U) & 0xFFU);
    order[2] = static_cast<uint8_t>((bits >> 16U) & 0xFFU);
}

std::vector<size_t> TensorFile::stored_order(std::string_view name) const {
    return find_entry(name).stored_order();
}

void TensorFile::set_stored_order(std::string_view name, std::vector<size_t> const &order) {
    auto const &entry = find_entry(name);
    auto const  index = static_cast<size_t>(&entry - _entries.data());
    _entries[index].set_stored_order(order);
    write_metadata();
}

void TensorFile::write_metadata() {
    // Update header. A file stays at the version every release reads unless an entry needs more.
    _header.version =
        std::ranges::any_of(_entries, [](TensorEntry const &e) { return e.permuted(); }) ? ETN_VERSION : ETN_VERSION_DECLARED_ORDER;
    _header.num_tensors        = static_cast<uint32_t>(_entries.size());
    _header.entry_table_offset = align_up(_next_data_offset, ETN_DATA_ALIGNMENT);
    _header.total_size         = _header.entry_table_offset + _entries.size() * sizeof(TensorEntry);

    // Write header at offset 0
    write_at(0, &_header, sizeof(FileHeader));

    // Write entry table at the end
    if (!_entries.empty()) {
        write_at(_header.entry_table_offset, _entries.data(), _entries.size() * sizeof(TensorEntry));
    }

    // Truncate file to exact size
    if (detail::truncate_file(_fd, static_cast<detail::file_offset_t>(_header.total_size)) < 0) {
        // Non-fatal: file may be slightly larger than needed
    }
}

void TensorFile::read_metadata() {
    // Read header
    read_at(0, &_header, sizeof(FileHeader));

    if (!_header.is_valid()) {
        EINSUMS_THROW_EXCEPTION(std::runtime_error, "TensorFile: '{}' is not a valid .etn file (bad magic or version)", _path);
    }

    // Read entry table
    _entries.resize(_header.num_tensors);
    if (_header.num_tensors > 0) {
        read_at(_header.entry_table_offset, _entries.data(), _header.num_tensors * sizeof(TensorEntry));
    }

    // A writer before entries were replaced in place appended a rewritten entry beside the old
    // one. Keep only the last record of each name and owning rank, which is the one lookups
    // already found; the next write of the table drops the rest.
    std::vector<TensorEntry>                   kept;
    std::set<std::pair<std::string, uint32_t>> seen;
    for (auto it = _entries.rbegin(); it != _entries.rend(); ++it) {
        if (seen.emplace(it->get_name(), it->owning_rank).second) {
            kept.push_back(*it);
        }
    }
    std::ranges::reverse(kept);
    _entries = std::move(kept);

    index_entries();

    // Compute next available data offset (after last data region)
    _next_data_offset = _header.data_offset;
    for (auto const &e : _entries) {
        uint64_t const end = e.data_offset + e.data_size;
        if (end > _next_data_offset)
            _next_data_offset = end;
    }
}

size_t TensorFile::add_entry(TensorEntry entry) {
    size_t const idx                 = _entries.size();
    _name_to_index[entry.get_name()] = idx;
    _entries.push_back(entry);
    return idx;
}

void TensorFile::index_entries() {
    _name_to_index.clear();
    for (size_t i = 0; i < _entries.size(); i++) {
        _name_to_index[_entries[i].get_name()] = i;
    }
}

void TensorFile::require_writable(std::string_view operation) const {
    if (_mode == Mode::Read) {
        EINSUMS_THROW_EXCEPTION(std::runtime_error, "{}: '{}' is open for reading only", operation, _path);
    }
}

void TensorFile::rename(std::string_view old_name, std::string_view new_name) {
    require_writable("TensorFile::rename");
    (void)find_entry(old_name);
    detail::check_entry_name(_path, "TensorFile::rename", new_name);
    if (old_name == new_name) {
        return;
    }
    if (contains(new_name)) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "TensorFile::rename: '{}' already holds an entry named '{}'; swap the two instead",
                                _path, new_name);
    }
    for (auto &entry : _entries) {
        if (entry.get_name() == old_name) {
            entry.set_name(new_name);
        }
    }
    index_entries();
    write_metadata();
}

void TensorFile::swap(std::string_view first, std::string_view second) {
    require_writable("TensorFile::swap");
    (void)find_entry(first);
    (void)find_entry(second);
    if (first == second) {
        return;
    }
    std::string const first_name(first);
    std::string const second_name(second);
    for (auto &entry : _entries) {
        if (std::string const name = entry.get_name(); name == first_name) {
            entry.set_name(second_name);
        } else if (name == second_name) {
            entry.set_name(first_name);
        }
    }
    index_entries();
    write_metadata();
}

namespace {

/// The buffer a copy moves elements through: @p asked bytes, or the work buffer when zero.
size_t copy_buffer_bytes(size_t asked) {
    return asked != 0 ? asked : einsums::detail::Einsums_BufferAllocator_vars::get_singleton().get_work_buffer_size();
}

} // namespace

void TensorFile::copy(TensorFile const &source, std::string_view source_name, std::string_view name, size_t buffer_bytes) {
    require_writable("TensorFile::copy");
    // By value: placing the destination may move the records of this file, which may be the source.
    TensorEntry const from  = source.find_entry(source_name);
    auto const        dtype = static_cast<DType>(from.dtype);
    detail::check_entry_type(source._path, "TensorFile::copy", from, dtype, std::nullopt);
    if (&source == this && source_name == name) {
        return;
    }

    std::vector<size_t> const dims(from.dims, from.dims + from.rank);
    if (contains(name)) {
        TensorEntry const &existing = find_entry(name);
        if (existing.dtype != from.dtype || existing.rank != from.rank || !std::equal(dims.begin(), dims.end(), existing.dims)) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                    "TensorFile::copy: entry '{}' of '{}' holds {} elements of extents {}, and entry '{}' of '{}' holds {} "
                                    "elements of extents {}; a copy never changes an entry's shape or type",
                                    source_name, source._path, dtype_name(dtype), dims, name, _path,
                                    dtype_name(static_cast<DType>(existing.dtype)),
                                    std::vector<size_t>(existing.dims, existing.dims + existing.rank));
        }
    } else {
        place_entry("TensorFile::copy", name, dtype, dims, ETN_ALL_RANKS).set_stored_order(from.stored_order());
    }
    TensorEntry const to = find_entry(name);

    size_t const elem   = dtype_size(dtype);
    size_t const budget = std::max(copy_buffer_bytes(buffer_bytes), elem);
    if (from.data_size != 0) {
        std::vector<size_t> const order = from.stored_order();
        if (order == to.stored_order()) {
            // The same layout: the data regions are the same bytes, moved as runs of the file.
            std::vector<std::byte> buffer(static_cast<size_t>(std::min<uint64_t>(budget, from.data_size)));
            for (uint64_t done = 0; done < from.data_size; done += buffer.size()) {
                size_t const bytes = static_cast<size_t>(std::min<uint64_t>(buffer.size(), from.data_size - done));
                source.read_at(from.data_offset + done, buffer.data(), bytes);
                write_at(to.data_offset + done, buffer.data(), bytes);
            }
        } else {
            // Boxes cut along the source's order: the axes it stores fastest whole while they fit,
            // the next in blocks, and the rest one index at a time, so each read is long runs.
            size_t const rank  = dims.size();
            size_t const limit = budget / elem;
            size_t       run   = 1;
            size_t       whole = 0;
            while (whole < rank && run * dims[order[whole]] <= limit) {
                run *= dims[order[whole]];
                ++whole;
            }
            size_t const block = whole < rank ? limit / run : 1;

            std::vector<std::pair<size_t, size_t>> ranges(rank);
            std::vector<size_t>                    at(rank, 0);
            for (size_t s = 0; s < whole; ++s) {
                ranges[order[s]] = {0, dims[order[s]]};
            }
            std::vector<std::byte> buffer(run * block * elem);
            for (bool more = true; more;) {
                for (size_t s = whole; s < rank; ++s) {
                    size_t const step = s == whole ? block : 1;
                    ranges[order[s]]  = {at[s], std::min(at[s] + step, dims[order[s]])};
                }
                source.read_slab(source_name, buffer.data(), ranges);
                write_slab(name, buffer.data(), ranges);

                more = false;
                for (size_t s = whole; s < rank; ++s) {
                    at[s] += s == whole ? block : 1;
                    if (at[s] < dims[order[s]]) {
                        more = true;
                        break;
                    }
                    at[s] = 0;
                }
            }
        }
    }
    write_metadata();
}

TensorEntry &TensorFile::place_entry(std::string_view operation, std::string_view name, DType dtype, std::vector<size_t> const &dims,
                                     uint32_t owning_rank) {
    detail::check_entry_name(_path, operation, name);

    auto const same = std::ranges::find_if(
        _entries, [&](TensorEntry const &entry) { return entry.get_name() == name && entry.owning_rank == owning_rank; });
    if (same != _entries.end() && static_cast<DType>(same->dtype) == dtype && same->rank == dims.size() &&
        std::equal(dims.begin(), dims.end(), same->dims)) {
        return *same;
    }

    TensorEntry entry;
    entry.init();
    entry.set_name(name);
    entry.dtype       = static_cast<uint8_t>(dtype);
    entry.rank        = static_cast<uint8_t>(dims.size());
    uint64_t elements = 1;
    for (size_t d = 0; d < dims.size(); ++d) {
        entry.dims[d] = dims[d];
        elements *= dims[d];
    }
    entry.data_offset = align_up(_next_data_offset, ETN_DATA_ALIGNMENT);
    entry.data_size   = elements * dtype_size(dtype);
    entry.owning_rank = owning_rank;
    _next_data_offset = entry.data_offset + entry.data_size;

    if (same != _entries.end()) {
        // The old data region is left unused: only the record says where an entry's data is.
        *same = entry;
        return *same;
    }
    return _entries[add_entry(entry)];
}

TensorEntry const &TensorFile::find_entry(std::string_view name) const {
    auto it = _name_to_index.find(std::string(name));
    if (it == _name_to_index.end()) {
        EINSUMS_THROW_EXCEPTION(std::runtime_error, "TensorFile: tensor '{}' not found in '{}'", name, _path);
    }
    // An entry of rank above the maximum was written by a writer that did not refuse it, and its
    // extents past the maximum overwrote the fields after them; nothing in it can be trusted.
    if (TensorEntry const &entry = _entries[it->second]; entry.rank > ETN_MAX_RANK) {
        EINSUMS_THROW_EXCEPTION(std::runtime_error, "TensorFile: entry '{}' in '{}' is damaged: stored rank {} exceeds the maximum {}",
                                name, _path, entry.rank, ETN_MAX_RANK);
    }
    return _entries[it->second];
}

namespace {
/// An element of @p Bytes bytes, for walking a slab when only the element SIZE matters.
template <size_t Bytes>
struct RawElement {
    std::array<std::byte, Bytes> bytes;
};
} // namespace

void TensorFile::read_slab(std::string_view name, void *destination, std::vector<std::pair<size_t, size_t>> const &ranges) const {
    auto const &entry = find_entry(name);
    detail::check_slab_ranges(_path, "TensorFile::read_slab", entry, ranges);

    // The destination holds the slab densely in the entry's declared order, which the walk restates
    // in its stored order; an entry stored as declared is read straight into it.
    size_t const        size = dtype_size(static_cast<DType>(entry.dtype));
    std::vector<size_t> extents;
    for (auto const &[first, last] : ranges) {
        extents.push_back(last - first);
    }
    auto const  box   = detail::stored_box(entry, ranges, detail::dense_layout(extents, size));
    auto *const dst   = static_cast<char *>(destination);
    auto const  visit = [&](uint64_t file_off, size_t dst_off, size_t bytes) {
        detail::read_elements(reader(), file_off, dst, box.layout, dst_off / size, bytes / size);
    };

    switch (size) {
    case 4:
        detail::walk_slab<RawElement<4>>(entry.data_offset, box.dims, box.ranges, visit);
        break;
    case 8:
        detail::walk_slab<RawElement<8>>(entry.data_offset, box.dims, box.ranges, visit);
        break;
    case 16:
        detail::walk_slab<RawElement<16>>(entry.data_offset, box.dims, box.ranges, visit);
        break;
    default:
        EINSUMS_THROW_EXCEPTION(std::runtime_error,
                                "TensorFile::read_slab: entry '{}' in '{}' has elements of {} bytes, which is not a size "
                                "this reads",
                                name, _path, size);
    }
}

void TensorFile::write_slab(std::string_view name, void const *source, std::vector<std::pair<size_t, size_t>> const &ranges) {
    auto const &entry = find_entry(name);
    detail::check_slab_ranges(_path, "TensorFile::write_slab", entry, ranges);

    size_t const        size = dtype_size(static_cast<DType>(entry.dtype));
    std::vector<size_t> extents;
    for (auto const &[first, last] : ranges) {
        extents.push_back(last - first);
    }
    auto const        box   = detail::stored_box(entry, ranges, detail::dense_layout(extents, size));
    auto const *const src   = static_cast<char const *>(source);
    auto const        visit = [&](uint64_t file_off, size_t src_off, size_t bytes) {
        detail::write_elements(writer(), file_off, src, box.layout, src_off / size, bytes / size);
    };

    switch (size) {
    case 4:
        detail::walk_slab<RawElement<4>>(entry.data_offset, box.dims, box.ranges, visit);
        break;
    case 8:
        detail::walk_slab<RawElement<8>>(entry.data_offset, box.dims, box.ranges, visit);
        break;
    case 16:
        detail::walk_slab<RawElement<16>>(entry.data_offset, box.dims, box.ranges, visit);
        break;
    default:
        EINSUMS_THROW_EXCEPTION(std::runtime_error,
                                "TensorFile::write_slab: entry '{}' in '{}' has elements of {} bytes, which is not a size "
                                "this writes",
                                name, _path, size);
    }
}

bool TensorFile::contains(std::string_view name) const {
    return _name_to_index.count(std::string(name)) > 0;
}

std::vector<size_t> TensorFile::dims(std::string_view name) const {
    auto const         &entry = find_entry(name);
    std::vector<size_t> result(entry.rank);
    for (size_t d = 0; d < entry.rank; d++)
        result[d] = entry.dims[d];
    return result;
}

DType TensorFile::dtype(std::string_view name) const {
    return static_cast<DType>(find_entry(name).dtype);
}

std::vector<std::string> TensorFile::tensor_names() const {
    std::vector<std::string> names;
    names.reserve(_entries.size());
    for (auto const &e : _entries)
        names.push_back(e.get_name());
    return names;
}

void TensorFile::flush() {
    if (_fd >= 0 && _mode != Mode::Read) {
        write_metadata();
        detail::sync_file(_fd);
    }
}

EINSUMS_NAMESPACE_END(tensor_io)
