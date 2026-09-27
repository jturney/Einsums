//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/TensorIO/TensorFile.hpp>

#include <cerrno>
#include <cstring>
#include <limits>
#include <stdexcept>

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

void TensorFile::write_at(uint64_t offset, void const *data, size_t bytes) {
    if (bytes == 0)
        return;
    auto  *ptr   = static_cast<char const *>(data);
    size_t total = 0;
    while (total < bytes) {
        detail::io_result_t const written =
            detail::pwrite_file(_fd, ptr + total, bytes - total, static_cast<detail::file_offset_t>(offset + total));
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
            detail::pread_file(_fd, ptr + total, bytes - total, static_cast<detail::file_offset_t>(offset + total));
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

void TensorFile::write_metadata() {
    // Update header
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

    // Build name index
    _name_to_index.clear();
    for (size_t i = 0; i < _entries.size(); i++) {
        _name_to_index[_entries[i].get_name()] = i;
    }

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

TensorEntry const &TensorFile::find_entry(std::string_view name) const {
    auto it = _name_to_index.find(std::string(name));
    if (it == _name_to_index.end()) {
        EINSUMS_THROW_EXCEPTION(std::runtime_error, "TensorFile: tensor '{}' not found in '{}'", name, _path);
    }
    return _entries[it->second];
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
