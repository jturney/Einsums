//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/Logging.hpp>
#include <Einsums/TensorIO/DistributedTensorFile.hpp>

// Always use POSIX I/O for file operations. MPI is only used for offset
// coordination (Exscan, Allreduce, Gather). This avoids MPI-IO reliability
// issues on local filesystems (especially macOS).
#include "PosixFileCompat.hpp"

#if defined(EINSUMS_HAVE_MPI)
#    include <mpi.h>
#endif

#include <cerrno>
#include <cstring>

namespace einsums::tensor_io {

DistributedTensorFile::DistributedTensorFile(std::string path, Mode mode)
    : _path(std::move(path)), _mode(mode), _my_rank(comm::world_rank()), _num_ranks(comm::world_size()) {

    // Rank 0 creates/truncates the file; all ranks then open it.
#if defined(EINSUMS_HAVE_MPI)
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    if (_mode == Mode::Write) {
        if (_my_rank == 0) {
            _fd = detail::open_file(_path, detail::OpenMode::WriteTruncate);
        }
#if defined(EINSUMS_HAVE_MPI)
        MPI_Barrier(MPI_COMM_WORLD); // Wait for rank 0 to create the file
#endif
        if (_my_rank != 0) {
            _fd = detail::open_file(_path, detail::OpenMode::ReadWrite);
        }
    } else {
        _fd = detail::open_file(_path, detail::OpenMode::Read);
    }

    if (_fd < 0) {
        EINSUMS_THROW_EXCEPTION(std::runtime_error, "DistributedTensorFile: cannot open '{}': {}", _path, std::strerror(errno));
    }

    if (_mode == Mode::Write) {
        _header.init();
        _header.num_ranks = static_cast<uint32_t>(_num_ranks);
        _next_data_offset = _header.data_offset;
        if (_my_rank == 0) {
            write_at(0, &_header, sizeof(FileHeader));
        }
#if defined(EINSUMS_HAVE_MPI)
        MPI_Barrier(MPI_COMM_WORLD);
#endif
    } else {
        read_metadata();
    }
}

// NOLINTNEXTLINE(bugprone-exception-escape)
DistributedTensorFile::~DistributedTensorFile() {
    if (!_closed) {
        close();
    }
}

void DistributedTensorFile::close() {
    if (_closed)
        return;

    if (_mode == Mode::Write) {
        write_metadata();
    }

    if (_fd >= 0) {
        detail::close_file(_fd);
        _fd = -1;
    }

    _closed = true;
    EINSUMS_LOG_INFO("DistributedTensorFile: closed '{}' ({} entries, {} ranks)", _path, _entries.size(), _num_ranks);
}

void DistributedTensorFile::write_at(uint64_t offset, void const *data, size_t bytes) {
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
            EINSUMS_THROW_EXCEPTION(std::runtime_error, "DistributedTensorFile: pwrite failed at offset {}: {}", offset,
                                    std::strerror(errno));
        }
        total += static_cast<size_t>(written);
    }
}

void DistributedTensorFile::read_at(uint64_t offset, void *data, size_t bytes) const {
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
            EINSUMS_THROW_EXCEPTION(std::runtime_error, "DistributedTensorFile: pread failed at offset {}: {}", offset,
                                    std::strerror(errno));
        }
        if (nread == 0)
            EINSUMS_THROW_EXCEPTION(std::runtime_error, "DistributedTensorFile: unexpected EOF at offset {}", offset + total);
        total += static_cast<size_t>(nread);
    }
}

uint64_t DistributedTensorFile::compute_distributed_offset(size_t local_bytes) {
    uint64_t const aligned_bytes = align_up(local_bytes, ETN_DATA_ALIGNMENT);
    uint64_t       my_offset     = 0;

#if defined(EINSUMS_HAVE_MPI)
    // Exclusive prefix sum: rank r gets offset = sum of all ranks < r
    uint64_t scan_result = 0;
    MPI_Exscan(&aligned_bytes, &scan_result, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
    if (my_rank_ == 0)
        scan_result = 0;

    my_offset = align_up(next_data_offset_, ETN_DATA_ALIGNMENT) + scan_result;

    // All ranks advance next_data_offset_ by the total across all ranks
    uint64_t total_bytes = 0;
    MPI_Allreduce(&aligned_bytes, &total_bytes, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
    next_data_offset_ = align_up(next_data_offset_, ETN_DATA_ALIGNMENT) + total_bytes;
#else
    my_offset                  = align_up(_next_data_offset, ETN_DATA_ALIGNMENT);
    _next_data_offset          = my_offset + aligned_bytes;
#endif

    return my_offset;
}

void DistributedTensorFile::write_metadata() {
#if defined(EINSUMS_HAVE_MPI)
    // Gather all entries from all ranks to rank 0
    int              local_count = static_cast<int>(entries_.size());
    std::vector<int> all_counts(num_ranks_);
    MPI_Gather(&local_count, 1, MPI_INT, all_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<TensorEntry> all_entries;
    if (my_rank_ == 0) {
        // Compute displacements for Gatherv
        int              total = 0;
        std::vector<int> byte_counts(num_ranks_);
        std::vector<int> displs(num_ranks_);
        for (int r = 0; r < num_ranks_; r++) {
            byte_counts[r] = all_counts[r] * static_cast<int>(sizeof(TensorEntry));
            displs[r]      = total;
            total += byte_counts[r];
        }
        all_entries.resize(total / static_cast<int>(sizeof(TensorEntry)));

        MPI_Gatherv(entries_.data(), local_count * static_cast<int>(sizeof(TensorEntry)), MPI_BYTE, all_entries.data(), byte_counts.data(),
                    displs.data(), MPI_BYTE, 0, MPI_COMM_WORLD);

        // Deduplicate replicated entries (all ranks added them)
        std::vector<TensorEntry>              unique_entries;
        std::unordered_map<std::string, bool> seen_replicated;
        for (auto const &e : all_entries) {
            std::string n = e.get_name();
            if (e.owning_rank == ETN_ALL_RANKS) {
                if (seen_replicated.count(n))
                    continue;
                seen_replicated[n] = true;
            }
            unique_entries.push_back(e);
        }

        header_.num_tensors        = static_cast<uint32_t>(unique_entries.size());
        header_.entry_table_offset = align_up(next_data_offset_, ETN_DATA_ALIGNMENT);
        header_.total_size         = header_.entry_table_offset + unique_entries.size() * sizeof(TensorEntry);

        write_at(0, &header_, sizeof(FileHeader));
        if (!unique_entries.empty()) {
            write_at(header_.entry_table_offset, unique_entries.data(), unique_entries.size() * sizeof(TensorEntry));
        }
        detail::truncate_file(fd_, static_cast<detail::file_offset_t>(header_.total_size));
    } else {
        MPI_Gatherv(entries_.data(), local_count * static_cast<int>(sizeof(TensorEntry)), MPI_BYTE, nullptr, nullptr, nullptr, MPI_BYTE, 0,
                    MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
#else
    _header.num_tensors        = static_cast<uint32_t>(_entries.size());
    _header.entry_table_offset = align_up(_next_data_offset, ETN_DATA_ALIGNMENT);
    _header.total_size         = _header.entry_table_offset + _entries.size() * sizeof(TensorEntry);

    write_at(0, &_header, sizeof(FileHeader));
    if (!_entries.empty()) {
        write_at(_header.entry_table_offset, _entries.data(), _entries.size() * sizeof(TensorEntry));
    }
    detail::truncate_file(_fd, static_cast<detail::file_offset_t>(_header.total_size));
#endif
}

void DistributedTensorFile::read_metadata() {
    read_at(0, &_header, sizeof(FileHeader));
    if (!_header.is_valid()) {
        EINSUMS_THROW_EXCEPTION(std::runtime_error, "DistributedTensorFile: '{}' is not a valid .etn file", _path);
    }

    _entries.resize(_header.num_tensors);
    if (_header.num_tensors > 0) {
        read_at(_header.entry_table_offset, _entries.data(), _header.num_tensors * sizeof(TensorEntry));
    }

    _name_to_index.clear();
    for (size_t i = 0; i < _entries.size(); i++) {
        _name_to_index[_entries[i].get_name()] = i;
    }

    _next_data_offset = _header.data_offset;
    for (auto const &e : _entries) {
        uint64_t const end = e.data_offset + e.data_size;
        if (end > _next_data_offset)
            _next_data_offset = end;
    }
}

bool DistributedTensorFile::contains(std::string_view name) const {
    return _name_to_index.count(std::string(name)) > 0;
}

std::vector<std::string> DistributedTensorFile::tensor_names() const {
    std::vector<std::string> names;
    names.reserve(_entries.size());
    for (auto const &e : _entries)
        names.push_back(e.get_name());
    return names;
}

} // namespace einsums::tensor_io
