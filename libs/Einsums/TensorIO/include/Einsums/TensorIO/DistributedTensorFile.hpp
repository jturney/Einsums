//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/// @file DistributedTensorFile.hpp
/// @brief Tensor file for distributed parallel I/O.
///
/// File operations use POSIX I/O on every rank; MPI handles only the
/// offset coordination, not the data transfer.
///
/// All ranks open the same file. Each rank writes its local tensor partitions
/// at computed offsets. The file is a standard .etn binary file, readable
/// by the serial TensorFile without MPI.
///
/// @code
/// using namespace einsums::tensor_io;
///
/// // All ranks collectively write
/// DistributedTensorFile out("checkpoint.etn", DistributedTensorFile::Mode::Write);
/// out.write("ERI", eri);           // Replicated tensor (rank 0 writes)
/// out.write_local("C", C_local);   // Each rank writes its partition
/// out.close();
///
/// // All ranks collectively read
/// DistributedTensorFile in("checkpoint.etn", DistributedTensorFile::Mode::Read);
/// in.read("ERI", eri);             // All ranks read the same data
/// in.read_local("C", C_local);     // Each rank reads its partition
/// @endcode

#include <Einsums/Config.hpp>

#include <Einsums/Comm/Collectives.hpp>
#include <Einsums/Comm/Runtime.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorIO/Format.hpp>

#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace einsums::tensor_io {

class EINSUMS_EXPORT DistributedTensorFile {
  public:
    enum class Mode : std::uint8_t { Read, Write };

    /// All ranks collectively open the file.
    explicit DistributedTensorFile(std::string path, Mode mode);

    /// Flushes and closes (collective).
    ~DistributedTensorFile();

    // Non-copyable
    DistributedTensorFile(DistributedTensorFile const &)            = delete;
    DistributedTensorFile &operator=(DistributedTensorFile const &) = delete;

    // ── Write (collective) ───────────────────────────────────────────────

    /// Write a replicated tensor. Rank 0 writes; others skip.
    template <typename T, size_t Rank>
    void write(std::string_view name, Tensor<T, Rank> const &tensor);

    /// Write each rank's local partition. All ranks participate.
    /// Each rank's data is stored at a unique offset. The entry table
    /// records one TensorEntry per rank with owning_rank set.
    template <typename T, size_t Rank>
    void write_local(std::string_view name, Tensor<T, Rank> const &tensor);

    // ── Read (collective) ────────────────────────────────────────────────

    /// Read a replicated tensor. All ranks read the same data.
    template <typename T, size_t Rank>
    void read(std::string_view name, Tensor<T, Rank> &tensor);

    /// Read this rank's local partition.
    template <typename T, size_t Rank>
    void read_local(std::string_view name, Tensor<T, Rank> &tensor);

    // ── Query ────────────────────────────────────────────────────────────

    [[nodiscard]] bool                     contains(std::string_view name) const;
    [[nodiscard]] std::vector<std::string> tensor_names() const;
    [[nodiscard]] size_t                   num_tensors() const { return _entries.size(); }

    /// Close the file (collective). Called automatically by destructor.
    void close();

  private:
    std::string                             _path;
    Mode                                    _mode;
    int                                     _my_rank;
    int                                     _num_ranks;
    FileHeader                              _header;
    std::vector<TensorEntry>                _entries;
    std::unordered_map<std::string, size_t> _name_to_index;
    uint64_t                                _next_data_offset;
    bool                                    _closed{false};

    int _fd{-1}; // POSIX file descriptor (used on all platforms)

    void write_at(uint64_t offset, void const *data, size_t bytes);
    void read_at(uint64_t offset, void *data, size_t bytes) const;
    void write_metadata();
    void read_metadata();

    /// Compute this rank's data offset via an MPI_Exscan exclusive prefix sum.
    uint64_t compute_distributed_offset(size_t local_bytes);
};

// ═══════════════════════════════════════════════════════════════════════════════
// Template implementations
// ═══════════════════════════════════════════════════════════════════════════════

template <typename T, size_t Rank>
void DistributedTensorFile::write(std::string_view name, Tensor<T, Rank> const &tensor) {
    // Replicated: only rank 0 writes the data
    size_t data_bytes = tensor.size() * sizeof(T);

    TensorEntry entry;
    entry.init();
    entry.set_name(name);
    entry.dtype = static_cast<uint8_t>(dtype_for<T>());
    entry.rank  = static_cast<uint8_t>(Rank);
    for (size_t d = 0; d < Rank; d++)
        entry.dims[d] = tensor.dim(d);
    entry.data_size   = data_bytes;
    entry.owning_rank = ETN_ALL_RANKS;

    // All ranks agree on the offset
    entry.data_offset = align_up(_next_data_offset, ETN_DATA_ALIGNMENT);
    _next_data_offset = entry.data_offset + data_bytes;

    // Only rank 0 writes the data
    if (_my_rank == 0) {
        write_at(entry.data_offset, tensor.data(), data_bytes);
    }

    _entries.push_back(entry);
    _name_to_index[std::string(name)] = _entries.size() - 1;
}

template <typename T, size_t Rank>
void DistributedTensorFile::write_local(std::string_view name, Tensor<T, Rank> const &tensor) {
    size_t local_bytes = tensor.size() * sizeof(T);

    // Use prefix sum to compute each rank's offset
    uint64_t my_offset = compute_distributed_offset(local_bytes);

    TensorEntry entry;
    entry.init();
    entry.set_name(name);
    entry.dtype = static_cast<uint8_t>(dtype_for<T>());
    entry.rank  = static_cast<uint8_t>(Rank);
    for (size_t d = 0; d < Rank; d++)
        entry.dims[d] = tensor.dim(d);
    entry.data_offset = my_offset;
    entry.data_size   = local_bytes;
    entry.owning_rank = static_cast<uint32_t>(_my_rank);

    // Each rank writes its own data at its own offset
    write_at(my_offset, tensor.data(), local_bytes);

    _entries.push_back(entry);

    _header.flags |= 0x1;
    _header.num_ranks = static_cast<uint32_t>(_num_ranks);
}

template <typename T, size_t Rank>
void DistributedTensorFile::read(std::string_view name, Tensor<T, Rank> &tensor) {
    // Find the replicated entry (owning_rank == ETN_ALL_RANKS)
    std::string const name_str(name);
    for (auto const &entry : _entries) {
        if (entry.get_name() == name_str && entry.owning_rank == ETN_ALL_RANKS) {
            Dim<Rank> dims;
            for (size_t d = 0; d < Rank; d++)
                dims[d] = entry.dims[d];
            tensor.resize(dims);
            read_at(entry.data_offset, tensor.data(), entry.data_size);
            return;
        }
    }
    throw std::runtime_error("DistributedTensorFile: replicated tensor '" + std::string(name) + "' not found");
}

template <typename T, size_t Rank>
void DistributedTensorFile::read_local(std::string_view name, Tensor<T, Rank> &tensor) {
    // Find the entry for this specific rank
    std::string const name_str(name);
    for (auto const &entry : _entries) {
        if (entry.get_name() == name_str && std::cmp_equal(entry.owning_rank, _my_rank)) {
            Dim<Rank> dims;
            for (size_t d = 0; d < Rank; d++)
                dims[d] = entry.dims[d];
            tensor.resize(dims);
            read_at(entry.data_offset, tensor.data(), entry.data_size);
            return;
        }
    }
    throw std::runtime_error("DistributedTensorFile: local tensor '" + std::string(name) + "' for rank " + std::to_string(_my_rank) +
                             " not found");
}

} // namespace einsums::tensor_io
