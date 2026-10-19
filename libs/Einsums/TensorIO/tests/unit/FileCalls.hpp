//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/// @file FileCalls.hpp
/// @brief Stop or record the calls a TensorFile makes that change a file, to test what a crash leaves.

#include <Einsums/Config/Namespace.hpp>
#include <Einsums/TensorIO/Format.hpp>
#include <Einsums/TensorIO/TensorFile.hpp>

#include <cstddef>
#include <fstream>
#include <functional>
#include <iterator>
#include <map>
#include <string>
#include <utility>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(tensor_io::file_calls)

/// Installs a file-call hook for its lifetime.
class FileCallHook {
  public:
    explicit FileCallHook(std::function<bool(detail::FileCall const &)> hook) { detail::set_file_call_hook(std::move(hook)); }
    ~FileCallHook() { detail::set_file_call_hook({}); }

    FileCallHook(FileCallHook const &)            = delete;
    FileCallHook &operator=(FileCallHook const &) = delete;
    FileCallHook(FileCallHook &&)                 = delete;
    FileCallHook &operator=(FileCallHook &&)      = delete;
};

/// The bytes of the file at @p path.
inline std::vector<char> file_bytes(std::string const &path) {
    std::ifstream in(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

/// Make the file at @p path hold exactly @p bytes.
inline void put_file_bytes(std::string const &path, std::vector<char> const &bytes) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

/// How many calls that change a file @p operation makes.
inline size_t count_file_calls(std::function<void()> const &operation) {
    size_t             calls = 0;
    FileCallHook const hook([&](detail::FileCall const &) {
        ++calls;
        return true;
    });
    operation();
    return calls;
}

/// Run @p operation with its calls from the @p n-th on (counting from zero) dropped, as a process
/// killed at that call leaves the file.
inline void run_killed_at(size_t n, std::function<void()> const &operation) {
    size_t             calls = 0;
    FileCallHook const hook([&](detail::FileCall const &) { return calls++ < n; });
    operation();
}

/// What a file holds, entry by entry: its element type, extents and bytes, in declared order.
struct EntryState {
    DType               dtype{};
    std::vector<size_t> dims;
    std::vector<char>   bytes;

    bool operator==(EntryState const &) const = default;
};
using FileState = std::map<std::string, EntryState>;

/// Read every entry of @p path. Throws when the file cannot be opened or an entry read.
inline FileState file_state(std::string const &path) {
    TensorFile const file(path, TensorFile::Mode::Read);
    FileState        state;
    for (auto const &name : file.tensor_names()) {
        EntryState entry{.dtype = file.dtype(name), .dims = file.dims(name), .bytes = {}};
        size_t     elements = 1;
        for (size_t const d : entry.dims) {
            elements *= d;
        }
        entry.bytes.resize(elements * dtype_size(entry.dtype));
        std::vector<std::pair<size_t, size_t>> ranges;
        ranges.reserve(entry.dims.size());
        for (size_t const d : entry.dims) {
            ranges.emplace_back(0, d);
        }
        if (elements != 0) {
            file.read_slab(name, entry.bytes.data(), ranges);
        }
        state.emplace(name, std::move(entry));
    }
    return state;
}

EINSUMS_NAMESPACE_END(tensor_io::file_calls)
