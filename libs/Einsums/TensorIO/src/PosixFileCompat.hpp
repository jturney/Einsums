//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/// @file PosixFileCompat.hpp
/// @brief Positional file I/O for TensorIO: POSIX pread/pwrite on Unix,
///        offset-explicit ReadFile/WriteFile on Windows.
///
/// The Windows implementations preserve the property TensorIO's slab reads
/// and writes rely on: positional I/O with NO shared file-position state, so
/// concurrent slab operations on one descriptor never race on a seek pointer.
/// (An OVERLAPPED offset on a synchronous handle is exactly Win32's pread.)
/// Files are always opened binary on Windows - text-mode newline translation
/// would corrupt tensor payloads.

#include <Einsums/Config.hpp>

#include <cstdint>
#include <string>

#if defined(EINSUMS_WINDOWS)
#    include <fcntl.h>
#    include <io.h>
#    include <sys/stat.h>
#    include <windows.h>
#else
#    include <fcntl.h>
#    include <sys/stat.h>
#    include <unistd.h>
#endif

namespace einsums::tensor_io::detail {

/// Open modes mirroring TensorFile's needs.
enum class OpenMode : std::uint8_t { Read, WriteTruncate, ReadWrite };

#if defined(EINSUMS_WINDOWS)

using file_offset_t = std::int64_t;
using io_result_t   = std::int64_t;

inline int open_file(std::string const &path, OpenMode mode) {
    int flags = _O_BINARY;
    switch (mode) {
    case OpenMode::Read:
        flags |= _O_RDONLY;
        break;
    case OpenMode::WriteTruncate:
        flags |= _O_RDWR | _O_CREAT | _O_TRUNC;
        break;
    case OpenMode::ReadWrite:
        flags |= _O_RDWR | _O_CREAT;
        break;
    }
    int fd = -1;
    (void)_sopen_s(&fd, path.c_str(), flags, _SH_DENYNO, _S_IREAD | _S_IWRITE);
    return fd;
}

inline void close_file(int fd) {
    _close(fd);
}

inline io_result_t pread_file(int fd, void *buf, std::size_t bytes, file_offset_t offset) {
    auto handle = reinterpret_cast<HANDLE>(_get_osfhandle(fd));
    if (handle == INVALID_HANDLE_VALUE) {
        return -1;
    }
    OVERLAPPED ov{};
    ov.Offset     = static_cast<DWORD>(offset & 0xFFFFFFFFu);
    ov.OffsetHigh = static_cast<DWORD>(static_cast<std::uint64_t>(offset) >> 32);
    DWORD got     = 0;
    if (!ReadFile(handle, buf, static_cast<DWORD>(bytes), &got, &ov)) {
        // Reading at/after EOF reports ERROR_HANDLE_EOF; POSIX pread returns 0.
        return GetLastError() == ERROR_HANDLE_EOF ? 0 : -1;
    }
    return static_cast<io_result_t>(got);
}

inline io_result_t pwrite_file(int fd, void const *buf, std::size_t bytes, file_offset_t offset) {
    auto handle = reinterpret_cast<HANDLE>(_get_osfhandle(fd));
    if (handle == INVALID_HANDLE_VALUE) {
        return -1;
    }
    OVERLAPPED ov{};
    ov.Offset     = static_cast<DWORD>(offset & 0xFFFFFFFFu);
    ov.OffsetHigh = static_cast<DWORD>(static_cast<std::uint64_t>(offset) >> 32);
    DWORD put     = 0;
    if (!WriteFile(handle, buf, static_cast<DWORD>(bytes), &put, &ov)) {
        return -1;
    }
    return static_cast<io_result_t>(put);
}

inline int truncate_file(int fd, file_offset_t length) {
    return _chsize_s(fd, length) == 0 ? 0 : -1;
}

#else

using file_offset_t = off_t;
using io_result_t   = ssize_t;

inline int open_file(std::string const &path, OpenMode mode) {
    int          flags = 0;
    mode_t const perms = S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH;
    switch (mode) {
    case OpenMode::Read:
        flags = O_RDONLY;
        break;
    case OpenMode::WriteTruncate:
        flags = O_RDWR | O_CREAT | O_TRUNC;
        break;
    case OpenMode::ReadWrite:
        flags = O_RDWR | O_CREAT;
        break;
    }
    return ::open(path.c_str(), flags, perms);
}

inline void close_file(int fd) {
    ::close(fd);
}

inline io_result_t pread_file(int fd, void *buf, std::size_t bytes, file_offset_t offset) {
    return ::pread(fd, buf, bytes, offset);
}

inline io_result_t pwrite_file(int fd, void const *buf, std::size_t bytes, file_offset_t offset) {
    return ::pwrite(fd, buf, bytes, offset);
}

inline int truncate_file(int fd, file_offset_t length) {
    return ::ftruncate(fd, length);
}

#endif

} // namespace einsums::tensor_io::detail
