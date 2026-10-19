//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/// @file PosixFileCompat.hpp
/// @brief Positional file I/O for TensorIO: POSIX pread/pwrite on Unix,
///        offset-explicit ReadFile/WriteFile on Windows.
///
/// No shared file position on either platform, so concurrent slab I/O on one descriptor cannot
/// race (on Windows, an OVERLAPPED offset on a synchronous handle). Windows files open binary.

#include <Einsums/Config.hpp>

#include <Einsums/Config/Namespace.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

#if defined(EINSUMS_WINDOWS)
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <fcntl.h>
#    include <io.h>
#    include <sys/stat.h>
#    include <windows.h>
#else
#    include <cerrno>
#    include <fcntl.h>
#    include <sys/file.h>
#    include <sys/stat.h>
#    include <unistd.h>
#endif

EINSUMS_NAMESPACE_BEGIN(tensor_io::detail)

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

inline int sync_file(int fd) {
    return _commit(fd);
}

/// Hold an exclusive lock on @p fd's file against other handles' commits. A byte far past any data
/// is locked, as a Windows byte-range lock also stops other handles writing the bytes it covers.
inline int lock_file(int fd) {
    auto handle = reinterpret_cast<HANDLE>(_get_osfhandle(fd));
    if (handle == INVALID_HANDLE_VALUE) {
        return -1;
    }
    OVERLAPPED ov{};
    ov.Offset     = 0;
    ov.OffsetHigh = 0x7FFFFFFFu;
    return LockFileEx(handle, LOCKFILE_EXCLUSIVE_LOCK, 0, 1, 0, &ov) ? 0 : -1;
}

inline int unlock_file(int fd) {
    auto handle = reinterpret_cast<HANDLE>(_get_osfhandle(fd));
    if (handle == INVALID_HANDLE_VALUE) {
        return -1;
    }
    OVERLAPPED ov{};
    ov.Offset     = 0;
    ov.OffsetHigh = 0x7FFFFFFFu;
    return UnlockFileEx(handle, 0, 1, 0, &ov) ? 0 : -1;
}

/// What tells two opens of one path apart: a file replaced under the path is another file. Windows
/// refuses to replace a file a handle holds open, so there the answer is always the same.
inline std::pair<std::uint64_t, std::uint64_t> file_identity(int /*fd*/) {
    return {0, 0};
}

inline std::pair<std::uint64_t, std::uint64_t> path_identity(std::string const & /*path*/) {
    return {0, 0};
}

/// Make a rename in @p directory survive a power loss. Windows records it with the file.
inline int sync_directory(std::string const & /*directory*/) {
    return 0;
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

inline int sync_file(int fd) {
#    if defined(__APPLE__)
    // fsync on macOS hands the data to the drive without flushing the drive's cache.
    if (::fcntl(fd, F_FULLFSYNC) == 0) {
        return 0;
    }
#    endif
    return ::fsync(fd);
}

/// Hold an exclusive lock on @p fd's file against other handles' commits.
inline int lock_file(int fd) {
    int result = 0;
    do {
        result = ::flock(fd, LOCK_EX);
    } while (result != 0 && errno == EINTR);
    return result;
}

inline int unlock_file(int fd) {
    return ::flock(fd, LOCK_UN);
}

/// What tells two opens of one path apart: a file replaced under the path is another file.
inline std::pair<std::uint64_t, std::uint64_t> file_identity(int fd) {
    struct stat st{};
    if (::fstat(fd, &st) != 0) {
        return {0, 0};
    }
    return {static_cast<std::uint64_t>(st.st_dev), static_cast<std::uint64_t>(st.st_ino)};
}

inline std::pair<std::uint64_t, std::uint64_t> path_identity(std::string const &path) {
    struct stat st{};
    if (::stat(path.c_str(), &st) != 0) {
        return {0, 0};
    }
    return {static_cast<std::uint64_t>(st.st_dev), static_cast<std::uint64_t>(st.st_ino)};
}

/// Make a rename in @p directory survive a power loss.
inline int sync_directory(std::string const &directory) {
    int const fd = ::open(directory.c_str(), O_RDONLY);
    if (fd < 0) {
        return -1;
    }
    int const result = ::fsync(fd);
    ::close(fd);
    return result;
}

#endif

EINSUMS_NAMESPACE_END(tensor_io::detail)
