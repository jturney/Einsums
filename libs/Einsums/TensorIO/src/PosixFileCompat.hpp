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

#include <cstddef>
#include <cstdint>
#include <string>
#include <system_error>
#include <utility>

#if defined(EINSUMS_WINDOWS)
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <cerrno>
#    include <cstring>
#    include <fcntl.h>
#    include <filesystem>
#    include <io.h>
#    include <sys/stat.h>
#    include <vector>
#    include <windows.h>
#else
#    include <cerrno>
#    include <cstdio>
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

/// errno for a Win32 error, so the callers' strerror(errno) reads the same on both platforms.
inline int errno_from_win32(DWORD error) {
    switch (error) {
    case ERROR_FILE_NOT_FOUND:
    case ERROR_PATH_NOT_FOUND:
        return ENOENT;
    case ERROR_ACCESS_DENIED:
    case ERROR_SHARING_VIOLATION:
    case ERROR_LOCK_VIOLATION:
        return EACCES;
    case ERROR_FILE_EXISTS:
    case ERROR_ALREADY_EXISTS:
        return EEXIST;
    case ERROR_DISK_FULL:
    case ERROR_HANDLE_DISK_FULL:
        return ENOSPC;
    case ERROR_TOO_MANY_OPEN_FILES:
        return EMFILE;
    default:
        return EIO;
    }
}

/// Open @p path as POSIX does: every handle shares read, write and delete with every other, so a
/// file another handle holds open can still be replaced under its path, as TensorFile::compact does.
/// The C runtime's _sopen_s cannot ask for delete sharing, so the handle comes from CreateFileW.
inline int open_file(std::string const &path, OpenMode mode) {
    DWORD access      = GENERIC_READ | GENERIC_WRITE;
    DWORD disposition = OPEN_ALWAYS;
    int   crt_flags   = _O_BINARY;
    switch (mode) {
    case OpenMode::Read:
        access      = GENERIC_READ;
        disposition = OPEN_EXISTING;
        crt_flags |= _O_RDONLY;
        break;
    case OpenMode::WriteTruncate:
        disposition = CREATE_ALWAYS;
        break;
    case OpenMode::ReadWrite:
        break;
    }
    HANDLE const handle = CreateFileW(std::filesystem::path(path).c_str(), access, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                                      nullptr, disposition, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (handle == INVALID_HANDLE_VALUE) {
        errno = errno_from_win32(GetLastError());
        return -1;
    }
    int const fd = _open_osfhandle(reinterpret_cast<intptr_t>(handle), crt_flags);
    if (fd < 0) {
        CloseHandle(handle);
        errno = EMFILE;
    }
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

/// The volume and file id of an open handle. ReFS ids are 128 bits; the two halves fold into one,
/// which on NTFS, whose ids fit the low half, is the id itself.
inline std::pair<std::uint64_t, std::uint64_t> handle_identity(HANDLE handle) {
    FILE_ID_INFO info{};
    if (GetFileInformationByHandleEx(handle, FileIdInfo, &info, sizeof(info))) {
        std::uint64_t halves[2]{};
        static_assert(sizeof(halves) == sizeof(info.FileId.Identifier));
        std::memcpy(halves, info.FileId.Identifier, sizeof(halves));
        return {info.VolumeSerialNumber, halves[0] ^ halves[1]};
    }
    BY_HANDLE_FILE_INFORMATION legacy{};
    if (GetFileInformationByHandle(handle, &legacy)) {
        return {legacy.dwVolumeSerialNumber, (static_cast<std::uint64_t>(legacy.nFileIndexHigh) << 32) | legacy.nFileIndexLow};
    }
    return {0, 0};
}

/// What tells two opens of one path apart: a file replaced under the path is another file.
inline std::pair<std::uint64_t, std::uint64_t> file_identity(int fd) {
    auto handle = reinterpret_cast<HANDLE>(_get_osfhandle(fd));
    if (handle == INVALID_HANDLE_VALUE) {
        return {0, 0};
    }
    return handle_identity(handle);
}

inline std::pair<std::uint64_t, std::uint64_t> path_identity(std::string const &path) {
    // No access is asked for, only the attributes, so no handle already open on the file blocks the
    // query. Backup semantics opens a directory as well, as stat does.
    HANDLE const handle = CreateFileW(std::filesystem::path(path).c_str(), 0, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                                      nullptr, OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS, nullptr);
    if (handle == INVALID_HANDLE_VALUE) {
        return {0, 0};
    }
    auto const identity = handle_identity(handle);
    CloseHandle(handle);
    return identity;
}

/// Put @p from at @p to, replacing any file there even while other handles hold it open, as POSIX
/// rename does: those handles keep the old file, and the path names the new one at once.
///
/// A rename with POSIX semantics (Windows 10 1709 and later, on NTFS) replaces a target other handles
/// hold open, provided they share delete, which every open_file handle does. MoveFileExW makes no such
/// promise, so it is only the fallback for a file system without POSIX semantics, where replacing an
/// open target may then be refused.
inline std::error_code replace_file(std::string const &from, std::string const &to) {
    std::wstring const target = std::filesystem::absolute(std::filesystem::path(to)).wstring();
    HANDLE const       handle = CreateFileW(std::filesystem::path(from).c_str(), DELETE | SYNCHRONIZE,
                                            FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING, 0, nullptr);
    if (handle == INVALID_HANDLE_VALUE) {
        return {static_cast<int>(GetLastError()), std::system_category()};
    }
    // FILE_RENAME_INFO ends in the name, so it is sized for this one.
    std::vector<std::byte> buffer(sizeof(FILE_RENAME_INFO) + target.size() * sizeof(wchar_t));
    auto                  *info = reinterpret_cast<FILE_RENAME_INFO *>(buffer.data());
    info->Flags                 = FILE_RENAME_FLAG_REPLACE_IF_EXISTS | FILE_RENAME_FLAG_POSIX_SEMANTICS;
    info->RootDirectory         = nullptr;
    info->FileNameLength        = static_cast<DWORD>(target.size() * sizeof(wchar_t));
    std::memcpy(info->FileName, target.c_str(), info->FileNameLength);
    bool const  renamed = SetFileInformationByHandle(handle, FileRenameInfoEx, info, static_cast<DWORD>(buffer.size())) != 0;
    DWORD const error   = renamed ? ERROR_SUCCESS : GetLastError();
    CloseHandle(handle);
    if (renamed) {
        return {};
    }
    if (error != ERROR_INVALID_PARAMETER && error != ERROR_NOT_SUPPORTED && error != ERROR_INVALID_FUNCTION) {
        return {static_cast<int>(error), std::system_category()};
    }
    if (MoveFileExW(std::filesystem::path(from).c_str(), target.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        return {};
    }
    return {static_cast<int>(GetLastError()), std::system_category()};
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

/// Put @p from at @p to, replacing any file there; handles open on the old file keep it.
inline std::error_code replace_file(std::string const &from, std::string const &to) {
    if (::rename(from.c_str(), to.c_str()) != 0) {
        return {errno, std::generic_category()};
    }
    return {};
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
