//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Debugging/Backtrace.hpp>
#include <Einsums/Debugging/CrashHandler.hpp>

#include <cstdio>
#include <cstring>
#include <string>

#if defined(EINSUMS_WINDOWS)
#    include <Windows.h>
// DbgHelp.h documents Windows.h as a prerequisite, so this include cannot be sorted
// above it.
#    include <DbgHelp.h>
#endif

EINSUMS_NAMESPACE_BEGIN(util)

#if defined(EINSUMS_WINDOWS)

namespace {

/// Where @ref install_crash_handler was asked to drop the minidump, and the filter it
/// displaced. Plain globals rather than anything with a destructor: the handler runs
/// during teardown, after static destructors have begun, so it may only touch storage
/// that cannot have been destroyed out from under it.
char                         g_dump_directory[MAX_PATH] = {};
LPTOP_LEVEL_EXCEPTION_FILTER g_previous_filter          = nullptr;
bool                         g_installed                = false;

/// Write straight to the standard error HANDLE.
///
/// Not std::cerr and not stderr: by the time this runs the iostreams objects may have
/// been destroyed and the CRT may have closed its streams, and a crash report that
/// disappears because the reporting path itself was torn down is worse than useless.
/// The raw handle stays valid until the process does.
void write_stderr(char const *text) {
    HANDLE const h = GetStdHandle(STD_ERROR_HANDLE);
    if (h == nullptr || h == INVALID_HANDLE_VALUE) {
        return;
    }
    DWORD written = 0;
    WriteFile(h, text, static_cast<DWORD>(std::strlen(text)), &written, nullptr);
}

/// Heap corruption detected by the allocator. Spelled out rather than taken from
/// ntstatus.h, which redefines a long list of macros Windows.h has already defined
/// unless the translation unit is arranged around WIN32_NO_STATUS.
///
/// Worth naming despite never reaching the filter: this status is raised through a
/// fail-fast path that bypasses the unhandled exception filter outright. Seeing it
/// here at all would mean it arrived some other way, and knowing that is the point.
constexpr DWORD kStatusHeapCorruption = 0xC0000374UL;

/// Spelling for the exception codes worth recognizing on sight. Anything else is
/// reported as its raw code, which is still enough to look up.
char const *exception_name(DWORD code) {
    switch (code) {
    case EXCEPTION_ACCESS_VIOLATION:
        return "EXCEPTION_ACCESS_VIOLATION";
    case EXCEPTION_ARRAY_BOUNDS_EXCEEDED:
        return "EXCEPTION_ARRAY_BOUNDS_EXCEEDED";
    case EXCEPTION_DATATYPE_MISALIGNMENT:
        return "EXCEPTION_DATATYPE_MISALIGNMENT";
    case EXCEPTION_FLT_DIVIDE_BY_ZERO:
        return "EXCEPTION_FLT_DIVIDE_BY_ZERO";
    case EXCEPTION_ILLEGAL_INSTRUCTION:
        return "EXCEPTION_ILLEGAL_INSTRUCTION";
    case EXCEPTION_IN_PAGE_ERROR:
        return "EXCEPTION_IN_PAGE_ERROR";
    case EXCEPTION_INT_DIVIDE_BY_ZERO:
        return "EXCEPTION_INT_DIVIDE_BY_ZERO";
    case EXCEPTION_PRIV_INSTRUCTION:
        return "EXCEPTION_PRIV_INSTRUCTION";
    case EXCEPTION_STACK_OVERFLOW:
        return "EXCEPTION_STACK_OVERFLOW";
    case kStatusHeapCorruption:
        return "STATUS_HEAP_CORRUPTION";
    default:
        return "unrecognized exception";
    }
}

/// Report which kind of access faulted. The operation lives in the first entry of an
/// access violation's ExceptionInformation and the address in the second.
char const *access_violation_operation(ULONG_PTR operation) {
    switch (operation) {
    case 0:
        return "reading";
    case 1:
        return "writing";
    case 8:
        return "executing (DEP)";
    default:
        return "accessing";
    }
}

/// Dump the faulting process to a file. Returns whether the file was written, and
/// fills @p path_out with where it went.
bool write_minidump(EXCEPTION_POINTERS *pointers, char (&path_out)[MAX_PATH]) {
    if (g_dump_directory[0] != '\0') {
        std::snprintf(path_out, MAX_PATH, "%s\\einsums-crash-%lu.dmp", g_dump_directory, GetCurrentProcessId());
    } else {
        std::snprintf(path_out, MAX_PATH, "einsums-crash-%lu.dmp", GetCurrentProcessId());
    }

    HANDLE const file = CreateFileA(path_out, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) {
        return false;
    }

    MINIDUMP_EXCEPTION_INFORMATION info;
    info.ThreadId          = GetCurrentThreadId();
    info.ExceptionPointers = pointers;
    info.ClientPointers    = FALSE;

    // Every thread's stack plus the module list, which is the minimum that answers the
    // question a teardown crash actually poses: which image was executing. The memory
    // options pull in what the stacks point at without paying for a full heap dump.
    auto const type = static_cast<MINIDUMP_TYPE>(MiniDumpWithThreadInfo | MiniDumpWithIndirectlyReferencedMemory | MiniDumpScanMemory);

    BOOL const ok = MiniDumpWriteDump(GetCurrentProcess(), GetCurrentProcessId(), file, type, &info, nullptr, nullptr);
    CloseHandle(file);
    return ok != FALSE;
}

LONG WINAPI crash_filter(EXCEPTION_POINTERS *pointers) {
    // Re-entrancy guard. A fault raised while reporting a fault would otherwise
    // recurse until the stack ran out, replacing a diagnosable crash with an
    // undiagnosable one.
    static LONG reporting = 0;
    if (InterlockedCompareExchange(&reporting, 1, 0) != 0) {
        return EXCEPTION_CONTINUE_SEARCH;
    }

    char buffer[1024];

    DWORD const code = pointers->ExceptionRecord->ExceptionCode;
    // %p is specified for void*, not void const*, so the cast belongs here rather than
    // at each use.
    void *const address = pointers->ExceptionRecord->ExceptionAddress;

    write_stderr("\n=== einsums: fatal exception ===\n");

    std::snprintf(buffer, sizeof(buffer), "code    : 0x%08lx (%s)\naddress : 0x%p\nthread  : %lu\nprocess : %lu\n", code,
                  exception_name(code), address, GetCurrentThreadId(), GetCurrentProcessId());
    write_stderr(buffer);

    if (code == EXCEPTION_ACCESS_VIOLATION && pointers->ExceptionRecord->NumberParameters >= 2) {
        std::snprintf(buffer, sizeof(buffer), "fault   : %s 0x%p\n",
                      access_violation_operation(pointers->ExceptionRecord->ExceptionInformation[0]),
                      reinterpret_cast<void *>(pointers->ExceptionRecord->ExceptionInformation[1]));
        write_stderr(buffer);
    }

    // The module the fault landed in. Without private symbols this is the single most
    // useful line in the report: it separates our code from the BLAS, the OpenMP
    // runtime, and the interpreter.
    HMODULE module = nullptr;
    if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           static_cast<LPCSTR>(address), &module) &&
        module != nullptr) {
        char module_path[MAX_PATH] = {};
        if (GetModuleFileNameA(module, module_path, MAX_PATH) != 0) {
            std::snprintf(
                buffer, sizeof(buffer), "module  : %s +0x%llx\n", module_path,
                static_cast<unsigned long long>(reinterpret_cast<char const *>(address) - reinterpret_cast<char const *>(module)));
            write_stderr(buffer);
        }
    }

    // Best effort, and allowed to fail. Walking the stack allocates and takes locks,
    // either of which can be exactly what is broken, so a backtrace that does not
    // arrive must not cost us the rest of the report.
    try {
        std::string const trace = backtrace();
        if (!trace.empty()) {
            write_stderr("\nbacktrace:\n");
            write_stderr(trace.c_str());
            write_stderr("\n");
        }
    } catch (...) { // NOLINT
        write_stderr("\nbacktrace: unavailable\n");
    }

    char dump_path[MAX_PATH] = {};
    if (write_minidump(pointers, dump_path)) {
        std::snprintf(buffer, sizeof(buffer), "\nminidump: %s\n", dump_path);
        write_stderr(buffer);
    } else {
        std::snprintf(buffer, sizeof(buffer), "\nminidump: failed to write %s (error %lu)\n", dump_path, GetLastError());
        write_stderr(buffer);
    }

    write_stderr("=== end einsums fatal exception ===\n");

    // Hand the exception on rather than swallowing it. The process still dies with the
    // right exception code, a debugger still gets its shot, and a machine configured
    // to collect dumps out of process still collects one - which is the more reliable
    // of the two dumps precisely when the fault came from teardown.
    InterlockedExchange(&reporting, 0);
    return EXCEPTION_CONTINUE_SEARCH;
}

} // namespace

void install_crash_handler(std::string const &dump_directory) {
    if (g_installed) {
        return;
    }

    g_dump_directory[0] = '\0';
    if (!dump_directory.empty() && dump_directory.size() < MAX_PATH) {
        std::memcpy(g_dump_directory, dump_directory.c_str(), dump_directory.size() + 1);
    }

    g_previous_filter = SetUnhandledExceptionFilter(&crash_filter);
    g_installed       = true;
}

void remove_crash_handler() {
    if (!g_installed) {
        return;
    }
    SetUnhandledExceptionFilter(g_previous_filter);
    g_previous_filter = nullptr;
    g_installed       = false;
}

#else

// POSIX carries its crash diagnostics on the signal handlers the Runtime module
// installs (see set_signal_handlers), which already report a backtrace. Adding a
// second mechanism here would only fight them, and would fight the interpreter's
// faulthandler in a Python process.
void install_crash_handler(std::string const & /*dump_directory*/) {
}

void remove_crash_handler() {
}

#endif

EINSUMS_NAMESPACE_END(util)
