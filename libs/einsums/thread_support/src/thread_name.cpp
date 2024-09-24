//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#include <einsums/config.hpp>

namespace einsums::detail {
std::string &thread_name() {
    static thread_local std::string thread_name_;
    return thread_name_;
}
} // namespace einsums::detail

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)

#    include <einsums/thread_support/set_thread_name.hpp>

namespace einsums::detail {
DWORD const MS_VC_EXCEPTION = 0x406D'1388;

#    pragma pack(push, 8)
typedef struct tagTHREADNAME_INFO {
    DWORD  dwType;     // Must be 0x1000.
    LPCSTR szName;     // Pointer to name (in user addr space).
    DWORD  dwThreadID; // Thread ID (-1=caller thread).
    DWORD  dwFlags;    // Reserved for future use, must be zero.
} THREADNAME_INFO;
#    pragma pack(pop)

// Set the name of the thread shown in the Visual Studio debugger
void set_thread_name(char const *threadName, DWORD dwThreadID) {
    THREADNAME_INFO info;
    info.dwType     = 0x1000;
    info.szName     = threadName;
    info.dwThreadID = dwThreadID;
    info.dwFlags    = 0;

    __try {
        RaiseException(MS_VC_EXCEPTION, 0, sizeof(info) / sizeof(ULONG_PTR), (ULONG_PTR *)&info);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
    }
}
} // namespace einsums::detail

#endif
