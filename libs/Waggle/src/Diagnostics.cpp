//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <cstdio>
#include <mutex>
#include <waggle/Config.hpp>
#include <waggle/Diagnostics.hpp>

WAGGLE_NAMESPACE_BEGIN

namespace {

std::mutex &handler_mutex() {
    static std::mutex mutex;
    return mutex;
}

DiagnosticHandler &handler_slot() {
    static DiagnosticHandler handler;
    return handler;
}

void write_to_stderr(DiagnosticLevel level, std::string_view message) {
    if (level < DiagnosticLevel::Warning) {
        return;
    }
    std::fprintf(stderr, "waggle: %.*s\n", static_cast<int>(message.size()), message.data());
}

} // namespace

void set_diagnostic_handler(DiagnosticHandler handler) {
    std::scoped_lock const lock(handler_mutex());
    handler_slot() = std::move(handler);
}

void diagnostic(DiagnosticLevel level, std::string_view message) {
    DiagnosticHandler handler;
    {
        std::scoped_lock const lock(handler_mutex());
        handler = handler_slot();
    }
    if (handler) {
        handler(level, message);
    } else {
        write_to_stderr(level, message);
    }
}

WAGGLE_NAMESPACE_END
