//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include "Diagnostics.hpp"

#include <Waggle/Config.hpp>

#include <cstdio>
#include <mutex>

WAGGLE_NAMESPACE_BEGIN

namespace {

/// Where messages go. Never destroyed: the profiler reports while it shuts down at exit, and a
/// static first used after the profiler was built would be destroyed before it.
struct HandlerState {
    std::mutex        mutex;
    DiagnosticHandler handler;
};

HandlerState &state() {
    static auto *const s = new HandlerState; // NOLINT(cppcoreguidelines-owning-memory): deliberately never freed
    return *s;
}

void write_to_stderr(DiagnosticLevel level, std::string_view message) {
    if (level < DiagnosticLevel::Warning) {
        return;
    }
    std::fprintf(stderr, "waggle: %.*s\n", static_cast<int>(message.size()), message.data());
}

} // namespace

void install_diagnostic_handler(DiagnosticHandler handler) {
    auto                  &s = state();
    std::scoped_lock const lock(s.mutex);
    s.handler = std::move(handler);
}

void diagnostic(DiagnosticLevel level, std::string_view message) {
    DiagnosticHandler handler;
    {
        auto                  &s = state();
        std::scoped_lock const lock(s.mutex);
        handler = s.handler;
    }
    if (handler) {
        handler(level, message);
    } else {
        write_to_stderr(level, message);
    }
}

WAGGLE_NAMESPACE_END
