//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Config/Namespace.hpp>

#if defined(EINSUMS_HAVE_PROFILER)

#    include <Einsums/Profile/Profile.hpp>

#    include <mutex>
#    include <spdlog/sinks/base_sink.h>
#    include <string_view>

EINSUMS_NAMESPACE_BEGIN(profile)

/**
 * @brief An spdlog sink that streams Einsums' log messages to the profiler's viewers.
 *
 * Installed for the life of the logger. With no server running each message costs one atomic
 * load, so a server started later still receives every message from then on.
 *
 * It must never log through EINSUMS_LOG_*, which would recurse into it.
 */
class SpdlogSink final : public spdlog::sinks::base_sink<std::mutex> {
  protected:
    void sink_it_(spdlog::details::log_msg const &msg) override {
        waggle::Profiler::instance().log(static_cast<int>(msg.level), msg.time,
                                         msg.source.filename != nullptr ? std::string_view(msg.source.filename) : std::string_view{},
                                         msg.source.line > 0 ? msg.source.line : 0,
                                         msg.source.funcname != nullptr ? std::string_view(msg.source.funcname) : std::string_view{},
                                         std::string_view(msg.payload.data(), msg.payload.size()));
    }

    void flush_() override {}
};

EINSUMS_NAMESPACE_END(profile)

#endif
