//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/Logging.hpp>
#include <Einsums/StringUtil/FromString.hpp>

#include <fmt/ostream.h>
#include <fmt/printf.h>

#include <iostream>
#include <spdlog/common.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <string>

#if defined(EINSUMS_HAVE_TRACY)
#    include <tracy/Tracy.hpp>
#endif

namespace einsums::detail {

EINSUMS_DETAIL_DEFINE_SPDLOG(einsums, warn)

#if defined(EINSUMS_HAVE_TRACY)
template <typename Mutex>
class tracy_sink : public spdlog::sinks::base_sink<Mutex> {
  protected:
    void sink_it_(spdlog::details::log_msg const &msg) override {
        spdlog::memory_buf_t formatted;
        this->formatter_->format(msg, formatted);

        auto str = fmt::to_string(formatted);
        TracyMessage(str.c_str(), str.size()); // send to Tracy
    }

    void flush_() override {}
};

// Convenience alias
using tracy_sink_mt = tracy_sink<std::mutex>;
using tracy_sink_st = tracy_sink<spdlog::details::null_mutex>;
#endif

spdlog::level::level_enum get_spdlog_level(std::string const &env) {
    try {
        return static_cast<spdlog::level::level_enum>(from_string<std::underlying_type_t<spdlog::level::level_enum>>(env));
    } catch (bad_lexical_cast const &) {
        fmt::print(std::cerr,
                   "Einsums given invalid log level: \"{}\". Using default level instead {} (warn). "
                   "Valid values are {} (trace) to {} (off).\n",
                   env, SPDLOG_LEVEL_WARN, SPDLOG_LEVEL_TRACE, SPDLOG_LEVEL_OFF);
        return spdlog::level::warn;
    }
}

std::shared_ptr<spdlog::sinks::sink> get_spdlog_sink(std::string const &env) {
    // In the future it might be useful to include a tcp sink option.
    // Could be useful when we are doing MPI/distributed development.
    if (env.empty()) {
        fmt::print(std::cerr, "Einsums given empty log destination. Using default instead (cerr).\n");
    } else if (env == "cout") {
        return std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    } else if (env == "cerr") {
        return std::make_shared<spdlog::sinks::stderr_color_sink_mt>();
    }
#if defined(EINSUMS_HAVE_TRACY)
    else if (env == "tracy") {
        return std::make_shared<tracy_sink_mt>();
    }
#endif
    return std::make_shared<spdlog::sinks::basic_file_sink_mt>(env);
}

} // namespace einsums::detail

namespace einsums {

void set_log_level(int level) {
    detail::get_einsums_logger().set_level(static_cast<spdlog::level::level_enum>(level));
}

int get_log_level() {
    return static_cast<int>(detail::get_einsums_logger().level());
}

} // namespace einsums