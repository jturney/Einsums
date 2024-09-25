//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#include <einsums/config.hpp>

#include <einsums/assert.hpp>
#include <einsums/debugging/print.hpp>
#include <einsums/init_runtime/detail/init_logging.hpp>
#include <einsums/runtime/get_worker_thread_num.hpp>
#include <einsums/runtime_configuration/runtime_configuration.hpp>
#include <einsums/string_util/from_string.hpp>
#include <einsums/threading_base/thread_data.hpp>

#if defined(EINSUMS_HAVE_MPI)
#    include <einsums/mpi_base/mpi.hpp>
#endif

#include <fmt/format.h>
#include <fmt/ostream.h>

#include <cstddef>
#include <ctime>
#include <memory>
#include <spdlog/pattern_formatter.h>
#include <spdlog/spdlog.h>
#include <string>
#include <string_view>
#include <utility>

namespace einsums::detail {

static void spdlog_format_thread_id(threads::detail::thread_id_type const &id, const spdlog::details::log_msg &, const std::tm &,
                                    spdlog::memory_buf_t                  &dest) {
    if (id) {
        dest.append(fmt::format("{}/{}", id,
#if defined(EINSUMS_HAVE_THREAD_DESCRIPTION)
                                id ? get_thread_id_data(id)->get_description() :
#endif
                                   "----"));
    } else {
        dest.append(std::string_view("----/----"));
    }
}

struct einsums_thread_id_formatter_flag final : spdlog::custom_flag_formatter {
    void format(const spdlog::details::log_msg &m, const std::tm &t, spdlog::memory_buf_t &dest) override {
        spdlog_format_thread_id(threads::detail::get_self_id(), m, t, dest);
    }

    [[nodiscard]] auto clone() const -> std::unique_ptr<custom_flag_formatter> override {
        return spdlog::details::make_unique<einsums_thread_id_formatter_flag>();
    }
};

class einsums_parent_thread_id_formatter_flag final : public spdlog::custom_flag_formatter {
  public:
    void format(const spdlog::details::log_msg &m, const std::tm &t, spdlog::memory_buf_t &dest) override {
        spdlog_format_thread_id(threads::detail::get_parent_id(), m, t, dest);
    }

    [[nodiscard]] auto clone() const -> std::unique_ptr<custom_flag_formatter> override {
        return spdlog::details::make_unique<einsums_parent_thread_id_formatter_flag>();
    }
};

class einsums_worker_thread_formatter_flag final : public spdlog::custom_flag_formatter {
    static void format_id(spdlog::memory_buf_t &dest, std::size_t i) {
        if (i != static_cast<std::size_t>(-1)) {
            dest.append(fmt::format("{:04}", i));
        } else {
            dest.append(std::string_view("----"));
        }
    }

  public:
    void format(const spdlog::details::log_msg &, const std::tm &, spdlog::memory_buf_t &dest) override {
        format_id(dest, get_thread_pool_num());
        dest.append(std::string_view("/"));
        format_id(dest, get_worker_thread_num());
        dest.append(std::string_view("/"));
        format_id(dest, get_local_worker_thread_num());
    }

    [[nodiscard]] auto clone() const -> std::unique_ptr<custom_flag_formatter> override {
        return spdlog::details::make_unique<einsums_worker_thread_formatter_flag>();
    }
};

class hostname_formatter_flag final : public spdlog::custom_flag_formatter {
  public:
    void format(const spdlog::details::log_msg &, const std::tm &, spdlog::memory_buf_t &dest) override {
        static EINSUMS_DETAIL_NS_DEBUG::hostname_print_helper helper{};
        static std::string_view                               hostname_str = helper.get_hostname();

        if (!hostname_str.empty()) {
            dest.append(hostname_str);
        } else {
            dest.append(std::string_view("----"));
        }

        static int rank = [&] {
        // First try to get the rank through MPI
#if defined(EINSUMS_HAVE_MPI)
            int mpi_initialized = 0;
            if (MPI_Initialized(&mpi_initialized) == MPI_SUCCESS && mpi_initialized) {
                int rank = 0;
                if (MPI_Comm_rank(MPI_COMM_WORLD, &rank) == MPI_SUCCESS) {
                    return rank;
                }
            }
#endif

            // Otherwise guess based on environment variables
            return helper.guess_rank();
        }();
        if (rank != -1) {
            dest.append(fmt::format("/{}", rank));
        } else {
            dest.append(std::string_view("/----"));
        }
    }

    [[nodiscard]] auto clone() const -> std::unique_ptr<custom_flag_formatter> override {
        return spdlog::details::make_unique<hostname_formatter_flag>();
    }
};

struct log_settings {
    std::string level;
    std::string dest;
    std::string format;
};

static auto get_log_settings(section const &ini, char const *sec) -> log_settings {
    log_settings result;
    if (ini.has_section(sec)) {
        section const *logini = ini.get_section(sec);
        EINSUMS_ASSERT(logini != nullptr);

        result.level  = logini->get_entry("level", "");
        result.dest   = logini->get_entry("destination", "");
        result.format = logini->get_entry("format", "");
    }
    return result;
}

void init_logging(const util::runtime_configuration &ini) {
    auto [level, dest, format] = get_log_settings(ini, "einsums.log");

    // Set log destination
    auto &sinks = get_einsums_logger().sinks();
    sinks.clear();
    sinks.push_back(get_spdlog_sink(dest));

    // Set log pattern
    auto formatter = std::make_unique<spdlog::pattern_formatter>();

    formatter->add_flag<einsums_thread_id_formatter_flag>('k');
    formatter->add_flag<einsums_parent_thread_id_formatter_flag>('q');
    formatter->add_flag<einsums_worker_thread_formatter_flag>('w');
    formatter->add_flag<hostname_formatter_flag>('j');

    formatter->set_pattern(format);
    get_einsums_logger().set_formatter(std::move(formatter));

    // Set log level
    get_einsums_logger().set_level(get_spdlog_level(level));
}

} // namespace einsums::detail