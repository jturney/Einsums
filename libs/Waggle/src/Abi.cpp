//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// The C interface (<Waggle/Waggle.h>), over the collector.

#include <Waggle/Config.hpp>

#include <Waggle/Waggle.h>

#include <chrono>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>

#include "Diagnostics.hpp"
#include "Profiler.hpp"
#include "Settings.hpp"

struct waggle_reply {
    std::string data;
};

namespace {

using waggle::Profiler;

auto profiler() -> Profiler & {
    return Profiler::instance();
}

/// @p s as a view, empty for null.
auto view(char const *s) -> std::string_view {
    return s != nullptr ? std::string_view(s) : std::string_view{};
}

/// Owns a C caller's user pointer, releasing it once when the last copy of the wrapper goes.
auto own(void *user, waggle_release_fn release) -> std::shared_ptr<void> {
    return {user, [release](void *p) {
                if (release != nullptr) {
                    release(p);
                }
            }};
}

/// A C handler as the collector's std::function.
auto wrap(waggle_handler_fn handler, void *user, waggle_release_fn release) -> waggle::RequestHandler {
    return [handler, owner = own(user, release)](std::string const &params) {
        waggle_reply reply;
        handler(owner.get(), params.data(), params.size(), &reply);
        return reply.data;
    };
}

/// Set settings by name as one update, honouring other libraries' settings unless @p override is
/// true. One update, so that the server starts on the port given beside it.
auto set_settings(char const *const *keys, char const *const *values, size_t count, bool override) -> int {
    waggle::SettingsUpdate update;
    for (size_t i = 0; i < count; ++i) {
        if (!waggle::SettingsStore::add_from_text(update, view(keys[i]), std::string(view(values[i])))) {
            return -1;
        }
    }
    if (override) {
        profiler().override_settings(update);
        return 0;
    }
    return static_cast<int>(profiler().configure(update));
}

} // namespace

extern "C" {

void waggle_abi_version(uint32_t *major, uint32_t *minor) {
    if (major != nullptr) {
        *major = WAGGLE_ABI_MAJOR;
    }
    if (minor != nullptr) {
        *minor = WAGGLE_ABI_MINOR;
    }
}

uint32_t waggle_intern(char const *s, size_t length) {
    return profiler().string_table().intern(std::string_view(s, length));
}

uint32_t waggle_register_domain(char const *name, size_t length) {
    return profiler().register_domain(std::string_view(name, length));
}

uint32_t waggle_register_site(char const *name, size_t name_length, char const *file, int line, char const *func, uint32_t domain) {
    return profiler().register_site(std::string_view(name, name_length), view(file), line, view(func), domain);
}

int waggle_enabled(void) {
    return profiler().enabled() ? 1 : 0;
}

void waggle_set_enabled(int on) {
    profiler().set_enabled(on != 0);
}

void waggle_zone_begin(uint32_t site, uint32_t name_id) {
    profiler().push_interned(site, name_id);
}

void waggle_zone_end(void) {
    profiler().pop();
}

void waggle_annotate_str(uint32_t key_id, uint32_t value_id) {
    profiler().annotate(key_id, waggle::AnnotateValueType::String, [&](waggle::AnnotationPayload &a) { a.string_id = value_id; });
}

void waggle_annotate_i64(uint32_t key_id, int64_t value) {
    profiler().annotate(key_id, waggle::AnnotateValueType::Int64, [&](waggle::AnnotationPayload &a) { a.int_val = value; });
}

void waggle_annotate_f64(uint32_t key_id, double value) {
    profiler().annotate(key_id, waggle::AnnotateValueType::Float64, [&](waggle::AnnotationPayload &a) { a.float_val = value; });
}

void waggle_mem_alloc(void const *address, int64_t bytes) {
    profiler().memory(waggle::EventType::MemAlloc, address, bytes);
}

void waggle_mem_free(void const *address, int64_t bytes) {
    profiler().memory(waggle::EventType::MemFree, address, bytes);
}

void waggle_set_thread_name(char const *name, size_t length) {
    profiler().set_thread_name(std::string(name, length));
}

uint32_t waggle_current_thread_id(void) {
    return Profiler::current_thread_id();
}

int waggle_config_set(char const *const *keys, char const *const *values, size_t count) {
    return set_settings(keys, values, count, false);
}

int waggle_config_override(char const *const *keys, char const *const *values, size_t count) {
    return set_settings(keys, values, count, true);
}

int64_t waggle_config_get(char const *key, char *buffer, size_t size) {
    auto const value = profiler().setting_text(view(key));
    if (!value) {
        return -1;
    }
    if (buffer != nullptr && size > value->size()) {
        std::memcpy(buffer, value->c_str(), value->size() + 1);
    }
    return static_cast<int64_t>(value->size());
}

void waggle_init(char const *name, char const *version, char const *git_commit, char const *git_branch, int git_dirty,
                 char const *build_type) {
    profiler().init({.name       = std::string(view(name)),
                     .version    = std::string(view(version)),
                     .git_commit = std::string(view(git_commit)),
                     .git_branch = std::string(view(git_branch)),
                     .git_dirty  = git_dirty != 0,
                     .build_type = std::string(view(build_type))});
}

void waggle_finalize(char const *name) {
    profiler().finalize(std::string(view(name)));
}

void waggle_flush(void) {
    profiler().flush();
}

void waggle_wait_for_viewer(void) {
    profiler().wait_for_viewer();
}

void waggle_server_start(uint16_t port) {
    profiler().start_server(port);
}

int waggle_server_running(void) {
    auto *srv = profiler().server();
    return srv != nullptr && srv->is_running() ? 1 : 0;
}

uint16_t waggle_server_port(void) {
    auto *srv = profiler().server();
    return srv != nullptr ? srv->port() : 0;
}

int waggle_viewer_connected(void) {
    auto *srv = profiler().server();
    return srv != nullptr && srv->has_client() ? 1 : 0;
}

void waggle_reply_set(waggle_reply *reply, char const *data, size_t length) {
    if (reply != nullptr) {
        reply->data.assign(data, length);
    }
}

void waggle_register_handler(char const *method, waggle_handler_fn handler, void *user, waggle_release_fn release) {
    profiler().register_handler(std::string(view(method)), wrap(handler, user, release));
}

void waggle_unregister_handler(char const *method) {
    profiler().unregister_handler(std::string(view(method)));
}

void waggle_register_session_section(char const *key, waggle_handler_fn section, void *user, waggle_release_fn release) {
    auto call = wrap(section, user, release);
    profiler().register_session_section(std::string(view(key)), [call = std::move(call)] { return call(""); });
}

void waggle_publish(char const *type, char const *json, size_t length) {
    profiler().publish(view(type), std::string_view(json, length));
}

void waggle_log(int level, int64_t unix_ns, char const *file, int line, char const *func, char const *message, size_t length) {
    auto const when = std::chrono::system_clock::time_point(
        std::chrono::duration_cast<std::chrono::system_clock::duration>(std::chrono::nanoseconds(unix_ns)));
    profiler().log(level, when, view(file), line, view(func), std::string_view(message, length));
}

void waggle_output(char const *message, size_t length) {
    profiler().output(std::string_view(message, length));
}

void waggle_set_diagnostic_handler(waggle_diagnostic_fn handler, void *user, waggle_release_fn release) {
    if (handler == nullptr) {
        waggle::install_diagnostic_handler({});
        return;
    }
    waggle::install_diagnostic_handler([handler, owner = own(user, release)](waggle::DiagnosticLevel level, std::string_view message) {
        handler(owner.get(), static_cast<int>(level), message.data(), message.size());
    });
}

void waggle_print_report(int detailed) {
    profiler().print(detailed != 0, std::cout);
    std::cout.flush();
}

int waggle_export_json(char const *path) {
    return profiler().export_json(std::string(view(path))) ? 0 : 1;
}

uint64_t waggle_total_push_count(void) {
    return profiler().total_push_count();
}

uint64_t waggle_total_pop_count(void) {
    return profiler().total_pop_count();
}

double waggle_push_overhead_ns(void) {
    return profiler().avg_push_overhead_ns();
}

double waggle_pop_overhead_ns(void) {
    return profiler().avg_pop_overhead_ns();
}

void waggle_open_zone_annotations(waggle_pair_fn fn, void *user) {
    auto &prof = profiler();
    prof.flush();
    auto *consumer = prof.consumer();
    auto  lock     = consumer->lock_shared();
    for (auto const &[key, value] : consumer->collect_zone_annotations(Profiler::current_thread_id())) {
        fn(user, key.data(), key.size(), value.data(), value.size());
    }
}

} // extern "C"
