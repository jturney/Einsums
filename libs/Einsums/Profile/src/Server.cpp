//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Profile/Server.hpp>

#if defined(EINSUMS_HAVE_PROFILER)

#    include <Einsums/Config.hpp>

#    include <Einsums/Logging.hpp>
#    include <Einsums/TypeSupport/JsonEscape.hpp>

#    ifndef _WIN32
#        include <arpa/inet.h>
#        include <fcntl.h>
#        include <netinet/in.h>
#        include <poll.h>
#        include <sys/socket.h>
#        include <unistd.h>
#    else
#        ifndef NOMINMAX
#            define NOMINMAX
#        endif
#        include <winsock2.h>
#        include <ws2tcpip.h>
#        pragma comment(lib, "ws2_32.lib")
#    endif

#    ifdef __APPLE__
#        include <mach-o/dyld.h>
#    elif defined(__linux__)
#        include <linux/limits.h>
#    endif

#    include <chrono>
#    include <cmath>
#    include <ctime>
#    include <filesystem>
#    include <fstream>
#    include <limits>

namespace einsums::profile {

namespace {

#    ifndef _WIN32
void set_nonblocking(int fd) {
    int const flags = fcntl(fd, F_GETFL, 0);
    if (flags >= 0)
        fcntl(fd, F_SETFL, flags | O_NONBLOCK);
}

void close_socket(int fd) {
    ::close(fd);
}
#    else
void set_nonblocking(SOCKET fd) {
    u_long mode = 1;
    ioctlsocket(fd, FIONBIO, &mode);
}

void close_socket(SOCKET fd) {
    closesocket(fd);
}
#    endif

// Use the shared json_escape from TypeSupport, aliased to keep call sites unchanged.
auto const &escape_json_str = ::einsums::json_escape;

// TODO: Don't we already have this in RuntimeConfiguration?
auto get_executable_path() -> std::string {
#    ifdef __APPLE__
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    char     path[1024];
    uint32_t size = sizeof(path);
    if (_NSGetExecutablePath(path, &size) == 0) {
        // Resolve symlinks
        // NOLINTNEXTLINE(modernize-avoid-c-arrays)
        char resolved[PATH_MAX];
        if (realpath(path, resolved)) {
            return {resolved};
        }
        return {path};
    }
#    elif defined(__linux__)
    char    path[PATH_MAX];
    ssize_t len = readlink("/proc/self/exe", path, sizeof(path) - 1);
    if (len > 0) {
        path[len] = '\0';
        return std::string(path);
    }
#    elif defined(_WIN32)
    char path[MAX_PATH];
    if (GetModuleFileNameA(nullptr, path, MAX_PATH) > 0) {
        return std::string(path);
    }
#    endif
    return "";
}

// TODO: Don't we already have this in RuntimeConfiguration?
auto get_executable_name() -> std::string {
#    ifdef __APPLE__
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    char     path[1024];
    uint32_t size = sizeof(path);
    if (_NSGetExecutablePath(path, &size) == 0) {
        return std::filesystem::path(path).filename().string();
    }
#    elif defined(__linux__)
    char    path[PATH_MAX];
    ssize_t len = readlink("/proc/self/exe", path, sizeof(path) - 1);
    if (len > 0) {
        path[len] = '\0';
        return std::filesystem::path(path).filename().string();
    }
#    elif defined(_WIN32)
    char path[MAX_PATH];
    if (GetModuleFileNameA(nullptr, path, MAX_PATH) > 0) {
        return std::filesystem::path(path).filename().string();
    }
#    endif
    return "unknown";
}

auto get_start_time_iso() -> std::string {
    auto        now = std::chrono::system_clock::now();
    std::time_t tt  = std::chrono::system_clock::to_time_t(now);
    std::tm     tm{};
#    ifdef _WIN32
    localtime_s(&tm, &tt);
#    else
    localtime_r(&tt, &tm);
#    endif
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", &tm);
    return {buf};
}

auto get_hostname() -> std::string {
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    char buf[256];
    if (gethostname(buf, sizeof(buf)) == 0) {
        return {buf};
    }
    return "unknown";
}

// Computed once at static init time
// NOLINTNEXTLINE(bugprone-throwing-static-initialization)
static std::string const s_executable_name = get_executable_name();
// NOLINTNEXTLINE(bugprone-throwing-static-initialization)
static std::string const s_executable_path = get_executable_path();
// NOLINTNEXTLINE(bugprone-throwing-static-initialization)
static std::string const s_start_time = get_start_time_iso();

auto ns_to_ms(ns t) -> double {
    return std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t).count();
}

auto compute_inclusive_ms(AggNode const &n) -> double { // NOLINT
    double total = ns_to_ms(n.total_exclusive);
    for (auto const &ch : n.children)
        total += compute_inclusive_ms(*ch.second);
    return total;
}

} // namespace

Server::Server(Consumer &consumer, StringTable &strings, std::string const &bind_addr, uint16_t port)
    : _consumer(consumer), _strings(strings) {
#    ifndef _WIN32
    _listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (_listen_fd < 0) {
        EINSUMS_LOG_WARN("Profile server: failed to create socket");
        return;
    }

    int opt = 1;
    setsockopt(_listen_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    set_nonblocking(_listen_fd);

    struct sockaddr_in addr {};
    addr.sin_family = AF_INET;
    inet_pton(AF_INET, bind_addr.c_str(), &addr.sin_addr);

    // Try the requested port, then auto-increment up to 16 times to find a free one
    static constexpr int kMaxPortAttempts = 16;
    uint16_t             bound_port       = 0;
    for (int attempt = 0; attempt < kMaxPortAttempts; ++attempt) {
        auto try_port = static_cast<uint16_t>(port + attempt);
        addr.sin_port = htons(try_port);
        if (bind(_listen_fd, reinterpret_cast<struct sockaddr *>(&addr), sizeof(addr)) == 0) {
            bound_port = try_port;
            break;
        }
        if (attempt == 0) {
            EINSUMS_LOG_INFO("Profile server: port {} in use, trying next...", try_port);
        }
    }

    if (bound_port == 0) {
        EINSUMS_LOG_WARN("Profile server: failed to bind to {}:{}-{}", bind_addr, port, port + kMaxPortAttempts - 1);
        close_socket(_listen_fd);
        _listen_fd = -1;
        return;
    }

    if (listen(_listen_fd, 4) < 0) {
        EINSUMS_LOG_WARN("Profile server: failed to listen");
        close_socket(_listen_fd);
        _listen_fd = -1;
        return;
    }

    EINSUMS_LOG_INFO("Profile server listening on {}:{}", bind_addr, bound_port);

    register_mdns(bound_port);
#    endif
}

auto Server::has_client() const -> bool {
    return !_client_fds.empty();
}

Server::~Server() {
    shutdown();
}

void Server::shutdown() {
    // Final flush: give connected viewers a chance to fetch data.
    // Process requests and send updates in a drain loop.

    // Determine drain duration: longer if wait-for-viewer was used (viewer is important),
    // shorter default for normal runs.
    int drain_iterations = 5; // default: 500ms
    try {
        auto &gc = GlobalConfigMap::get_singleton();
        if (gc.get_bool("profiler-wait-for-viewer", false))
            drain_iterations = 30; // 3 seconds when viewer was explicitly requested
    } catch (...) {                // NOLINT
    }

    if (!_client_fds.empty()) {
        EINSUMS_LOG_INFO("Profile server: draining to {} connected viewer(s)...", _client_fds.size());
        for (int i = 0; i < drain_iterations; i++) {
            accept_clients();
            recv_requests();
            send_updates();
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        // Final pass
        recv_requests();
        send_updates();
    } else if (drain_iterations > 5) {
        // No client connected yet but wait-for-viewer was set, so give viewer a chance to connect late
        EINSUMS_LOG_INFO("Profile server: waiting briefly for late viewer connections...");
        for (int i = 0; i < drain_iterations; i++) {
            accept_clients();
            if (!_client_fds.empty()) {
                // Client just connected; drain remaining data
                EINSUMS_LOG_INFO("Profile server: late viewer connected, draining...");
                for (int j = 0; j < 10; j++) {
                    recv_requests();
                    send_updates();
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                }
                recv_requests();
                send_updates();
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }

    unregister_mdns();

    for (int const fd : _client_fds) {
        close_socket(fd);
    }
    _client_fds.clear();

    if (_listen_fd >= 0) {
        close_socket(_listen_fd);
        _listen_fd = -1;
    }
}

void Server::tick() {
    if (_listen_fd < 0)
        return;

    accept_clients();
    recv_requests();
    send_updates();
}

void Server::accept_clients() {
#    ifndef _WIN32
    while (static_cast<int>(_client_fds.size()) < kMaxClients) {
        struct sockaddr_in addr {};
        socklen_t          len = sizeof(addr);
        int                fd  = accept(_listen_fd, reinterpret_cast<struct sockaddr *>(&addr), &len);
        if (fd < 0)
            break; // no pending connections

        set_nonblocking(fd);
#        ifdef __APPLE__
        int val = 1;
        setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &val, sizeof(val));
#        endif
        _client_fds.push_back(fd);
        EINSUMS_LOG_INFO("Profile server: client connected (fd={})", fd);

        // Send initial snapshot
        send_snapshot_to(fd);
    }
#    endif
}

void Server::write_node_json(std::string &out, AggNode const &n) { // NOLINT
    static constexpr int64_t kMinSentinel = (std::numeric_limits<int64_t>::max)();

    out += R"({"name":")" + escape_json_str(n.name) + "\"";
    out += ",\"call_count\":" + std::to_string(n.call_count);
    out += ",\"exclusive_ms\":" + std::to_string(ns_to_ms(n.total_exclusive));
    out += ",\"inclusive_ms\":" + std::to_string(compute_inclusive_ms(n));

    // Min exclusive
    if (n.call_count == 0 || n.exclusive_min.count() == kMinSentinel)
        out += ",\"exclusive_min_ms\":0.0";
    else
        out += ",\"exclusive_min_ms\":" + std::to_string(ns_to_ms(n.exclusive_min));

    // Max exclusive
    out += ",\"exclusive_max_ms\":" + std::to_string(ns_to_ms(n.exclusive_max));

    // Stddev
    if (n.call_count > 1) {
        double const stddev_ns = std::sqrt(static_cast<double>(n.total_exclusive_M2) / static_cast<double>(n.call_count - 1));
        out += ",\"stddev_ms\":" + std::to_string(stddev_ns / 1e6);
    } else {
        out += ",\"stddev_ms\":0.0";
    }

    // Source location
    out += R"(,"file":")" + escape_json_str(n.file) + "\"";
    out += ",\"line\":" + std::to_string(n.line);
    out += R"(,"function":")" + escape_json_str(n.function) + "\"";

    // Annotations (with numeric stats where available)
    if (!n.annotations.empty()) {
        out += ",\"annotations\":{";
        bool first = true;
        for (auto const &a : n.annotations) {
            if (!first)
                out += ",";
            first    = false;
            auto nit = n.numeric_annotations.find(a.first);
            if (nit != n.numeric_annotations.end()) {
                auto const  &na  = nit->second;
                double const avg = na.count > 0 ? na.total / static_cast<double>(na.count) : 0.0;
                out += "\"" + escape_json_str(a.first) + "\":{";
                out += R"("last":")" + escape_json_str(a.second) + "\"";
                out += ",\"avg\":" + std::to_string(avg);
                out += ",\"min\":" + std::to_string(na.min_val);
                out += ",\"max\":" + std::to_string(na.max_val);
                out += "}";
            } else {
                out += "\"" + escape_json_str(a.first) + "\":\"" + escape_json_str(a.second) + "\"";
            }
        }
        out += "}";
    }

    // Hardware counters
    if (!n.counters_total.empty()) {
        out += ",\"counters\":{";
        bool first = true;
        for (auto const &ct : n.counters_total) {
            if (!first)
                out += ",";
            first = false;
            out += "\"" + escape_json_str(ct.first) + "\":{";
            out += "\"total\":" + std::to_string(ct.second);
            auto mit = n.counters_min.find(ct.first);
            if (mit != n.counters_min.end())
                out += ",\"min\":" + std::to_string(mit->second);
            auto xit = n.counters_max.find(ct.first);
            if (xit != n.counters_max.end())
                out += ",\"max\":" + std::to_string(xit->second);
            out += "}";
        }
        out += "}";
    }

    // Memory tracking
    if (n.mem_alloc_count > 0 || n.mem_free_count > 0) {
        out += ",\"memory\":{";
        out += "\"alloc_count\":" + std::to_string(n.mem_alloc_count);
        out += ",\"free_count\":" + std::to_string(n.mem_free_count);
        out += ",\"alloc_bytes\":" + std::to_string(n.mem_alloc_bytes);
        out += ",\"free_bytes\":" + std::to_string(n.mem_free_bytes);
        out += ",\"current_bytes\":" + std::to_string(n.mem_current_bytes);
        out += ",\"peak_bytes\":" + std::to_string(n.mem_peak_bytes);
        out += "}";
    }

    // Per-call histogram
    {
        bool has_data = false;
        for (unsigned long long const i : n.histogram) {
            if (i > 0) {
                has_data = true;
                break;
            }
        }
        if (has_data) {
            out += ",\"histogram\":{";
            // Bucket labels: 1us, 2us, 4us, ..., 1s, 2s
            // NOLINTNEXTLINE(modernize-avoid-c-arrays)
            static char const *const bucket_labels[] = {"1us",   "2us",   "4us",   "8us",   "16us",  "32us",  "64us",
                                                        "128us", "256us", "512us", "1ms",   "2ms",   "4ms",   "8ms",
                                                        "16ms",  "32ms",  "64ms",  "128ms", "256ms", "512ms", "1s"};
            bool                     first           = true;
            for (int i = 0; i < AggNode::kHistogramBuckets; ++i) {
                if (n.histogram[i] > 0) {
                    if (!first)
                        out += ",";
                    first = false;
                    out += "\"";
                    out += bucket_labels[i];
                    out += "\":";
                    out += std::to_string(n.histogram[i]);
                }
            }
            out += "}";
        }
    }

    // Children
    if (!n.children.empty()) {
        out += ",\"children\":[";
        bool first = true;
        for (auto const &ch : n.children) {
            if (!first)
                out += ",";
            first = false;
            write_node_json(out, *ch.second);
        }
        out += "]";
    }

    out += "}";
}

void Server::send_snapshot_to(int fd) {
    auto        lock       = _consumer.lock_shared();
    auto const &thread_map = _consumer.thread_data();

    // Build JSON Lines: meta line + snapshot line
    std::string msg;

    // Meta line
    msg += R"({"type":"meta","pid":)";
#    ifndef _WIN32
    msg += std::to_string(getpid());
#    else
    msg += std::to_string(GetCurrentProcessId());
#    endif

    msg += R"(,"executable":")" + escape_json_str(s_executable_name) + "\"";
    msg += R"(,"executable_path":")" + escape_json_str(s_executable_path) + "\"";
    msg += R"(,"start_time":")" + escape_json_str(s_start_time) + "\"";
    msg += R"(,"git_commit":")" + escape_json_str(EINSUMS_HAVE_GIT_COMMIT) + "\"";
    msg += R"(,"git_branch":")" + escape_json_str(EINSUMS_HAVE_GIT_BRANCH) + "\"";
    msg += std::string(",\"git_dirty\":") + (EINSUMS_HAVE_GIT_DIRTY ? "true" : "false");
    {
        // EINSUMS_BUILD_TYPE is a bare identifier (e.g. release, debug), so stringify it.
#    define EINSUMS_STRINGIFY_HELPER_(x) #x
#    define EINSUMS_STRINGIFY_(x)        EINSUMS_STRINGIFY_HELPER_(x)
        msg += R"(,"build_type":")" + escape_json_str(EINSUMS_STRINGIFY_(EINSUMS_BUILD_TYPE)) + "\"";
#    undef EINSUMS_STRINGIFY_
#    undef EINSUMS_STRINGIFY_HELPER_
    }

    auto &cb = get_counter_backend();
    msg += ",\"counters\":[";
    for (int i = 0; i < kNumCounterSlots; ++i) {
        if (i > 0)
            msg += ",";
        msg += "\"" + escape_json_str(cb.slot_name(i)) + "\"";
    }
    msg += "]}\n";

    // Snapshot line
    msg += R"({"type":"snapshot","seq":)" + std::to_string(++_seq);
    msg += ",\"dropped\":" + std::to_string(_consumer.dropped_count());
    msg += ",\"threads\":{";
    bool first_thread = true;
    for (auto const &tkv : thread_map) {
        if (!first_thread)
            msg += ",";
        first_thread = false;
        msg += "\"" + std::to_string(tkv.first) + R"(":{"name":")" + escape_json_str(tkv.second.name) + R"(","children":[)";
        bool first_child = true;
        for (auto const &c : tkv.second.root.children) {
            if (!first_child)
                msg += ",";
            first_child = false;
            write_node_json(msg, *c.second);
        }
        msg += "]}";
    }
    msg += "}}\n";

    // Append any accumulated log messages so the initial snapshot includes them
    auto logs = _log_queue.drain();
    for (auto const &entry : logs) {
        msg += R"({"type":"log","level":)" + std::to_string(entry.level);
        msg += R"(,"timestamp":")" + escape_json_str(entry.timestamp) + "\"";
        msg += R"(,"file":")" + escape_json_str(entry.file) + "\"";
        msg += ",\"line\":" + std::to_string(entry.line);
        msg += R"(,"function":")" + escape_json_str(entry.function) + "\"";
        msg += R"(,"message":")" + escape_json_str(entry.message) + "\"}\n";
    }

    // Append any accumulated println output messages
    auto outputs = _output_queue.drain();
    for (auto const &entry : outputs) {
        msg += R"({"type":"output")";
        msg += R"(,"timestamp":")" + escape_json_str(entry.timestamp) + "\"";
        msg += R"(,"message":")" + escape_json_str(entry.message) + "\"}\n";
    }

#    ifndef _WIN32
    // Best-effort send; drop if client can't keep up
    ::send(fd, msg.data(), msg.size(), 0);
#    endif
}

void Server::write_timeline_json(std::string &out) {
    auto const &events = _consumer.timeline_events();
    if (events.empty())
        return;

    out += R"({"type":"timeline","events":[)";
    bool first = true;
    for (auto const &te : events) {
        if (!first)
            out += ",";
        first = false;
        out += "{\"tid\":" + std::to_string(te.thread_id);
        out += R"(,"name":")" + escape_json_str(te.name) + "\"";
        out += ",\"start_ms\":" + std::to_string(te.start_ms);
        out += ",\"end_ms\":" + std::to_string(te.end_ms);
        out += "}";
    }
    out += "]}\n";
}

void Server::send_updates() {
    if (_client_fds.empty())
        return;

    auto        lock       = _consumer.lock_shared();
    auto const &thread_map = _consumer.thread_data();

    // Build a snapshot update line
    std::string msg;
    msg += R"({"type":"snapshot","seq":)" + std::to_string(++_seq);
    msg += ",\"dropped\":" + std::to_string(_consumer.dropped_count());
    msg += ",\"threads\":{";
    bool first_thread = true;
    for (auto const &tkv : thread_map) {
        if (!first_thread)
            msg += ",";
        first_thread = false;
        msg += "\"" + std::to_string(tkv.first) + R"(":{"name":")" + escape_json_str(tkv.second.name) + R"(","children":[)";
        bool first_child = true;
        for (auto const &c : tkv.second.root.children) {
            if (!first_child)
                msg += ",";
            first_child = false;
            write_node_json(msg, *c.second);
        }
        msg += "]}";
    }
    msg += "}}\n";

    // Append timeline events
    write_timeline_json(msg);

    // Append log messages
    auto logs = _log_queue.drain();
    for (auto const &entry : logs) {
        msg += R"({"type":"log","level":)" + std::to_string(entry.level);
        msg += R"(,"timestamp":")" + escape_json_str(entry.timestamp) + "\"";
        msg += R"(,"file":")" + escape_json_str(entry.file) + "\"";
        msg += ",\"line\":" + std::to_string(entry.line);
        msg += R"(,"function":")" + escape_json_str(entry.function) + "\"";
        msg += R"(,"message":")" + escape_json_str(entry.message) + "\"}\n";
    }

    // Append println output messages
    auto outputs = _output_queue.drain();
    for (auto const &entry : outputs) {
        msg += R"({"type":"output")";
        msg += R"(,"timestamp":")" + escape_json_str(entry.timestamp) + "\"";
        msg += R"(,"message":")" + escape_json_str(entry.message) + "\"}\n";
    }

    // Append benchmark result events
    auto bench_results = _benchmark_queue.drain();
    for (auto const &entry : bench_results) {
        msg += R"({"type":"benchmark_result")";
        msg += R"(,"label":")" + escape_json_str(entry.label) + "\"";
        msg += R"(,"metric":")" + escape_json_str(entry.metric) + "\"";
        msg += ",\"value_us\":" + std::to_string(entry.value_us);
        msg += ",\"min_us\":" + std::to_string(entry.min_us);
        msg += ",\"max_us\":" + std::to_string(entry.max_us);
        msg += ",\"stddev_us\":" + std::to_string(entry.stddev_us);
        msg += ",\"warmup_us\":" + std::to_string(entry.warmup_us);
        msg += ",\"reps\":" + std::to_string(entry.reps);
        if (!entry.annotations_json.empty()) {
            msg += ",\"annotations\":" + entry.annotations_json;
        }
        msg += "}\n";
    }

    // Send to all clients, remove disconnected ones
    std::vector<int> alive;
    for (int fd : _client_fds) {
#    ifndef _WIN32
        ssize_t const sent = ::send(fd, msg.data(), msg.size(), 0);
        if (sent > 0) {
            alive.push_back(fd);
        } else {
            EINSUMS_LOG_INFO("Profile server: client disconnected (fd={})", fd);
            close_socket(fd);
        }
#    endif
    }
    _client_fds = std::move(alive);
}

void Server::register_handler(std::string const &method, RequestHandler handler) {
    _request_handlers[method] = std::move(handler);
}

void Server::recv_requests() {
#    ifndef _WIN32
    for (int const fd : _client_fds) {
        // NOLINTNEXTLINE(modernize-avoid-c-arrays)
        char          buf[4096];
        ssize_t const n = ::recv(fd, buf, sizeof(buf), MSG_DONTWAIT);
        if (n > 0) {
            _recv_buffers[fd].append(buf, static_cast<size_t>(n));
            // Process complete lines
            auto &rb = _recv_buffers[fd];
            while (true) {
                auto pos = rb.find('\n');
                if (pos == std::string::npos)
                    break;
                std::string const line = rb.substr(0, pos);
                rb.erase(0, pos + 1);
                if (!line.empty()) {
                    process_request(fd, line);
                }
            }
        }
    }
    // Clean up recv buffers for disconnected clients
    for (auto it = _recv_buffers.begin(); it != _recv_buffers.end();) {
        bool found = false;
        for (int const fd : _client_fds) {
            if (fd == it->first) {
                found = true;
                break;
            }
        }
        if (!found) {
            it = _recv_buffers.erase(it);
        } else {
            ++it;
        }
    }
#    endif
}

void Server::process_request(int fd, std::string const &line) {
#    ifndef _WIN32
    // Minimal JSON parsing for request: {"type":"request","id":"...","method":"...","params":{...}}
    // We don't have a JSON library, so do simple string extraction.

    // Check it's a request
    if (line.find("\"type\"") == std::string::npos || line.find("\"request\"") == std::string::npos) {
        return;
    }

    // Extract "id" value
    std::string req_id;
    {
        auto pos = line.find("\"id\"");
        if (pos != std::string::npos) {
            pos = line.find(':', pos + 4);
            if (pos != std::string::npos) {
                auto start = line.find('"', pos + 1);
                if (start != std::string::npos) {
                    auto end = line.find('"', start + 1);
                    if (end != std::string::npos) {
                        req_id = line.substr(start + 1, end - start - 1);
                    }
                }
            }
        }
    }

    // Extract "method" value
    std::string method;
    {
        auto pos = line.find("\"method\"");
        if (pos != std::string::npos) {
            pos = line.find(':', pos + 8);
            if (pos != std::string::npos) {
                auto start = line.find('"', pos + 1);
                if (start != std::string::npos) {
                    auto end = line.find('"', start + 1);
                    if (end != std::string::npos) {
                        method = line.substr(start + 1, end - start - 1);
                    }
                }
            }
        }
    }

    if (req_id.empty() || method.empty()) {
        return;
    }

    // Extract "params" object (everything between the braces after "params":)
    std::string params = "{}";
    {
        auto pos = line.find("\"params\"");
        if (pos != std::string::npos) {
            pos = line.find(':', pos + 8);
            if (pos != std::string::npos) {
                // Skip whitespace
                pos++;
                while (pos < line.size() && (line[pos] == ' ' || line[pos] == '\t'))
                    pos++;
                if (pos < line.size() && line[pos] == '{') {
                    // Find matching closing brace (simple depth counting)
                    int          depth     = 0;
                    bool         in_string = false;
                    size_t const start     = pos;
                    for (size_t i = pos; i < line.size(); ++i) {
                        if (line[i] == '"' && (i == 0 || line[i - 1] != '\\')) {
                            in_string = !in_string;
                        }
                        if (!in_string) {
                            if (line[i] == '{')
                                depth++;
                            if (line[i] == '}') {
                                depth--;
                                if (depth == 0) {
                                    params = line.substr(start, i - start + 1);
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Dispatch to handler
    auto        it = _request_handlers.find(method);
    std::string response;
    if (it != _request_handlers.end()) {
        std::string const data = it->second(params);
        response               = R"({"type":"response","id":")" + escape_json_str(req_id) + R"(","data":)" + data + "}\n";
    } else {
        response = R"({"type":"response","id":")" + escape_json_str(req_id) + "\",\"data\":{\"error\":\"unknown method\"}}\n";
    }

    ::send(fd, response.data(), response.size(), 0);
#    else
    (void)fd;
    (void)line;
#    endif
}

void Server::register_mdns(uint16_t port) {
#    ifdef __APPLE__
    // Build TXT record with executable name, PID, and start time
    TXTRecordRef txt;
    TXTRecordCreate(&txt, 0, nullptr);
    TXTRecordSetValue(&txt, "exe", static_cast<uint8_t>(s_executable_name.size()), s_executable_name.c_str());
    auto pid_str = std::to_string(getpid());
    TXTRecordSetValue(&txt, "pid", static_cast<uint8_t>(pid_str.size()), pid_str.c_str());
    TXTRecordSetValue(&txt, "start", static_cast<uint8_t>(s_start_time.size()), s_start_time.c_str());

    // Service name: "exe-PID" to distinguish multiple instances
    std::string service_name = s_executable_name + "-" + pid_str;

    DNSServiceErrorType err = DNSServiceRegister(&_mdns_ref, 0, 0, service_name.c_str(), "_einsums-profile._tcp", nullptr, nullptr,
                                                 htons(port), TXTRecordGetLength(&txt), TXTRecordGetBytesPtr(&txt), nullptr, nullptr);

    TXTRecordDeallocate(&txt);

    if (err == kDNSServiceErr_NoError) {
        EINSUMS_LOG_INFO("Profile server: registered mDNS service '{}' on port {}", service_name, port);
    } else {
        EINSUMS_LOG_WARN("Profile server: mDNS registration failed (error {})", static_cast<int>(err));
        _mdns_ref = nullptr;
    }
#    else
    (void)port;
#    endif
}

void Server::unregister_mdns() {
#    ifdef __APPLE__
    if (_mdns_ref) {
        DNSServiceRefDeallocate(_mdns_ref);
        _mdns_ref = nullptr;
        EINSUMS_LOG_INFO("Profile server: unregistered mDNS service");
    }
#    endif
}

void Server::export_session(std::string const &path, std::string const &label,
                            std::vector<std::pair<std::string, std::string>> const &extra_json) {
    // Flush all pending events from ring buffers into the AggNode tree.
    _consumer.flush();

    auto        lock       = _consumer.lock_shared();
    auto const &thread_map = _consumer.thread_data();

    // Build a single JSON object in the viewer's session format:
    // { "label": "...", "type": "snapshot", "meta": {...}, "seq": N, "dropped": N, "threads": {...} }
    std::string json;
    json.reserve(65536);

    std::string const session_label = label.empty() ? s_executable_name : label;

    json += "{\n";
    json += R"(  "label": ")" + escape_json_str(session_label) + "\",\n";
    json += "  \"type\": \"snapshot\",\n";

    // Meta
    json += "  \"meta\": {\n";
    json += "    \"pid\": ";
#    ifndef _WIN32
    json += std::to_string(getpid());
#    else
    json += std::to_string(GetCurrentProcessId());
#    endif
    json += ",\n";
    json += R"(    "hostname": ")" + escape_json_str(get_hostname()) + "\",\n";
    json += R"(    "executable": ")" + escape_json_str(s_executable_name) + "\",\n";
    json += R"(    "start_time": ")" + escape_json_str(s_start_time) + "\",\n";
    json += R"(    "executable_path": ")" + escape_json_str(s_executable_path) + "\",\n";
    json += "    \"counters\": [";
    auto &cb = get_counter_backend();
    for (int i = 0; i < kNumCounterSlots; ++i) {
        if (i > 0)
            json += ", ";
        json += "\"" + escape_json_str(cb.slot_name(i)) + "\"";
    }
    json += "]\n";
    json += "  },\n";

    // Snapshot fields
    json += "  \"seq\": " + std::to_string(_seq) + ",\n";
    json += "  \"dropped\": " + std::to_string(_consumer.dropped_count()) + ",\n";

    // Threads
    json += "  \"threads\": {";
    bool first_thread = true;
    for (auto const &tkv : thread_map) {
        if (!first_thread)
            json += ",";
        first_thread = false;
        json += "\n    \"" + std::to_string(tkv.first) + R"(": {"name": ")" + escape_json_str(tkv.second.name) + R"(", "children": [)";
        bool first_child = true;
        for (auto const &c : tkv.second.root.children) {
            if (!first_child)
                json += ",";
            first_child = false;
            json += "\n      ";
            write_node_json(json, *c.second);
        }
        json += "\n    ]}";
    }
    json += "\n  }";

    // Include any extra JSON fields provided by the caller.
    for (auto const &[key, value] : extra_json) {
        json += ",\n  \"" + escape_json_str(key) + "\": " + value;
    }

    // Include compute graph data if the handler is registered.
    {
        auto it = _request_handlers.find("get_compute_graphs");
        if (it != _request_handlers.end()) {
            std::string const graphs_json = it->second("");
            auto              arr_start   = graphs_json.find('[');
            auto              arr_end     = graphs_json.rfind(']');
            if (arr_start != std::string::npos && arr_end != std::string::npos && arr_end > arr_start) {
                json += ",\n  \"compute_graphs\": ";
                json += graphs_json.substr(arr_start, arr_end - arr_start + 1);
            }
        }
    }

    json += "\n}\n";

    // If the file already exists, read existing sessions and append.
    // Output uses the multi-session format: {"sessions": [s1, s2, ...]}
    std::string output;
    {
        std::ifstream existing(path);
        if (existing.is_open()) {
            // Read the existing file content.
            std::string content((std::istreambuf_iterator<char>(existing)), std::istreambuf_iterator<char>());
            existing.close();

            if (!content.empty()) {
                // Find the existing sessions array. Two formats:
                // 1. {"sessions": [...]}  is multi-session
                // 2. {"label": ...}       is single session
                //
                // Strategy: find "sessions" key. If present, insert before the closing ']}'.
                // If single session, wrap both in multi-session format.
                auto sessions_pos = content.find("\"sessions\"");
                if (sessions_pos != std::string::npos) {
                    // Multi-session: find the last ']' before the final '}'
                    auto last_bracket = content.rfind(']');
                    if (last_bracket != std::string::npos) {
                        output = content.substr(0, last_bracket);
                        output += ",\n";
                        output += json;
                        output += content.substr(last_bracket); // "]}" and any trailing whitespace
                    }
                } else {
                    // Single session: wrap both in multi-session format.
                    output = "{\"sessions\": [\n";
                    // Trim trailing whitespace from existing content.
                    auto end = content.find_last_not_of(" \t\n\r");
                    if (end != std::string::npos)
                        content = content.substr(0, end + 1);
                    output += content;
                    output += ",\n";
                    output += json;
                    output += "\n]}\n";
                }
            }
        }
    }

    // If no existing file or parsing failed, write as single session.
    if (output.empty()) {
        output = json;
    }

    std::ofstream file(path, std::ios::trunc);
    if (file.is_open()) {
        file << output;
        file.close();
        EINSUMS_LOG_INFO("Profile session exported to '{}' ({} bytes)", path, output.size());
    } else {
        EINSUMS_LOG_WARN("Failed to export profile session to '{}'", path);
    }
}

} // namespace einsums::profile

#endif
