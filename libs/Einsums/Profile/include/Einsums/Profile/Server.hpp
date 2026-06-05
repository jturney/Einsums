//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#if defined(EINSUMS_HAVE_PROFILER)

#    include <Einsums/Profile/Consumer.hpp>
#    include <Einsums/Profile/LogSink.hpp>
#    include <Einsums/Profile/StringTable.hpp>

#    include <cstdint>
#    include <functional>
#    include <string>
#    include <unordered_map>
#    include <vector>

#    ifdef __APPLE__
#        include <dns_sd.h>
#    endif

namespace einsums::profile {

/// TCP server that streams profiling data as JSON Lines to connected clients.
/// Binds to localhost by default, accepts up to 4 simultaneous clients.
/// On macOS, advertises via Bonjour/mDNS as "_einsums-profile._tcp".
class EINSUMS_EXPORT Server {
  public:
    Server(Consumer &consumer, StringTable &strings, std::string const &bind_addr = "127.0.0.1", uint16_t port = 19216);
    ~Server();

    Server(Server const &)            = delete;
    Server &operator=(Server const &) = delete;

    /// Call periodically from consumer event loop. Accepts new connections and sends updates.
    void tick();

    /// Shut down the server, close all connections.
    void shutdown();

    /// Request handler: method name → handler function.
    /// Handler receives (params_json) and returns data_json string.
    using RequestHandler = std::function<std::string(std::string const &)>;

    /// Register a request handler for a given method name.
    void register_handler(std::string const &method, RequestHandler handler);

    /// Whether the server is active.
    [[nodiscard]] auto is_running() const -> bool { return _listen_fd >= 0; }

    /// Whether at least one viewer client is connected.
    [[nodiscard]] auto has_client() const -> bool;

    /// Access the log message queue (for wiring the profiler_sink).
    LogMessageQueue &log_queue() { return _log_queue; }

    /// Access the output message queue (for forwarding println output).
    LogMessageQueue &output_queue() { return _output_queue; }

    /// Access the benchmark result queue (for forwarding benchmark_result events).
    BenchmarkResultQueue &benchmark_queue() { return _benchmark_queue; }

    /// Export the current profiling session to a JSON file compatible with the
    /// imgui profile viewer's "Load Session" feature. Flushes all pending events
    /// before exporting.
    /// @param path Output file path.
    /// @param label Session label (shown in viewer).
    /// @param extra_json Additional JSON fields to embed at the top level
    ///        (e.g., compute graph data). Each entry is "key": json_value.
    void export_session(std::string const &path, std::string const &label = "",
                        std::vector<std::pair<std::string, std::string>> const &extra_json = {});

  private:
    void accept_clients();
    void send_snapshot_to(int fd);
    void send_updates();
    void write_node_json(std::string &out, AggNode const &n);
    void write_timeline_json(std::string &out);

    void register_mdns(uint16_t port);
    void unregister_mdns();

    void recv_requests();
    void process_request(int fd, std::string const &line);

    Consumer    &_consumer;
    StringTable &_strings;

    int                  _listen_fd = -1;
    std::vector<int>     _client_fds;
    uint64_t             _seq        = 0;
    static constexpr int kMaxClients = 4;

    /// Per-client receive buffer for incoming requests.
    std::unordered_map<int, std::string> _recv_buffers;

    /// Registered request handlers.
    std::unordered_map<std::string, RequestHandler> _request_handlers;

    LogMessageQueue      _log_queue;
    LogMessageQueue      _output_queue;
    BenchmarkResultQueue _benchmark_queue;

#    ifdef __APPLE__
    DNSServiceRef _mdns_ref = nullptr;
#    endif
};

} // namespace einsums::profile

#endif
