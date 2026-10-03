//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Config/Namespace.hpp>

#if defined(EINSUMS_HAVE_PROFILER)

#    include <Einsums/Profile/Consumer.hpp>
#    include <Einsums/Profile/LogQueue.hpp>
#    include <Einsums/Profile/RequestHandlers.hpp>
#    include <Einsums/Profile/StringTable.hpp>

#    include <atomic>
#    include <cstdint>
#    include <deque>
#    include <mutex>
#    include <string>
#    include <string_view>
#    include <unordered_map>
#    include <vector>

#    ifdef __APPLE__
#        include <dns_sd.h>
#    endif

EINSUMS_NAMESPACE_BEGIN(profile)

/// TCP server that streams profiling data as JSON Lines to connected clients.
/// Binds to localhost by default, accepts up to 4 simultaneous clients.
/// On macOS, advertises via Bonjour/mDNS as "_einsums-profile._tcp".
class EINSUMS_EXPORT Server {
  public:
    /// @param consumer  The aggregated trees and timeline the server streams.
    /// @param strings   The string table the trees' ids resolve through.
    /// @param handlers  The request handlers and session sections libraries registered; read, never
    ///                  owned, and may outlive or predate the server.
    /// @param bind_addr The address to listen on; loopback, since the server is unauthenticated.
    /// @param port      The port to listen on.
    Server(Consumer &consumer, StringTable &strings, RequestHandlers const &handlers, std::string const &bind_addr = "127.0.0.1",
           uint16_t port = 19216);
    ~Server();

    Server(Server const &)            = delete;
    Server &operator=(Server const &) = delete;

    /// Call periodically from consumer event loop. Accepts new connections and sends updates.
    void tick();

    /// Shut down the server, close all connections.
    /// @param viewer_requested Whether the program was told to wait for a viewer, which earns a
    ///        longer final drain and a short wait for one that is late.
    void shutdown(bool viewer_requested = false);

    /// The port the server listens on, which is the requested one or the first free one after it;
    /// 0 when not listening.
    [[nodiscard]] auto port() const -> uint16_t { return _bound_port; }

    /// Whether the server is active.
    [[nodiscard]] auto is_running() const -> bool { return _listen_fd >= 0; }

    /// Whether a viewer is connected. Lock-free and callable from any thread; every ComputeGraph
    /// destruction asks.
    [[nodiscard]] auto has_client() const -> bool;

    /// Access the log message queue (for wiring the profiler_sink).
    LogMessageQueue &log_queue() { return _log_queue; }

    /// Access the output message queue (for forwarding println output).
    LogMessageQueue &output_queue() { return _output_queue; }

    /// Queue a message for every connected viewer: @p json_object (a JSON object, braces included)
    /// with a ``"type"`` member of @p type added. Thread-safe. The oldest messages are dropped
    /// past @ref kMaxPublished.
    void publish(std::string_view type, std::string_view json_object);

    /// Messages @ref publish holds for viewers before dropping the oldest.
    static constexpr size_t kMaxPublished = 10000;

    /// Flush, then export the session to a JSON file the viewer loads with ``--load``.
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

    int      _listen_fd  = -1;
    uint16_t _bound_port = 0;

    /// Connected viewers. Owned by the thread driving tick(): the consumer thread, then the main
    /// thread in shutdown() after the consumer is joined. Other threads use _has_client.
    std::vector<int> _client_fds;

    /// `!_client_fds.empty()`, republished after every change. May be a tick stale, which only
    /// affects a caching decision.
    std::atomic<bool> _has_client{false};

    uint64_t             _seq        = 0;
    static constexpr int kMaxClients = 4;

    /// Per-client receive buffer for incoming requests.
    std::unordered_map<int, std::string> _recv_buffers;

    RequestHandlers const &_handlers;

    LogMessageQueue _log_queue;
    LogMessageQueue _output_queue;

    /// Lines @ref publish queued, each a complete JSON Lines record.
    std::mutex              _published_mutex;
    std::deque<std::string> _published;

#    ifdef __APPLE__
    DNSServiceRef _mdns_ref = nullptr;
#    endif
};

EINSUMS_NAMESPACE_END(profile)

#endif
