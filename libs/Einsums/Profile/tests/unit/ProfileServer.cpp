//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/Profile/Profile.hpp>

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <waggle/Consumer.hpp>
#include <waggle/LogQueue.hpp>
#include <waggle/RequestHandlers.hpp>
#include <waggle/Server.hpp>
#include <waggle/StringTable.hpp>

#ifndef _WIN32
#    include <arpa/inet.h>
#    include <netinet/in.h>
#    include <sys/ioctl.h>
#    include <sys/socket.h>
#    include <unistd.h>
#    ifdef __linux__
#        include <linux/sockios.h>
#    endif
#endif

using namespace waggle;

#ifndef _WIN32
namespace {

/// A port nothing is listening on (the kernel's pick; the Server takes no port 0).
uint16_t free_port() {
    int const   fd = ::socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    ::bind(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr));
    socklen_t len = sizeof(addr);
    ::getsockname(fd, reinterpret_cast<sockaddr *>(&addr), &len);
    ::close(fd);
    return ntohs(addr.sin_port);
}

int connect_to(uint16_t port) {
    int const   fd = ::socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_port        = htons(port);
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    REQUIRE(::connect(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) == 0);
    return fd;
}

/// Bytes sent on `fd` that the peer's kernel has not yet acknowledged.
int unacknowledged_bytes(int fd) {
    int n = 0;
#    ifdef __APPLE__
    socklen_t len = sizeof(n);
    REQUIRE(::getsockopt(fd, SOL_SOCKET, SO_NWRITE, &n, &len) == 0);
#    else
    REQUIRE(::ioctl(fd, SIOCOUTQ, &n) == 0);
#    endif
    return n;
}

/// Returns once the server's kernel holds the connection in its listen backlog. connect() returns
/// when the client has the SYN-ACK, which can be before the server has processed the final ACK
/// (macOS hands loopback input to a separate thread), and until then accept() finds nothing. A
/// byte the server's kernel acknowledged arrived after that ACK, so it proves the connection is
/// queued, and the kernel acknowledges it without anyone calling accept().
void wait_until_queued(int fd) {
    REQUIRE(::send(fd, "\n", 1, 0) == 1); // an empty request line, which the server skips
    auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (unacknowledged_bytes(fd) > 0) {
        REQUIRE(std::chrono::steady_clock::now() < deadline);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

/// Send one request line and return the response to it, waiting up to 10 s for the consumer's next
/// tick (every ~500 ms) to answer. Snapshot and other lines streamed meanwhile are skipped.
std::string request(int fd, std::string const &method) {
    std::string const line = R"({"type":"request","id":"t1","method":")" + method + R"(","params":{}})" + "\n";
    REQUIRE(::send(fd, line.data(), line.size(), 0) == static_cast<ssize_t>(line.size()));

    timeval timeout{};
    timeout.tv_sec = 10;
    ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));

    std::string received;
    char        buffer[4096];
    while (true) {
        for (size_t start = 0, end; (end = received.find('\n', start)) != std::string::npos; start = end + 1) {
            std::string const record = received.substr(start, end - start);
            if (record.find(R"("type":"response")") != std::string::npos && record.find(R"("id":"t1")") != std::string::npos) {
                return record;
            }
        }
        ssize_t const n = ::recv(fd, buffer, sizeof(buffer), 0);
        REQUIRE(n > 0); // a timeout or a closed connection fails here
        received.append(buffer, static_cast<size_t>(n));
    }
}

} // namespace

// The handler table used to belong to the server, and TaskPool and ComputeGraph registered only if
// a server already existed when they first ran: one started later, or a profiler constructed
// before the options were read, left the viewer's TaskPool and graph panels empty for good. The
// profiler owns the table now, and a server started at any time answers from it.
TEST_CASE("A handler registered before the server starts is answered", "[profiler][server]") {
    auto &prof = Profiler::instance();
    prof.register_handler("test_early_handler", [](std::string const &) { return std::string(R"({"answer":42})"); });

    prof.start_server(free_port()); // a no-op if another case started it first
    REQUIRE(prof.server() != nullptr);
    REQUIRE(prof.server()->is_running());

    int const         client   = connect_to(prof.server()->port());
    std::string const response = request(client, "test_early_handler");
    CHECK(response.find(R"("data":{"answer":42})") != std::string::npos);

    prof.unregister_handler("test_early_handler");
    CHECK(request(client, "test_early_handler").find("unknown method") != std::string::npos);
    ::close(client);
}

TEST_CASE("A published message reaches a connected viewer with its type", "[profiler][server]") {
    StringTable     strings;
    SiteTable       sites;
    Consumer        consumer(strings, sites);
    RequestHandlers handlers;
    uint16_t const  port = free_port();
    Server          server(consumer, strings, handlers, "127.0.0.1", port);
    REQUIRE(server.is_running());

    server.publish("test_event", R"({"value":7,"label":"x"})");
    server.publish("test_empty", "{}");

    int const client = connect_to(port);
    wait_until_queued(client);
    server.shutdown();

    std::string received;
    char        buffer[4096];
    for (ssize_t n; (n = ::recv(client, buffer, sizeof(buffer), 0)) > 0;) {
        received.append(buffer, static_cast<size_t>(n));
    }
    ::close(client);

    INFO("received: " << received);
    CHECK(received.find(R"({"type":"test_event","value":7,"label":"x"})") != std::string::npos);
    CHECK(received.find(R"({"type":"test_empty"})") != std::string::npos);
}

TEST_CASE("A session file embeds every registered section", "[profiler][server]") {
    StringTable     strings;
    SiteTable       sites;
    Consumer        consumer(strings, sites);
    RequestHandlers handlers;
    handlers.add_session_section("test_section", [] { return std::string(R"([1,2,3])"); });
    Server server(consumer, strings, handlers, "127.0.0.1", free_port());

    auto const path = std::filesystem::temp_directory_path() / "einsums_profile_session_sections.json";
    std::filesystem::remove(path);
    server.export_session(path.string(), "sections");
    server.shutdown();

    std::ifstream     in(path);
    std::stringstream contents;
    contents << in.rdbuf();
    std::filesystem::remove(path);
    std::string const text = contents.str();
    CHECK(text.find(R"("format": "waggle-session")") != std::string::npos);
    CHECK(text.find(R"("version": 1)") != std::string::npos);
    // A library's data sits under "extensions", by its namespaced key, not beside the profiler's.
    auto const extensions = text.find(R"("extensions": {)");
    REQUIRE(extensions != std::string::npos);
    CHECK(text.find(R"("test_section": [1,2,3])", extensions) != std::string::npos);
}

/// Reads a server's JSON Lines stream, keeping what arrived past the line it returned.
class LineReader {
  public:
    explicit LineReader(int fd) : _fd(fd) {
        timeval timeout{};
        timeout.tv_sec = 1;
        ::setsockopt(_fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    }

    /// The next line holding every string in @p wanted, skipping others; fails after 10 s. The
    /// deadline is overall, since a server streaming snapshots never lets a per-recv timeout expire.
    std::string next(std::initializer_list<std::string_view> wanted) {
        auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (true) {
            for (size_t end; (end = _buffer.find('\n')) != std::string::npos;) {
                std::string record = _buffer.substr(0, end);
                _buffer.erase(0, end + 1);
                if (std::ranges::all_of(wanted, [&](std::string_view w) { return record.find(w) != std::string::npos; })) {
                    return record;
                }
            }
            REQUIRE(std::chrono::steady_clock::now() < deadline);
            char          chunk[4096];
            ssize_t const n = ::recv(_fd, chunk, sizeof(chunk), 0);
            if (n > 0) {
                _buffer.append(chunk, static_cast<size_t>(n));
            } else {
                REQUIRE((n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK))); // a timeout, not a closed connection
            }
        }
    }

  private:
    int         _fd;
    std::string _buffer;
};

TEST_CASE("The meta message lists every client", "[profiler][server]") {
    StringTable     strings;
    SiteTable       sites;
    Consumer        consumer(strings, sites);
    RequestHandlers handlers;
    handlers.add_client({.name = "first-lib", .version = "1.2.3", .git_commit = "abc123"});
    handlers.add_client({.name = "second-lib", .version = "0.1"});
    handlers.add("zeta_method", [](std::string const &) { return std::string("{}"); });
    handlers.add("alpha_method", [](std::string const &) { return std::string("{}"); });
    uint16_t const port = free_port();
    Server         server(consumer, strings, handlers, "127.0.0.1", port);
    REQUIRE(server.is_running());
    CHECK(server.port() == port);

    int const client = connect_to(port);
    wait_until_queued(client);
    server.tick(); // accepts the viewer and sends it the meta line

    LineReader        reader(client);
    std::string const meta = reader.next({R"("type":"meta")"});
    CHECK(meta.find(R"({"name":"first-lib","version":"1.2.3","git_commit":"abc123")") != std::string::npos);
    CHECK(meta.find(R"({"name":"second-lib","version":"0.1")") != std::string::npos);
    // The first client's build stays at the top level, where viewers read it.
    CHECK(meta.find(R"("git_commit":"abc123","git_branch":"","git_dirty":false,"build_type":"","clients":[)") != std::string::npos);
    // What the viewer may ask this program, sorted, so it shows only the panels it can fill.
    CHECK(meta.find(R"("handlers":["alpha_method","zeta_method"])") != std::string::npos);
    ::close(client);
    server.shutdown();
}

// Einsums wired its log sink and println forwarding only if a server existed while logging was
// set up, so a server started later showed an empty log panel. Both now go through the profiler,
// which forwards to whatever server runs when the message arrives.
TEST_CASE("Log messages and program output reach a server started later", "[profiler][server]") {
    auto &prof = Profiler::instance();
    prof.start_server(free_port()); // a no-op if an earlier case started it
    REQUIRE(prof.server() != nullptr);
    REQUIRE(prof.server()->is_running());

    int const client = connect_to(prof.server()->port());
    // Messages go to the viewers connected when the server next ticks, so wait until the consumer's
    // tick has accepted this one: its meta line says so.
    LineReader reader(client);
    (void)reader.next({R"("type":"meta")"});
    prof.log(3, std::chrono::system_clock::now(), "/some/dir/source.cpp", 42, "a_function", "late-server log line");
    prof.output("late-server output line");

    std::string const log = reader.next({R"("type":"log")", "late-server log line"});
    CHECK(log.find(R"("level":3)") != std::string::npos);
    CHECK(log.find(R"("file":"source.cpp")") != std::string::npos); // the basename
    CHECK(log.find(R"("line":42)") != std::string::npos);
    CHECK_FALSE(reader.next({R"("type":"output")", "late-server output line"}).empty());
    ::close(client);
}

// Regression: shutdown() drained only to clients the server had already accepted, so one still
// in the listen backlog - a client that connected after the last tick, which for a program that
// finishes between two ticks is every client - received nothing, and `einsums bench run` lost
// the benchmark results of every short performance test.
TEST_CASE("Server shutdown delivers queued results to a client it has not yet accepted", "[profiler][server]") {
    StringTable     strings;
    SiteTable       sites;
    Consumer        consumer(strings, sites);
    RequestHandlers handlers;
    uint16_t const  port = free_port();
    Server          server(consumer, strings, handlers, "127.0.0.1", port);
    REQUIRE(server.is_running());

    server.publish("benchmark_result", R"({"label":"short-test N=8","metric":"t_einsum","value_us":12.5})");

    int const client = connect_to(port); // connected, but no tick() has run to accept it
    wait_until_queued(client);
    server.shutdown();

    std::string received;
    char        buffer[4096];
    for (ssize_t n; (n = ::recv(client, buffer, sizeof(buffer), 0)) > 0;) {
        received.append(buffer, static_cast<size_t>(n));
    }
    ::close(client);

    INFO("received: " << received);
    REQUIRE(received.find(R"("type":"benchmark_result")") != std::string::npos);
    REQUIRE(received.find(R"("label":"short-test N=8")") != std::string::npos);
}
#else
// Placeholder: the server has no Winsock implementation, so it never listens on Windows. Once it
// does, the POSIX case above (with Winsock socket calls in its helpers) replaces this one.
TEST_CASE("Server does not listen on Windows", "[profiler][server]") {
    StringTable     strings;
    SiteTable       sites;
    Consumer        consumer(strings, sites);
    RequestHandlers handlers;
    Server          server(consumer, strings, handlers, "127.0.0.1", 19216);
    REQUIRE_FALSE(server.is_running());
}
#endif
