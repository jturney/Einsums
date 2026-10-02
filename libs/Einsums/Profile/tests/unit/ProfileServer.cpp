//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/Profile/Consumer.hpp>
#include <Einsums/Profile/LogSink.hpp>
#include <Einsums/Profile/Profile.hpp>
#include <Einsums/Profile/RequestHandlers.hpp>
#include <Einsums/Profile/Server.hpp>
#include <Einsums/Profile/StringTable.hpp>

#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>

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

using namespace einsums::profile;

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

    uint16_t const port = free_port();
    prof.start_server(port);
    REQUIRE(prof.server() != nullptr);
    if (!prof.server()->is_running()) {
        // Another case started the singleton's server first; this one cannot pick the port.
        SKIP("the profiler's server already runs on another port");
    }

    int const         client   = connect_to(port);
    std::string const response = request(client, "test_early_handler");
    CHECK(response.find(R"("data":{"answer":42})") != std::string::npos);

    prof.unregister_handler("test_early_handler");
    CHECK(request(client, "test_early_handler").find("unknown method") != std::string::npos);
    ::close(client);
}

TEST_CASE("A published message reaches a connected viewer with its type", "[profiler][server]") {
    StringTable     strings;
    Consumer        consumer(strings);
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
    Consumer        consumer(strings);
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
    CHECK(contents.str().find(R"("test_section": [1,2,3])") != std::string::npos);
}

// Regression: shutdown() drained only to clients the server had already accepted, so one still
// in the listen backlog - a client that connected after the last tick, which for a program that
// finishes between two ticks is every client - received nothing, and `einsums bench run` lost
// the benchmark results of every short performance test.
TEST_CASE("Server shutdown delivers queued results to a client it has not yet accepted", "[profiler][server]") {
    StringTable     strings;
    Consumer        consumer(strings);
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
    Consumer        consumer(strings);
    RequestHandlers handlers;
    Server          server(consumer, strings, handlers, "127.0.0.1", 19216);
    REQUIRE_FALSE(server.is_running());
}
#endif
