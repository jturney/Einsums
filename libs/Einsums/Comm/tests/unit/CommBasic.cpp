//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file CommBasic.cpp
/// @brief Tests for the Comm module (works with both mock and real MPI backends).

#include <Einsums/Comm/Collectives.hpp>
#include <Einsums/Comm/Communicator.hpp>
#include <Einsums/Comm/DistributionMap.hpp>
#include <Einsums/Comm/Platform.hpp>
#include <Einsums/Comm/Runtime.hpp>

#include <array>
#include <cmath>
#include <complex>
#include <numeric>
#include <span>
#include <vector>

#include <Einsums/Testing.hpp>

namespace comm = einsums::comm;

// Helper: check that a comm operation succeeded (for tests where we don't expect errors).
#define COMM_OK(expr) REQUIRE((expr).has_value())

// ═══════════════════════════════════════════════════════════════════════════════
// Platform detection
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("Comm Platform - compile-time flags consistent", "[Comm][Platform]") {
    // Either has_mpi or is_mock, never both
    CHECK(comm::has_mpi != comm::is_mock);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Runtime
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("Comm Runtime - world rank and size", "[Comm][Runtime]") {
    // On mock: rank=0, size=1. On MPI: whatever the launch config is.
    int rank = comm::world_rank();
    int size = comm::world_size();

    CHECK(rank >= 0);
    CHECK(size >= 1);
    CHECK(rank < size);
}

TEST_CASE("Comm Runtime - is_root", "[Comm][Runtime]") {
    CHECK(comm::is_root() == (comm::world_rank() == 0));
}

// ═══════════════════════════════════════════════════════════════════════════════
// Communicator
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("Communicator - world is valid", "[Comm][Communicator]") {
    auto &world = comm::Communicator::world();
    CHECK(static_cast<bool>(world));
    CHECK(world.rank() >= 0);
    CHECK(world.size() >= 1);
    CHECK(world.is_root() == (world.rank() == 0));
}

TEST_CASE("Communicator - split", "[Comm][Communicator]") {
    auto &world = comm::Communicator::world();
    auto  sub   = world.split(/*color=*/0, /*key=*/world.rank());

    CHECK(static_cast<bool>(sub));
    CHECK(sub.size() >= 1);
}

TEST_CASE("Communicator - dup", "[Comm][Communicator]") {
    auto &world = comm::Communicator::world();
    auto  copy  = world.dup();

    CHECK(static_cast<bool>(copy));
    CHECK(copy.size() == world.size());
    CHECK(copy.rank() == world.rank());
}

// ═══════════════════════════════════════════════════════════════════════════════
// Blocking collectives (mock: single rank → trivial)
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("Allreduce - double sum", "[Comm][Collectives]") {
    std::vector<double> send = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> recv(4, 0.0);

    COMM_OK(comm::allreduce<double>(send, recv, comm::ReduceOp::Sum));

    // Each rank sends the same values, so sum == nprocs * value
    int nprocs = comm::world_size();
    for (size_t i = 0; i < 4; i++) {
        CHECK(recv[i] == send[i] * nprocs);
    }
}

TEST_CASE("Allreduce inplace - float sum", "[Comm][Collectives]") {
    std::vector<float> buf    = {10.0f, 20.0f, 30.0f};
    int                nprocs = comm::world_size();

    COMM_OK(comm::allreduce_inplace<float>(buf, comm::ReduceOp::Sum));

    CHECK(buf[0] == 10.0f * nprocs);
    CHECK(buf[1] == 20.0f * nprocs);
    CHECK(buf[2] == 30.0f * nprocs);
}

TEST_CASE("Broadcast - double", "[Comm][Collectives]") {
    std::vector<double> buf = {42.0, 43.0};

    COMM_OK(comm::broadcast<double>(buf, /*root=*/0));

    CHECK(buf[0] == 42.0);
    CHECK(buf[1] == 43.0);
}

TEST_CASE("Allgather - int", "[Comm][Collectives]") {
    std::vector<int> send = {comm::world_rank()};
    std::vector<int> recv(comm::world_size(), -1);

    COMM_OK(comm::allgather<int>(send, recv));

    // Each rank[i] contributed i, so recv[i] == i
    for (int i = 0; i < comm::world_size(); i++) {
        CHECK(recv[i] == i);
    }
}

TEST_CASE("Scatter - double", "[Comm][Collectives]") {
    auto                nprocs = comm::world_size();
    int                 rank   = comm::world_rank();
    std::vector<double> send(nprocs * 2, 0.0);
    for (int i = 0; i < nprocs * 2; i++)
        send[i] = static_cast<double>(i);

    std::vector<double> recv(2, -1.0);

    COMM_OK(comm::scatter<double>(send, recv, /*root=*/0));

    // Rank r gets elements [2*r, 2*r+1]
    CHECK(recv[0] == static_cast<double>(rank * 2));
    CHECK(recv[1] == static_cast<double>(rank * 2 + 1));
}

TEST_CASE("Barrier - does not hang", "[Comm][Collectives]") {
    COMM_OK(comm::barrier());
}

// ═══════════════════════════════════════════════════════════════════════════════
// Non-blocking collectives
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("Iallreduce - async double sum", "[Comm][Collectives]") {
    std::vector<double> send = {5.0, 6.0};
    std::vector<double> recv(2, 0.0);
    int                 nprocs = comm::world_size();

    auto req_result = comm::iallreduce<double>(send, recv, comm::ReduceOp::Sum);
    REQUIRE(req_result.has_value());
    auto req = std::move(req_result.value());

    // Wait for completion
    req.wait();

    CHECK(recv[0] == 5.0 * nprocs);
    CHECK(recv[1] == 6.0 * nprocs);
}

TEST_CASE("Request - test polls correctly", "[Comm][Collectives]") {
    std::vector<double> send = {1.0};
    std::vector<double> recv(1, 0.0);

    auto req_result = comm::iallreduce<double>(send, recv, comm::ReduceOp::Sum);
    REQUIRE(req_result.has_value());
    auto req = std::move(req_result.value());

    if constexpr (comm::is_mock) {
        CHECK(req.test());
    } else {
        req.wait();
        CHECK(req.test());
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Complex type support
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("Allreduce - complex<double>", "[Comm][Collectives]") {
    using C               = std::complex<double>;
    int            nprocs = comm::world_size();
    std::vector<C> send   = {{1.0, 2.0}, {3.0, 4.0}};
    std::vector<C> recv(2);

    COMM_OK(comm::allreduce<C>(send, recv, comm::ReduceOp::Sum));

    CHECK(recv[0].real() == 1.0 * nprocs);
    CHECK(recv[0].imag() == 2.0 * nprocs);
    CHECK(recv[1].real() == 3.0 * nprocs);
    CHECK(recv[1].imag() == 4.0 * nprocs);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Additional coverage: untested functions and edge cases
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("Comm Runtime - is_initialized", "[Comm][Runtime]") {
    // Should be true since module hooks run at startup
    CHECK(comm::is_initialized());
}

TEST_CASE("Communicator - native_handle", "[Comm][Communicator]") {
    auto &world = comm::Communicator::world();
    if constexpr (comm::is_mock) {
        // Mock: native handle may be nullptr
        (void)world.native_handle(); // Just verify it doesn't crash
    } else {
        CHECK(world.native_handle() != nullptr);
    }
}

TEST_CASE("Communicator - equality", "[Comm][Communicator]") {
    auto &world = comm::Communicator::world();
    CHECK(world == world);

    auto dup = world.dup();
    // dup is a different communicator object
    // On mock, both share same impl so might be equal; platform dependent
}

TEST_CASE("Send and Recv - double", "[Comm][Collectives]") {
    int nprocs = comm::world_size();
    int rank   = comm::world_rank();

    if (nprocs < 2) {
        // Single rank (mock or mpirun -np 1): just verify no crash
        std::vector<double> buf = {10.0, 20.0, 30.0};
        COMM_OK(comm::send<double>(buf, /*dest=*/0, /*tag=*/0));
        COMM_OK(comm::recv<double>(buf, /*src=*/0, /*tag=*/0));
    } else {
        // Ring send: rank r sends to (r+1) % nprocs, receives from (r-1+nprocs) % nprocs
        int dest = (rank + 1) % nprocs;
        int src  = (rank - 1 + nprocs) % nprocs;

        std::vector<double> send_buf = {static_cast<double>(rank)};
        std::vector<double> recv_buf(1, -1.0);

        if (rank % 2 == 0) {
            COMM_OK(comm::send<double>(send_buf, dest, /*tag=*/0));
            COMM_OK(comm::recv<double>(recv_buf, src, /*tag=*/0));
        } else {
            COMM_OK(comm::recv<double>(recv_buf, src, /*tag=*/0));
            COMM_OK(comm::send<double>(send_buf, dest, /*tag=*/0));
        }

        CHECK(recv_buf[0] == static_cast<double>(src));
    }
}

TEST_CASE("Ibroadcast - non-blocking", "[Comm][Collectives]") {
    std::vector<double> buf = {42.0, 43.0};

    auto req_result = comm::ibroadcast<double>(buf, /*root=*/0);
    REQUIRE(req_result.has_value());
    req_result.value().wait();

    CHECK(buf[0] == 42.0);
    CHECK(buf[1] == 43.0);
}

TEST_CASE("Iallgather - non-blocking", "[Comm][Collectives]") {
    std::vector<int> send = {comm::world_rank()};
    std::vector<int> recv(comm::world_size(), -1);

    auto req_result = comm::iallgather<int>(send, recv);
    REQUIRE(req_result.has_value());
    req_result.value().wait();

    for (int i = 0; i < comm::world_size(); i++) {
        CHECK(recv[i] == i);
    }
}

TEST_CASE("Request - default constructor", "[Comm][Collectives]") {
    comm::Request req;
    // Default request should be testable without crashing
    CHECK(req.test());
    REQUIRE_NOTHROW(req.wait());
}

TEST_CASE("Allreduce - empty buffer", "[Comm][Collectives]") {
    std::vector<double> empty_send;
    std::vector<double> empty_recv;

    // Zero-size allreduce should not crash
    COMM_OK(comm::allreduce<double>(empty_send, empty_recv, comm::ReduceOp::Sum));
}

TEST_CASE("Broadcast - long long type", "[Comm][Collectives]") {
    std::vector<long long> buf = {9999999999LL};

    COMM_OK(comm::broadcast<long long>(buf, 0));

    CHECK(buf[0] == 9999999999LL);
}

TEST_CASE("Allreduce - ReduceOp Max", "[Comm][Collectives]") {
    // Each rank sends rank-dependent values; max picks the highest
    int                 rank = comm::world_rank();
    std::vector<double> send = {static_cast<double>(rank), 100.0 - rank, static_cast<double>(rank * rank)};
    std::vector<double> recv(3, 0.0);

    COMM_OK(comm::allreduce<double>(send, recv, comm::ReduceOp::Max));

    int nprocs = comm::world_size();
    CHECK(recv[0] == static_cast<double>(nprocs - 1));                  // max of ranks
    CHECK(recv[1] == 100.0);                                            // max of (100-rank): rank 0
    CHECK(recv[2] == static_cast<double>((nprocs - 1) * (nprocs - 1))); // max of rank^2
}

TEST_CASE("Allreduce - ReduceOp Min", "[Comm][Collectives]") {
    int                 rank = comm::world_rank();
    std::vector<double> send = {static_cast<double>(rank + 1), 100.0 + rank};
    std::vector<double> recv(2, 0.0);

    COMM_OK(comm::allreduce<double>(send, recv, comm::ReduceOp::Min));

    CHECK(recv[0] == 1.0);   // min of (rank+1): rank 0's value
    CHECK(recv[1] == 100.0); // min of (100+rank): rank 0's value
}

TEST_CASE("Allreduce - ReduceOp Prod", "[Comm][Collectives]") {
    int                 nprocs = comm::world_size();
    std::vector<double> send   = {2.0, 3.0, 4.0};
    std::vector<double> recv(3, 0.0);

    COMM_OK(comm::allreduce<double>(send, recv, comm::ReduceOp::Prod));

    // All ranks send same values, so prod = value^nprocs
    CHECK(recv[0] == std::pow(2.0, nprocs));
    CHECK(recv[1] == std::pow(3.0, nprocs));
    CHECK(recv[2] == std::pow(4.0, nprocs));
}

TEST_CASE("Allreduce - complex<float>", "[Comm][Collectives]") {
    using CF               = std::complex<float>;
    int             nprocs = comm::world_size();
    std::vector<CF> send   = {{1.0f, 2.0f}};
    std::vector<CF> recv(1);

    COMM_OK(comm::allreduce<CF>(send, recv, comm::ReduceOp::Sum));

    CHECK(recv[0].real() == 1.0f * nprocs);
    CHECK(recv[0].imag() == 2.0f * nprocs);
}

// ═══════════════════════════════════════════════════════════════════════════════
// DistributionMap
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("DistributionMap - auto replicate small tensor", "[Comm][DistributionMap]") {
    // 10x10 doubles = 800 bytes, well below 64MB threshold
    auto map = comm::auto_distribute<2>({10, 10}, sizeof(double));

    CHECK(map.is_replicated());
    CHECK(map.strategy[0] == comm::DistStrategy::Replicated);
    CHECK(map.strategy[1] == comm::DistStrategy::Replicated);
    CHECK(map.global_dims[0] == 10);
    CHECK(map.global_dims[1] == 10);
}

TEST_CASE("DistributionMap - auto distribute large tensor", "[Comm][DistributionMap]") {
    // Force distribution with a tiny threshold
    auto map = comm::auto_distribute<2>({100, 100}, sizeof(double), /*threshold=*/100);

    if (comm::world_size() > 1) {
        CHECK_FALSE(map.is_replicated());
    } else {
        // Single rank: everything is replicated regardless
        CHECK(map.is_replicated());
    }
}

TEST_CASE("DistributionMap - local_range for blocked", "[Comm][DistributionMap]") {
    comm::DistributionMap<2> map;
    map.global_dims = {100, 50};
    map.strategy    = {comm::DistStrategy::Blocked, comm::DistStrategy::Replicated};
    map.block_size  = {25, 50};
    map.num_ranks   = 4;

    auto [s0, e0] = map.local_range(0, 0);
    CHECK(s0 == 0);
    CHECK(e0 == 25);

    auto [s1, e1] = map.local_range(0, 1);
    CHECK(s1 == 25);
    CHECK(e1 == 50);

    auto [s3, e3] = map.local_range(0, 3);
    CHECK(s3 == 75);
    CHECK(e3 == 100);

    // Replicated dimension: always full range
    auto [sr, er] = map.local_range(1, 0);
    CHECK(sr == 0);
    CHECK(er == 50);
}

TEST_CASE("DistributionMap - local_dims", "[Comm][DistributionMap]") {
    comm::DistributionMap<3> map;
    map.global_dims = {120, 80, 60};
    map.strategy    = {comm::DistStrategy::Blocked, comm::DistStrategy::Replicated, comm::DistStrategy::Replicated};
    map.block_size  = {30, 80, 60};
    map.num_ranks   = 4;

    auto dims_0 = map.local_dims(0);
    CHECK(dims_0[0] == 30);
    CHECK(dims_0[1] == 80);
    CHECK(dims_0[2] == 60);

    auto dims_3 = map.local_dims(3);
    CHECK(dims_3[0] == 30);
}
