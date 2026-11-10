//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/BufferAllocator/MemoryPool.hpp>
#include <Einsums/Config/Namespace.hpp>

#include <cstddef>
#include <new>

EINSUMS_NAMESPACE_BEGIN(compute_graph::detail)

/// One shared, 64-byte-aligned block per graph, in which MemoryPlanning places the graph's planned
/// intermediates. The graph holds it (Graph::arena), and the attach executors built from the
/// planned Materialize nodes hold it too, so it outlives a re-plan that replaces it while nodes
/// built against it remain, and every replay reuses it verbatim.
struct ArenaBlock {
    explicit ArenaBlock(std::size_t bytes) : size(bytes), data(static_cast<std::byte *>(memory::aligned_alloc(bytes))) {
        if (data == nullptr && bytes != 0) {
            throw std::bad_alloc{};
        }
    }
    ~ArenaBlock() { memory::aligned_free(data); }
    ArenaBlock(ArenaBlock const &)            = delete;
    ArenaBlock &operator=(ArenaBlock const &) = delete;
    ArenaBlock(ArenaBlock &&)                 = delete;
    ArenaBlock &operator=(ArenaBlock &&)      = delete;

    std::size_t size;
    std::byte  *data;
};

EINSUMS_NAMESPACE_END(compute_graph::detail)
