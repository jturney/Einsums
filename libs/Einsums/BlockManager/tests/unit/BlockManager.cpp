//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/BlockManager/BlockManager.hpp>
#include <Einsums/BufferAllocator/BufferAllocator.hpp>
#include <Einsums/BufferAllocator/ModuleVars.hpp>
#include <Einsums/Config/Types.hpp>

#include <mutex>

#include <Einsums/Testing.hpp>

TEST_CASE("Block manager") {
    using namespace einsums;

    // Set the max buffer size to 4MB.
    {
        auto &singleton = GlobalConfigMap::get_singleton();
        auto  lock      = std::lock_guard(singleton);
        singleton.set_string("gpu_buffer_size", "4MB");
        singleton.set_string("buffer_size", "4MB");
    }

    // Get the manager.
    auto &manager = BlockManager::get_singleton();

    // Now, allocate a block.
    auto first_weak_block = manager.request_block(65536);

    {
        // Hold onto the first block.
        auto first_block = first_weak_block.lock();

        // Now, go through and keep allocating a bunch of 64kB blocks. Stale blocks should be removed, so this should never throw.
        for (size_t i = 0; i < 1024; i++) {
            REQUIRE_NOTHROW(manager.request_block(65536));
        }
    }

    // Finally, check to make sure that the first block was never freed.
    REQUIRE(!first_weak_block.expired());

    // And check to make sure that the throwing mechanism works as well. Request 2GB.
    REQUIRE_THROWS(manager.request_block(2147483648));
}
