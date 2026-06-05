//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/BlockManager/BlockManager.hpp>

#include <memory>
#include <thread>

namespace einsums {
using std::weak_ptr;

EINSUMS_SINGLETON_IMPL(BlockManager);

// NOLINTNEXTLINE(modernize-avoid-c-arrays)
std::weak_ptr<uint8_t[]> BlockManager::request_block(size_t bytes) {
    // Try to clean up stale blocks.
    for (int i = 0; i < 10 && _alloc.available_size() < bytes; i++) {
        bool erased = false;
        {
            std::scoped_lock const guard(*this);

            for (auto it = _block_list.begin(); it != _block_list.end(); it++) {
                if (it->use_count() == 1) {
                    _block_list.erase(it);
                    erased = true;
                    break;
                }
            }
        }

        // Wait a bit to see if any blocks are freed up.
        if (!erased) {
            std::this_thread::yield();
        }
    }

    if (_alloc.available_size() < bytes) {
        EINSUMS_THROW_EXCEPTION(std::runtime_error,
                                "Could not allocate enough memory for the requested block, even after stale blocks were removed! "
                                "Requested {} bytes, but there are only {} bytes available.",
                                bytes, _alloc.available_size());
    }

    std::scoped_lock const guard(*this);

    _block_list.push_back(std::allocate_shared_for_overwrite<uint8_t[]>(_alloc, bytes)); // NOLINT(modernize-avoid-c-arrays)

    return {_block_list.back()};
}

} // namespace einsums