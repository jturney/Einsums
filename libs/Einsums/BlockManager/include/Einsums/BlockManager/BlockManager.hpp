//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/BufferAllocator/BufferAllocator.hpp>
#include <Einsums/TypeSupport/Lockable.hpp>
#include <Einsums/TypeSupport/Singleton.hpp>

#include <mutex>
#include <stdexcept>

namespace einsums {

struct EINSUMS_EXPORT BlockManager final : public design_pats::Lockable<std::mutex> {
    EINSUMS_SINGLETON_DEF(BlockManager)
  public:
    ~BlockManager() = default;
    using BlockType = std::shared_ptr<uint8_t[]>;

    std::weak_ptr<uint8_t[]> request_block(size_t bytes);

  private:
    BlockManager() = default;

    BufferList<BlockType> _block_list;

    BufferAllocator<uint8_t> _alloc;
};

} // namespace einsums