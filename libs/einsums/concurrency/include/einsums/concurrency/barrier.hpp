//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <condition_variable>
#include <cstddef>
#include <mutex>

namespace einsums::concurrency::detail {

struct EINSUMS_EXPORT barrier {

    barrier(std::size_t number_of_threads);
    ~barrier();

    void wait();

  private:
    using mutex_type = std::mutex;

    static constexpr std::size_t barrier_flag = static_cast<std::size_t>(1) << (CHAR_BIT * sizeof(std::size_t) - 1);

    std::size_t const _number_of_threads;
    std::size_t       _total;

    mutable mutex_type      _mtx;
    std::condition_variable _cond;
};

} // namespace einsums::concurrency::detail