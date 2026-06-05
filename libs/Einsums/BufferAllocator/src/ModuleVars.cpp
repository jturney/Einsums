//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/BufferAllocator/ModuleVars.hpp>
#include <Einsums/Logging.hpp>
#include <Einsums/Print.hpp>
#include <Einsums/StringUtil/MemoryString.hpp>

#include <omp.h>

namespace einsums::detail {

EINSUMS_SINGLETON_IMPL(Einsums_BufferAllocator_vars)

void Einsums_BufferAllocator_vars::update_max_size(ConfigMappingType<std::string> const &options) {
    auto       &singleton = get_singleton();
    auto        lock      = std::lock_guard(singleton);
    auto const &value     = options.at("buffer-size");

    singleton._max_size = string_util::memory_string(value);

    if (singleton._max_size == 0) {
        singleton._max_size = std::allocator_traits<std::allocator<uint8_t>>::max_size(std::allocator<uint8_t>());
    }

    auto const &work_buffer = options.at("work-buffer-size");

    singleton._work_buffer = string_util::memory_string(work_buffer);

    if (singleton._work_buffer == 0) {
        singleton._work_buffer = singleton._max_size / 8 / omp_get_num_threads();
    }
}

size_t Einsums_BufferAllocator_vars::get_max_size() const {
    return _max_size;
}

void Einsums_BufferAllocator_vars::set_max_size(size_t val) {
    _max_size = val;
}

bool Einsums_BufferAllocator_vars::request_bytes(size_t bytes) {
    auto lock = std::lock_guard(_lock);

    if (bytes + _curr_size > _max_size) {
        EINSUMS_LOG_WARN("An allocator attempted to request too much memory! Einsums currently has {} bytes allocated to it, of which {} "
                         "are in use. The requested allocation would require {} more bytes for a total of {} bytes.",
                         _max_size, _curr_size, bytes, bytes + _curr_size);
        return false;
    }

    _curr_size += bytes;
    return true;
}

void Einsums_BufferAllocator_vars::release_bytes(size_t bytes) {
    auto lock = std::lock_guard(_lock);

    if (bytes > _curr_size) {
        _curr_size = 0;
    } else {
        _curr_size -= bytes;
    }
}

size_t Einsums_BufferAllocator_vars::get_available() const {
    if (_curr_size > _max_size) {
        return 0;
    }
    return _max_size - _curr_size;
}

size_t Einsums_BufferAllocator_vars::get_work_buffer_size() const {
    return _work_buffer;
}

} // namespace einsums::detail