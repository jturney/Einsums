//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/BufferAllocator/InitModule.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/TypeSupport/Lockable.hpp>
#include <Einsums/TypeSupport/Singleton.hpp>

#include <source_location>

namespace einsums {
namespace detail {

/**
 * Contains variables for the BufferAllocator module.
 *
 * @versionadded{1.1.0}
 */
class EINSUMS_EXPORT Einsums_BufferAllocator_vars final : public design_pats::Lockable<std::recursive_mutex> {
    EINSUMS_SINGLETON_DEF(Einsums_BufferAllocator_vars)

  public:
    // Put module-global variables here.

    /**
     * @brief Requests a number of bytes from the counter.
     *
     * @versionadded{1.1.0}
     */
    bool request_bytes(size_t bytes);

    /**
     * @brief Releases a number of bytes counter.
     *
     * @versionadded{1.1.0}
     */
    void release_bytes(size_t bytes);

    /**
     * @brief Update the maximum size of the counter.
     *
     * @versionadded{1.1.0}
     */
    static void update_max_size(ConfigMappingType<std::string> const &options);

    /**
     * @brief Get the maximum size of the counter.
     *
     * @versionadded{1.1.0}
     */
    size_t get_max_size() const;

    /**
     * @brief Get the size for a work buffer.
     *
     * @versionadded{2.0.0}
     */
    size_t get_work_buffer_size() const;

    /**
     * @brief Set the maximum size of the counter.
     */
    void set_max_size(size_t val);

    /**
     * @brief Get available bytes.
     *
     * @versionadded{1.1.0}
     */
    size_t get_available() const;

    ~Einsums_BufferAllocator_vars() { ; }

  private:
    explicit Einsums_BufferAllocator_vars() = default;

    /**
     * The maximum number of bytes available to the buffers.
     *
     * @versionadded{1.1.0}
     */
    size_t _max_size{4294967296};

    /**
     * The current number of bytes allocated by Einsums.
     *
     * @versionadded{1.1.0}
     */
    size_t _curr_size{0};

    /**
     * The recommended size for a work buffer.
     *
     * @versionadded{2.0.0}
     */
    size_t _work_buffer{1024};

    /**
     * Mutex for avoiding race conditions.
     *
     * @versionadded{1.1.0}
     */
    std::mutex _lock;
};

} // namespace detail
} // namespace einsums