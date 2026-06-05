//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>

namespace einsums::hashes {

/**
 * @struct container_hash
 *
 * Hashses an arbitrary container object. Containers need to be iterable.
 *
 * Examples of allowed containers include std::array, std::vector, std::queue, etc.
 *
 * The default implementation is a PJW hash, based on the Wikipedia page.
 *
 * @tparam ContainerType The type of container.
 *
 * @versionadded{1.0.0}
 */
template <typename ContainerType>
struct ContainerHash {
  public:
    ContainerHash()          = default;
    virtual ~ContainerHash() = default;

    /**
     * Hashes the contents of a container.
     *
     * @param[in] cont The container to hash.
     * @return The container's hash.
     *
     * @versionadded{1.0.0}
     */
    size_t operator()(ContainerType const &cont) const {
        size_t hash = 0;

        // Calculate the mask. If size_t is N bytes, mask for the top N bits.
        // The first part creates a Mersenne value with the appropriate number of bits.
        // The second shifts it to the top.
        constexpr size_t mask = (((size_t)1 << sizeof(size_t)) - 1) << (7 * sizeof(size_t));

        for (auto const &val : cont) {
            auto const *bytes = reinterpret_cast<uint8_t const *>(std::addressof(val));

            for (int i = 0; i < sizeof(std::decay_t<decltype(val)>); i++) {
                hash <<= sizeof(size_t); // Shift left a number of bits equal to the number of bytes in size_t.
                hash += bytes[i];

                if ((hash & mask) != (size_t)0) {
                    hash ^= mask >> (6 * sizeof(size_t));
                    hash &= ~mask;
                }
            }
        }
        return hash;
    }
};

} // namespace einsums::hashes
