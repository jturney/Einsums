//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/**
 * @file BufferAllocator.hpp
 */

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/BufferAllocator/ModuleVars.hpp>
#include <Einsums/Errors.hpp>

#include <complex>
#include <cstdlib>
#include <deque>
#include <forward_list>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_set>

#if defined(EINSUMS_HAVE_TRACY)
#    include <tracy/Tracy.hpp>
#endif

namespace einsums {

namespace detail {

EINSUMS_EXPORT void *allocate(size_t n);
EINSUMS_EXPORT void  deallocate(void *);

} // namespace detail
/**
 * @struct BufferAllocator
 *
 * @brief Allocator whose maximum size can be restricted by runtime variables.
 *
 * The maximum size is controlled by the global configuration options. In particular, the option to control the size is
 * @c --einsums:buffer-size and the string in the configuration mapping is @c buffer-size with the standard variants,
 * such as @c BUFFER_SIZE .
 *
 * Use in dynamic allocations with frequent resizing is discouraged, such as in a vector. Resizing requires the buffer
 * to be reallocated while still being allocated. If the buffers are big when resized, this means that the buffer
 * allocation will fail even though the result would be expected to fall within the range of acceptable sizes.
 * However, when using this to allocate a buffer once, this should be fine.
 *
 * This allocator follows the C++ standard for allocators, and can be used in templates that use them, such as
 * containers, smart pointers, and more.
 *
 * @tparam T The data type returned by this allocator.
 *
 * @versionadded{1.1.0}
 */
template <typename T>
struct BufferAllocator {
    /**
     * @typedef pointer
     *
     * Represents the type of pointer handled by this allocator.
     *
     * @versionadded{1.1.0}
     */
    using pointer = T *; // NOLINT(readability-identifier-naming)

    /**
     * @typedef const_pointer
     *
     * Represents the type of pointer handled by this allocator, but for const operations.
     *
     * @versionadded{1.1.0}
     */
    using const_pointer = T const *; // NOLINT(readability-identifier-naming)

    /**
     * @typedef void_pointer
     *
     * Represents the type of typeless pointer handled by this allocator.
     *
     * @versionadded{1.1.0}
     */
    using void_pointer = void *; // NOLINT(readability-identifier-naming)

    /**
     * @typedef const_void_pointer
     *
     * Represents the type of typeless pointer handled by this allocator, but for const operations.
     *
     * @versionadded{1.1.0}
     */
    using const_void_pointer = void const *; // NOLINT(readability-identifier-naming)

    /**
     * @typedef value_type
     *
     * The type of buffers this allocator makes.
     *
     * @versionadded{1.1.0}
     */
    using value_type = T; // NOLINT(readability-identifier-naming)

    /**
     * @typedef size_type
     *
     * The type used to represent sizes for this buffer.
     *
     * @versionadded{1.1.0}
     */
    using size_type = size_t; // NOLINT(readability-identifier-naming)

    /**
     * @typedef difference_type
     *
     * The type used to represent address offsets.
     *
     * @versionadded{1.1.0}
     */
    using difference_type = ptrdiff_t; // NOLINT(readability-identifier-naming)

    /**
     * @typedef is_always_equal
     *
     * Flag that indicates that all allocators of this type are considered to be equal.
     *
     * @versionadded{1.1.0}
     */
    using is_always_equal = std::true_type; // NOLINT(readability-identifier-naming)

    /**
     * @property type_size
     *
     * The size of the types handled by this allocator. This is needed to handle void pointers, which traditionally
     * don't have a size, but should be treated as one byte.
     *
     * @versionadded{1.1.0}
     */
    constexpr static size_t type_size = sizeof(std::conditional_t<std::is_void_v<std::remove_cv_t<T>>, char, T>);

    constexpr BufferAllocator() noexcept = default;

    constexpr BufferAllocator(BufferAllocator const &) noexcept = default;

    template <typename U>
    constexpr BufferAllocator(BufferAllocator<U> const &) noexcept {}

    constexpr BufferAllocator(BufferAllocator &&) = default;

    template <typename U>
    constexpr BufferAllocator(BufferAllocator<U> &&) {}

    constexpr BufferAllocator &operator=(BufferAllocator const &) noexcept = default;

    template <typename U>
    constexpr BufferAllocator &operator=(BufferAllocator<U> const &) noexcept {
        return *this;
    }

    constexpr BufferAllocator &operator=(BufferAllocator &&) = default;

    template <typename U>
    constexpr BufferAllocator &operator=(BufferAllocator<U> &&) {
        return *this;
    }

    /**
     * @brief Allocate an array of values.
     *
     * This does not initialize the values in the array. If given zero, this will return the null pointer.
     *
     * @param n The number of elements in the array.
     *
     * @return A pointer to the newly allocated memory.
     *
     * @throws std::runtime_error When the allocation size is too large or the allocation returns an unexpected null pointer.
     *
     * @versionadded{1.1.0}
     */
    pointer allocate(size_type n) {
        if (n == 0) {
            return nullptr;
        }

        pointer out;

        if (!reserve(n)) {
            EINSUMS_THROW_EXCEPTION(std::runtime_error,
                                    "Could not allocate enough memory for buffers. Requested {} elements or {} bytes, but only {} bytes "
                                    "available out of {} bytes maximum.",
                                    n, n * type_size, available_size(), max_size());
        }

        out = static_cast<pointer>(detail::allocate(n * type_size));
        if (out == nullptr) {
            EINSUMS_THROW_EXCEPTION(
                std::runtime_error,
                "Could not allocate enough memory for buffers. Requested {} elements or {} bytes, but malloc returned a null pointer.", n,
                n * type_size);
        }

#if defined(EINSUMS_HAVE_TRACY)
        TracyAlloc(out, n * type_size);
#endif

        return out;
    }

    /**
     * @brief Reserve a number of elements without allocating.
     *
     * This function is used when you want to allocate some memory with some other allocator, such as the
     * default allocator, but still want to track and limit the memory being used.
     *
     * @param n The number of elements to reserve.
     *
     * @return True if the allocation will be successful, false if it will fail.
     *
     * @versionadded{1.1.0}
     */
    [[nodiscard("This function tells you when an allocation fails due to being out of memory. Don't ignore its return value. It is bad "
                "form.")]] bool
    reserve(size_type n) {
        if (n == 0) {
            return true;
        }

        try {
            auto &vars = detail::Einsums_BufferAllocator_vars::get_singleton();

            return vars.request_bytes(n * type_size);
        } catch (std::runtime_error &) {
            return false;
        }
    }

    /**
     * @brief Release a number of elements without freeing.
     *
     * This function is used when you want to deallocate some memory with some other allocator, such as the
     * default allocator, but still want to track the memory being used.
     *
     * @param n The number of elements to release.
     *
     * @versionadded{1.1.0}
     */
    void release(size_type n) {
        if (n == 0) {
            return;
        }

        try {
            auto &vars = detail::Einsums_BufferAllocator_vars::get_singleton();
            vars.release_bytes(n * type_size);
        } catch (std::runtime_error &) {
            return;
        }
    }

    /**
     * @brief Deallocate a number of elements.
     *
     * @param p The pointer to free.
     * @param n The number of elements the pointer points to.
     *
     * @versionadded{1.1.0}
     */
    void deallocate(pointer p, size_type n) {
        release(n);

        if (p != nullptr) {
#if defined(EINSUMS_HAVE_TRACY)
            TracyFree(p);
#endif
            detail::deallocate(p);
        }
    }

    /**
     * @brief Query the maximum number of elements the allocator can accept.
     *
     * This will return the number of elements as determined by the global buffer size limit.
     *
     * @return The number of elements specified by the buffer size option.
     *
     * @versionadded{1.1.0}
     */
    [[nodiscard]] size_type max_size() const {
        try {
            return detail::Einsums_BufferAllocator_vars::get_singleton().get_max_size() / type_size;
        } catch (std::runtime_error &) {
            return 0;
        }
    }

    /**
     * @brief Query the number of elements the allocator has free.
     *
     * This will return the number of elements that have not yet been allocated.
     *
     * @return The number of elements available to allocate.
     *
     * @versionadded{1.1.0}
     */
    [[nodiscard]] size_type available_size() const {

        try {
            return detail::Einsums_BufferAllocator_vars::get_singleton().get_available() / type_size;

        } catch (std::runtime_error &) {
            return 0;
        }
    }

    /**
     * @brief Query the recommended work buffer size.
     *
     * This will give the recommended size of a work buffer.
     *
     * @versionadded{2.0.0}
     */
    [[nodiscard]] size_type work_buffer_size() const {
        try {
            return detail::Einsums_BufferAllocator_vars::get_singleton().get_work_buffer_size() / type_size;
        } catch (std::runtime_error &) {
            return 0;
        }
    }

    /**
     * @brief Test whether two buffer allocators are the same.
     *
     * All buffer allocators are considered to be the same, so this will always return true.
     *
     * @param other The allocator to compare to.
     * @return Always returns true.
     *
     * @versionadded{1.1.0}
     */
    constexpr bool operator==(BufferAllocator<T> const & /*other*/) const { return true; }

    /**
     * @brief Test whether two buffer allocators are not the same.
     *
     * All buffer allocators are considered to be the same, so this will always return false.
     *
     * @param other The allocator to compare to.
     * @return Always returns false.
     *
     * @versionadded{1.1.0}
     */
    constexpr bool operator!=(BufferAllocator<T> const & /*other*/) const { return false; }
};

#ifndef EINSUMS_WINDOWS

extern template struct EINSUMS_EXPORT BufferAllocator<void>;
extern template struct EINSUMS_EXPORT BufferAllocator<signed char>;
extern template struct EINSUMS_EXPORT BufferAllocator<signed short>;
extern template struct EINSUMS_EXPORT BufferAllocator<signed int>;
extern template struct EINSUMS_EXPORT BufferAllocator<signed long>;
extern template struct EINSUMS_EXPORT BufferAllocator<signed long long>;
extern template struct EINSUMS_EXPORT BufferAllocator<unsigned char>;
extern template struct EINSUMS_EXPORT BufferAllocator<unsigned short>;
extern template struct EINSUMS_EXPORT BufferAllocator<unsigned int>;
extern template struct EINSUMS_EXPORT BufferAllocator<unsigned long>;
extern template struct EINSUMS_EXPORT BufferAllocator<unsigned long long>;
extern template struct EINSUMS_EXPORT BufferAllocator<float>;
extern template struct EINSUMS_EXPORT BufferAllocator<double>;
extern template struct EINSUMS_EXPORT BufferAllocator<std::complex<float>>;
extern template struct EINSUMS_EXPORT BufferAllocator<std::complex<double>>;

#endif

/**
 * Alias for a vector that uses a BufferAllocator<T> as its allocator.
 *
 * @tparam T The data type stored by the container.
 *
 * @versionadded{1.1.0}
 */
template <typename T>
using BufferVector = std::vector<T, BufferAllocator<T>>;

/**
 * Alias for a deque that uses a BufferAllocator<T> as its allocator.
 *
 * @tparam T The data type stored by the container.
 *
 * @versionadded{1.1.0}
 */
template <typename T>
using BufferDeque = std::deque<T, BufferAllocator<T>>;

/**
 * Alias for a forward list that uses a BufferAllocator<T> as its allocator.
 *
 * @tparam T The data type stored by the container.
 *
 * @versionadded{1.1.0}
 */
template <typename T>
using BufferForwardList = std::forward_list<T, BufferAllocator<T>>;

/**
 * Alias for a linked list that uses a BufferAllocator<T> as its allocator.
 *
 * @tparam T The data type stored by the container.
 *
 * @versionadded{1.1.0}
 */
template <typename T>
using BufferList = std::list<T, BufferAllocator<T>>;

/**
 * Alias for a set that uses a BufferAllocator<T> as its allocator.
 *
 * @tparam Key The data stored by the container.
 * @tparam Compare A comparison function for the elements.
 *
 * @versionadded{1.1.0}
 */
template <typename Key, typename Compare = std::less<Key>>
using BufferSet = std::set<Key, Compare, BufferAllocator<Key>>;

/**
 * Alias for a map that uses a BufferAllocator<T> as its allocator.
 *
 * @tparam Key The type for the keys for indexing the container.
 * @tparam T The data type stored by the container.
 * @tparam Compare A function that checks keys against each other.
 *
 * @versionadded{1.1.0}
 */
template <typename Key, typename T, typename Compare = std::less<Key>>
using BufferMap = std::map<Key, T, Compare, BufferAllocator<std::pair<Key const, T>>>;

/**
 * Alias for a multiset that uses a BufferAllocator<T> as its allocator.
 *
 * @tparam Key The data stored by the container.
 * @tparam Compare A comparison function for the elements.
 *
 * @versionadded{1.1.0}
 */
template <typename Key, typename Compare = std::less<Key>>
using BufferMultiSet = std::multiset<Key, Compare, BufferAllocator<Key>>;

/**
 * Alias for a multimap that uses a BufferAllocator<T> as its allocator.
 *
 * @tparam Key The type for the keys for indexing the container.
 * @tparam T The data type stored by the container.
 * @tparam Compare A function that checks keys against each other.
 *
 * @versionadded{1.1.0}
 */
template <typename Key, typename T, typename Compare = std::less<Key>>
using BufferMultiMap = std::multimap<Key, T, Compare, BufferAllocator<std::pair<Key const, T>>>;

/**
 * Alias for an unordered set that uses a BufferAllocator<T> as its allocator.
 *
 * @tparam Key The type for the keys for indexing the container.
 * @tparam Hash A function for hashing keys.
 * @tparam KeyEqual A function to determine if two keys are the same.
 *
 * @versionadded{1.1.0}
 */
template <typename Key, typename Hash = std::hash<Key>, typename KeyEqual = std::equal_to<Key>>
using BufferUnorderedSet = std::unordered_set<Key, Hash, KeyEqual, BufferAllocator<Key>>;

/**
 * Alias for an unordered map that uses a BufferAllocator<T> as its allocator.
 *
 * @tparam Key The type for the keys for indexing the container.
 * @tparam T The data type stored by the container.
 * @tparam Hash A function that hashes keys.
 * @tparam KeyEqual A function to determine if two keys are the same.
 *
 * @versionadded{1.1.0}
 */
template <typename Key, typename T, typename Hash = std::hash<Key>, typename KeyEqual = std::equal_to<Key>>
using BufferUnorderedMap = std::unordered_map<Key, T, Hash, KeyEqual, BufferAllocator<std::pair<Key const, T>>>;

/**
 * Alias for an unordered multiset that uses a BufferAllocator<T> as its allocator.
 *
 * @tparam Key The type for the keys for indexing the container.
 * @tparam Hash A function for hashing keys.
 * @tparam KeyEqual A function to determine if two keys are the same.
 *
 * @versionadded{1.1.0}
 */
template <typename Key, typename Hash = std::hash<Key>, typename KeyEqual = std::equal_to<Key>>
using BufferUnorderedMultiSet = std::unordered_multiset<Key, Hash, KeyEqual, BufferAllocator<Key>>;

/**
 * Alias for an unordered multimap that uses a BufferAllocator<T> as its allocator.
 *
 * @tparam Key The type for the keys for indexing the container.
 * @tparam T The data type stored by the container.
 * @tparam Hash A function that hashes keys.
 * @tparam KeyEqual A function to determine if two keys are the same.
 *
 * @versionadded{1.1.0}
 */
template <typename Key, typename T, typename Hash = std::hash<Key>, typename KeyEqual = std::equal_to<Key>>
using BufferUnorderedMultiMap = std::unordered_multimap<Key, T, Hash, KeyEqual, BufferAllocator<std::pair<Key const, T>>>;

/**
 * Alias for a basic string that uses a BufferAllocator<T> as its allocator.
 *
 * @tparam CharT The type of character stored by the string.
 * @tparam Traits The character traits of the character type.
 *
 * @versionadded{2.0.0}
 */
template <typename CharT, typename Traits = std::char_traits<CharT>>
using BufferBasicString = std::basic_string<CharT, Traits, BufferAllocator<CharT>>;

/**
 * Alias for a string that uses a BufferAllocator<T> as its allocator.
 *
 * @versionadded{2.0.0}
 */
using BufferString = BufferBasicString<char>;

} // namespace einsums