//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/BufferAllocator/BufferAllocator.hpp>
#include <Einsums/Config/Namespace.hpp>

using namespace einsums;

EINSUMS_NAMESPACE_BEGIN()

static void check_requirements() {
    constexpr bool requirement = requires(BufferAllocator<int> alloc, size_t n) {
        { *alloc.allocate(n) } -> std::same_as<typename BufferAllocator<int>::value_type &>;
        { alloc.deallocate(alloc.allocate(n), n) };
    } && std::copy_constructible<BufferAllocator<int>> && std::equality_comparable<BufferAllocator<int>>;

    static_assert(requirement);
}

#ifndef EINSUMS_WINDOWS

template struct BufferAllocator<void>;
template struct BufferAllocator<signed char>;
template struct BufferAllocator<signed short>;
template struct BufferAllocator<signed int>;
template struct BufferAllocator<signed long>;
template struct BufferAllocator<signed long long>;
template struct BufferAllocator<unsigned char>;
template struct BufferAllocator<unsigned short>;
template struct BufferAllocator<unsigned int>;
template struct BufferAllocator<unsigned long>;
template struct BufferAllocator<unsigned long long>;
template struct BufferAllocator<float>;
template struct BufferAllocator<double>;
template struct BufferAllocator<std::complex<float>>;
template struct BufferAllocator<std::complex<double>>;

#endif

EINSUMS_NAMESPACE_END()