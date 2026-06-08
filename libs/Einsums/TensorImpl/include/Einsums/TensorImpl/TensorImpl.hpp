//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/BlockManager/BlockManager.hpp>
#include <Einsums/BufferAllocator/BufferAllocator.hpp>
#include <Einsums/Concepts/Complex.hpp>
#include <Einsums/Concepts/File.hpp>
#include <Einsums/Concepts/NamedRequirements.hpp>
#include <Einsums/Config/CompilerSpecific.hpp>
#include <Einsums/Logging.hpp>
#include <Einsums/Print.hpp>
#include <Einsums/Tensor/TensorForward.hpp>
#include <Einsums/TensorBase/Common.hpp>
#include <Einsums/TensorBase/IndexUtilities.hpp>

#include <type_traits>

namespace einsums {

namespace detail {
template <typename T>
    requires(!std::is_const_v<T>)
struct TensorImpl;

template <typename T, typename TOther>
void copy_to(TensorImpl<TOther> const &in, TensorImpl<T> &out);
} // namespace detail

// Forward declaration of the Tensor printing function.
template <typename T>
void println(detail::TensorImpl<T> const &A, TensorPrintOptions options = {});

template <FileOrOStream Output, typename T>
void fprintln(Output &fp, detail::TensorImpl<T> const &A, TensorPrintOptions options = {});
namespace detail {

/**
 * @struct TensorImpl
 *
 * @brief Underlying implementation details for tensor objects.
 */
template <typename T>
    requires(!std::is_const_v<T>)
struct TensorImpl final {
    /**
     * @typedef pointer
     *
     * @brief The type for pointers returned by this class.
     */
    using pointer = T *; // NOLINT(readability-identifier-naming)

    /**
     * @typedef const_pointer
     *
     * @brief The type for const pointers returned by this class.
     */
    using const_pointer = T const *; // NOLINT(readability-identifier-naming)

    /**
     * @typedef void_pointer
     *
     * @brief The type for void pointers returned by this class.
     */
    using void_pointer = void *; // NOLINT(readability-identifier-naming)

    /**
     * @typedef const_void_pointer
     *
     * @brief The type for const void pointers returned by this class.
     */
    using const_void_pointer = void const *; // NOLINT(readability-identifier-naming)

    /**
     * @typedef value_type
     *
     * @brief The type of data stored by this class.
     */
    using value_type = T; // NOLINT(readability-identifier-naming)

    /**
     * @typedef reference
     *
     * @brief The reference type returned by this class.
     */
    using reference = T &; // NOLINT(readability-identifier-naming)

    /**
     * @typedef const_reference
     *
     * @brief The const reference type returned by this class.
     */
    using const_reference = T const &; // NOLINT(readability-identifier-naming)

    // Rule of five methods.

    /**
     * @brief Default constructor.
     */
    constexpr TensorImpl() noexcept : _dims(), _strides(), _rank{0}, _size{0}, _row_major{false} {};

    /**
     * @brief Copy constructor.
     */
    constexpr TensorImpl(TensorImpl<T> const &other)
        : _ptr{other._ptr}, _rank{other._rank}, _strides{other._strides}, _dims{other._dims}, _size{other._size},
          _row_major{other._row_major} {}

    /**
     * @brief Move constructor.
     */
    constexpr TensorImpl(TensorImpl<T> &&other) noexcept
        : _ptr{other._ptr}, _rank{other._rank}, _strides{std::move(other._strides)}, _dims{std::move(other._dims)}, _size{other._size},
          _row_major{other._row_major} {
        other._ptr  = nullptr;
        other._rank = 0;
        other._strides.clear();
        other._dims.clear();
        other._size = 0;
    }

    /**
     * @brief Copy assignment.
     */
    constexpr TensorImpl<T> &operator=(TensorImpl<T> const &other) {
        _ptr       = other._ptr;
        _rank      = other._rank;
        _strides   = other._strides;
        _dims      = other._dims;
        _size      = other._size;
        _row_major = other._row_major;
        return *this;
    }

    /**
     * @brief Move assignment.
     */
    constexpr TensorImpl<T> &operator=(TensorImpl<T> &&other) noexcept {
        _ptr       = other._ptr;
        _rank      = other._rank;
        _strides   = std::move(other._strides);
        _dims      = std::move(other._dims);
        _size      = other._size;
        _row_major = other._row_major;

        other._ptr  = nullptr;
        other._rank = 0;
        other._strides.clear();
        other._dims.clear();
        other._size = 0;
        return *this;
    }

    // Other constructors.

    /**
     * @brief Create a tensor with the given pointer and dimensions.
     *
     * This will calculate the strides using the given memory order.
     *
     * @param ptr The pointer to wrap.
     * @param dims The dimensions of the tensor.
     * @param row_major Whether to compute the strides in row-major or column-major ordering.
     */
    template <Container Dims>
    constexpr TensorImpl(pointer ptr, Dims const &dims, bool row_major)
        : _ptr{ptr}, _rank{dims.size()}, _dims(dims.begin(), dims.end()), _row_major{row_major} {
        compute_strides();
    }

    /**
     * @brief Create a tensor with the given pointer and dimensions.
     *
     * This will calculate the strides using the given memory order.
     *
     * @param ptr The pointer to wrap.
     * @param dims The dimensions of the tensor.
     */
    template <Container Dims>
    constexpr TensorImpl(pointer ptr, Dims const &dims)
        : _ptr{ptr}, _rank{dims.size()}, _dims(dims.begin(), dims.end()),
          _row_major{GlobalConfigMap::get_singleton().get_bool("row-major")} {
        compute_strides();
    }

    /**
     * @brief Create a tensor with the given pointer, dimensions, and strides.
     *
     * @param ptr The pointer to wrap.
     * @param dims The dimensions for the tensor.
     * @param strides The strides for the tensor.
     */
    template <Container Dims, Container Strides>
    constexpr TensorImpl(pointer ptr, Dims const &dims, Strides const &strides)
        : _ptr{ptr}, _rank{dims.size()}, _dims(dims.begin(), dims.end()), _strides(strides.begin(), strides.end()),
          _size{std::accumulate(dims.begin(), dims.end(), static_cast<size_t>(1), std::multiplies<>())} {
        infer_row_major();
    }

    /**
     * @brief Create a tensor with the given pointer and dimensions.
     *
     * This will calculate the strides using the given memory order.
     *
     * @param ptr The pointer to wrap.
     * @param dims The dimensions of the tensor.
     * @param row_major Whether to compute the strides in row-major or column-major ordering.
     */
    constexpr TensorImpl(pointer ptr, std::initializer_list<size_t> const &dims, bool row_major)
        : _ptr{ptr}, _rank{dims.size()}, _dims(dims.begin(), dims.end()), _row_major{row_major} {
        compute_strides();
    }

    /**
     * @brief Create a tensor with the given pointer and dimensions.
     *
     * This will calculate the strides using the given memory order.
     *
     * @param ptr The pointer to wrap.
     * @param dims The dimensions of the tensor.
     */
    constexpr TensorImpl(pointer ptr, std::initializer_list<size_t> const &dims)
        : _ptr{ptr}, _rank{dims.size()}, _dims(dims.begin(), dims.end()),
          _row_major{GlobalConfigMap::get_singleton().get_bool("row-major")} {
        compute_strides();
    }

    /**
     * @brief Create a tensor with the given pointer, dimensions, and strides.
     *
     * @param ptr The pointer to wrap.
     * @param dims The dimensions for the tensor.
     * @param strides The strides for the tensor.
     */
    template <Container Strides>
    constexpr TensorImpl(pointer ptr, std::initializer_list<size_t> const &dims, Strides const &strides)
        : _ptr{ptr}, _rank{dims.size()}, _dims(dims.begin(), dims.end()), _strides(strides.begin(), strides.end()),
          _size{std::accumulate(dims.begin(), dims.end(), static_cast<size_t>(1), std::multiplies<>())} {
        infer_row_major();
    }

    /**
     * @brief Create a tensor with the given pointer, dimensions, and strides.
     *
     * @param ptr The pointer to wrap.
     * @param dims The dimensions for the tensor.
     * @param strides The strides for the tensor.
     */
    template <Container Dims>
    constexpr TensorImpl(pointer ptr, Dims const &dims, std::initializer_list<size_t> const &strides)
        : _ptr{ptr}, _rank{dims.size()}, _dims(dims.begin(), dims.end()), _strides(strides.begin(), strides.end()),
          _size{std::accumulate(dims.begin(), dims.end(), static_cast<size_t>(1), std::multiplies<>())} {
        infer_row_major();
    }

    /**
     * @brief Create a tensor with the given pointer, dimensions, and strides.
     *
     * @param ptr The pointer to wrap.
     * @param dims The dimensions for the tensor.
     * @param strides The strides for the tensor.
     */
    constexpr TensorImpl(pointer ptr, std::initializer_list<size_t> const &dims, std::initializer_list<size_t> const &strides)
        : _ptr{ptr}, _rank{dims.size()}, _dims(dims.begin(), dims.end()), _strides(strides.begin(), strides.end()),
          _size{std::accumulate(dims.begin(), dims.end(), static_cast<size_t>(1), std::multiplies<>())} {
        infer_row_major();
    }

    ~TensorImpl() = default;

    // Getters and setters.

    /**
     * @brief Get the pointer being wrapped.
     */
    [[nodiscard]] constexpr pointer data() noexcept { return _ptr; }

    /**
     * @brief Get the pointer being wrapped.
     */
    [[nodiscard]] constexpr const_pointer data() const noexcept { return _ptr; }

    /**
     * @brief Get the rank of the tensor.
     */
    [[nodiscard]] constexpr size_t rank() const noexcept { return _rank; }

    /**
     * @brief Get the dimensions of the tensor.
     */
    [[nodiscard]] constexpr BufferVector<size_t> const &dims() const noexcept { return _dims; }

    /**
     * @brief Get the strides of the tensor.
     */
    [[nodiscard]] constexpr BufferVector<size_t> const &strides() const noexcept { return _strides; }

    /**
     * @brief Get the size of the tensor.
     */
    [[nodiscard]] constexpr size_t size() const noexcept { return _size; }

    /**
     * @brief Change the pointer being wrapped by the tensor.
     */
    constexpr void set_data(pointer ptr) noexcept { _ptr = ptr; }

    /// @brief Override the strides (used by local view to preserve original memory layout).
    template <size_t N>
    constexpr void set_strides(std::array<size_t, N> const &new_strides) noexcept {
        _strides.assign(new_strides.begin(), new_strides.end());
    }

    // Indexed getters and setters.

    /**
     * @brief Get the pointer to the given index.
     *
     * Negative indices will be made positive by adding the dimension along an axis.
     *
     * @param index The indices to use for the offset.
     */
    template <typename... MultiIndex>
        requires(std::is_integral_v<std::remove_cvref_t<MultiIndex>> && ... && true)
    [[nodiscard]] constexpr pointer data(MultiIndex &&...index) {
        if (sizeof...(MultiIndex) < _rank) {
            EINSUMS_THROW_EXCEPTION(not_enough_args, "Not enough indices passed to data!");
        } else if (sizeof...(MultiIndex) > _rank) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to data!");
        }

        if (_ptr == nullptr) {
            return nullptr;
        }

        return _ptr + indices_to_sentinel_negative_check(_strides, _dims, BufferVector<size_t>{static_cast<size_t>(index)...});
    }

    /**
     * @brief Get the pointer to the given index.
     *
     * Negative indices will be made positive by adding the dimension along an axis.
     *
     * @param index The indices to use for the offset.
     */
    template <typename... MultiIndex>
        requires(std::is_integral_v<std::remove_cvref_t<MultiIndex>> && ... && true)
    [[nodiscard]] constexpr const_pointer data(MultiIndex &&...index) const {
        if (sizeof...(MultiIndex) < _rank) {
            EINSUMS_THROW_EXCEPTION(not_enough_args, "Not enough indices passed to data!");
        } else if (sizeof...(MultiIndex) > _rank) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to data!");
        }

        if (_ptr == nullptr) {
            return nullptr;
        }

        return _ptr + indices_to_sentinel_negative_check(_strides, _dims, BufferVector<size_t>{static_cast<size_t>(index)...});
    }

    /**
     * @brief Get the pointer to the given index.
     *
     * Note that this does not do any checks, including index wrapping or bounds checking. Only use when
     * you are certain the indices will not go out of range.
     *
     * @param index The indices to use for the offset.
     */
    template <typename... MultiIndex>
        requires(std::is_integral_v<std::remove_cvref_t<MultiIndex>> && ... && true)
    [[nodiscard]] constexpr pointer data_no_check(MultiIndex &&...index) {
        if (sizeof...(MultiIndex) < _rank) {
            EINSUMS_THROW_EXCEPTION(not_enough_args, "Not enough indices passed to data!");
        } else if (sizeof...(MultiIndex) > _rank) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to data!");
        }

        if (_ptr == nullptr) {
            return nullptr;
        }

        return _ptr + indices_to_sentinel(_strides, BufferVector<size_t>{static_cast<size_t>(index)...});
    }

    /**
     * @brief Get the pointer to the given index.
     *
     * Note that this does not do any checks, including index wrapping or bounds checking. Only use when
     * you are certain the indices will not go out of range.
     *
     * @param index The indices to use for the offset.
     */
    template <typename... MultiIndex>
        requires(std::is_integral_v<std::remove_cvref_t<MultiIndex>> && ... && true)
    [[nodiscard]] constexpr const_pointer data_no_check(MultiIndex &&...index) const {
        if (sizeof...(MultiIndex) < _rank) {
            EINSUMS_THROW_EXCEPTION(not_enough_args, "Not enough indices passed to data!");
        } else if (sizeof...(MultiIndex) > _rank) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to data!");
        }

        if (_ptr == nullptr) {
            return nullptr;
        }

        return _ptr + indices_to_sentinel(_strides, BufferVector<size_t>{static_cast<size_t>(index)...});
    }

    /**
     * @brief Get the pointer to the given index.
     *
     * Negative indices will be made positive by adding the dimension along an axis.
     *
     * @param index The indices to use for the offset.
     */
    template <Container Index>
        requires(!std::is_same_v<Range, Index>)
    [[nodiscard]] constexpr pointer data(Index const &index) {
        if (index.size() < _rank) {
            EINSUMS_THROW_EXCEPTION(not_enough_args, "Not enough indices passed to data!");
        } else if (index.size() > _rank) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to data!");
        }

        if (_ptr == nullptr) {
            return nullptr;
        }

        return _ptr + indices_to_sentinel_negative_check(_strides, _dims, index);
    }

    /**
     * @brief Get the pointer to the given index.
     *
     * Negative indices will be made positive by adding the dimension along an axis.
     *
     * @param index The indices to use for the offset.
     */
    template <Container Index>
        requires(!std::is_same_v<Range, Index>)
    [[nodiscard]] constexpr const_pointer data(Index const &index) const {
        if (index.size() < _rank) {
            EINSUMS_THROW_EXCEPTION(not_enough_args, "Not enough indices passed to data!");
        } else if (index.size() > _rank) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to data!");
        }

        if (_ptr == nullptr) {
            return nullptr;
        }

        return _ptr + indices_to_sentinel_negative_check(_strides, _dims, index);
    }

    /**
     * @brief Get the pointer to the given index.
     *
     * Note that this does not do any checks, including index wrapping or bounds checking. Only use when
     * you are certain the indices will not go out of range.
     *
     * @param index The indices to use for the offset.
     */
    template <Container Index>
        requires(!std::is_same_v<Range, Index>)
    [[nodiscard]] constexpr pointer data_no_check(Index const &index) {
        if (index.size() < _rank) {
            EINSUMS_THROW_EXCEPTION(not_enough_args, "Not enough indices passed to data!");
        } else if (index.size() > _rank) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to data!");
        }

        if (_ptr == nullptr) {
            return nullptr;
        }

        return _ptr + indices_to_sentinel(_strides, index);
    }

    /**
     * @brief Get the pointer to the given index.
     *
     * Note that this does not do any checks, including index wrapping or bounds checking. Only use when
     * you are certain the indices will not go out of range.
     *
     * @param index The indices to use for the offset.
     */
    template <Container Index>
        requires(!std::is_same_v<Range, Index>)
    [[nodiscard]] constexpr const_pointer data_no_check(Index const &index) const {
        if (index.size() < _rank) {
            EINSUMS_THROW_EXCEPTION(not_enough_args, "Not enough indices passed to data!");
        } else if (index.size() > _rank) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to data!");
        }

        if (_ptr == nullptr) {
            return nullptr;
        }

        return _ptr + indices_to_sentinel(_strides, index);
    }

    /**
     * @brief Get the pointer to the given index.
     *
     * Negative indices will be made positive by adding the dimension along an axis.
     *
     * @param index The indices to use for the offset.
     */
    template <std::integral IntType>
    [[nodiscard]] constexpr pointer data(std::initializer_list<IntType> const &index) {
        if (index.size() < _rank) {
            EINSUMS_THROW_EXCEPTION(not_enough_args, "Not enough indices passed to data!");
        } else if (index.size() > _rank) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to data!");
        }

        if (_ptr == nullptr) {
            return nullptr;
        }

        return _ptr + indices_to_sentinel_negative_check(_strides, _dims, index);
    }

    /**
     * @brief Get the pointer to the given index.
     *
     * Negative indices will be made positive by adding the dimension along an axis.
     *
     * @param index The indices to use for the offset.
     */
    template <std::integral IntType>
    [[nodiscard]] constexpr const_pointer data(std::initializer_list<IntType> const &index) const {
        if (index.size() < _rank) {
            EINSUMS_THROW_EXCEPTION(not_enough_args, "Not enough indices passed to data!");
        } else if (index.size() > _rank) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to data!");
        }

        if (_ptr == nullptr) {
            return nullptr;
        }

        return _ptr + indices_to_sentinel_negative_check(_strides, _dims, index);
    }

    /**
     * @brief Get the pointer to the given index.
     *
     * Note that this does not do any checks, including index wrapping or bounds checking. Only use when
     * you are certain the indices will not go out of range.
     *
     * @param index The indices to use for the offset.
     */
    template <std::integral IntType>
    [[nodiscard]] constexpr pointer data_no_check(std::initializer_list<IntType> const &index) {

        return _ptr + indices_to_sentinel(_strides, index);
    }

    /**
     * @brief Get the pointer to the given index.
     *
     * Note that this does not do any checks, including index wrapping or bounds checking. Only use when
     * you are certain the indices will not go out of range.
     *
     * @param index The indices to use for the offset.
     */
    template <std::integral IntType>
    [[nodiscard]] constexpr const_pointer data_no_check(std::initializer_list<IntType> const &index) const {

        return _ptr + indices_to_sentinel(_strides, index);
    }

    /**
     * @brief Get the dimension along an axis.
     *
     * Negative values will be wrapped around.
     *
     * @param i The axis to check.
     */
    [[nodiscard]] constexpr size_t dim(std::integral auto i) const {
        // Rank 0 is a default-constructed, moved-from, or allocated scalar: report
        // 0 when there is no storage (dead tensor) and 1 for an allocated scalar.
        // Handle it before touching _dims (which is empty here).
        if (_rank == 0) {
            return (_ptr == nullptr) ? 0 : 1;
        }
        // A null data pointer with non-zero size is an unmaterialized (deferred)
        // tensor — report 0 so it reads as not-yet-allocated (and the buffer
        // protocol exposes nothing to dereference). A null pointer with size 0 is
        // a *legitimately empty* tensor: its storage holds no elements, so the
        // backing std::vector's data() is null, but its extents are real and must
        // be reported (e.g. a (0, N) tensor is not a (0, 0) tensor).
        if (_ptr == nullptr && _size != 0) {
            return 0;
        }
        auto temp = i;
        if (temp < 0) {
            temp += _rank;
        }

        if (temp < 0 || temp >= _rank) {
            EINSUMS_THROW_EXCEPTION(std::out_of_range, "The index passed to dim was out of range! Got {}, expected between {} and {}.", i,
                                    -static_cast<ptrdiff_t>(_rank), _rank - 1);
        }

        return _dims[temp];
    }

    /**
     * @brief Get the stride along an axis.
     *
     * Negative values will be wrapped around.
     *
     * @param i The axis to check.
     */
    [[nodiscard]] constexpr size_t stride(std::integral auto i) const {
        if (_rank == 0) {
            return 0;
        }
        auto temp = i;
        if (temp < 0) {
            temp += _rank;
        }

        if (temp < 0 || temp >= _rank) {
            EINSUMS_THROW_EXCEPTION(std::out_of_range, "The index passed to stride was out of range! Got {}, expected between {} and {}.",
                                    i, -static_cast<ptrdiff_t>(_rank), _rank - 1);
        }

        return _strides[temp];
    }

    // More complicated getters.

    /**
     * @brief Check whether the tensor is contiguous in memory.
     *
     * For a tensor to be contiguous in memory, there must not be
     * any data outside of any dimension. Views are often not contiguous, though they sometimes can be.
     */
    [[nodiscard]] constexpr bool is_contiguous() const {
        if (_rank == 0) {
            return true;
        }
        if (stride(0) < stride(-1)) {
            return dim(-1) * stride(-1) == _size;
        } else {
            return dim(0) * stride(0) == _size;
        }
    }

    /**
     * @brief Check if a tensor is able to be used as a vector argument to BLAS level 1 calls.
     *
     * @param[out] incx If not nullptr, this will be set to contain the spacing between items.
     */
    [[nodiscard]] bool is_totally_vectorable(size_t *incx = nullptr) const {
        if (_rank == 0) {
            if (incx != nullptr) {
                *incx = 0;
            }
            return false;
        }
        // Ignore size-1 dimensions (their stride is irrelevant — a permuted view
        // can leave one at a boundary with an inflated stride). The tensor is
        // totally vectorable iff the extent>1 dimensions tile memory with a
        // single increment and no gaps, i.e. the span (largest dim * largest
        // stride) equals element_count * smallest_stride.
        size_t min_s = 0, max_s = 0, max_dim = 0;
        bool   found = false;
        for (size_t i = 0; i < _rank; ++i) {
            if (_dims[i] <= 1) {
                continue;
            }
            if (!found) {
                min_s = max_s = _strides[i];
                max_dim       = _dims[i];
                found         = true;
            } else {
                if (_strides[i] < min_s) {
                    min_s = _strides[i];
                }
                if (_strides[i] > max_s) {
                    max_s   = _strides[i];
                    max_dim = _dims[i];
                }
            }
        }
        if (!found) { // at most one element
            if (incx != nullptr) {
                *incx = _strides[0];
            }
            return true;
        }
        if (incx != nullptr) {
            *incx = min_s;
        }
        return max_dim * max_s == _size * min_s;
    }

    /**
     * @brief Checks to see if a tensor can be passed to gemm.
     *
     * For this to be the case, the tensor needs to be rank 2 and its smallest stride must be 1.
     *
     * @param[out] lda The leading dimension which can be passed into gemm and similar calls.
     */
    [[nodiscard]] bool is_gemmable(size_t *lda = nullptr) const {
        if (_rank != 2 || (_strides[0] != 1 && _strides[1] != 1) || (_strides[0] == _strides[1])) {
            return false;
        } else {
            if (lda != nullptr) {
                *lda = std::max(_strides[0], _strides[1]);
            }
            return true;
        }
    }

    /**
     * @brief Get the smallest stride for the tensor.
     */
    [[nodiscard]] constexpr size_t get_incx() const {
        if (_rank == 0) {
            return 0;
        }
        // The vectorization increment is the smallest stride among dimensions
        // with extent > 1. A size-1 dimension is never traversed, so its stride
        // is irrelevant and must be ignored: a permuted view can leave a size-1
        // axis at a boundary with an inflated stride, which previously fooled the
        // stride(0)/stride(-1) shortcut into returning the wrong increment.
        size_t inc   = 0;
        bool   found = false;
        for (size_t i = 0; i < _rank; ++i) {
            if (_dims[i] > 1 && (!found || _strides[i] < inc)) {
                inc   = _strides[i];
                found = true;
            }
        }
        return found ? inc : _strides[0];
    }

    /**
     * @brief Get the smallest stride for the tensor. Equivalent to get_incx.
     */
    [[nodiscard]] constexpr size_t get_incy() const { return get_incx(); }

    /**
     * @brief Gets the largest stride for a rank-2 tensor only.
     */
    [[nodiscard]] constexpr size_t get_lda() const {
        if (_rank != 2) {
            EINSUMS_THROW_EXCEPTION(rank_error, "Can not get the leading dimension of a tensor whose rank is not 2!");
        } else {
            return std::max(stride(0), stride(1));
        }
    }

    /**
     * @brief Gets the largest stride for a rank-2 tensor only. Equivalent to get_lda
     */
    [[nodiscard]] constexpr size_t get_ldb() const { return get_lda(); }

    /**
     * @brief Gets the largest stride for a rank-2 tensor only. Equivalent to get_lda.
     */
    [[nodiscard]] constexpr size_t get_ldc() const { return get_lda(); }

    /**
     * @brief Checks to see if a tensor is general row-major.
     *
     * Rank-1 tensors are always both row-major and column-major, so don't assume the two
     * are logically exclusive.
     */
    [[nodiscard]] constexpr bool is_row_major() const {
        if (_rank <= 1) {
            return true;
        } else {
            return _row_major;
        }
    }

    /**
     * @brief Checks to see if a tensor is general column-major.
     *
     * Rank-1 tensors are always both row-major and column-major, so don't assume the two
     * are logically exclusive.
     */
    [[nodiscard]] constexpr bool is_column_major() const {
        if (_rank <= 1) {
            return true;
        } else {
            return !_row_major;
        }
    }

    /**
     * @brief Returns the raw row-major flag as stored, without the rank-1 collapse.
     *
     * Use this when you need to *preserve* the originally-requested layout
     * across operations like resize, where the user may grow a rank-≤1
     * tensor into something higher-rank and expects the same layout
     * (column-major by default in einsums) to carry over. ``is_row_major()``
     * collapses rank-≤1 to ``true`` and would silently flip the layout.
     */
    constexpr bool stored_row_major() const noexcept { return _row_major; }

    /**
     * @brief Calculate the parameters for looping over a BLAS call.
     *
     * A quick overview of how this might be used is something like this.
     *
     * @code
     * for(size_t i = 0; i < hard_size; i++) {
     *     blas_call(easy_size, data + i * stride);
     * }
     * @endcode
     *
     * For more complete examples, take a look at the implementation in @c TensorImplOperations.hpp .
     *
     * @param[out] easy_size The number of elements that can be passed into a BLAS call at any given time.
     * @param[out] hard_size The number of times the BLAS call will need to be made.
     * @param[out] easy_rank The largest rank that can be passed into a BLAS call. Any rank less than this
     * can also be passed.
     * @param[out] incx The spacing between elements for the BLAS call.
     */
    void query_vectorable_params(size_t *easy_size, size_t *hard_size, size_t *easy_rank, size_t *incx) const {
        if (_rank == 0) {
            *easy_size = 0;
            *hard_size = 0;
            *easy_rank = 0;
            *incx      = 0;
        } else if (_rank == 1) {
            *easy_size = _size;
            *hard_size = 0;
            *easy_rank = 1;
            *incx      = _strides[0];
        } else {

            *easy_size = 1;
            *hard_size = 1;
            *easy_rank = 0;

            // Vectorization increment = smallest stride among extent>1 dims
            // (size-1 dims have an irrelevant stride; see get_incx). The walk
            // grows the contiguous "easy" block from the innermost end (suffix
            // for row-major, prefix for column-major, matching impl_axpy's hard-
            // dim extraction) by accepting each dim whose stride equals the
            // running product of the easy extents so far. A size-1 dim folds in
            // for free while we are still inside the easy block; once a gap is
            // hit, the remaining dims (including any size-1 ones) are "hard".
            // (Previously `incx` came from stride(0)/stride(-1) and the running
            // product was a never-updated `const size = 1`, both wrong for
            // permuted/degenerate views.)
            size_t inc   = 0;
            bool   found = false;
            for (size_t i = 0; i < _rank; ++i) {
                if (_dims[i] > 1 && (!found || _strides[i] < inc)) {
                    inc   = _strides[i];
                    found = true;
                }
            }
            *incx = found ? inc : _strides[0];

            size_t expected   = *incx;
            bool   still_easy = true;
            if (is_row_major()) {
                for (int i = static_cast<int>(_rank) - 1; i >= 0; i--) {
                    if (_dims[i] == 1) {
                        if (still_easy) {
                            *easy_rank += 1;
                        }
                        continue;
                    }
                    if (still_easy && _strides[i] == expected) {
                        *easy_rank += 1;
                        *easy_size *= _dims[i];
                        expected *= _dims[i];
                    } else {
                        still_easy = false;
                        *hard_size *= _dims[i];
                    }
                }
            } else {
                for (int i = 0; std::cmp_less(i, _rank); i++) {
                    if (_dims[i] == 1) {
                        if (still_easy) {
                            *easy_rank += 1;
                        }
                        continue;
                    }
                    if (still_easy && _strides[i] == expected) {
                        *easy_rank += 1;
                        *easy_size *= _dims[i];
                        expected *= _dims[i];
                    } else {
                        still_easy = false;
                        *hard_size *= _dims[i];
                    }
                }
            }
        }
    }

    /**
     * @brief Checks to see if the pointer is associated.
     */
    [[nodiscard]] constexpr bool is_empty_view() const noexcept { return _ptr == nullptr; }

    // Subscript.

    [[nodiscard]] constexpr reference subscript() noexcept { return *_ptr; }

    [[nodiscard]] constexpr const_reference subscript() const noexcept { return *_ptr; }

    /**
     * @brief Subscript the tensor.
     *
     * @param index The indices for the subscript.
     */
    template <bool IgnoreRemoveRange = false, typename... MultiIndex>
        requires(std::is_integral_v<std::remove_cvref_t<MultiIndex>> && ... && true)
    [[nodiscard]] constexpr reference subscript(MultiIndex &&...index) {
        return *data(std::forward<MultiIndex>(index)...);
    }

    /**
     * @brief Subscript the tensor.
     *
     * @param index The indices for the subscript.
     */
    template <bool IgnoreRemoveRange = false, typename... MultiIndex>
        requires(std::is_integral_v<std::remove_cvref_t<MultiIndex>> && ... && true)
    [[nodiscard]] constexpr const_reference subscript(MultiIndex &&...index) const {
        return *data(std::forward<MultiIndex>(index)...);
    }

    /**
     * @brief Subscript the tensor.
     *
     * @param index The indices for the subscript.
     */
    template <bool IgnoreRemoveRange = false, Container MultiIndex>
        requires requires {
            requires std::is_integral_v<typename MultiIndex::value_type>;
            requires !std::is_base_of_v<Range, MultiIndex>;
        }
    [[nodiscard]] constexpr reference subscript(MultiIndex const &index) {
        return *data(index);
    }

    /**
     * @brief Subscript the tensor.
     *
     * @param index The indices for the subscript.
     */
    template <bool IgnoreRemoveRange = false, Container MultiIndex>
        requires requires {
            requires std::is_integral_v<typename MultiIndex::value_type>;
            requires !std::is_base_of_v<Range, MultiIndex>;
        }
    [[nodiscard]] constexpr const_reference subscript(MultiIndex const &index) const {
        return *data(index);
    }

    /**
     * @brief Subscript the tensor.
     *
     * @param index The indices for the subscript.
     */
    template <typename... MultiIndex>
        requires(std::is_integral_v<std::remove_cvref_t<MultiIndex>> && ... && true)
    [[nodiscard]] constexpr reference subscript_no_check(MultiIndex &&...index) {
        return *data_no_check(std::forward<MultiIndex>(index)...);
    }

    /**
     * @brief Subscript the tensor.
     *
     * @param index The indices for the subscript.
     */
    template <typename... MultiIndex>
        requires(std::is_integral_v<std::remove_cvref_t<MultiIndex>> && ... && true)
    [[nodiscard]] constexpr const_reference subscript_no_check(MultiIndex &&...index) const {
        return *data_no_check(std::forward<MultiIndex>(index)...);
    }

    /**
     * @brief Subscript the tensor.
     *
     * @param index The indices for the subscript.
     */
    template <Container MultiIndex>
        requires requires {
            requires std::is_integral_v<typename MultiIndex::value_type>;
            requires !std::is_same_v<Range, MultiIndex>;
        }
    [[nodiscard]] constexpr reference subscript_no_check(MultiIndex const &index) {
        return *data_no_check(index);
    }

    /**
     * @brief Subscript the tensor.
     *
     * @param index The indices for the subscript.
     */
    template <Container MultiIndex>
        requires requires {
            requires std::is_integral_v<typename MultiIndex::value_type>;
            requires !std::is_same_v<Range, MultiIndex>;
        }
    [[nodiscard]] constexpr const_reference subscript_no_check(MultiIndex const &index) const {
        return *data_no_check(index);
    }

    /**
     * @brief Subscript the tensor.
     *
     * @param index The indices for the subscript.
     */
    template <bool IgnoreRemoveRange = false, std::integral IntType>
    [[nodiscard]] constexpr reference subscript(std::initializer_list<IntType> const &index) {
        return *data(index);
    }

    /**
     * @brief Subscript the tensor.
     *
     * @param index The indices for the subscript.
     */
    template <bool IgnoreRemoveRange = false, std::integral IntType>
    [[nodiscard]] constexpr const_reference subscript(std::initializer_list<IntType> const &index) const {
        return *data(index);
    }

    /**
     * @brief Subscript the tensor.
     *
     * @param index The indices for the subscript.
     */
    template <std::integral IntType>
    [[nodiscard]] constexpr reference subscript_no_check(std::initializer_list<IntType> const &index) {
        return *data_no_check(index);
    }

    /**
     * @brief Subscript the tensor.
     *
     * @param index The indices for the subscript.
     */
    template <std::integral IntType>
    [[nodiscard]] constexpr const_reference subscript_no_check(std::initializer_list<IntType> const &index) const {
        return *data_no_check(index);
    }

    // Slicing.

    /**
     * @brief Create a view using the given data.
     *
     * The inputs can either be @c Range for a slice of data, @c All for a whole axis, or an integer for a single element along an axis.
     *
     * @param index The slice parameters for the view.
     */
    template <bool IgnoreRemoveRange = false, typename... MultiIndex>
        requires requires {
            requires((std::is_integral_v<std::remove_cvref_t<MultiIndex>> || std::is_base_of_v<Range, std::remove_cvref_t<MultiIndex>> ||
                      std::is_base_of_v<AllT, std::remove_cvref_t<MultiIndex>>) &&
                     ... && true);
            requires(!std::is_integral_v<std::remove_cvref_t<MultiIndex>> || ... || false);
        }
    [[nodiscard]] constexpr TensorImpl<T> subscript(MultiIndex &&...index) {
        auto index_tuple = std::make_tuple(std::forward<MultiIndex>(index)...);

        adjust_ranges<0>(index_tuple);

        BufferVector<size_t> new_dims, new_strides;

        new_dims.reserve(_rank);
        new_strides.reserve(_rank);

        size_t offset = compute_view<IgnoreRemoveRange, 0>(new_dims, new_strides, index_tuple);

        if (_ptr == nullptr) {
            return TensorImpl<T>(nullptr, new_dims, new_strides);
        } else {
            return TensorImpl<T>(_ptr + offset, new_dims, new_strides);
        }
    }

    /**
     * @brief Create a view using the given data.
     *
     * The inputs can either be @c Range for a slice of data, @c All for a whole axis, or an integer for a single element along an axis.
     *
     * @param index The slice parameters for the view.
     */
    template <bool IgnoreRemoveRange = false, typename... MultiIndex>
        requires requires {
            requires((std::is_integral_v<std::remove_cvref_t<MultiIndex>> || std::is_base_of_v<Range, std::remove_cvref_t<MultiIndex>> ||
                      std::is_base_of_v<AllT, std::remove_cvref_t<MultiIndex>>) &&
                     ... && true);
            requires(!std::is_integral_v<std::remove_cvref_t<MultiIndex>> || ... || false);
        }
    [[nodiscard]] constexpr TensorImpl<T> const subscript(MultiIndex &&...index) const {
        auto index_tuple = std::make_tuple(std::forward<MultiIndex>(index)...);

        adjust_ranges<0>(index_tuple);

        BufferVector<size_t> new_dims, new_strides;

        new_dims.reserve(_rank);
        new_strides.reserve(_rank);

        size_t offset = compute_view<IgnoreRemoveRange, 0>(new_dims, new_strides, index_tuple);

        if (_ptr == nullptr) {
            return TensorImpl<T>(nullptr, new_dims, new_strides);
        } else {
            return TensorImpl<T>(_ptr + offset, new_dims, new_strides);
        }
    }

    /**
     * @brief Create a view using the given data.
     *
     * The inputs can either be @c Range for a slice of data, @c All for a whole axis, or an integer for a single element along an axis.
     *
     * @param index The slice parameters for the view.
     */
    template <bool IgnoreRemoveRange = false, Container MultiIndex>
        requires(std::is_base_of_v<Range, typename MultiIndex::value_type>)
    [[nodiscard]] constexpr TensorImpl<T> subscript(MultiIndex const &index) {
        BufferVector<Range> index_list{index.begin(), index.end()};
        adjust_ranges(index_list);

        BufferVector<size_t> new_dims, new_strides;

        new_dims.reserve(_rank);
        new_strides.reserve(_rank);

        size_t offset = compute_view<IgnoreRemoveRange>(new_dims, new_strides, index_list);

        if (_ptr == nullptr) {
            return TensorImpl<T>(nullptr, new_dims, new_strides);
        } else {
            return TensorImpl<T>(_ptr + offset, new_dims, new_strides);
        }
    }

    /**
     * @brief Create a view using the given data.
     *
     * The inputs can either be @c Range for a slice of data, @c All for a whole axis, or an integer for a single element along an axis.
     *
     * @param index The slice parameters for the view.
     */
    template <bool IgnoreRemoveRange = false, Container MultiIndex>
        requires(std::is_base_of_v<Range, typename MultiIndex::value_type>)
    [[nodiscard]] constexpr TensorImpl<T> const subscript(MultiIndex const &index) const {
        BufferVector<Range> index_list{index.begin(), index.end()};
        adjust_ranges(index_list);

        BufferVector<size_t> new_dims, new_strides;

        new_dims.reserve(_rank);
        new_strides.reserve(_rank);

        size_t offset = compute_view<IgnoreRemoveRange>(new_dims, new_strides, index_list);

        if (_ptr == nullptr) {
            return TensorImpl<T>(nullptr, new_dims, new_strides);
        } else {
            return TensorImpl<T>(_ptr + offset, new_dims, new_strides);
        }
    }

    /**
     * @brief Create a view using the given data.
     *
     * The inputs can either be @c Range for a slice of data, @c All for a whole axis, or an integer for a single element along an axis.
     *
     * @param index The slice parameters for the view.
     */
    [[nodiscard]] constexpr TensorImpl<T> subscript(std::initializer_list<Range> const &index) {
        BufferVector<Range> index_list{index.begin(), index.end()};
        adjust_ranges(index_list);

        BufferVector<size_t> new_dims, new_strides;

        new_dims.reserve(_rank);
        new_strides.reserve(_rank);

        size_t offset = compute_view<true>(new_dims, new_strides, index_list);

        if (_ptr == nullptr) {
            return TensorImpl<T>(nullptr, new_dims, new_strides);
        } else {
            return TensorImpl<T>(_ptr + offset, new_dims, new_strides);
        }
    }

    /**
     * @brief Create a view using the given data.
     *
     * The inputs can either be @c Range for a slice of data, @c All for a whole axis, or an integer for a single element along an axis.
     *
     * @param index The slice parameters for the view.
     */
    template <bool IgnoreRemoveRange = false>
    [[nodiscard]] constexpr TensorImpl<T> const subscript(std::initializer_list<Range> const &index) const {
        BufferVector<Range> index_list{index.begin(), index.end()};
        adjust_ranges(index_list);

        BufferVector<size_t> new_dims, new_strides;

        new_dims.reserve(_rank);
        new_strides.reserve(_rank);

        size_t offset = compute_view<true>(new_dims, new_strides, index_list);

        if (_ptr == nullptr) {
            return TensorImpl<T>(nullptr, new_dims, new_strides);
        } else {
            return TensorImpl<T>(_ptr + offset, new_dims, new_strides);
        }
    }

    /**
     * @brief Create a view using the given data.
     *
     * The inputs can either be @c Range for a slice of data, @c All for a whole axis, or an integer for a single element along an axis.
     *
     * @param index The slice parameters for the view.
     */
    template <typename... MultiIndex>
        requires requires {
            requires((std::is_integral_v<std::remove_cvref_t<MultiIndex>> || std::is_base_of_v<Range, std::remove_cvref_t<MultiIndex>> ||
                      std::is_base_of_v<AllT, std::remove_cvref_t<MultiIndex>>) &&
                     ... && true);
            requires(!std::is_integral_v<std::remove_cvref_t<MultiIndex>> || ... || false);
        }
    [[nodiscard]] constexpr TensorImpl<T> subscript_no_check(MultiIndex &&...index) {
        auto index_tuple = std::make_tuple(std::forward<MultiIndex>(index)...);

        BufferVector<size_t> new_dims, new_strides;

        new_dims.reserve(_rank);
        new_strides.reserve(_rank);

        size_t offset = compute_view<true, 0>(new_dims, new_strides, index_tuple);

        if (_ptr == nullptr) {
            return TensorImpl<T>(nullptr, new_dims, new_strides);
        } else {
            return TensorImpl<T>(_ptr + offset, new_dims, new_strides);
        }
    }

    /**
     * @brief Create a view using the given data.
     *
     * The inputs can either be @c Range for a slice of data, @c All for a whole axis, or an integer for a single element along an axis.
     *
     * @param index The slice parameters for the view.
     */
    template <typename... MultiIndex>
        requires requires {
            requires((std::is_integral_v<std::remove_cvref_t<MultiIndex>> || std::is_base_of_v<Range, std::remove_cvref_t<MultiIndex>> ||
                      std::is_base_of_v<AllT, std::remove_cvref_t<MultiIndex>>) &&
                     ... && true);
            requires(!std::is_integral_v<std::remove_cvref_t<MultiIndex>> || ... || false);
        }
    [[nodiscard]] constexpr TensorImpl<T> const subscript_no_check(MultiIndex &&...index) const {
        auto index_tuple = std::make_tuple(std::forward<MultiIndex>(index)...);

        BufferVector<size_t> new_dims, new_strides;

        new_dims.reserve(_rank);
        new_strides.reserve(_rank);

        size_t offset = compute_view<true, 0>(new_dims, new_strides, index_tuple);

        if (_ptr == nullptr) {
            return TensorImpl<T>(nullptr, new_dims, new_strides);
        } else {
            return TensorImpl<T>(_ptr + offset, new_dims, new_strides);
        }
    }

    /**
     * @brief Create a view using the given data.
     *
     * The inputs can either be @c Range for a slice of data, @c All for a whole axis, or an integer for a single element along an axis.
     *
     * @param index The slice parameters for the view.
     */
    template <Container MultiIndex>
        requires(std::is_base_of_v<Range, typename MultiIndex::value_type>)
    [[nodiscard]] constexpr TensorImpl<T> subscript_no_check(MultiIndex const &index) {
        BufferVector<size_t> new_dims, new_strides;

        new_dims.reserve(_rank);
        new_strides.reserve(_rank);

        size_t offset = compute_view<true>(new_dims, new_strides, index);

        if (_ptr == nullptr) {
            return TensorImpl<T>(nullptr, new_dims, new_strides);
        } else {
            return TensorImpl<T>(_ptr + offset, new_dims, new_strides);
        }
    }

    /**
     * @brief Create a view using the given data.
     *
     * The inputs can either be @c Range for a slice of data, @c All for a whole axis, or an integer for a single element along an axis.
     *
     * @param index The slice parameters for the view.
     */
    template <Container MultiIndex>
        requires(std::is_base_of_v<Range, typename MultiIndex::value_type>)
    [[nodiscard]] constexpr TensorImpl<T> const subscript_no_check(MultiIndex const &index) const {
        BufferVector<size_t> new_dims, new_strides;

        new_dims.reserve(_rank);
        new_strides.reserve(_rank);

        size_t offset = compute_view<true>(new_dims, new_strides, index);

        if (_ptr == nullptr) {
            return TensorImpl<T>(nullptr, new_dims, new_strides);
        } else {
            return TensorImpl<T>(_ptr + offset, new_dims, new_strides);
        }
    }

    /**
     * @brief Create a view using the given data.
     *
     * The inputs can either be @c Range for a slice of data, @c All for a whole axis, or an integer for a single element along an axis.
     *
     * @param index The slice parameters for the view.
     */
    [[nodiscard]] constexpr TensorImpl<T> subscript_no_check(std::initializer_list<Range> const &index) {
        BufferVector<size_t> new_dims, new_strides;

        new_dims.reserve(_rank);
        new_strides.reserve(_rank);

        size_t offset = compute_view<true>(new_dims, new_strides, index);

        if (_ptr == nullptr) {
            return TensorImpl<T>(nullptr, new_dims, new_strides);
        } else {
            return TensorImpl<T>(_ptr + offset, new_dims, new_strides);
        }
    }

    /**
     * @brief Create a view using the given data.
     *
     * The inputs can either be @c Range for a slice of data, @c All for a whole axis, or an integer for a single element along an axis.
     *
     * @param index The slice parameters for the view.
     */
    [[nodiscard]] constexpr TensorImpl<T> const subscript_no_check(std::initializer_list<Range> const &index) const {
        BufferVector<size_t> new_dims, new_strides;

        new_dims.reserve(_rank);
        new_strides.reserve(_rank);

        size_t offset = compute_view<true>(new_dims, new_strides, index);

        if (_ptr == nullptr) {
            return TensorImpl<T>(nullptr, new_dims, new_strides);
        } else {
            return TensorImpl<T>(_ptr + offset, new_dims, new_strides);
        }
    }

    [[nodiscard]] constexpr TensorImpl<T> transpose_view() {
        return TensorImpl<T>(_ptr, BufferVector<size_t>(_dims.rbegin(), _dims.rend()),
                             BufferVector<size_t>(_strides.rbegin(), _strides.rend()));
    }

    [[nodiscard]] constexpr TensorImpl<T> const transpose_view() const {
        return TensorImpl<T>(_ptr, BufferVector<size_t>(_dims.rbegin(), _dims.rend()),
                             BufferVector<size_t>(_strides.rbegin(), _strides.rend()));
    }

    /**
     * @brief Create an axis-permuted view: result axis i takes parent axis ``perm[i]``.
     *
     * Like @ref transpose_view but for an arbitrary permutation of the axes
     * (``transpose_view`` is the full-reversal special case). Does not move data;
     * only the dims/strides are reordered. @p perm must be a permutation of
     * ``[0, rank)`` (unchecked here — callers validate).
     */
    [[nodiscard]] constexpr TensorImpl<T> permute_view(std::vector<size_t> const &perm) {
        BufferVector<size_t> new_dims, new_strides;
        new_dims.reserve(perm.size());
        new_strides.reserve(perm.size());
        for (size_t const ax : perm) {
            new_dims.push_back(_dims[ax]);
            new_strides.push_back(_strides[ax]);
        }
        return TensorImpl<T>(_ptr, std::move(new_dims), std::move(new_strides));
    }

    [[nodiscard]] constexpr TensorImpl<T> const permute_view(std::vector<size_t> const &perm) const {
        BufferVector<size_t> new_dims, new_strides;
        new_dims.reserve(perm.size());
        new_strides.reserve(perm.size());
        for (size_t const ax : perm) {
            new_dims.push_back(_dims[ax]);
            new_strides.push_back(_strides[ax]);
        }
        return TensorImpl<T>(_ptr, std::move(new_dims), std::move(new_strides));
    }

    /**
     * @brief Create a row-major view.
     *
     * This does not permute the data. It only reverses the dimensions and strides,
     * and only if the tensor is not already row major.
     */
    [[nodiscard]] constexpr TensorImpl<T> to_row_major() {
        if (stride(0) >= stride(-1)) {
            return *this;
        } else {
            return transpose_view();
        }
    }

    /**
     * @brief Create a column-major view.
     *
     * This does not permute the data. It only reverses the dimensions and strides,
     * and only if the tensor is not already column major.
     */
    [[nodiscard]] constexpr TensorImpl<T> to_column_major() {
        if (stride(0) <= stride(-1)) {
            return *this;
        } else {
            return transpose_view();
        }
    }

    /**
     * @brief Create a row-major view.
     *
     * This does not permute the data. It only reverses the dimensions and strides,
     * and only if the tensor is not already row major.
     */
    [[nodiscard]] constexpr TensorImpl<T> const to_row_major() const {
        if (stride(0) >= stride(-1)) {
            return *this;
        } else {
            return transpose_view();
        }
    }

    /**
     * @brief Create a column-major view.
     *
     * This does not permute the data. It only reverses the dimensions and strides,
     * and only if the tensor is not already column major.
     */
    [[nodiscard]] constexpr TensorImpl<T> const to_column_major() const {
        if (stride(0) <= stride(-1)) {
            return *this;
        } else {
            return transpose_view();
        }
    }

    /**
     * @brief Creates a view with the given indices tied together.
     *
     * This could be useful for Hadamard indices.
     *
     * @param index The index positions to tie together. Negative numbers will be wrapped.
     */
    template <typename... MultiIndex>
        requires(std::is_integral_v<MultiIndex> && ... && true)
    [[nodiscard]] constexpr TensorImpl<T> tie_indices(MultiIndex &&...index) {
        if constexpr (sizeof...(MultiIndex) <= 1) {
            return *this;
        } else {
            size_t new_stride = 0, new_dim = std::numeric_limits<size_t>::max();

            auto index_array = std::to_array<ptrdiff_t>({static_cast<ptrdiff_t>(index)...});

            BufferVector<size_t> new_strides(_strides), new_dims(_dims);

            // Calculate the tied stride.
            for (size_t i = 0; i < index_array.size(); i++) {
                // Deal with negatives.
                auto temp = index_array[i];
                if (index_array[i] < 0) {
                    index_array[i] += _rank;
                }

                if (index_array[i] < 0 || index_array[i] >= _rank) {
                    EINSUMS_THROW_EXCEPTION(std::out_of_range,
                                            "Attempting to tie indices that are out of bounds! Got {}, expected between {} and {}.", temp,
                                            -static_cast<ptrdiff_t>(_rank), _rank - 1);
                }

                new_stride += _strides[index_array[i]];
                new_strides[index_array[i]] = 0;

                if (_dims[index_array[i]] < new_dim) {
                    new_dim = _dims[index_array[i]];
                }
            }

            // Insert the dim and stride.
            if (is_row_major()) {
                bool found = false;
                for (int i = 0; i < _rank - 1; i++) {
                    if (new_strides[i] == 0 && new_stride >= _strides[i + 1]) {
                        new_strides[i] = new_stride;
                        new_dims[i]    = new_dim;
                        found          = true;
                        break;
                    }
                }

                if (!found) {
                    new_strides[_rank - 1] = new_stride;
                    new_dims[_rank - 1]    = new_dim;
                }
            } else {
                bool found = false;
                for (int i = 0; i < _rank - 1; i++) {
                    if (new_strides[i] == 0 && new_stride < _strides[i + 1]) {
                        new_strides[i] = new_stride;
                        new_dims[i]    = new_dim;
                        found          = true;
                        break;
                    }
                }

                if (!found) {
                    new_strides[_rank - 1] = new_stride;
                    new_dims[_rank - 1]    = new_dim;
                }
            }

            // Remove all the zero indices.
            BufferVector<size_t> temp_strides, temp_dims;
            temp_strides.reserve(_rank);
            temp_dims.reserve(_rank);

            for (int i = 0; std::cmp_less(i, _rank); i++) {
                if (new_strides[i] != 0) {
                    temp_strides.push_back(new_strides[i]);
                    temp_dims.push_back(new_dims[i]);
                }
            }

            return TensorImpl<T>(_ptr, temp_dims, temp_strides);
        }
    }

    /**
     * @brief Creates a view with the given indices tied together.
     *
     * This could be useful for Hadamard indices.
     *
     * @param index The index positions to tie together. Negative numbers will be wrapped.
     */
    template <typename... MultiIndex>
        requires(std::is_integral_v<MultiIndex> && ... && true)
    [[nodiscard]] constexpr TensorImpl<T> const tie_indices(MultiIndex &&...index) const {
        if constexpr (sizeof...(MultiIndex) <= 1) {
            return *this;
        } else {
            size_t new_stride = 0, new_dim = std::numeric_limits<size_t>::max();

            auto index_array = std::to_array<ptrdiff_t>({static_cast<ptrdiff_t>(index)...});

            BufferVector<size_t> new_strides(_strides), new_dims(_dims);

            // Calculate the tied stride.
            for (size_t i = 0; i < index_array.size(); i++) {
                // Deal with negatives.
                auto temp = index_array[i];
                if (index_array[i] < 0) {
                    index_array[i] += _rank;
                }

                if (index_array[i] < 0 || index_array[i] >= _rank) {
                    EINSUMS_THROW_EXCEPTION(std::out_of_range,
                                            "Attempting to tie indices that are out of bounds! Got {}, expected between {} and {}.", temp,
                                            -static_cast<ptrdiff_t>(_rank), _rank - 1);
                }

                // This takes care of duplicates.
                new_stride += new_strides[index_array[i]];
                new_strides[index_array[i]] = 0;

                if (_dims[index_array[i]] < new_dim) {
                    new_dim = _dims[index_array[i]];
                }
            }

            // Insert the dim and stride.
            if (is_row_major()) {
                bool found = false;
                for (int i = 0; i < _rank - 1; i++) {
                    if (new_strides[i] == 0 && new_stride >= _strides[i + 1]) {
                        new_strides[i] = new_stride;
                        new_dims[i]    = new_dim;
                        found          = true;
                        break;
                    }
                }

                if (!found) {
                    new_strides[_rank - 1] = new_stride;
                    new_dims[_rank - 1]    = new_dim;
                }
            } else {
                bool found = false;
                for (int i = 0; i < _rank - 1; i++) {
                    if (new_strides[i] == 0 && new_stride < _strides[i + 1]) {
                        new_strides[i] = new_stride;
                        new_dims[i]    = new_dim;
                        found          = true;
                        break;
                    }
                }

                if (!found) {
                    new_strides[_rank - 1] = new_stride;
                    new_dims[_rank - 1]    = new_dim;
                }
            }

            // Remove all the zero strides.
            BufferVector<size_t> temp_strides, temp_dims;
            temp_strides.reserve(_rank);
            temp_dims.reserve(_rank);

            for (int i = 0; std::cmp_less(i, _rank); i++) {
                if (new_strides[i] != 0) {
                    temp_strides.push_back(new_strides[i]);
                    temp_dims.push_back(new_dims[i]);
                }
            }

            return TensorImpl<T>(_ptr, temp_dims, temp_strides);
        }
    }

    void lock() const { _mutex.lock(); }

    void unlock() const { _mutex.unlock(); }

    [[nodiscard]] bool try_lock() const { return _mutex.try_lock(); }

  private:
    /// Compute strides from _dims and _row_major. Sets _strides and _size.
    constexpr void compute_strides() {
        _strides.resize(_rank);
        _size = 1;
        if (_row_major) {
            for (int i = static_cast<int>(_rank) - 1; i >= 0; i--) {
                _strides[i] = _size;
                _size *= _dims[i];
            }
        } else {
            for (size_t i = 0; i < _rank; i++) {
                _strides[i] = _size;
                _size *= _dims[i];
            }
        }
    }

    /// Infer _row_major from existing _strides and _dims.
    constexpr void infer_row_major() {
        if (_rank < 2 || (stride(0) > stride(-1))) {
            _row_major = true;
        } else if (stride(0) == stride(-1)) {
            _row_major = (dim(0) > dim(-1));
        } else {
            _row_major = false;
        }
    }

    template <size_t __I, typename... MultiIndex>
    constexpr void adjust_ranges(std::tuple<MultiIndex...> &indices) const {
        if constexpr (__I >= sizeof...(MultiIndex)) {
            return;
        } else if constexpr (std::is_integral_v<std::tuple_element_t<__I, std::remove_cvref_t<decltype(indices)>>>) {
            auto &index = std::get<__I>(indices);
            auto  temp  = index;

            if (index < 0) {
                index += _dims[__I];
            }

            if (index < 0 || index >= _dims[__I]) {
                EINSUMS_THROW_EXCEPTION(std::out_of_range,
                                        "Index passed to view creation is out of range! Got {}, expected between {} and {}.", temp,
                                        -static_cast<ptrdiff_t>(_dims[__I]), _dims[__I] - 1);
            }
            adjust_ranges<__I + 1>(indices);
        } else if constexpr (std::is_base_of_v<Range, std::tuple_element_t<__I, std::remove_cvref_t<decltype(indices)>>>) {
            auto &index = std::get<__I>(indices);
            auto  temp  = index;

            if (index[0] < 0) {
                index[0] += _dims[__I];
            }

            if (index[1] < 0) {
                index[1] += _dims[__I];
            }

            if (index[0] < 0 || index[0] > _dims[__I]) {
                EINSUMS_THROW_EXCEPTION(std::out_of_range,
                                        "Lower bound of range passed to view creation is out of range! Got {}, expected between {} and {}.",
                                        temp[0], -static_cast<ptrdiff_t>(_dims[__I]), _dims[__I] - 1);
            }

            if (index[1] < index[0] || index[1] > _dims[__I]) {
                EINSUMS_THROW_EXCEPTION(std::out_of_range,
                                        "Upper bound of range passed to view creation is out of range! Got {}, expected between {} and {}.",
                                        temp[1], index[0], _dims[__I]);
            }
            adjust_ranges<__I + 1>(indices);
        } else {
            adjust_ranges<__I + 1>(indices);
        }
    }

    template <bool IgnoreRemoveRange, size_t __I, typename... MultiIndex>
        requires((std::is_integral_v<MultiIndex> || std::is_base_of_v<Range, MultiIndex> || std::is_base_of_v<AllT, MultiIndex>) && ... &&
                 true)
    constexpr size_t compute_view(BufferVector<size_t> &out_dims, BufferVector<size_t> &out_strides,
                                  std::tuple<MultiIndex...> const &indices) const {
        if constexpr (__I >= sizeof...(MultiIndex)) {
            return 0;
        } else {
            if constexpr (std::is_integral_v<std::tuple_element_t<__I, std::remove_cvref_t<decltype(indices)>>>) {
                return std::get<__I>(indices) * _strides[__I] + compute_view<IgnoreRemoveRange, __I + 1>(out_dims, out_strides, indices);
            } else if constexpr (!IgnoreRemoveRange &&
                                 std::is_base_of_v<RemovableRange, std::tuple_element_t<__I, std::remove_cvref_t<decltype(indices)>>>) {
                auto range = std::get<__I>(indices);

                if (range[0] != range[1] && range.is_removable()) {
                    out_dims.push_back(range[1] - range[0]);
                    out_strides.push_back(_strides[__I]);
                }
                return range[0] * _strides[__I] + compute_view<IgnoreRemoveRange, __I + 1>(out_dims, out_strides, indices);
            } else if constexpr (std::is_base_of_v<Range, std::tuple_element_t<__I, std::remove_cvref_t<decltype(indices)>>>) {
                auto range = std::get<__I>(indices);

                out_dims.push_back(range[1] - range[0]);
                out_strides.push_back(_strides[__I]);
                return range[0] * _strides[__I] + compute_view<IgnoreRemoveRange, __I + 1>(out_dims, out_strides, indices);
            } else if constexpr (std::is_base_of_v<AllT, std::tuple_element_t<__I, std::remove_cvref_t<decltype(indices)>>>) {
                out_dims.push_back(_dims[__I]);
                out_strides.push_back(_strides[__I]);
                return compute_view<IgnoreRemoveRange, __I + 1>(out_dims, out_strides, indices);
            }
        }
    }

    template <ContainerOrInitializer MultiIndex>
        requires(std::is_base_of_v<Range, typename MultiIndex::value_type>)
    constexpr void adjust_ranges(MultiIndex &indices) const {
        for (auto &&[item, dim] : Zip(indices, std::as_const(_dims))) {
            auto temp = item;

            if (item[0] < 0) {
                item[0] += dim;
            }

            if (item[1] < 0) {
                item[1] += dim;
            }

            if (item[0] < 0 || item[0] > dim) {
                EINSUMS_THROW_EXCEPTION(std::out_of_range,
                                        "Lower bound of range passed to view creation is out of range! Got {}, expected between {} and {}.",
                                        temp[0], -static_cast<ptrdiff_t>(dim), dim - 1);
            }

            if (item[1] < item[0] || item[1] > dim) {
                EINSUMS_THROW_EXCEPTION(std::out_of_range,
                                        "Upper bound of range passed to view creation is out of range! Got {}, expected between {} and {}.",
                                        temp[1], item[0], dim);
            }
        }
    }

    template <bool IgnoreRemoveRange, ContainerOrInitializer MultiIndex>
        requires(std::is_base_of_v<Range, typename MultiIndex::value_type>)
    constexpr size_t compute_view(BufferVector<size_t> &out_dims, BufferVector<size_t> &out_strides, MultiIndex const &indices) const {
        size_t out = 0;
        for (auto const &[range, stride] : Zip(indices, _strides)) {
            if constexpr (IgnoreRemoveRange) {
                out_dims.push_back(range[1] - range[0]);
                out_strides.push_back(stride);
                out += range[0] * stride;
            } else {
                if (!range.is_removable() || range[1] - range[0] != 0) {
                    out_dims.push_back(range[1] - range[0]);
                    out_strides.push_back(stride);
                }
                out += range[0] * stride;
            }
        }

        return out;
    }

    pointer              _ptr{nullptr};
    size_t               _rank, _size;
    BufferVector<size_t> _dims, _strides;
    bool                 _row_major;
    std::mutex mutable _mutex;
};

} // namespace detail

namespace detail {

/**
 * Count the number of digits in a number.
 */
template <std::integral T>
auto ndigits(T number) -> int {
    int digits{0};
    if (number < 0)
        digits = 1; // Remove this line if '-' counts as a digit
    while (number) {
        number /= 10;
        digits++;
    }
    return digits;
}
} // namespace detail

template <FileOrOStream Output, typename T>
void fprintln(Output &fp, detail::TensorImpl<T> const &A, TensorPrintOptions options) {
    size_t Rank = A.rank();

    {
        print::Indent const indent{};

        fprintln(fp, "Data Type: {}", type_name<T>());

        if (Rank > 0) {
            std::ostringstream oss;
            for (size_t i = 0; i < Rank; i++) {
                oss << A.dim(i) << " ";
            }
            fprintln(fp, "Dims{{{}}}", oss.str().c_str());
        }

        if (Rank > 0) {
            std::ostringstream oss;
            for (size_t i = 0; i < Rank; i++) {
                oss << A.stride(i) << " ";
            }
            fprintln(fp, "Strides{{{}}}", oss.str());
        }

        if (options.full_output) {
            fprintln(fp);

            if (Rank == 0) {
                T value = A.subscript();

                std::ostringstream oss;
                oss << "              ";
                if constexpr (std::is_floating_point_v<T>) {
                    if (std::abs(value) < 1.0E-4) {
                        oss << fmt::format("{:14.4e} ", value);
                    } else {
                        oss << fmt::format("{:14.8f} ", value);
                    }
                } else if constexpr (IsComplexV<T>) {
                    oss << fmt::format("({:14.8f} ", value.real()) << " + " << fmt::format("{:14.8f}i)", value.imag());
                } else
                    oss << fmt::format("{:14} ", value);

                fprintln(fp, "{}", oss.str());
                fprintln(fp);
            } else if (Rank > 1) {
                BufferVector<size_t> index_strides(Rank - 1);
                size_t elements = dims_to_strides(BufferVector<size_t>(A.dims().begin(), std::prev(A.dims().end())), index_strides, true);

                auto                 final_dim = A.dim(Rank - 1);
                auto                 ndigits   = detail::ndigits(final_dim);
                BufferVector<size_t> target_combination(Rank);

                for (size_t item = 0; item < elements; item++) {
                    sentinel_to_indices(item, index_strides, target_combination);

                    std::ostringstream oss;
                    for (int j = 0; j < final_dim; j++) {
                        if (j % options.width == 0) {
                            std::ostringstream tmp;
                            tmp << fmt::format("{}", fmt::join(target_combination.begin(), std::prev(target_combination.end()), ", "));
                            if (final_dim >= j + options.width)
                                oss << fmt::format(
                                    "{:<14}", fmt::format("({}, {:{}d}-{:{}d}): ", tmp.str(), j, ndigits, j + options.width - 1, ndigits));
                            else
                                oss << fmt::format("{:<14}",
                                                   fmt::format("({}, {:{}d}-{:{}d}): ", tmp.str(), j, ndigits, final_dim - 1, ndigits));
                        }
                        target_combination.at(Rank - 1) = j;
                        T value                         = A.subscript(target_combination);
                        if (std::abs(value) > 1.0E+10) {
                            if constexpr (std::is_floating_point_v<T>)
                                oss << "\x1b[0;37;41m" << fmt::format("{:14.8f} ", value) << "\x1b[0m";
                            else if constexpr (IsComplexV<T>)
                                oss << "\x1b[0;37;41m(" << fmt::format("{:14.8f} ", value.real()) << " + "
                                    << fmt::format("{:14.8f}i)", value.imag()) << "\x1b[0m";
                            else
                                oss << "\x1b[0;37;41m" << fmt::format("{:14d} ", value) << "\x1b[0m";
                        } else {
                            if constexpr (std::is_floating_point_v<T>) {
                                if (std::abs(value) < 1.0E-4) {
                                    oss << fmt::format("{:14.4e} ", value);
                                } else {
                                    oss << fmt::format("{:14.8f} ", value);
                                }
                            } else if constexpr (IsComplexV<T>) {
                                oss << fmt::format("({:14.8f} ", value.real()) << " + " << fmt::format("{:14.8f}i)", value.imag());
                            } else
                                oss << fmt::format("{:14} ", value);
                        }
                        if (j % options.width == options.width - 1 && j != final_dim - 1) {
                            oss << "\n";
                        }
                    }
                    fprintln(fp, "{}", oss.str());
                    fprintln(fp);
                }
            } else if (Rank == 1) {

                size_t elements = A.size();

                for (size_t item = 0; item < elements; item++) {
                    std::ostringstream oss;
                    oss << "(";
                    oss << fmt::format("{}, ", item);
                    oss << "): ";

                    T value = A.subscript(item);
                    if (std::abs(value) > 1.0E+5) {
                        if constexpr (std::is_floating_point_v<T>)
                            oss << fmt::format(fg(fmt::color::white) | bg(fmt::color::red), "{:14.8f} ", value);
                        else if constexpr (IsComplexV<T>) {
                            oss << fmt::format(fg(color::white) | bg(color::red), "({:14.8f} + {:14.8f})", value.real(), value.imag());
                        } else
                            oss << fmt::format(fg(color::white) | bg(color::red), "{:14} ", value);
                    } else {
                        if constexpr (std::is_floating_point_v<T>)
                            if (std::abs(value) < 1.0E-4) {
                                oss << fmt::format("{:14.4e} ", value);
                            } else {
                                oss << fmt::format("{:14.8f} ", value);
                            }
                        else if constexpr (IsComplexV<T>) {
                            oss << fmt::format("({:14.8f} ", value.real()) << " + " << fmt::format("{:14.8f}i)", value.imag());
                        } else
                            oss << fmt::format("{:14} ", value);
                    }

                    fprintln(fp, "{}", oss.str());
                }
            }
        }
    }
    fprintln(fp);
}

template <typename T>
void println(detail::TensorImpl<T> const &A, TensorPrintOptions options) {
    fprintln(std::cout, A, options);
}

} // namespace einsums

#include <Einsums/TensorImpl/TensorImplOperations.hpp>
