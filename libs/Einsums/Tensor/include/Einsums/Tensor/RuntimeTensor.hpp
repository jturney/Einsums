//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/BufferAllocator/BufferAllocator.hpp>
#include <Einsums/Concepts/File.hpp>
#include <Einsums/Concepts/SubscriptChooser.hpp>
#include <Einsums/Concepts/TensorConcepts.hpp>
#include <Einsums/Errors/Error.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/GPU/DeviceVector.hpp>
#include <Einsums/GPU/Runtime.hpp>
#include <Einsums/Python/Protocol.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/Tensor/TensorForward.hpp>
#include <Einsums/TensorBase/IndexUtilities.hpp>
#include <Einsums/TensorBase/TensorBase.hpp>
#include <Einsums/TensorImpl/TensorImpl.hpp>
#include <Einsums/TensorImpl/TensorImplOperations.hpp>

#include <fmt/format.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace einsums {

/**
 * @class GeneralRuntimeTensor
 *
 * @brief Represents a tensor whose properties can be determined at runtime but not compile time.
 *
 * This kind of tensor is unable to be used in many of the same ways as a tensor with compile-time rank. It is mostly used for communication
 * with the Python interface.
 *
 * @tparam T The data type stored by the tensor.
 * @tparam Alloc The allocator used for the internal data.
 *
 * @versionadded{1.0.0}
 * @versionchangeddesc{2.0.0}
 *      This used to be RuntimeTensor. An allocator parameter was added, and RuntimeTensor is now an alias to this with the standard
 *      allocator.
 * @endversion
 */
template <typename T, typename Alloc>
struct
    // clang-format off
EINSUMS_PYBIND_EXPOSE
// GeneralRuntimeTensor's allocator depends on T, so we pin each
// (T, std::allocator<T>) tuple individually rather than using a flat
// INSTANTIATE_TEMPLATE cross-product. BUFFER_PROTOCOL_STD lets the
// codegen synthesize the buffer-info builder from the named pure-C++
// accessors — Tensor.hpp has zero pybind11 references.
EINSUMS_PYBIND_BUFFER_PROTOCOL
EINSUMS_PYBIND_BUFFER_PROTOCOL_STD(data = data, rank = rank, dim = dim, stride = stride, element_type = T)
EINSUMS_PYBIND_ITERATOR_STD(begin = begin, end = end)
EINSUMS_PYBIND_INDEX_PROTOCOL_STD(element_type = T, rank = rank, dim = dim, at_element = at_element, set_element = set_element, at_view = at_view, view_type = einsums::RuntimeTensorView<T>)
EINSUMS_PYBIND_INSTANTIATE_AS("RuntimeTensorF", GeneralRuntimeTensor<float, std::allocator<float>>)
EINSUMS_PYBIND_INSTANTIATE_AS("RuntimeTensorD", GeneralRuntimeTensor<double, std::allocator<double>>)
EINSUMS_PYBIND_INSTANTIATE_AS("RuntimeTensorC", GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>)
EINSUMS_PYBIND_INSTANTIATE_AS("RuntimeTensorZ", GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
    // clang-format on
    GeneralRuntimeTensor : public tensor_base::CoreTensor,
                           tensor_base::RuntimeTensorNoType,
                           design_pats::Lockable<std::recursive_mutex> {
  public:
    /**
     * @typedef Vector
     *
     * @brief Represents how the data is stored in the tensor.
     *
     * For device allocators, gpu::DeviceVector is used instead of std::vector
     * (mirrors the storage selection in GeneralTensor). On a device tensor,
     * host-only operations (iterators, scalar subscript, set_all) are
     * disabled — see IsDeviceTensor below.
     */
    using Vector =
        std::conditional_t<gpu::IsDeviceAllocatorV<Alloc>, gpu::DeviceVector<std::remove_cv_t<T>>, std::vector<std::remove_cv_t<T>, Alloc>>;

    using Allocator = Alloc;

    /// True if this runtime tensor uses device (GPU) memory. Mirrors
    /// GeneralTensor::IsDeviceTensor and gates the same set of host-only
    /// operations.
    static constexpr bool IsDeviceTensor = gpu::IsDeviceAllocatorV<Alloc>;

    /**
     * @typedef ValueType
     *
     * @brief Represents the data type stored in the tensor.
     */
    using ValueType = T;

    /**
     * @brief Compile-time rank sentinel; the actual rank is only known at
     *        runtime via @ref rank(). See @ref einsums::dynamic_rank for
     *        the semantics this triggers in compile-time rank checks.
     */
    static constexpr int Rank = dynamic_rank;

    /**
     * @typedef Pointer
     *
     * @brief Type for pointers contained by this object.
     */
    using Pointer = typename detail::TensorImpl<T>::pointer;

    /**
     * @typedef ConstPointer
     *
     * @brief Type for const pointers contained by this object.
     */
    using ConstPointer = typename detail::TensorImpl<T>::const_pointer;

    /**
     * @typedef Reference
     *
     * @brief Type for references to items in the object.
     */
    using Reference = typename detail::TensorImpl<T>::reference;

    /**
     * @typedef ConstReference
     *
     * @brief Type for const references to items in the object.
     */
    using ConstReference = typename detail::TensorImpl<T>::const_reference;

    EINSUMS_PYBIND_EXPOSE GeneralRuntimeTensor() noexcept = default;

    /**
     * @brief Tag type for deferred allocation.
     *
     * Constructs a shell tensor (valid metadata, no data) without allocating
     * backing storage. Mirrors GeneralTensor::DeferredAlloc — same semantics:
     * data() returns a sentinel pointer until materialize() is called.
     */
    struct DeferredAlloc {};

    /// Tag value for deferred allocation constructors.
    static constexpr DeferredAlloc deferred_alloc{};

    /**
     * @brief Construct a shell runtime tensor with deferred allocation.
     *
     * Creates a tensor with valid dims/strides but no backing data. Used by
     * Workspace::declare_runtime_tensor and Graph::declare_runtime_tensor to
     * register a tensor that the MaterializationPass will allocate later
     * (potentially at a different size after distribution planning).
     */
    template <Container Dim>
    GeneralRuntimeTensor(DeferredAlloc, std::string name, Dim const &dims)
        : _name{std::move(name)}, _impl(reinterpret_cast<T *>(0x1), dims, GlobalConfigMap::get_singleton().get_bool("row-major")) {}

    GeneralRuntimeTensor(DeferredAlloc, std::string name, std::initializer_list<size_t> dims)
        : GeneralRuntimeTensor(DeferredAlloc{}, std::move(name), std::vector<size_t>(dims)) {}

    /**
     * @brief Default copy constructor.
     */
    GeneralRuntimeTensor(GeneralRuntimeTensor<T, Alloc> const &copy) : _name{copy._name}, _impl(copy.impl()), _data(copy.vector_data()) {
        _impl.set_data(_data.data());
    }

    /**
     * @brief Copy with a different allocator.
     *
     * Handles host↔device transfers automatically when copying between
     * host runtime tensors and GPU runtime tensors (mirrors
     * GeneralTensor's cross-allocator ctor).
     */
    template <typename Alloc2>
    GeneralRuntimeTensor(GeneralRuntimeTensor<T, Alloc2> const &copy) : _name{copy.name()}, _impl(copy.impl()) {
        constexpr bool other_is_device = gpu::IsDeviceAllocatorV<Alloc2>;
        constexpr bool this_is_device  = IsDeviceTensor;

        _data.resize(_impl.size());
        _impl.set_data(_data.data());

        size_t const bytes = _impl.size() * sizeof(T);

        if constexpr (this_is_device && !other_is_device) {
            gpu::memcpy_host_to_device(_data.data(), copy.data(), bytes);
        } else if constexpr (!this_is_device && other_is_device) {
            gpu::memcpy_device_to_host(_data.data(), copy.data(), bytes);
        } else if constexpr (this_is_device && other_is_device) {
            gpu::memcpy_device_to_device(_data.data(), copy.data(), bytes);
        } else {
            std::memcpy(_data.data(), copy.data(), bytes);
        }
    }

    /**
     * @brief Create a new runtime tensor with the given name and dimensions.
     *
     * @param name the new name of the tensor.
     * @param dims The dimensions of the tensor.
     */
    template <Container Dim>
    GeneralRuntimeTensor(std::string name, Dim const &dims, bool row_major) : _name{std::move(name)}, _impl(nullptr, dims, row_major) {
        _data.resize(_impl.size());

        _impl.set_data(_data.data());
    }

    /**
     * @brief Create a new runtime tensor with the given dimensions.
     *
     * @param dims The dimensions of the tensor.
     */
    template <Container Dim>
    explicit GeneralRuntimeTensor(Dim const &dims, bool row_major) : _impl(nullptr, dims, row_major) {
        _data.resize(_impl.size());

        _impl.set_data(_data.data());
    }

    /**
     * @brief Create a new runtime tensor with the given name and dimensions using an initializer list.
     *
     * @param name the new name of the tensor.
     * @param dims The dimensions of the tensor as an initializer list.
     */
    GeneralRuntimeTensor(std::string name, std::initializer_list<size_t> dims, bool row_major)
        : GeneralRuntimeTensor(name, std::vector<size_t>(dims), row_major) {}

    /**
     * @brief Create a new runtime tensor with the given dimensions using an initializer list.
     *
     * @param dims The dimensions of the tensor as an initializer list.
     */
    explicit GeneralRuntimeTensor(std::initializer_list<size_t> dims, bool row_major)
        : GeneralRuntimeTensor(std::vector<size_t>(dims), row_major) {}

    /**
     * @brief Create a new runtime tensor with the given name and dimensions.
     *
     * @param name the new name of the tensor.
     * @param dims The dimensions of the tensor.
     */
    template <Container Dim>
    EINSUMS_PYBIND_EXPOSE EINSUMS_PYBIND_INSTANTIATE_MEMBER(Dim = std::vector<size_t>)
        GeneralRuntimeTensor(std::string name, Dim const &dims)
        : _name{std::move(name)}, _impl(nullptr, dims, GlobalConfigMap::get_singleton().get_bool("row-major")) {
        _data.resize(_impl.size());

        _impl.set_data(_data.data());
    }

    /**
     * @brief Create a new runtime tensor with the given dimensions.
     *
     * @param dims The dimensions of the tensor.
     */
    template <Container Dim>
    explicit GeneralRuntimeTensor(Dim const &dims) : _impl(nullptr, dims, GlobalConfigMap::get_singleton().get_bool("row-major")) {
        _data.resize(_impl.size());

        _impl.set_data(_data.data());
    }

    /**
     * @brief Create a new runtime tensor with the given name and dimensions using an initializer list.
     *
     * @param name the new name of the tensor.
     * @param dims The dimensions of the tensor as an initializer list.
     */
    GeneralRuntimeTensor(std::string name, std::initializer_list<size_t> dims)
        : GeneralRuntimeTensor(name, std::vector<size_t>(dims), GlobalConfigMap::get_singleton().get_bool("row-major")) {}

    /**
     * @brief Create a new runtime tensor with the given dimensions using an initializer list.
     *
     * @param dims The dimensions of the tensor as an initializer list.
     */
    explicit GeneralRuntimeTensor(std::initializer_list<size_t> dims)
        : GeneralRuntimeTensor(std::vector<size_t>(dims), GlobalConfigMap::get_singleton().get_bool("row-major")) {}

    /**
     * @brief Copy a tensor into a runtime tensor.
     *
     * The data from the tensor will be copied, not mapped. If you want to alias the data, use a RuntimeTensorView instead.
     *
     * @param copy The tensor to copy.
     */
    template <size_t Rank, typename Alloc2>
    GeneralRuntimeTensor(GeneralTensor<T, Rank, Alloc2> const &copy) : _name{copy.name()}, _impl(nullptr, copy.dims(), copy.strides()) {
        constexpr bool other_is_device = gpu::IsDeviceAllocatorV<Alloc2>;
        constexpr bool this_is_device  = IsDeviceTensor;

        _data.resize(copy.size());
        _impl.set_data(_data.data());

        size_t const bytes = copy.size() * sizeof(T);

        if constexpr (this_is_device && !other_is_device) {
            gpu::memcpy_host_to_device(_data.data(), copy.data(), bytes);
        } else if constexpr (!this_is_device && other_is_device) {
            gpu::memcpy_device_to_host(_data.data(), copy.data(), bytes);
        } else if constexpr (this_is_device && other_is_device) {
            gpu::memcpy_device_to_device(_data.data(), copy.data(), bytes);
        } else {
            std::memcpy(_data.data(), copy.data(), bytes);
        }
    }

    template <size_t Rank, typename Alloc2>
    GeneralRuntimeTensor(GeneralTensor<T, Rank, Alloc2> &&copy) noexcept
        : _name{std::move(copy.name())}, _impl{std::move(copy.impl())}, _data{std::move(copy.vector_data())} {
        _impl.set_data(_data.data());
    }

    /**
     * @brief Copy a tensor view into a runtime tensor.
     *
     * The data from the tensor will be copied, not mapped. If you want to alias the data, use a RuntimeTensorView instead.
     *
     * @param copy The tensor view to copy.
     */
    template <size_t Rank>
    GeneralRuntimeTensor(TensorView<T, Rank> const &copy) : _impl(nullptr, copy.dims()) {
        static_assert(!IsDeviceTensor, "Constructing a device runtime tensor from a host TensorView is not supported. "
                                       "Construct a host RuntimeTensor first, then cross-allocator-copy into a RuntimeGPUTensor.");
        _data.resize(_impl.size());

        _impl.set_data(_data.data());

        detail::copy_to(copy.impl(), _impl);
    }

    /**
     * @brief Copy a tensor view into a runtime tensor.
     *
     * The data from the tensor will be copied, not mapped. If you want to alias the data, use a RuntimeTensorView instead.
     *
     * @param copy The tensor view to copy.
     */
    GeneralRuntimeTensor(RuntimeTensorView<T> const &copy) : _impl(nullptr, copy.dims()) {
        static_assert(!IsDeviceTensor, "Constructing a device runtime tensor from a RuntimeTensorView is not supported. "
                                       "Views carry no allocator information; copy into a host RuntimeTensor first.");
        _data.resize(_impl.size());

        _impl.set_data(_data.data());

        detail::copy_to(copy.impl(), _impl);
    }

    // HIP clang doesn't like it when this is defaulted.
    virtual ~GeneralRuntimeTensor() { _alive_canary = 0; }

    /**
     * @brief Set all of the data in the tensor to zero.
     */
    EINSUMS_PYBIND_EXPOSE virtual void zero() {
        if constexpr (IsDeviceTensor) {
            gpu::device_memset(_data.data(), 0, _data.size() * sizeof(T));
        } else {
            std::memset(_data.data(), 0, _data.size() * sizeof(T));
        }
    }

    /**
     * @brief Set all of the data in the tensor to the same value.
     *
     * @param val The value to fill the tensor with.
     *
     * Not supported on device runtime tensors — throws at runtime.
     * (set_all is virtual, so its body is instantiated for the v-table
     * even if never called; we branch with if constexpr instead of
     * static_assert to keep the device variant compilable.)
     */
    EINSUMS_PYBIND_EXPOSE virtual void set_all(T val) {
        if constexpr (IsDeviceTensor) {
            EINSUMS_THROW_EXCEPTION(std::runtime_error, "set_all() is not supported for device runtime tensors. Use a GPU kernel instead.");
        } else {
            std::fill(_data.begin(), _data.end(), val);
        }
    }

    /**
     * @brief Flat-element iterators (LegacyForwardIterator).
     *
     * Iterate over the storage in linear memory order. Used by Python's
     * iteration protocol via EINSUMS_PYBIND_ITERATOR_STD; for C++ callers,
     * also enables range-for and STL algorithms over the tensor's elements.
     * This iterates *elements*, not rows — for row-iteration, take
     * sub-views first.
     *
     * Disabled for device runtime tensors (gpu::DeviceVector has no
     * iterators — device memory is not host-accessible).
     */
    auto begin() noexcept
        requires(!IsDeviceTensor)
    {
        return _data.begin();
    }
    auto end() noexcept
        requires(!IsDeviceTensor)
    {
        return _data.end();
    }
    auto begin() const noexcept
        requires(!IsDeviceTensor)
    {
        return _data.begin();
    }
    auto end() const noexcept
        requires(!IsDeviceTensor)
    {
        return _data.end();
    }
    auto cbegin() const noexcept
        requires(!IsDeviceTensor)
    {
        return _data.cbegin();
    }
    auto cend() const noexcept
        requires(!IsDeviceTensor)
    {
        return _data.cend();
    }

    /**
     * @brief Get the pointer to the stored data.
     */
    [[nodiscard]] Pointer data() noexcept { return _data.data(); }

    /**
     * @copydoc data()
     */
    [[nodiscard]] ConstPointer data() const noexcept { return _data.data(); }

    /**
     * @brief Get the pointer to the stored data starting at the given index.
     *
     * @param index A collection of integers to use as the index.
     */
    template <Container Storage>
    [[nodiscard]] Pointer data(Storage const &index) {
        return _impl.data(index);
    }

    /**
     * @brief Get the pointer to the stored data starting at the given index.
     *
     * @param index A collection of integers to use as the index.
     */
    template <Container Storage>
    [[nodiscard]] ConstPointer data(Storage const &index) const {
        return _impl.data(index);
    }

    /**
     * @brief Subscript into the tensor, checking for validity of the index.
     *
     * This function will check the indices. If an index is negative, it will be wrapped around.
     * It will also make sure that the indices aren't too big. It will also check to see that
     * the correct number of indices were passed.
     *
     * @param index The index to use for the subscript.
     */
    template <Container Storage>
        requires(!std::is_base_of_v<Range, typename Storage::value_type> && !std::is_base_of_v<Range, Storage> && !IsDeviceTensor)
    Reference operator()(Storage const &index) {
        return _impl.subscript(index);
    }

    /**
     * @brief Subscript into the tensor, checking for validity of the index.
     *
     * This function will check the indices. If an index is negative, it will be wrapped around.
     * It will also make sure that the indices aren't too big. It will also check to see that
     * the correct number of indices were passed.
     *
     * @param index The index to use for the subscript.
     */
    template <Container Storage>
        requires(!std::is_base_of_v<Range, typename Storage::value_type> && !std::is_base_of_v<Range, Storage> && !IsDeviceTensor)
    ConstReference operator()(Storage const &index) const {
        return _impl.subscript(index);
    }

    /**
     * @brief Read a single element by full integer index.
     *
     * Pure-C++ entry point used by EINSUMS_PYBIND_INDEX_PROTOCOL_STD.
     * The index vector must have one entry per axis (length == rank()),
     * each in [0, dim(i)). Negative-index normalization happens at the
     * Python layer before this is called; the C++ entry point assumes
     * non-negative, in-range indices. Disabled for device tensors —
     * scalar reads of GPU memory must go through gpu::memcpy_device_to_host.
     */
    T at_element(std::vector<std::int64_t> const &idx) const {
        static_assert(!IsDeviceTensor, "at_element() is not supported for device runtime tensors.");
        return _impl.subscript(idx);
    }

    /**
     * @brief Write a single element by full integer index.
     *
     * Pure-C++ entry point used by EINSUMS_PYBIND_INDEX_PROTOCOL_STD.
     * Same preconditions as at_element.
     */
    void set_element(std::vector<std::int64_t> const &idx, T value) {
        static_assert(!IsDeviceTensor, "set_element() is not supported for device runtime tensors.");
        _impl.subscript(idx) = value;
    }

    /**
     * @brief Build a sub-view from a per-axis SliceSpec vector.
     *
     * Pure-C++ entry point used by EINSUMS_PYBIND_INDEX_PROTOCOL_STD's
     * view-returning paths. The spec vector must have one entry per axis
     * (length == rank()). Each axis's contribution to the resulting
     * view depends on its kind:
     *   - ``Index``  collapses the axis (rank reduces by 1)
     *   - ``Range``  keeps the axis with new dim ``ceil((stop-start)/step)``
     *                and stride ``parent.stride(axis) * step``
     *   - ``Full``   keeps the axis verbatim
     *
     * Negative indices and slice bounds are normalized at the Python
     * layer; this entry point assumes already-normalized non-negative
     * values.
     */
    RuntimeTensorView<T> at_view(std::vector<einsums::SliceSpec> const &specs) const {
        std::size_t const r = _impl.rank();
        if (specs.size() != r) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument, "at_view: spec count {} does not match tensor rank {}", specs.size(), r);
        }
        std::vector<std::size_t> view_dims;
        std::vector<std::size_t> view_strides;
        std::vector<std::size_t> offsets(r, 0);
        view_dims.reserve(r);
        view_strides.reserve(r);
        for (std::size_t i = 0; i < r; ++i) {
            auto const &s = specs[i];
            switch (s.kind) {
            case einsums::SliceSpec::Kind::Index:
                offsets[i] = static_cast<std::size_t>(s.index);
                // Axis collapses — omit from view_dims/view_strides.
                break;
            case einsums::SliceSpec::Kind::Range: {
                offsets[i]              = static_cast<std::size_t>(s.start);
                std::int64_t const span = s.stop - s.start;
                std::int64_t const step = s.step != 0 ? s.step : 1;
                std::int64_t const n    = step > 0 ? (span + step - 1) / step : 0;
                view_dims.push_back(static_cast<std::size_t>(n < 0 ? 0 : n));
                view_strides.push_back(_impl.stride(i) * static_cast<std::size_t>(step));
                break;
            }
            case einsums::SliceSpec::Kind::Full:
                offsets[i] = 0;
                view_dims.push_back(_impl.dim(i));
                view_strides.push_back(_impl.stride(i));
                break;
            }
        }
        return RuntimeTensorView<T>(*this, view_dims, view_strides, offsets);
    }

    /**
     * @brief Get the pointer to the stored data starting at the given index.
     *
     * @param args A collection of integers to use as the index.
     */
    template <std::integral... Args>
    [[nodiscard]] Pointer data(Args &&...args) {
        return _impl.data(std::forward<Args>(args)...);
    }

    /**
     * @brief Get the pointer to the stored data starting at the given index.
     *
     * @param args A collection of integers to use as the index.
     */
    template <std::integral... Args>
    [[nodiscard]] ConstPointer data(Args &&...args) const {
        return _impl.data(std::forward<Args>(args)...);
    }

    /**
     * @brief Get the pointer to the stored data starting at the given index.
     *
     * @param args A collection of integers to use as the index.
     */
    template <std::integral... Args>
    [[nodiscard]] Pointer data(Args const &...args) {
        return _impl.data(args...);
    }

    /**
     * @brief Get the pointer to the stored data starting at the given index.
     *
     * @param args A collection of integers to use as the index.
     */
    template <std::integral... Args>
    [[nodiscard]] ConstPointer data(Args const &...args) const {
        return _impl.data(args...);
    }

    Reference operator()()
        requires(!IsDeviceTensor)
    {
        return *_impl.data();
    }

    ConstReference operator()() const
        requires(!IsDeviceTensor)
    {
        return *_impl.data();
    }

    /**
     * @brief Subscript into the tensor, checking for validity of the index.
     *
     * This function will check the indices. If an index is negative, it will be wrapped around.
     * It will also make sure that the indices aren't too big. If fewer indices than necessary
     * are passed, it will throw an error. This will hopefully change in the future to allow for
     * the creation of views. It will still throw an error when too many arguments are passed.
     *
     * @param args The index to use for the subscript.
     *
     * @todo std::variant can't handle references. We may be able to make our own, but for right now,
     * this will not be able to handle the wrong number of arguments.
     */
    template <std::integral... Args>
        requires(!IsDeviceTensor)
    Reference operator()(Args const &...args) {
        return _impl.subscript(args...);
    }

    /**
     * @brief Subscript into the tensor, checking for validity of the index.
     *
     * This function will check the indices. If an index is negative, it will be wrapped around.
     * It will also make sure that the indices aren't too big. If too few indices are passed,
     * it will create a view.
     *
     * @param args The index to use for the subscript.
     */
    template <std::integral... Args>
        requires(!IsDeviceTensor)
    ConstReference operator()(Args const &...args) const {
        return _impl.subscript(args...);
    }

    /**
     * @brief Subscripts into the tensor with ranges.
     *
     * This function creates a view based on the ranges passed in.
     *
     * @param args The indices to use. Can be integers, ranges, or All.
     */
    template <typename... Args>
        requires requires {
            requires(std::is_same_v<Range, Args> || ... || false) || (std::is_same_v<AllT, Args> || ... || false);
            requires !(std::is_integral_v<Args> && ... && true);
        }
    RuntimeTensorView<T> const operator()(Args const &...args) const {
        return RuntimeTensorView<T>(_impl.subscript(args...));
    }

    /**
     * @brief Subscripts into the tensor with ranges.
     *
     * This function creates a view based on the ranges passed in.
     *
     * @param args The indices to use. Can be integers, ranges, or All.
     */
    template <typename... Args>
        requires requires {
            requires(std::is_same_v<Range, Args> || ... || false) || (std::is_same_v<AllT, Args> || ... || false);
            requires !(std::is_integral_v<Args> && ... && true);
        }
    RuntimeTensorView<T> operator()(Args const &...args) {
        return RuntimeTensorView<T>(_impl.subscript(args...));
    }

    /**
     * @brief Copy the data from one tensor into this tensor.
     *
     * @param other The tensor to copy from.
     */
    template <size_t Rank, typename Alloc2>
    GeneralRuntimeTensor &operator=(GeneralTensor<T, Rank, Alloc2> const &other) {
        _impl = other.impl();

        _data.resize(_impl.size());

        _impl.set_data(_data.data());

        constexpr bool other_is_device = gpu::IsDeviceAllocatorV<Alloc2>;
        size_t const   bytes           = _impl.size() * sizeof(T);

        if constexpr (IsDeviceTensor && !other_is_device) {
            gpu::memcpy_host_to_device(_data.data(), other.data(), bytes);
        } else if constexpr (!IsDeviceTensor && other_is_device) {
            gpu::memcpy_device_to_host(_data.data(), other.data(), bytes);
        } else if constexpr (IsDeviceTensor && other_is_device) {
            gpu::memcpy_device_to_device(_data.data(), other.data(), bytes);
        } else {
            detail::copy_to(other.impl(), _impl);
        }

        return *this;
    }

    /**
     * @brief Copy the data from one tensor into this tensor.
     *
     * @param other The tensor to copy from.
     */
    template <typename TOther, size_t Rank, typename Alloc2>
    GeneralRuntimeTensor &operator=(GeneralTensor<TOther, Rank, Alloc2> const &other) {
        static_assert(!IsDeviceTensor,
                      "Element-type-converting copy is not supported for device runtime tensors — would require a GPU kernel.");
        _impl = other.impl();

        _data.resize(_impl.size());

        _impl.set_data(_data.data());

        detail::copy_to(other.impl(), _impl);

        return *this;
    }

    /**
     * @brief Copy the data from one tensor view into this tensor.
     *
     * @param other The tensor view to copy from.
     */
    template <typename TOther, size_t Rank>
    GeneralRuntimeTensor &operator=(TensorView<TOther, Rank> const &other) {
        static_assert(!IsDeviceTensor, "Copy from a host TensorView is not supported for device runtime tensors.");
        _impl = detail::TensorImpl<T>(nullptr, other.dims());

        _data.resize(_impl.size());

        _impl.set_data(_data.data());

        detail::copy_to(other.impl(), _impl);

        return *this;
    }

    /**
     * @brief Copy the data from one tensor into this tensor.
     *
     * @param other The tensor to copy from.
     */
    GeneralRuntimeTensor &operator=(GeneralRuntimeTensor<T, Alloc> const &other) {
        if (this == &other) {
            return *this;
        }
        _impl = other.impl();

        _data.resize(_impl.size());

        _impl.set_data(_data.data());

        if constexpr (IsDeviceTensor) {
            gpu::memcpy_device_to_device(_data.data(), other.data(), _impl.size() * sizeof(T));
        } else {
            detail::copy_to(other.impl(), _impl);
        }

        return *this;
    }

    template <typename Alloc2>
    GeneralRuntimeTensor &operator=(GeneralRuntimeTensor<T, Alloc2> const &other) {
        _impl = other.impl();

        _data.resize(_impl.size());

        _impl.set_data(_data.data());

        constexpr bool other_is_device = gpu::IsDeviceAllocatorV<Alloc2>;
        size_t const   bytes           = _impl.size() * sizeof(T);

        if constexpr (IsDeviceTensor && !other_is_device) {
            gpu::memcpy_host_to_device(_data.data(), other.data(), bytes);
        } else if constexpr (!IsDeviceTensor && other_is_device) {
            gpu::memcpy_device_to_host(_data.data(), other.data(), bytes);
        } else if constexpr (IsDeviceTensor && other_is_device) {
            gpu::memcpy_device_to_device(_data.data(), other.data(), bytes);
        } else {
            detail::copy_to(other.impl(), _impl);
        }

        return *this;
    }

    /**
     * @brief Copy the data from one tensor view into this tensor.
     *
     * @param other The tensor view to copy from.
     *
     * Not supported on device runtime tensors — throws at runtime.
     * (operator= is virtual, so its body is instantiated for the v-table
     * even on device. Branch with if constexpr to keep the device
     * variant compilable.)
     */
    virtual GeneralRuntimeTensor &operator=(RuntimeTensorView<T> const &other) {
        if constexpr (IsDeviceTensor) {
            EINSUMS_THROW_EXCEPTION(std::runtime_error, "Copy from a RuntimeTensorView is not supported for device runtime tensors.");
        } else {
            _impl = detail::TensorImpl<T>(nullptr, other.dims());

            _data.resize(_impl.size());

            _impl.set_data(_data.data());

            detail::copy_to(other.impl(), _impl);
        }

        return *this;
    }

    /**
     * @brief Copy the data from one tensor into this tensor.
     *
     * @param other The tensor to copy from.
     */
    template <typename TOther, typename Alloc2>
    GeneralRuntimeTensor &operator=(GeneralRuntimeTensor<TOther, Alloc2> const &other) {
        static_assert(!IsDeviceTensor,
                      "Element-type-converting copy is not supported for device runtime tensors — would require a GPU kernel.");
        _impl = other.impl();

        _data.resize(_impl.size());

        _impl.set_data(_data.data());

        detail::copy_to(other.impl(), _impl);

        return *this;
    }

    /**
     * @brief Copy the data from one tensor view into this tensor.
     *
     * @param other The tensor view to copy from.
     */
    template <typename TOther>
    GeneralRuntimeTensor &operator=(RuntimeTensorView<TOther> const &other) {
        static_assert(!IsDeviceTensor, "Copy from a RuntimeTensorView is not supported for device runtime tensors.");
        _impl = detail::TensorImpl<T>(nullptr, other.dims());

        _data.resize(_impl.size());

        _impl.set_data(_data.data());

        detail::copy_to(other.impl(), _impl);

        return *this;
    }

    /**
     * @brief Fill the tensor with the given value.
     *
     * @param value The value to fill the tensor with.
     */
    virtual GeneralRuntimeTensor &operator=(T value) {
        set_all(value);
        return *this;
    }

    // Element-wise compound-assignment operators are host-only — they go
    // through detail::add_assign/sub_assign/etc., which iterate scalar
    // elements. For device tensors, do GPU-side equivalents through
    // ComputeGraph or BLAS calls.
    template <typename TOther>
        requires(!IsDeviceTensor)
    GeneralRuntimeTensor &operator+=(TOther const &b) {
        detail::add_assign(b, _impl);

        return *this;
    }

    template <typename TOther>
        requires(!IsDeviceTensor)
    GeneralRuntimeTensor &operator-=(TOther const &b) {
        detail::sub_assign(b, _impl);

        return *this;
    }

    template <typename TOther>
        requires(!IsDeviceTensor)
    GeneralRuntimeTensor &operator*=(TOther const &b) {
        detail::mult_assign(b, _impl);

        return *this;
    }

    template <typename TOther>
        requires(!IsDeviceTensor)
    GeneralRuntimeTensor &operator/=(TOther const &b) {
        detail::div_assign(b, _impl);

        return *this;
    }

    template <typename TOther>
        requires(!IsDeviceTensor) && requires(TOther t) {
            { t.impl() };
        }
    GeneralRuntimeTensor &operator+=(TOther const &b) {
        detail::add_assign(b.impl(), _impl);

        return *this;
    }

    template <typename TOther>
        requires(!IsDeviceTensor) && requires(TOther t) {
            { t.impl() };
        }
    GeneralRuntimeTensor &operator-=(TOther const &b) {
        detail::sub_assign(b.impl(), _impl);

        return *this;
    }

    template <typename TOther>
        requires(!IsDeviceTensor) && requires(TOther t) {
            { t.impl() };
        }
    GeneralRuntimeTensor &operator*=(TOther const &b) {
        detail::mult_assign(b.impl(), _impl);

        return *this;
    }

    template <typename TOther>
        requires(!IsDeviceTensor) && requires(TOther t) {
            { t.impl() };
        }
    GeneralRuntimeTensor &operator/=(TOther const &b) {
        detail::div_assign(b.impl(), _impl);

        return *this;
    }

    template <size_t Rank>
    operator TensorView<T, Rank>() {
        return TensorView<T, Rank>(_impl);
    }

    template <size_t Rank>
    operator TensorView<T, Rank> const() const {
        return TensorView<T, Rank>(_impl);
    }
    /**
     * @brief Get the length of the tensor along a given axis.
     *
     * @param d The axis to query. Negative values will wrap around.
     */
    [[nodiscard]] EINSUMS_PYBIND_EXPOSE virtual size_t dim(int d) const { return _impl.dim(d); }

    /**
     * @brief Get the dimensions of the tensor.
     */
    [[nodiscard]] virtual BufferVector<size_t> dims() const noexcept { return _impl.dims(); }

    /**
     * @brief Return the vector containing the data stored by the tensor.
     */
    [[nodiscard]] virtual Vector const &vector_data() const { return _data; }

    /**
     * @brief Return the vector containing the data stored by the tensor.
     */
    [[nodiscard]] virtual Vector &vector_data() { return _data; }

    /**
     * @brief Get the stride along a given axis.
     *
     * @param d The axis to query. Negative values will wrap around.
     */
    [[nodiscard]] EINSUMS_PYBIND_EXPOSE virtual size_t stride(int d) const { return _impl.stride(d); }

    /**
     * @brief Return the strides of the tensor.
     */
    [[nodiscard]] virtual BufferVector<size_t> strides() const noexcept { return _impl.strides(); }

    /**
     * @brief Create a rank-1 view of the tensor.
     */
    [[nodiscard]] virtual auto to_rank_1_view() const -> RuntimeTensorView<T> {
        std::vector<size_t> dim{size()};

        return RuntimeTensorView<T>{*this, dim};
    }

    /**
     * @brief Returns the linear size of the tensor.
     */
    [[nodiscard]] EINSUMS_PYBIND_EXPOSE EINSUMS_PYBIND_GETTER("size") virtual auto size() const -> size_t { return _data.size(); }

    /**
     * @brief Returns whether the tensor sees all of the underlying data.
     *
     * This type of tensor will always see all of its underlying data, so this will always be true.
     */
    [[nodiscard]] virtual bool full_view_of_underlying() const noexcept { return true; }

    /**
     * @brief Get the rank of the tensor.
     */
    [[nodiscard]] EINSUMS_PYBIND_EXPOSE virtual size_t rank() const noexcept { return _impl.rank(); }

    /**
     * @brief Set the name of the tensor.
     *
     * @param new_name The new name of the tensor.
     */
    EINSUMS_PYBIND_SETTER("name") virtual void set_name(std::string const &new_name) { this->_name = new_name; }

    /**
     * @brief Get the name of the tensor.
     */
    [[nodiscard]] EINSUMS_PYBIND_GETTER("name") virtual std::string const &name() const noexcept { return this->_name; }

    [[nodiscard]] virtual detail::TensorImpl<T> &impl() noexcept { return _impl; }

    [[nodiscard]] virtual detail::TensorImpl<T> const &impl() const noexcept { return _impl; }

    [[nodiscard]] bool is_row_major() const { return _impl.is_row_major(); }

    [[nodiscard]] bool is_column_major() const { return _impl.is_column_major(); }

    [[nodiscard]] RuntimeTensorView<T> transpose_view() { return RuntimeTensorView<T>(_impl.transpose_view()); }

    [[nodiscard]] RuntimeTensorView<T> const transpose_view() const { return RuntimeTensorView<T>(_impl.transpose_view()); }

    [[nodiscard]] RuntimeTensorView<T> to_row_major() { return RuntimeTensorView<T>(_impl.to_row_major()); }

    [[nodiscard]] RuntimeTensorView<T> const to_row_major() const { return RuntimeTensorView<T>(_impl.to_row_major()); }

    [[nodiscard]] RuntimeTensorView<T> to_column_major() { return RuntimeTensorView<T>(_impl.to_column_major()); }

    [[nodiscard]] RuntimeTensorView<T> const to_column_major() const { return RuntimeTensorView<T>(_impl.to_column_major()); }

    template <std::integral... MultiIndex>
    [[nodiscard]] RuntimeTensorView<T> tie_indices(MultiIndex &&...index) {
        return RuntimeTensorView<T>(_impl.tie_indices(std::forward<MultiIndex>(index)...));
    }

    template <std::integral... MultiIndex>
    [[nodiscard]] RuntimeTensorView<T> const tie_indices(MultiIndex &&...index) const {
        return RuntimeTensorView<T>(_impl.tie_indices(std::forward<MultiIndex>(index)...));
    }

    // ── Symmetry metadata ──────────────────────────────────────────────
    // Mirrors GeneralTensor's symmetry API so runtime-rank tensors (the
    // Python-facing path) can declare the same invariants. See
    // Tensor/SymmetryOps.hpp for symmetrize / check_symmetry (they're
    // compile-time-rank and only apply to GeneralTensor).

    void set_symmetry(SymmetryDescriptor desc) {
        if (desc.empty())
            _symmetry.reset();
        else
            _symmetry = std::make_unique<SymmetryDescriptor>(std::move(desc));
    }

    [[nodiscard]] SymmetryDescriptor const *symmetry() const noexcept { return _symmetry.get(); }

    void clear_symmetry() { _symmetry.reset(); }

    [[nodiscard]] bool has_symmetry() const { return _symmetry && !_symmetry->empty(); }

    // ── Deferred-allocation lifecycle ────────────────────────────────
    //
    // Mirrors GeneralTensor's API so ComputeGraph passes (Materialization,
    // FreeInsertion) and TensorHandle::make_handle's SFINAE probes pick
    // up the same capabilities on RuntimeTensor.

    /// Allocate backing storage (no-op if already materialized).
    /// Idempotent; safe to call repeatedly.
    void materialize() {
        if (is_materialized())
            return;
        _data.resize(_impl.size());
        _impl.set_data(_data.data());
    }

    /// True iff backing storage is allocated (or the tensor is rank-0
    /// with no data needed). Mirrors GeneralTensor's contract.
    [[nodiscard]] bool is_materialized() const { return !_data.empty() || _impl.size() == 0; }

    /// Release backing storage, returning to the deferred state. Dims and
    /// strides are preserved. data() returns a sentinel pointer until
    /// materialize() is called again. Used by FreeInsertion to free
    /// intermediates after their last consumer.
    void release() {
        if (_data.empty())
            return;
        _data.clear();
        _data.shrink_to_fit();
        _impl.set_data(reinterpret_cast<T *>(0x1)); // sentinel — dims/strides preserved
    }

    /// Override the internal data pointer. Used by the GPU executor's
    /// swap_data callback (TensorHandle.hpp:240) to redirect a tensor
    /// to a device shadow allocation, then restore the original later.
    /// Caller is responsible for the pointer's lifetime.
    void set_data(Pointer ptr) noexcept { _impl.set_data(ptr); }

    /// Change dimensions of an already-materialized tensor.
    /// No-op if dims match the current shape. Otherwise allocates a
    /// new buffer (existing data is discarded — same contract as
    /// GeneralTensor's resize). Argument is any range of size_t.
    template <typename Dims>
        requires requires(Dims const &d) {
            d.begin();
            d.end();
            d.size();
        }
    void resize(Dims const &dims) {
        // No-op if shape unchanged.
        if (dims.size() == _impl.rank() && std::equal(_impl.dims().cbegin(), _impl.dims().cend(), dims.begin())) {
            return;
        }
        // Build new impl to compute the required size, but don't commit yet.
        // NB: ``stored_row_major()``, not ``is_row_major()`` — the latter
        // collapses to true for rank-≤1 tensors regardless of how the
        // tensor was originally constructed, so resizing a column-major
        // ``[0]`` placeholder up to ``[3,4]`` would silently flip layout.
        detail::TensorImpl<T> new_impl(nullptr, dims, _impl.stored_row_major());
        // Resize data first — if this throws, _impl and _data remain consistent.
        _data.resize(new_impl.size());
        // Data resize succeeded — now commit.
        _impl = std::move(new_impl);
        _impl.set_data(_data.data());
    }

    /// Initializer-list overload for ergonomic call sites:
    /// ``t.resize({3, 4, 5})``. Same contract as the range form.
    void resize(std::initializer_list<size_t> dims) { resize(std::vector<size_t>(dims)); }

    /// Variadic ergonomic overload: ``t.resize(3, 4, 5)``.
    template <std::integral... Dims>
        requires(sizeof...(Dims) >= 1)
    void resize(Dims... dims) {
        resize(std::vector<size_t>{static_cast<size_t>(dims)...});
    }

    /// Change dimensions of a deferred (un-materialized) tensor without
    /// allocating storage. Used by DistributionPlanning + Materialization
    /// passes to shrink a globally-declared tensor to a local partition
    /// before allocating. Asserts the tensor is not yet materialized,
    /// matching GeneralTensor's contract.
    template <typename Dims>
        requires requires(Dims const &d) {
            d.begin();
            d.end();
            d.size();
        }
    void resize_deferred(Dims const &dims) {
        assert(!is_materialized() && "resize_deferred() must be called before materialize()");
        detail::TensorImpl<T> new_impl(reinterpret_cast<T *>(0x1), dims, _impl.stored_row_major());
        _impl = std::move(new_impl);
    }

    void resize_deferred(std::initializer_list<size_t> dims) { resize_deferred(std::vector<size_t>(dims)); }

    template <std::integral... Dims>
        requires(sizeof...(Dims) >= 1)
    void resize_deferred(Dims... dims) {
        resize_deferred(std::vector<size_t>{static_cast<size_t>(dims)...});
    }

    /// Destruction canary used by ComputeGraph's runtime validator
    /// (TensorHandle.hpp:308). Compares against kAliveCanary; the
    /// destructor overwrites with 0 so use-after-free is detectable.
    [[nodiscard]] bool is_alive() const { return _alive_canary == kAliveCanary; }

    static constexpr uint64_t kAliveCanary = 0xC0FFEE'DEAD'BEEF'42ULL;

  protected:
    Vector _data{};

    std::string _name{"(unnamed)"};

    detail::TensorImpl<T> _impl{};

    std::unique_ptr<SymmetryDescriptor> _symmetry{};

    uint64_t _alive_canary{kAliveCanary};

    template <typename TOther>
    friend class RuntimeTensorView;

    template <typename TOther, typename Alloc2>
    friend class GeneralRuntimeTensor;
};

/**
 * @class RuntimeTensorView
 *
 * @brief Represents a view of a tensor whose properties can be determined at runtime but not compile time.
 */
template <typename T>
struct EINSUMS_PYBIND_EXPOSE
    // Same Plan C protocol surface as GeneralRuntimeTensor: zero-copy
    // numpy interop via buffer protocol, Python iter, scalar subscript.
    // Slice/partial-tuple subscript on a view is not yet emitted (would
    // need a nested at_view returning another RuntimeTensorView).
    EINSUMS_PYBIND_BUFFER_PROTOCOL EINSUMS_PYBIND_BUFFER_PROTOCOL_STD(data = data, rank = rank, dim = dim, stride = stride,
                                                                      element_type = T)
        EINSUMS_PYBIND_INDEX_PROTOCOL_STD(element_type = T, rank = rank, dim = dim, at_element = at_element, set_element = set_element)
            EINSUMS_PYBIND_INSTANTIATE_AS("RuntimeTensorViewF", RuntimeTensorView<float>)
                EINSUMS_PYBIND_INSTANTIATE_AS("RuntimeTensorViewD", RuntimeTensorView<double>)
                    EINSUMS_PYBIND_INSTANTIATE_AS("RuntimeTensorViewC", RuntimeTensorView<std::complex<float>>)
                        EINSUMS_PYBIND_INSTANTIATE_AS("RuntimeTensorViewZ", RuntimeTensorView<std::complex<double>>) RuntimeTensorView
    : public tensor_base::CoreTensor,
      public tensor_base::RuntimeTensorNoType,
      public tensor_base::RuntimeTensorViewNoType,
      public design_pats::Lockable<std::recursive_mutex> {
  public:
    /**
     * @typedef ValueType
     *
     * @brief The data type stored by the tensor.
     */
    using ValueType = T;

    /**
     * @brief Compile-time rank sentinel; see @ref einsums::dynamic_rank.
     */
    static constexpr int Rank = dynamic_rank;

    /**
     * @typedef Pointer
     *
     * @brief Type for pointers contained by this object.
     */
    using Pointer = typename detail::TensorImpl<T>::pointer;

    /**
     * @typedef ConstPointer
     *
     * @brief Type for const pointers contained by this object.
     */
    using ConstPointer = typename detail::TensorImpl<T>::const_pointer;

    /**
     * @typedef Reference
     *
     * @brief Type for references to items in the object.
     */
    using Reference = typename detail::TensorImpl<T>::reference;

    /**
     * @typedef ConstReference
     *
     * @brief Type for const references to items in the object.
     */
    using ConstReference = typename detail::TensorImpl<T>::const_reference;

    RuntimeTensorView() = default;

    /**
     * @brief Default copy constructor.
     *
     * @param copy The tensor to copy.
     */
    RuntimeTensorView(RuntimeTensorView<T> const &copy) = default;

    /**
     * @brief Creates a new view based on another view.
     *
     * This view and the other view will share the same data pointer.
     *
     * @param view The tensor to view.
     */
    RuntimeTensorView(RuntimeTensor<T> const &view) : _impl{view.impl()}, _name{view.name()} {}

    /**
     * @brief Creates a view of a tensor with new dimensions specified.
     *
     * @param other The tensor to view.
     * @param dims The new dimensions for the view.
     */
    template <Container Dim>
    RuntimeTensorView(RuntimeTensor<T> const &other, Dim const &dims)
        : _impl{const_cast<Pointer>(other.data()), dims, other.impl().is_row_major()} {}

    /**
     * @brief Creates a view of a tensor with new dimensions specified.
     *
     * @param other The tensor to view.
     * @param dims The new dimensions for the view.
     */
    template <Container Dim>
    RuntimeTensorView(RuntimeTensorView<T> const &other, Dim const &dims)
        : _impl(const_cast<Pointer>(other.data()), dims, other.impl().is_row_major()) {}

    /**
     * @brief Creates a view of a tensor with new dimensions, strides, and offsets specified.
     *
     * @param other The tensor to view.
     * @param dims The new dimensions for the view.
     * @param strides The new strides for the view.
     * @param offsets The offsets for the view.
     */
    template <Container Dim, Container Stride, Container Offset, typename Alloc>
    RuntimeTensorView(GeneralRuntimeTensor<T, Alloc> const &other, Dim const &dims, Stride const &strides, Offset const &offsets)
        : _impl(const_cast<Pointer>(other.data(offsets)), dims, strides) {}

    /**
     * @brief Creates a view of a tensor with new dimensions, strides, and offsets specified.
     *
     * @param other The tensor to view.
     * @param dims The new dimensions for the view.
     * @param strides The new strides for the view.
     * @param offsets The offsets for the view.
     */
    template <Container Dim, Container Stride, Container Offset>
    RuntimeTensorView(RuntimeTensorView<T> const &other, Dim const &dims, Stride const &strides, Offset const &offsets)
        : _impl(const_cast<Pointer>(other.data(offsets)), dims, strides) {}

    /**
     * @brief Creates a view of a tensor with compile-time rank.
     *
     * @param copy The tensor to view.
     */
    template <size_t Rank>
    RuntimeTensorView(TensorView<T, Rank> const &copy) : _impl(copy.impl()) {}

    /**
     * @brief Creates a view of a tensor with compile-time rank.
     *
     * @param copy The tensor to view.
     */
    template <size_t Rank, typename Alloc>
    RuntimeTensorView(GeneralTensor<T, Rank, Alloc> const &copy) : _impl(copy.impl()) {}

    /**
     * @brief Creates a view around an implementation.
     */
    RuntimeTensorView(detail::TensorImpl<T> const &impl) : _impl(impl) {}

    // HIP clang doesn't like it when this is defaulted.
    virtual ~RuntimeTensorView() = default;

    /**
     * @brief Set all the entries in the tensor to zero.
     */
    virtual void zero() { detail::copy_to(T{0.0}, _impl); }

    /**
     * @brief Fill the tensor with the specified value.
     *
     * @param val The value to fill the tensor with.
     */
    virtual void set_all(T val) { detail::copy_to(val, _impl); }

    /**
     * @brief Return a pointer to the beginning of the data.
     */
    [[nodiscard]] Pointer data() { return _impl.data(); }

    /**
     * @brief Return a pointer to the beginning of the data.
     */
    [[nodiscard]] ConstPointer data() const { return _impl.data(); }

    /**
     * @brief Return a pointer to the data starting at the given index.
     */
    template <Container Storage>
    [[nodiscard]] Pointer data(Storage const &index) {
        return _impl.data(index);
    }

    /**
     * @brief Return a pointer to the data starting at the given index.
     */
    template <Container Storage>
    [[nodiscard]] ConstPointer data(Storage const &index) const {
        return _impl.data(index);
    }

    Reference operator()() { return *_impl.data(); }

    ConstReference operator()() const { return *_impl.data(); }

    /**
     * @brief Subscript into the tensor.
     *
     * This version checks for negative values and does bounds checking.
     *
     * @param index The index to use for subscripting.
     */
    template <Container Storage>
        requires(!std::is_base_of_v<Range, typename Storage::value_type> && !std::is_base_of_v<Range, Storage>)
    Reference operator()(Storage const &index) {
        return _impl.subscript(index);
    }

    /**
     * @brief Subscript into the tensor.
     *
     * This version checks for negative values and does bounds checking.
     *
     * @param index The index to use for subscripting.
     */
    template <Container Storage>
        requires(!std::is_base_of_v<Range, typename Storage::value_type> && !std::is_base_of_v<Range, Storage>)
    ConstReference operator()(Storage const &index) const {
        return _impl.subscript(index);
    }

    /**
     * @brief Subscript into the tensor.
     *
     * This version checks for negative values and does bounds checking.
     *
     * @param index The index to use for subscripting.
     */
    template <Container Storage>
        requires(std::is_base_of_v<Range, typename Storage::value_type>)
    RuntimeTensorView<T> operator()(Storage const &index) {
        return RuntimeTensorView<T>(_impl.subscript(index));
    }

    /**
     * @brief Subscript into the tensor.
     *
     * This version checks for negative values and does bounds checking.
     *
     * @param index The index to use for subscripting.
     */
    template <Container Storage>
        requires(std::is_base_of_v<Range, typename Storage::value_type>)
    RuntimeTensorView<T> const operator()(Storage const &index) const {
        return RuntimeTensorView<T>(_impl.subscript(index));
    }

    /**
     * @brief Get the data starting at the given index.
     *
     * @param args The indices for the starting point.
     */
    template <std::integral... Args>
    [[nodiscard]] Pointer data(Args &&...args) {
        _impl.data(std::forward<Args>(args)...);
    }

    /**
     * @brief Get the data starting at the given index.
     *
     * @param args The indices for the starting point.
     */
    template <std::integral... Args>
    [[nodiscard]] ConstPointer data(Args &&...args) const {
        _impl.data(std::forward<Args>(args)...);
    }

    /**
     * @brief Get the data starting at the given index.
     *
     * @param args The indices for the starting point.
     */
    template <std::integral... Args>
    [[nodiscard]] Pointer data(Args const &...args) {
        _impl.data(args...);
    }

    /**
     * @brief Get the data starting at the given index.
     *
     * @param args The indices for the starting point.
     */
    template <std::integral... Args>
    [[nodiscard]] ConstPointer data(Args const &...args) const {
        _impl.data(args...);
    }

    /**
     * @brief Subscript into the tensor.
     *
     * If there aren't enough indices, an error will be thrown. This version checks for negative indices and does bounds checking.
     *
     * @param args The indices to use for the subscript.
     */
    template <std::integral... Args>
    Reference operator()(Args const &...args) {
        return _impl.subscript(args...);
    }

    /**
     * @brief Subscript into the tensor.
     *
     * If there aren't enough indices, an error will be thrown. This version checks for negative indices and does bounds checking.
     *
     * @param args The indices to use for the subscript.
     */
    template <std::integral... Args>
    ConstReference operator()(Args const &...args) const {
        return _impl.subscript(args...);
    }

    /**
     * @brief Create a view with the given parameters.
     *
     * @param args The indices for the subscript. Can contain Range and All.
     */
    template <typename... Args>
        requires requires {
            requires(std::is_same_v<Range, Args> || ... || false) || (std::is_same_v<AllT, Args> || ... || false);
            requires !(std::is_integral_v<Args> && ... && true);
        }
    RuntimeTensorView<T> const operator()(Args const &...args) const {
        return RuntimeTensorView<T>(_impl.subscript(args...));
    }

    /**
     * @brief Create a view with the given parameters.
     *
     * @param args The indices for the subscript. Can contain Range and All.
     */
    template <typename... Args>
        requires requires {
            requires(std::is_same_v<Range, Args> || ... || false) || (std::is_same_v<AllT, Args> || ... || false);
            requires !(std::is_integral_v<Args> && ... && true);
        }
    RuntimeTensorView<T> operator()(Args const &...args) {
        return _impl.subscript(args...);
    }

    /**
     * @brief Copy data from one tensor into this tensor.
     *
     * @param other The tensor to copy from.
     */
    template <typename TOther, size_t Rank>
    RuntimeTensorView<T> &operator=(Tensor<TOther, Rank> const &other) {
        detail::copy_to(other.impl(), _impl);

        return *this;
    }

    /**
     * @brief Copy data from one tensor into this tensor.
     *
     * @param other The tensor to copy from.
     */
    template <typename TOther, size_t Rank>
    RuntimeTensorView<T> &operator=(TensorView<TOther, Rank> const &other) {
        detail::copy_to(other.impl(), _impl);

        return *this;
    }

    /**
     * @brief Copy data from one tensor into this tensor.
     *
     * @param other The tensor to copy from.
     */
    virtual RuntimeTensorView<T> &operator=(RuntimeTensor<T> const &other) {
        detail::copy_to(other.impl(), _impl);

        return *this;
    }

    virtual RuntimeTensorView<T> &operator=(BufferRuntimeTensor<T> const &other) {
        detail::copy_to(other.impl(), _impl);

        return *this;
    }

    template <typename Alloc>
    RuntimeTensorView<T> &operator=(GeneralRuntimeTensor<T, Alloc> const &other) {
        detail::copy_to(other.impl(), _impl);

        return *this;
    }

    /**
     * @brief Copy data from one tensor into this tensor.
     *
     * @param other The tensor to copy from.
     */
    RuntimeTensorView<T> &operator=(RuntimeTensorView<T> const &other) {
        detail::copy_to(other.impl(), _impl);

        return *this;
    }

    /**
     * @brief Copy data from one tensor into this tensor.
     *
     * @param other The tensor to copy from.
     */
    template <typename TOther, typename Alloc>
    RuntimeTensorView<T> &operator=(GeneralRuntimeTensor<TOther, Alloc> const &other) {
        detail::copy_to(other.impl(), _impl);

        return *this;
    }

    /**
     * @brief Copy data from one tensor into this tensor.
     *
     * @param other The tensor to copy from.
     */
    template <typename TOther>
    RuntimeTensorView<T> &operator=(RuntimeTensorView<TOther> const &other) {
        detail::copy_to(other.impl(), _impl);

        return *this;
    }

    /**
     * @brief Fill the tensor with the given value.
     *
     * @param value The value to fill the tensor with.
     */
    virtual RuntimeTensorView<T> &operator=(T value) {
        set_all(value);
        return *this;
    }

    template <typename TOther>
    RuntimeTensorView<T> &operator+=(TOther const &b) {
        detail::add_assign(b, _impl);

        return *this;
    }

    template <typename TOther>
    RuntimeTensorView<T> &operator-=(TOther const &b) {
        detail::sub_assign(b, _impl);

        return *this;
    }

    template <typename TOther>
    RuntimeTensorView<T> &operator*=(TOther const &b) {
        detail::mult_assign(b, _impl);

        return *this;
    }

    template <typename TOther>
    RuntimeTensorView<T> &operator/=(TOther const &b) {
        detail::div_assign(b, _impl);

        return *this;
    }

    template <typename TOther>
        requires requires(TOther t) {
            { t.impl() };
        }
    RuntimeTensorView<T> &operator+=(TOther const &b) {
        detail::add_assign(b.impl(), _impl);

        return *this;
    }

    template <typename TOther>
        requires requires(TOther t) {
            { t.impl() };
        }
    RuntimeTensorView<T> &operator-=(TOther const &b) {
        detail::sub_assign(b.impl(), _impl);

        return *this;
    }

    template <typename TOther>
        requires requires(TOther t) {
            { t.impl() };
        }
    RuntimeTensorView<T> &operator*=(TOther const &b) {
        detail::mult_assign(b.impl(), _impl);

        return *this;
    }

    template <typename TOther>
        requires requires(TOther t) {
            { t.impl() };
        }
    RuntimeTensorView<T> &operator/=(TOther const &b) {
        detail::div_assign(b.impl(), _impl);

        return *this;
    }

    template <size_t Rank>
    operator TensorView<T, Rank>() {
        if (rank() != Rank) {
            EINSUMS_THROW_EXCEPTION(dimension_error, "Can not convert a rank-{} RuntimeTensorView into a rank-{} TensorView!", rank(),
                                    Rank);
        }

        return TensorView<T, Rank>(_impl);
    }

    template <size_t Rank>
    operator TensorView<T, Rank>() const {
        if (rank() != Rank) {
            EINSUMS_THROW_EXCEPTION(dimension_error, "Can not convert a rank-{} RuntimeTensorView into a rank-{} TensorView!", rank(),
                                    Rank);
        }

        return TensorView<T, Rank>(_impl);
    }

    template <size_t Rank, typename Alloc>
    operator GeneralTensor<T, Rank, Alloc>() const {
        if (rank() != Rank) {
            EINSUMS_THROW_EXCEPTION(dimension_error, "Can not convert a rank-{} RuntimeTensorView into a rank-{} TensorView!", rank(),
                                    Rank);
        }

        return TensorView<T, Rank>(_impl);
    }

    /**
     * @brief Get the length of the tensor along the given axis.
     *
     * @param d The axis to query. Negative indices will be wrapped around.
     */
    [[nodiscard]] virtual auto dim(int d) const -> size_t { return _impl.dim(d); }

    /**
     * @brief Gets the dimensions of the tensor.
     */
    [[nodiscard]] virtual auto dims() const noexcept -> BufferVector<size_t> { return _impl.dims(); }

    /**
     * @brief Gets the stride of the tensor along the given axis.
     *
     * @param d The axis to query. Negative indices will be wrapped around.
     */
    [[nodiscard]] virtual auto stride(int d) const -> size_t { return _impl.stride(d); }

    /**
     * @brief Gets the strides of the tensor.
     */
    [[nodiscard]] virtual auto strides() const noexcept -> BufferVector<size_t> { return _impl.strides(); }

    /**
     * @brief Gets the rank-1 veiw of the tensor.
     *
     * This does not work well for tensor views due to the variation in strides.
     */
    [[nodiscard]] virtual auto to_rank_1_view() const -> RuntimeTensorView<T> {
        std::vector<size_t> dim{_impl.size()};

        return RuntimeTensorView<T>{*this, dim};
    }

    /**
     * @brief Returns the linear size of the tensor.
     */
    [[nodiscard]] virtual auto size() const noexcept -> size_t { return _impl.size(); }

    /**
     * @brief Checks whether the tensor sees all of the underlying data.
     */
    [[nodiscard]] virtual bool full_view_of_underlying() const noexcept { return _impl.is_contiguous(); }

    /**
     * @brief Returns the name of the tensor.
     */
    [[nodiscard]] virtual std::string const &name() const { return _name; };

    /**
     * @brief Sets the name of the tensor.
     *
     * @param new_name The new name for the tensor.
     */
    virtual void set_name(std::string const &new_name) { _name = new_name; };

    /**
     * @brief Gets the rank of the tensor.
     */
    [[nodiscard]] EINSUMS_PYBIND_EXPOSE virtual size_t rank() const noexcept { return _impl.rank(); }

    /**
     * @brief Read a single element by full integer index.
     *
     * Pure-C++ entry point used by the codegen index protocol.
     */
    T at_element(std::vector<std::int64_t> const &idx) const { return _impl.subscript(idx); }

    /**
     * @brief Write a single element by full integer index.
     */
    void set_element(std::vector<std::int64_t> const &idx, T value) { _impl.subscript(idx) = value; }

    /**
     * @brief Gets the implementation details.
     */
    [[nodiscard]] virtual detail::TensorImpl<T> &impl() { return _impl; }

    [[nodiscard]] virtual detail::TensorImpl<T> const &impl() const { return _impl; }

    [[nodiscard]] bool is_row_major() const { return _impl.is_row_major(); }

    [[nodiscard]] bool is_column_major() const { return _impl.is_column_major(); }

    [[nodiscard]] RuntimeTensorView<T> transpose_view() { return RuntimeTensorView<T>(_impl.transpose_view()); }

    [[nodiscard]] RuntimeTensorView<T> const transpose_view() const { return RuntimeTensorView<T>(_impl.transpose_view()); }

    [[nodiscard]] RuntimeTensorView<T> to_row_major() { return RuntimeTensorView<T>(_impl.to_row_major()); }

    [[nodiscard]] RuntimeTensorView<T> const to_row_major() const { return RuntimeTensorView<T>(_impl.to_row_major()); }

    [[nodiscard]] RuntimeTensorView<T> to_column_major() { return RuntimeTensorView<T>(_impl.to_column_major()); }

    [[nodiscard]] RuntimeTensorView<T> const to_column_major() const { return RuntimeTensorView<T>(_impl.to_column_major()); }

    template <std::integral... MultiIndex>
    [[nodiscard]] RuntimeTensorView<T> tie_indices(MultiIndex &&...index) {
        return RuntimeTensorView<T>(_impl.tie_indices(std::forward<MultiIndex>(index)...));
    }

    template <std::integral... MultiIndex>
    [[nodiscard]] RuntimeTensorView<T> const tie_indices(MultiIndex &&...index) const {
        return RuntimeTensorView<T>(_impl.tie_indices(std::forward<MultiIndex>(index)...));
    }

  protected:
    /**
     * @property _name
     *
     * @brief The name of the tensor.
     */
    std::string _name{"(unnamed view)"};

    detail::TensorImpl<T> _impl{};
};

#ifndef DOXYGEN
template <einsums::FileOrOStream Output, einsums::TensorConcept AType>
    requires requires {
        requires einsums::BasicTensorConcept<AType> || !einsums::AlgebraTensorConcept<AType>;
        requires !einsums::RankTensorConcept<AType>;
    }
void fprintln(Output &fp, AType const &A, einsums::TensorPrintOptions options = {}) {
    using namespace einsums;
    using T          = typename AType::ValueType;
    std::size_t Rank = A.rank();

    fprintln(fp, "Name: {}", A.name());
    {
        print::Indent const indent{};

        if constexpr (!TensorViewConcept<AType>)
            fprintln(fp, "Type: In Core Tensor");
        else
            fprintln(fp, "Type: In Core Tensor View");
    }
    fprintln(fp, A.impl(), options);
}

template <einsums::TensorConcept AType>
    requires requires {
        requires einsums::BasicTensorConcept<AType> || !einsums::AlgebraTensorConcept<AType>;
        requires !einsums::RankTensorConcept<AType>;
    }
void println(AType const &A, einsums::TensorPrintOptions options = {}) {
    fprintln(std::cout, A, options);
}

#endif

#if !defined(EINSUMS_WINDOWS) && !defined(DOXYGEN)
extern template class EINSUMS_EXPORT GeneralRuntimeTensor<float, std::allocator<float>>;
extern template class EINSUMS_EXPORT GeneralRuntimeTensor<double, std::allocator<double>>;
extern template class EINSUMS_EXPORT GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>;
extern template class EINSUMS_EXPORT GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>;

extern template class EINSUMS_EXPORT GeneralRuntimeTensor<float, BufferAllocator<float>>;
extern template class EINSUMS_EXPORT GeneralRuntimeTensor<double, BufferAllocator<double>>;
extern template class EINSUMS_EXPORT GeneralRuntimeTensor<std::complex<float>, BufferAllocator<std::complex<float>>>;
extern template class EINSUMS_EXPORT GeneralRuntimeTensor<std::complex<double>, BufferAllocator<std::complex<double>>>;

// gpu::DeviceAllocator variants are implicitly instantiated where used —
// see TensorDefs.cpp for rationale (mirrors GeneralTensor's pattern).

extern template class EINSUMS_EXPORT RuntimeTensorView<float>;
extern template class EINSUMS_EXPORT RuntimeTensorView<double>;
extern template class EINSUMS_EXPORT RuntimeTensorView<std::complex<float>>;
extern template class EINSUMS_EXPORT RuntimeTensorView<std::complex<double>>;
#endif
} // namespace einsums
