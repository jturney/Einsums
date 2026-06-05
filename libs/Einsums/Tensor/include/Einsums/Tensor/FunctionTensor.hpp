//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Concepts/TensorConcepts.hpp>
#include <Einsums/Iterator/Enumerate.hpp>
#include <Einsums/Tensor/TensorForward.hpp>
#include <Einsums/TensorBase/IndexUtilities.hpp>
#include <Einsums/TensorBase/TensorBase.hpp>
#include <Einsums/TypeSupport/AreAllConvertible.hpp>
#include <Einsums/TypeSupport/Arguments.hpp>
#include <Einsums/TypeSupport/CountOfType.hpp>
#include <Einsums/TypeSupport/TypeName.hpp>

#include <type_traits>
#include <utility>

namespace einsums {

// forward declaration.
template <typename T, size_t Rank, size_t BaseRank>
struct FunctionTensorView;

#ifdef __HIP__
template <typename T, size_t Rank, size_t BaseRank>
struct DeviceFunctionTensorView;
#endif

namespace tensor_base {

/**
 * @struct FunctionTensor
 *
 * @brief Optional base class for function tensors. This class provides some useful functionality.
 *
 * A function tensor is one which takes advantage of the use of the operator() to index tensors to provide a way to
 * call a function. An example of this might be the Kronecker delta, which has such simple structure that creating
 * a whole new tensor object for it would be wasteful. This class does not define all function tensors. Instead,
 * it provides some useful behavior so that users can create their own with minimal effort. The only thing that
 * is required of function tensors is that they follow TensorConcept and they provide a function call operator
 * that can take as many arguments as the rank of the tensor. This does mean that other tensor classes, like
 * Tensor and BlockTensor, are also function tensors, though they do not inherit this class.
 */
template <typename T, size_t rank>
struct FunctionTensor : public CoreTensor {
  protected:
    /**
     * @property _dims
     *
     * @brief The dimensions of the tensor.
     */
    Dim<rank> _dims;

    /**
     * @property _name
     *
     * @brief The name of the tensor.
     */
    std::string _name{"(unnamed)"};

    /**
     * @property _size
     *
     * @brief The size of the tensor. Equal to the product of the dimensions.
     */
    size_t _size;

    /**
     * @brief Checks for negative indices and makes them positive, then performs range checking.
     */
    virtual void fix_indices(std::array<ptrdiff_t, rank> *inds) const {
        for (int i = 0; std::cmp_less(i, Rank); i++) {
            int orig = inds->at(i);
            if (inds->at(i) < 0) {
                inds->at(i) += _dims[i];
            }
            if (inds->at(i) >= _dims[i] || inds->at(i) < 0) {
                std::string message = fmt::format("Function tensor index out of range! Index at rank {} ", i);
                if (orig != inds->at(i)) {
                    message = fmt::format("{}({} -> {}) ", message, orig, inds->at(i));
                } else {
                    message = fmt::format("{}({}) ", message, inds->at(i));
                }
                EINSUMS_THROW_EXCEPTION(std::out_of_range, "{}is too far below zero or is greater than {}", message, _dims[i]);
            }
        }
    }

  public:
    /**
     * @typedef ValueType
     *
     * @brief The type of data returned by the tensor.
     */
    using ValueType = T;

    /**
     * @property Rank
     *
     * @brief The rank of the tensor.
     */
    constexpr static size_t Rank = rank;

    /**
     * @brief Create a new function tensor with the given name and dimensions.
     *
     * @param name The name of the tensor.
     * @param dims The dimensions of the tensor.
     */
    template <typename... Args>
        requires requires {
            requires(!std::is_same_v<Args, Dim<Rank>> || ...);
            requires sizeof...(Args) == Rank;
        }
    FunctionTensor(std::string name, Args... dims) : _dims{dims...}, _name{std::move(name)} {
        _size = 1;

        for (int i = 0; std::cmp_less(i, Rank); i++) {
            _size *= _dims[i];
        }
    }

    /**
     * @brief Create a new function tensor with the given name and dimensions.
     *
     * @param name The name of the tensor.
     * @param dims The dimensions of the tensor.
     */
    FunctionTensor(std::string name, Dim<Rank> dims) : _dims(dims), _name{std::move(name)} {
        _size = 1;

        for (int i = 0; std::cmp_less(i, Rank); i++) {
            _size *= _dims[i];
        }
    }

    /**
     * @brief Create a new function tensor with the given dimensions.
     *
     * @param dims The dimensions of the tensor.
     */
    FunctionTensor(Dim<Rank> dims) : _dims(dims) {
        _size = 1;

        for (int i = 0; std::cmp_less(i, Rank); i++) {
            _size *= _dims[i];
        }
    }

    virtual ~FunctionTensor() = default;

    /**
     * @brief Call the function.
     *
     * This is the method that should be overloaded in child classes to perform the actual
     * function call. Due to the inability to override methods with variable parameters,
     * this method with a set input type is the workaround.
     *
     * @param inds The index for the function.
     */
    virtual T call(std::array<ptrdiff_t, Rank> const &inds) const = 0;

    /**
     * @brief Subscript into the function tensor, wrapping negative indices and performing bounds checks.
     *
     * @param inds The index to use for the subscript.
     */
    template <typename... MultiIndex>
        requires requires {
            requires(sizeof...(MultiIndex) == Rank);
            requires(std::is_integral_v<MultiIndex> && ...);
        }
    T operator()(MultiIndex... inds) const {
        auto new_inds = std::array<ptrdiff_t, Rank>{static_cast<int>(inds)...};

        fix_indices(&new_inds);

        return this->call(new_inds);
    }

    /**
     * @brief Subscript into the function tensor without checking for negative indices or bounds.
     *
     * @param inds The index to use for the subscript.
     */
    template <typename... MultiIndex>
        requires requires {
            requires(sizeof...(MultiIndex) == Rank);
            requires(std::is_integral_v<std::remove_cvref_t<MultiIndex>> && ...);
        }
    T subscript(MultiIndex... inds) const {
        return this->call(std::array<ptrdiff_t, Rank>{inds...});
    }

    /**
     * @brief Subscript into the function tensor without checking for negative indices or bounds.
     *
     * @param inds The index to use for the subscript.
     */
    template <typename int_type>
        requires requires { requires(std::is_integral_v<int_type>); }
    T subscript(std::array<int_type, Rank> const &inds) const {
        if constexpr (std::is_same_v<int_type, ptrdiff_t>) {
            return this->call(inds);
        } else {
            std::array<ptrdiff_t, Rank> new_inds;
            for (size_t i = 0; i < Rank; i++) {
                new_inds[i] = static_cast<ptrdiff_t>(inds[i]);
            }
            return this->call(new_inds);
        }
    }

    /**
     * @brief Subscript into the function tensor, wrapping negative indices and performing bounds checks.
     *
     * @param inds The index to use for the subscript.
     */
    template <typename Storage>
        requires requires {
            requires !std::is_integral_v<Storage>;
            requires !std::is_same_v<Storage, AllT>;
            requires !std::is_same_v<Storage, Range>;
            requires !std::is_same_v<Storage, std::array<int, Rank>>;
        }
    T operator()(Storage const &inds) const {
        auto new_inds = std::array<ptrdiff_t, Rank>();

        for (int i = 0; std::cmp_less(i, Rank); i++) {
            new_inds[i] = static_cast<ptrdiff_t>(inds.at(i));
        }

        fix_indices(&new_inds);

        return this->call(new_inds);
    }

    /**
     * @brief Subscript into the function tensor, wrapping negative indices and performing bounds checks.
     *
     * Creates a view when one of the indices is All.
     *
     * @param inds The index to use for the subscript.
     */
    template <typename... MultiIndex>
        requires requires {
            requires(sizeof...(MultiIndex) == Rank);
            requires(AtLeastOneOfType<AllT, MultiIndex...>);
            requires(NoneOfType<std::array<int, Rank>, MultiIndex...>);
        }
    auto operator()(MultiIndex... inds) const
        -> FunctionTensorView<T, count_of_type<AllT, MultiIndex...>() + count_of_type<Range, MultiIndex...>(), Rank> {
        auto const &indices = std::forward_as_tuple(inds...);

        std::array<ptrdiff_t, Rank> index_template;

        Offset<count_of_type<AllT, MultiIndex...>() + count_of_type<Range, MultiIndex...>()> offsets;
        Dim<count_of_type<AllT, MultiIndex...>() + count_of_type<Range, MultiIndex...>()>    dims{};

        int counter{0};
        for_sequence<sizeof...(MultiIndex)>([&](auto i) {
            // println("looking at {}", i);
            if constexpr (std::is_convertible_v<std::tuple_element_t<i, std::tuple<MultiIndex...>>, ptrdiff_t>) {
                auto tmp = static_cast<ptrdiff_t>(std::get<i>(indices));
                if (tmp < 0)
                    tmp = _dims[i] + tmp;
                index_template[i] = tmp;
            } else if constexpr (std::is_same_v<AllT, std::tuple_element_t<i, std::tuple<MultiIndex...>>>) {
                dims[counter]     = _dims[i];
                offsets[counter]  = 0;
                index_template[i] = -1;
                counter++;

            } else if constexpr (std::is_same_v<Range, std::tuple_element_t<i, std::tuple<MultiIndex...>>>) {
                auto range       = std::get<i>(indices);
                offsets[counter] = range[0];
                if (range[1] < 0) {
                    auto temp = _dims[i] + range[1];
                    range[1]  = temp;
                }
                dims[counter]     = range[1] - range[0];
                index_template[i] = -1;
                counter++;
            }
        });

        return FunctionTensorView<T, count_of_type<AllT, MultiIndex...>() + count_of_type<Range, MultiIndex...>(), Rank>(
            this, offsets, dims, index_template);
    }

    /**
     * @brief Subscript into the function tensor, wrapping negative indices and performing bounds checks.
     *
     * Creates a view based on the indices, which can be ranges or single values.
     *
     * @param index The index to use for the subscript.
     */
    template <typename... MultiIndex>
        requires NumOfType<Range, Rank, MultiIndex...>
    auto operator()(MultiIndex... index) const -> FunctionTensorView<T, Rank, Rank> {
        Dim<Rank>                   dims{};
        Offset<Rank>                offset{};
        std::array<ptrdiff_t, Rank> index_template;

        auto ranges = arguments::get_array_from_tuple<std::array<Range, Rank>>(std::forward_as_tuple(index...));

        for (int r = 0; std::cmp_less(r, Rank); r++) {
            auto range = ranges[r];
            offset[r]  = range[0];
            if (range[1] < 0) {
                auto temp = _dims[r] + range[1];
                range[1]  = temp;
            }
            dims[r]           = range[1] - range[0];
            index_template[r] = -1;
        }

        return FunctionTensorView<T, Rank, Rank>{this, std::move(offset), std::move(dims), index_template};
    }

    /**
     * @brief Get the dimensions of the tensor.
     */
    virtual Dim<Rank> dims() const { return _dims; }

    /**
     * @brief Get the dimension of the tensor along a given axis.
     *
     * @param d The axis to query.
     */
    [[nodiscard]] virtual auto dim(int d) const -> size_t { return _dims[d]; }

    /**
     * @brief Get the name of the tensor.
     */
    [[nodiscard]] virtual std::string const &name() const { return _name; }

    /**
     * @brief Set the name of the tensor.
     *
     * @param str The new name.
     */
    virtual void set_name(std::string const &str) { _name = str; }

    /**
     * @brief Convert the function tensor into a regular tensor.
     */
    virtual BufferTensor<T, Rank> get() const {
        BufferTensor<T, Rank> out(dims());
        out.set_name(name());

        Stride<Rank> index_strides;
        size_t       elements = dims_to_strides(dims(), index_strides);

        EINSUMS_OMP_PARALLEL_FOR
        for (size_t item = 0; item < elements; item++) {
            thread_local std::array<ptrdiff_t, Rank> index;

            sentinel_to_indices(item, index_strides, index);

            out.data()[item] = call(index);
        }

        return out;
    }

    operator BufferTensor<T, Rank>() const { return get(); }

    /**
     * Returns whether the tensor contains all elements or only some subset of a whole.
     *
     * @return False if the tensor is a view of a larger tensor. True if this is the entire tensor.
     */
    [[nodiscard]] virtual bool full_view_of_underlying() const { return true; }
};

} // namespace tensor_base

/**
 * @class FuncPointerTensor
 *
 * @brief This is a function tensor that wraps a function pointer.
 *
 * Function tensors of this type wrap function pointers. Whenever their subscript method is
 * called, the arguments will ultimately be passed on to the function pointer contained within.
 *
 * @tparam T The return type of the function.
 * @tparam Rank The rank of the tensor.
 */
template <typename T, size_t Rank>
struct FuncPointerTensor : public tensor_base::FunctionTensor<T, Rank> {
  protected:
    /**
     * @property _func_ptr
     *
     * @brief The function pointer called by the subscript operator.
     */
    T (*_func_ptr)(std::array<ptrdiff_t, Rank> const &);

  public:
    /**
     * @brief Construct a new function tensor with the given dimensions and function pointer.
     *
     * @param name The new name of the tensor.
     * @param func_ptr The function pointer that will actually be doing the work.
     * @param dims The dimensions of the tensor.
     */
    template <typename... Args>
    FuncPointerTensor(std::string name, T (*func_ptr)(std::array<ptrdiff_t, Rank> const &), Args... dims)
        : tensor_base::FunctionTensor<T, Rank>(name, dims...), _func_ptr(func_ptr) {}

    /**
     * @brief Copy a function pointer tensor.
     *
     * @param copy The tensor to copy.
     */
    FuncPointerTensor(FuncPointerTensor<T, Rank> const &copy) : tensor_base::FunctionTensor<T, Rank>(copy) { _func_ptr = copy._func_ptr; }

    virtual ~FuncPointerTensor() = default;

    /**
     * @brief Call the function with the given indices.
     */
    T call(std::array<ptrdiff_t, Rank> const &inds) const override { return _func_ptr(inds); }
};

/**
 * @struct FunctionTensorView
 *
 * @brief Acts as a view of a FunctionTensor.
 *
 * This class allows function tensors to define an offset and different dimensions for the indices.
 * When called, the indices passed will be converted into what they would be in the original function
 * tensor.
 */
template <typename T, size_t rank, size_t UnderlyingRank>
struct FunctionTensorView : public tensor_base::FunctionTensor<T, rank> {
  protected:
    /**
     * @property _func_tensor
     *
     * @brief A pointer to the underlying function tensor.
     */
    tensor_base::FunctionTensor<T, UnderlyingRank> const *_func_tensor;

    /**
     * @property _offsets
     *
     * @brief A list of the index offsets.
     */
    Offset<rank> _offsets;

    /**
     * @property _index_template
     *
     * @brief Contains a template for the indices.
     *
     * The template will contain negative values where indices need to be replaced. Non-negative values
     * will be left intact when being passed to the viewed tensor.
     */
    std::array<ptrdiff_t, UnderlyingRank> _index_template;

    /**
     * @property _full_view
     *
     * @brief Determines whether the view sees all of the data of the underlying tensor.
     */
    bool _full_view{true};

    /**
     * @brief Takes the input indices and applies the offsets and template to prepare it to be passed to the
     * underlying tensor.
     */
    virtual std::array<ptrdiff_t, UnderlyingRank> apply_view(std::array<ptrdiff_t, rank> const &inds) const {
        std::array<ptrdiff_t, UnderlyingRank> out;
        int                                   curr_rank = 0;
        for (int i = 0; std::cmp_less(i, Rank) && curr_rank < UnderlyingRank; i++) {
            while (_index_template.at(curr_rank) >= 0) {
                out[curr_rank] = _index_template[curr_rank];
                curr_rank++;
            }
            out.at(curr_rank) = inds.at(i);
            if (inds.at(i) < 0) {
                out.at(curr_rank) = inds.at(i) + this->_dims[i];
            }
            if (out.at(curr_rank) >= this->_dims[i] || out.at(curr_rank) < 0) {
                EINSUMS_THROW_EXCEPTION(std::out_of_range,
                                        "Function tensor view index out of range! Index of rank {} is {}, which is < 0 or >= {}.", i,
                                        inds.at(i), this->_dims[i]);
            }

            out.at(curr_rank) += _offsets[i];
            curr_rank++;
        }

        return out;
    }

  public:
    /**
     * @typedef ValueType
     *
     * @brief The type of data returned by the tensor.
     */
    using ValueType = T;

    /**
     * @property Rank
     *
     * @brief The rank of the view.
     */
    constexpr static size_t Rank = rank;

    /**
     * @brief Create a function tensor view with the given name, offsets, dimensions, etc.
     *
     * @param name The name of the view.
     * @param func_tens The underlying tensor that this object views.
     * @param offsets The offset for each axis in the view.
     * @param dims The dimensions of the view.
     * @param index_template A template for the indices. It contains negative numbers where values should be filled in, and full indices
     * where they have been explicitly specified.
     */
    FunctionTensorView(std::string name, tensor_base::FunctionTensor<T, UnderlyingRank> *func_tens, Offset<Rank> const &offsets,
                       Dim<Rank> const &dims, std::array<ptrdiff_t, UnderlyingRank> const &index_template)
        : _offsets{offsets}, _func_tensor(func_tens), _index_template{index_template}, tensor_base::FunctionTensor<T, Rank>(name, dims) {
        if constexpr (Rank != UnderlyingRank) {
            _full_view = false;
        } else {
            for (int i = 0; i < UnderlyingRank; i++) {
                if (index_template.at(i) >= 0) {
                    _full_view = false;
                    break;
                }
                if (dims[i] != func_tens->dim(i)) {
                    _full_view = false;
                    break;
                }
                if (_offsets.at(i) != 0) {
                    _full_view = false;
                    break;
                }
            }
        }
    }

    /**
     * @brief Create a function tensor view with the given offsets, dimensions, etc.
     *
     * @param func_tens The underlying tensor that this object views.
     * @param offsets The offset for each axis in the view.
     * @param dims The dimensions of the view.
     * @param index_template A template for the indices. It contains negative numbers where values should be filled in, and full indices
     * where they have been explicitly specified.
     */
    FunctionTensorView(tensor_base::FunctionTensor<T, UnderlyingRank> const *func_tens, Offset<Rank> const &offsets, Dim<Rank> const &dims,
                       std::array<ptrdiff_t, UnderlyingRank> const &index_template)
        : _offsets{offsets}, _func_tensor(func_tens), _index_template{index_template}, tensor_base::FunctionTensor<T, Rank>(dims) {}

    /**
     * @brief Function tensor view copy constructor.
     */
    FunctionTensorView(FunctionTensorView const &copy) : tensor_base::FunctionTensor<T, Rank>(copy) {
        _func_tensor    = copy._func_tensor;
        _offsets        = copy._offsets;
        _index_template = copy._index_template;
        _full_view      = copy._full_view;
    }

    /**
     * @brief Call the underlying function of the function tensor.
     */
    T call(std::array<ptrdiff_t, Rank> const &inds) const override {
        auto fixed_inds = apply_view(inds);
        return _func_tensor->call(fixed_inds);
    }

    /**
     * @brief Returns whether the view sees all of the data of the underlying tensor.
     */
    [[nodiscard]] bool full_view_of_underlying() const override { return _full_view; }
};

/**
 * @struct KroneckerDelta
 *
 * @brief This function tensor represents the Kronecker delta, and can be used in einsum calls.
 *
 * This function tensor can act as an example of what can be done with the more general function tensors.
 */
template <typename T>
struct KroneckerDelta {
  public:
    /**
     * @property Rank
     *
     * @brief The rank of the tensor.
     */
    constexpr static size_t Rank = 2;

    /**
     * @typedef ValueType
     *
     * @brief The type of data returned by the tensor.
     */
    using ValueType = T;

    /**
     * Default constructs the tensor to have dimensions of zero.
     */
    constexpr KroneckerDelta() : _dim{0} {}

    /**
     * @brief Construct a new Kronecker delta tensor with the specified dimension.
     *
     * @param dim The length of one side of the tensor. The Kronecker delta tensor is square, so this will be the same for both dimensions.
     */
    constexpr KroneckerDelta(size_t dim) : _dim{dim} {}

    constexpr ~KroneckerDelta() = default;

    /**
     * @brief Compute the Kronecker delta.
     *
     * This just compares the indices. If the indices are the same, then it returns 1. Otherwise, it returns zero.
     *
     * @param i The first index.
     * @param j The second index.
     */
    template <typename T1, typename T2>
        requires(std::is_integral_v<T1> && std::is_integral_v<T2>)
    constexpr T operator()(T1 i, T2 j) const {
        if (i == j) {
            return T{1.0};
        } else {
            return T{0.0};
        }
    }

    /**
     * @brief Compute the Kronecker delta.
     *
     * This just compares the indices. If the indices are the same, then it returns 1. Otherwise, it returns zero.
     *
     * @param index The indices for the tensor.
     */
    template <typename int_type>
        requires(std::is_integral_v<int_type>)
    constexpr T operator()(std::array<int_type, 2> const &index) const {
        if (index[0] == index[1]) {
            return T{1.0};
        } else {
            return T{0.0};
        }
    }

    /**
     * @brief Compute the Kronecker delta.
     *
     * This just compares the indices. If the indices are the same, then it returns 1. Otherwise, it returns zero.
     *
     * @param i The first index.
     * @param j The second index.
     */
    template <typename T1, typename T2>
        requires(std::is_integral_v<T1> && std::is_integral_v<T2>)
    constexpr T subscript(T1 i, T2 j) const {
        if (i == j) {
            return T{1.0};
        } else {
            return T{0.0};
        }
    }

    /**
     * @brief Compute the Kronecker delta.
     *
     * This just compares the indices. If the indices are the same, then it returns 1. Otherwise, it returns zero.
     *
     * @param index The indices for the tensor.
     */
    template <typename int_type>
        requires(std::is_integral_v<int_type>)
    constexpr T subscript(std::array<int_type, 2> const &index) const {
        if (index[0] == index[1]) {
            return T{1.0};
        } else {
            return T{0.0};
        }
    }

    /**
     * @brief Returns the length of one side of the tensor.
     *
     * Because the tensor is square, the argument is ignored.
     *
     * @param i The axis to query. Ignored because the tensor is square.
     */
    [[nodiscard]] constexpr size_t dim(int /*i*/) const { return _dim; }

    /**
     * @brief Get the dimensions of the tensor.
     */
    [[nodiscard]] constexpr Dim<2> dims() const { return Dim{_dim, _dim}; }

    /**
     * @brief Get the name of the tensor.
     *
     * The name of the tensor is always <tt>"Kronecker delta"</tt>.
     */
    [[nodiscard]] constexpr std::string name() const { return "Kronecker delta"; }

    /**
     * @brief Checks to see if the tensor sees all of its data.
     *
     * Because this is the full tensor, this will always return true.
     */
    [[nodiscard]] constexpr bool full_view_of_underlying() const { return true; }

  private:
    /**
     * @property dim_
     *
     * @brief This is the length of one side of the tensor.
     */
    size_t _dim;
};

} // namespace einsums