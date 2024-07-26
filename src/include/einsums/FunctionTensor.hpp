#pragma once

#include "einsums/_Common.hpp"

#include "einsums/Tensor.hpp"
#include "einsums/utility/TensorBases.hpp"
#include "einsums/utility/TensorTraits.hpp"

#include <stdexcept>
#include <type_traits>

namespace einsums {

// forward declaration.
template <typename T, size_t Rank, size_t BaseRank>
struct FunctionTensorView;

namespace tensor_props {
template <typename T, size_t Rank>
struct FunctionTensorBase : public virtual TensorBase<T, Rank>, virtual FunctionTensorBaseNoExtra {
  protected:
    Dim<Rank>   _dims;
    std::string _name;
    size_t      _size;

    virtual void fix_indices(std::array<int, Rank> *inds) const {
        for (int i = 0; i < Rank; i++) {
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
                throw std::out_of_range(fmt::format("{}is too far less than zero or is greater than {}", message, _dims[i]));
            }
        }
    }

  public:
    FunctionTensorBase(const FunctionTensorBase &) = default;
    FunctionTensorBase(FunctionTensorBase &&)      = default;

    template <typename... Args>
        requires(!std::is_same_v<Args, Dim<Rank>> || ...)
    FunctionTensorBase(std::string name, Args... dims) : _dims{dims...}, _name{name} {
        _size = 1;

        // Not parallel. Just vectorize.
#pragma omp for simd reduction(* : _size)
        for (int i = 0; i < Rank; i++) {
            _size *= _dims[i];
        }
    }

    FunctionTensorBase(std::string name, Dim<Rank> dims) : _dims(dims), _name{name} {
        _size = 1;

        // Not parallel. Just vectorize.
#pragma omp for simd reduction(* : _size)
        for (int i = 0; i < Rank; i++) {
            _size *= _dims[i];
        }
    }

    FunctionTensorBase(Dim<Rank> dims) : _dims(dims), _name{"(unnamed)"} {
        _size = 1;

        // Not parallel. Just vectorize.
#pragma omp for simd reduction(* : _size)
        for (int i = 0; i < Rank; i++) {
            _size *= _dims[i];
        }
    }

    virtual ~FunctionTensorBase() = default;

    virtual T call(const std::array<int, Rank> &inds) const = 0;

    template <typename... MultiIndex>
        requires requires {
            requires(sizeof...(MultiIndex) == Rank);
            requires(std::is_integral_v<MultiIndex> && ...);
        }
    T operator()(MultiIndex... inds) const {
        auto new_inds = std::array<int, Rank>{static_cast<int>(inds)...};

        fix_indices(&new_inds);

        return this->call(new_inds);
    }

    template <typename Storage>
        requires requires {
            requires !std::is_integral_v<Storage>;
            requires !std::is_same_v<Storage, AllT>;
            requires !std::is_same_v<Storage, Range>;
            requires !std::is_same_v<Storage, std::array<int, Rank>>;
        }
    T operator()(const Storage &inds) const {
        auto new_inds = std::array<int, Rank>();

        for (int i = 0; i < Rank; i++) {
            new_inds[i] = (int)inds.at(i);
        }

        fix_indices(&new_inds);

        return this->call(new_inds);
    }

    template <typename... MultiIndex>
        requires requires {
            requires(sizeof...(MultiIndex) == Rank);
            requires(AtLeastOneOfType<AllT, MultiIndex...>);
            requires(NoneOfType<std::array<int, Rank>, MultiIndex...>);
        }
    auto operator()(MultiIndex... inds) const
        -> FunctionTensorView<T, count_of_type<AllT, MultiIndex...>() + count_of_type<Range, MultiIndex...>(), Rank> {
        const auto &indices = std::forward_as_tuple(inds...);

        std::vector<int> index_template(Rank);

        Offset<count_of_type<AllT, MultiIndex...>() + count_of_type<Range, MultiIndex...>()> offsets;
        Dim<count_of_type<AllT, MultiIndex...>() + count_of_type<Range, MultiIndex...>()>    dims{};

        int counter{0};
        for_sequence<sizeof...(MultiIndex)>([&](auto i) {
            // println("looking at {}", i);
            if constexpr (std::is_convertible_v<std::tuple_element_t<i, std::tuple<MultiIndex...>>, std::int64_t>) {
                auto tmp = static_cast<std::int64_t>(std::get<i>(indices));
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

    template <typename... MultiIndex>
        requires NumOfType<Range, Rank, MultiIndex...>
    auto operator()(MultiIndex... index) const -> FunctionTensorView<T, Rank, Rank> {
        Dim<Rank>        dims{};
        Offset<Rank>     offset{};
        std::vector<int> index_template(Rank);

        auto ranges = get_array_from_tuple<std::array<Range, Rank>>(std::forward_as_tuple(index...));

        for (int r = 0; r < Rank; r++) {
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

    Dim<Rank> dims() const { return _dims; }

    auto dim(int d) const -> size_t { return _dims[d]; }

    const std::string &name() const { return _name; }

    void set_name(std::string &str) { _name = str; }
};

} // namespace tensor_props

template <typename T, size_t Rank>
struct FuncPointerTensor : public virtual tensor_props::FunctionTensorBase<T, Rank>, virtual tensor_props::CoreTensorBase {
  private:
    T (*_func_ptr)(const std::array<int, Rank> &);

  public:
    template <typename... Args>
    FuncPointerTensor(std::string name, T (*func_ptr)(const std::array<int, Rank> &), Args... dims)
        : tensor_props::FunctionTensorBase<T, Rank>(name, dims...), _func_ptr(func_ptr) {}

    virtual ~FuncPointerTensor() = default;

    virtual T call(const std::array<int, Rank> &inds) const override { return _func_ptr(inds); }

    size_t dim(int d) const override { return tensor_props::FunctionTensorBase<T, Rank>::dim(d); }
};

template <typename T, size_t Rank, size_t UnderlyingRank>
struct FunctionTensorView : public virtual tensor_props::FunctionTensorBase<T, Rank>,
                            virtual tensor_props::TensorViewBase<T, Rank, tensor_props::FunctionTensorBase<T, UnderlyingRank>> {
  protected:
    const tensor_props::FunctionTensorBase<T, UnderlyingRank> *_func_tensor;
    Offset<Rank>                                               _offsets;
    std::vector<int>                                           _index_template;
    bool                                                       _full_view{true};

    virtual std::vector<int> fix_indices(const std::array<int, Rank> &inds) const {
        std::vector<int> out(_index_template);
        int              curr_rank = 0;
        for (int i = 0; i < Rank && curr_rank < UnderlyingRank; i++) {
            while (out.at(curr_rank) >= 0) {
                curr_rank++;
            }
            out.at(curr_rank) = inds.at(i);
            if (out.at(curr_rank) < 0) {
                out.at(curr_rank) += this->_dims[i];
            }
            if (out.at(curr_rank) >= this->_dims[i] || out.at(curr_rank) < 0) {
                throw std::out_of_range(
                    fmt::format("Function tensor view index out of range! Index of rank {} is {}, which is < 0 or >= {}.", i, inds.at(i),
                                this->_dims[i]));
            }

            out.at(curr_rank) += _offsets[i];
            curr_rank++;
        }

        return out;
    }

  public:
    FunctionTensorView(std::string name, const tensor_props::FunctionTensorBase<T, UnderlyingRank> *func_tens, const Offset<Rank> &offsets,
                       const Dim<Rank> &dims, std::vector<int> index_template)
        : _offsets{offsets}, _func_tensor(func_tens), _index_template(index_template),
          tensor_props::FunctionTensorBase<T, Rank>(name, dims) {
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

    FunctionTensorView(const tensor_props::FunctionTensorBase<T, UnderlyingRank> *func_tens, const Offset<Rank> &offsets,
                       const Dim<Rank> &dims, std::vector<int> index_template)
        : _offsets{offsets}, _func_tensor(func_tens), _index_template(index_template), tensor_props::FunctionTensorBase<T, Rank>(dims) {}

    virtual T call(const std::array<int, Rank> &inds) const override { return _func_tensor->call(inds); }

    template <typename... MultiIndex>
        requires requires {
            requires(sizeof...(MultiIndex) == Rank);
            requires(std::is_integral_v<MultiIndex> && ...);
        }
    T operator()(MultiIndex... inds) const {
        auto new_inds = std::array<int, Rank>{static_cast<int>(inds)...};

        auto fixed_inds = fix_indices(&new_inds);

        return (*_func_tensor)(fixed_inds);
    }

    template <typename Storage>
        requires requires {
            requires !std::is_integral_v<Storage>;
            requires !std::is_same_v<Storage, AllT>;
            requires !std::is_same_v<Storage, Range>;
        }
    T operator()(const Storage &inds) const {
        auto new_inds = std::array<int, Rank>();

        for (int i = 0; i < Rank; i++) {
            new_inds[i] = (int)inds.at(i);
        }

        fix_indices(&new_inds);

        return (*_func_tensor)(new_inds);
    }

    template <typename... MultiIndex>
        requires requires {
            requires(sizeof...(MultiIndex) == Rank);
            requires(AtLeastOneOfType<AllT, MultiIndex...>);
            requires(NoneOfType<std::array<int, Rank>, MultiIndex...>);
        }
    auto operator()(MultiIndex... inds) const
        -> FunctionTensorView<T, count_of_type<AllT, MultiIndex...>() + count_of_type<Range, MultiIndex...>(), UnderlyingRank> {
        const auto &indices = std::forward_as_tuple(inds...);

        std::vector<int> index_template(_index_template);

        Offset<count_of_type<AllT, MultiIndex...>() + count_of_type<Range, MultiIndex...>()> offsets;
        Dim<count_of_type<AllT, MultiIndex...>() + count_of_type<Range, MultiIndex...>()>    dims{};

        int counter{0}, counter_2{0};
        for_sequence<sizeof...(MultiIndex)>([&](auto i) {
            // println("looking at {}", i);
            while (index_template[counter_2] >= 0) { // Skip already assigned values.
                counter_2++;
            }
            if constexpr (std::is_convertible_v<std::tuple_element_t<i, std::tuple<MultiIndex...>>, std::int64_t>) {
                auto tmp = static_cast<std::int64_t>(std::get<i>(indices));
                if (tmp < 0)
                    tmp = this->_dims[i] + tmp;
                index_template[counter_2] = tmp;
            } else if constexpr (std::is_same_v<AllT, std::tuple_element_t<i, std::tuple<MultiIndex...>>>) {
                dims[counter]             = this->_dims[i];
                offsets[counter]          = 0;
                index_template[counter_2] = -1;
                counter++;

            } else if constexpr (std::is_same_v<Range, std::tuple_element_t<i, std::tuple<MultiIndex...>>>) {
                auto range       = std::get<i>(indices);
                offsets[counter] = range[0];
                if (range[1] < 0) {
                    auto temp = this->_dims[i] + range[1];
                    range[1]  = temp;
                }
                dims[counter]     = range[1] - range[0];
                index_template[i] = -1;
                counter++;
            }
            counter_2++;
        });

        return FunctionTensorView<T, count_of_type<AllT, MultiIndex...>() + count_of_type<Range, MultiIndex...>(), UnderlyingRank>(
            this, offsets, dims, index_template);
    }

    template <typename... MultiIndex>
        requires NumOfType<Range, Rank, MultiIndex...>
    auto operator()(MultiIndex... index) const -> FunctionTensorView<T, Rank, UnderlyingRank> {
        Dim<Rank>    dims{};
        Offset<Rank> offset{};

        auto ranges = get_array_from_tuple<std::array<Range, Rank>>(std::forward_as_tuple(index...));

        for (int r = 0; r < Rank; r++) {
            auto range = ranges[r];
            offset[r]  = range[0];
            if (range[1] < 0) {
                auto temp = this->_dims[r] + range[1];
                range[1]  = temp;
            }
            dims[r] = range[1] - range[0];
        }
        // index template doesn't change. No single indices to modify it.

        return FunctionTensorView<T, Rank, Rank>{*this, std::move(offset), std::move(dims), _index_template};
    }

    bool full_view_of_underlying() const override { return _full_view; }
};

} // namespace einsums