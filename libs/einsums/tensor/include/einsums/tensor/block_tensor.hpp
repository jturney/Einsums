//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/assert.hpp>
#include <einsums/concepts/complex.hpp>
#include <einsums/concepts/file.hpp>
#include <einsums/concepts/tensor.hpp>
#include <einsums/errors/throw_exception.hpp>
#include <einsums/print.hpp>
#include <einsums/tensor/tensor.hpp>
#include <einsums/tensor/tensor_fwd.hpp> // for tensor_print_options
#include <einsums/tensor_base/tensor_base.hpp>
#include <einsums/util/type_name.hpp>

#if defined(EINSUMS_COMPUTE_CODE)
#    include <einsums/tensor/device_tensor.hpp>
#endif

#include <numeric>
#include <vector>

namespace einsums {

namespace tensor_base {

/**
 * @struct block_tensor
 *
 * Represents a block-diagonal tensor.
 *
 * @tparam T The data type stored in this tensor.
 * @tparam Rank The rank of the tensor
 * @tparam TensorType The underlying type for the tensors.
 */
template <typename T, size_t Rank, typename TensorType>
struct block_tensor : public virtual collected_tensor<T, Rank, TensorType>,
                      virtual tensor<T, Rank>,
                      virtual block_tensor_no_extra,
                      virtual lockable_tensor,
                      virtual algebra_optimized_tensor {
  protected:
    std::string _name{"(Unnamed)"};
    size_t      _dim{0}; // Only allowing square tensors.

    std::vector<TensorType> _blocks{};
    std::vector<range>      _ranges{};
    std::vector<size_t>     _dims;

    template <typename OtherT, size_t OtherRank, typename OtherTensorType>
    friend struct block_tensor;

    T _zero_value{0.0};

    void update_dims() {
        if (_dims.size() != _blocks.size()) {
            _dims.resize(_blocks.size());
        }
        if (_ranges.size() != _blocks.size()) {
            _ranges.resize(_blocks.size());
        }

        size_t sum = 0;

        for (int i = 0; i < _blocks.size(); i++) {
            _dims[i]   = _blocks[i].dim(0);
            _ranges[i] = range{sum, sum + _dims[i]};
            sum += _dims[i];
        }

        _dim = sum;
    }

  public:
    /**
     * @brief Construct a new BlockTensor object. Default constructor.
     */
    block_tensor() = default;

    /**
     * @brief Construct a new BlockTensor object. Default copy constructor
     */
    block_tensor(const block_tensor &other) : _ranges{other._ranges}, _dims{other._dims}, _blocks{}, _dim{other._dim} {
        for (int i = 0; i < other._blocks.size(); i++) {
            _blocks.emplace_back((const TensorType &)other._blocks[i]);
        }

        update_dims();
    }

    /**
     * @brief Destroy the BlockTensor object.
     */
    virtual ~block_tensor() = default;

    /**
     * @brief Construct a new BlockTensor object with the given name and blockdimensions.
     *
     * Constructs a new BlockTensor object using the information provided in \p name and \p block_dims .
     *
     * @code
     * // Constructs a rank 4 tensor with two blocks, the first is 2x2x2x2, the second is 3x3x3x3.
     * auto A = BlockTensor<double, 4>("A", 2, 3);
     * @endcode
     *
     * The newly constructed Tensor is NOT zeroed out for you. If you start having NaN issues
     * in your code try calling Tensor.zero() or zero(Tensor) to see if that resolves it.
     *
     * @tparam Dims Variadic template arguments for the dimensions. Must be castable to size_t.
     * @param name Name of the new tensor.
     * @param block_dims The size of each block.
     */
    template <typename... Dims>
    explicit block_tensor(std::string name, Dims... block_dims)
        : _name{std::move(name)}, _dim{(static_cast<size_t>(block_dims) + ... + 0)}, _blocks(), _ranges(), _dims(sizeof...(Dims)) {
        auto dim_array   = einsums::dim{block_dims...};
        auto _block_dims = einsums::dim<Rank>();

        for (int i = 0; i < sizeof...(Dims); i++) {
            _block_dims.fill(dim_array[i]);

            _blocks.emplace_back(_block_dims);
        }

        update_dims();
    }

    /**
     * @brief Construct a new BlockTensor object with the given name and blockdimensions.
     *
     * Constructs a new BlockTensor object using the information provided in \p name and \p block_dims .
     *
     * @code
     * // Constructs a rank 4 tensor with two blocks, the first is 2x2x2x2, the second is 3x3x3x3.
     * auto A = BlockTensor<double, 4>("A", 2, 3);
     * @endcode
     *
     * The newly constructed Tensor is NOT zeroed out for you. If you start having NaN issues
     * in your code try calling Tensor.zero() or zero(Tensor) to see if that resolves it.
     *
     * @tparam Dims Variadic template arguments for the dimensions. Must be castable to size_t.
     * @param name Name of the new tensor.
     * @param block_dims The size of each block.
     */
    template <typename ArrayArg>
    explicit block_tensor(std::string name, const ArrayArg &block_dims)
        : _name{std::move(name)}, _dim{0}, _blocks(), _ranges(), _dims(block_dims.cbegin(), block_dims.cend()) {

        auto _block_dims = dim<Rank>();

        for (int i = 0; i < block_dims.size(); i++) {
            _block_dims.fill(block_dims[i]);

            _blocks.emplace_back(_block_dims);
        }

        update_dims();
    }

    /**
     * @brief Construct a new BlockTensor object using the dimensions given by Dim object.
     *
     * @param block_dims The dimension of each block.
     */
    template <size_t Dims>
    explicit block_tensor(dim<Dims> block_dims) : _blocks(), _ranges(), _dims(block_dims) {
        auto _block_dims = dim<Rank>();

        for (int i = 0; i < Dims; i++) {
            _block_dims.fill(_block_dims[i]);

            _blocks.emplace_back(_block_dims);
        }

        update_dims();
    }

    /**
     * @brief Find the block which can be indexed by the given index.
     *
     * This finds the block for which the index is within its range.
     */
    int block_of(size_t index) const {
        for (int i = 0; i < _ranges.size(); i++) {
            if (_ranges[i][0] <= index && _ranges[i][1] > index) {
                return i;
            }
        }

        EINSUMS_THROW_EXCEPTION(error::bad_parameter, "Index out of range!");
    }

    /**
     * @brief Zeroes out the tensor data.
     */
    void zero() {
        EINSUMS_OMP_PARALLEL_FOR
        for (int i = 0; i < _blocks.size(); i++) {
            _blocks[i].zero();
        }
    }

    /**
     * @brief Set the all entries in the blocks to the given value.
     *
     * @param value Value to set the elements to.
     */
    void set_all(T value) {
        EINSUMS_OMP_PARALLEL_FOR
        for (int i = 0; i < _blocks.size(); i++) {
            _blocks[i].set_all(value);
        }
    }

    /**
     * @brief Return the selected block with an integer ID.
     */
    const TensorType &block(int id) const { return _blocks.at(id); }

    /**
     * @brief Return the selected block with an integer ID.
     */
    TensorType &block(int id) { return _blocks.at(id); }

    /**
     * @brief Return the selected block with an integer ID.
     */
    const TensorType &block(const std::string &name) const {
        for (int i = 0; i < _blocks.size(); i++) {
            if (_blocks[i].name() == name) {
                return _blocks[i];
            }
        }
        if (_blocks.size() == 0) {
            EINSUMS_THROW_EXCEPTION(error::bad_parameter, "Could not find block with the name '{}': no blocks in tensor.", name);
        }
        EINSUMS_THROW_EXCEPTION(error::bad_parameter, "Could not find block with the name '{}': no blocks with given name.", name);
    }

    /**
     * @brief Return the selected block with an integer ID.
     */
    TensorType &block(const std::string &name) {
        for (int i = 0; i < _blocks.size(); i++) {
            if (_blocks[i].name() == name) {
                return _blocks[i];
            }
        }
        if (_blocks.size() == 0) {
            EINSUMS_THROW_EXCEPTION(error::bad_parameter, "Could not find block with the name '{}': no blocks in tensor.", name);
        }
        EINSUMS_THROW_EXCEPTION(error::bad_parameter, "Could not find block with the name '{}': no blocks with given name.", name);
    }

    /**
     * @brief Add a block to the end of the list of blocks.
     */
    void push_block(TensorType &&value) {
        for (int i = 0; i < Rank; i++) {
            if (value.dim(i) != value.dim(0)) {
                EINSUMS_THROW_EXCEPTION(
                    error::bad_parameter,
                    "Can only push square/hypersquare tensors to a block tensor. Make sure all dimensions are the same.");
            }
        }
        _blocks.push_back(value);
        update_dims();
    }

    /**
     * @brief Add a block to the specified position in the
     */
    void insert_block(int pos, TensorType &&value) {
        for (int i = 0; i < Rank; i++) {
            if (value.dim(i) != value.dim(0)) {
                EINSUMS_THROW_EXCEPTION(
                    error::bad_parameter,
                    "Can only push square/hypersquare tensors to a block tensor. Make sure all dimensions are the same.");
            }
        }
        // Add the block.
        _blocks.insert(std::next(_blocks.begin(), pos), value);

        update_dims();
    }

    /**
     * @brief Add a block to the end of the list of blocks.
     */
    void push_block(const TensorType &value) {
        for (int i = 0; i < Rank; i++) {
            if (value.dim(i) != value.dim(0)) {
                EINSUMS_THROW_EXCEPTION(
                    error::bad_parameter,
                    "Can only push square/hypersquare tensors to a block tensor. Make sure all dimensions are the same.");
            }
        }
        _blocks.push_back(value);
        update_dims();
    }

    /**
     * @brief Add a block to the specified position in the list of blocks.
     */
    void insert_block(int pos, const TensorType &value) {
        for (int i = 0; i < Rank; i++) {
            if (value.dim(i) != value.dim(0)) {
                EINSUMS_THROW_EXCEPTION(
                    error::bad_parameter,
                    "Can only push square/hypersquare tensors to a block tensor. Make sure all dimensions are the same.");
            }
        }
        // Add the block.
        _blocks.insert(std::next(_blocks.begin(), pos), value);

        update_dims();
    }

    /**
     * Returns a pointer into the tensor at the given location.
     *
     *
     * @tparam MultiIndex The datatypes of the passed parameters. Must be castable to
     * @param index The explicit desired index into the tensor. Must be castable to std::int64_t.
     * @return A pointer into the tensor at the requested location.
     */
    template <typename... MultiIndex>
        requires requires {
            requires NoneOfType<all_t, MultiIndex...>;
            requires NoneOfType<range, MultiIndex...>;
        }
    auto data(MultiIndex... index) -> T * {
#if !defined(DOXYGEN_SHOULD_SKIP_THIS)
        EINSUMS_ASSERT(sizeof...(MultiIndex) <= Rank);

        auto index_list = std::array{static_cast<std::int64_t>(index)...};
        int  block      = -1;

        for (auto [i, _index] : enumerate(index_list)) {
            if (_index < 0) {
                index_list[i] = _dim + _index;
            }

            // Find the block.
            if (block == -1) {
                for (int j = 0; j < _ranges.size(); j++) {
                    if (_ranges[j][0] <= _index && _index < _ranges[j][1]) {
                        block = j;
                        break;
                    }
                }
            }

            if (_ranges[block][0] <= _index && _index < _ranges[block][1]) {
                // Remap the index to be in the block.
                index_list[i] -= _ranges[block][0];
            } else {
                return nullptr; // The indices point outside of all the blocks.
            }
        }

        size_t ordinal = std::inner_product(index_list.begin(), index_list.end(), _blocks[block].strides().begin(), size_t{0});
        return &(_blocks[block].data()[ordinal]);
#endif
    }

    /**
     * @brief Subscripts into the tensor.
     *
     * This version works when all elements are explicit values into the tensor.
     * It does not work with the All or Range tags.
     *
     * @tparam MultiIndex Datatype of the indices. Must be castable to std::int64_t.
     * @param index The explicit desired index into the tensor. Elements must be castable to std::int64_t.
     * @return const T&
     */
    template <typename... MultiIndex>
        requires requires {
            requires NoneOfType<all_t, MultiIndex...>;
            requires NoneOfType<range, MultiIndex...>;
        }
    auto operator()(MultiIndex... index) const -> const T & {

        static_assert(sizeof...(MultiIndex) == Rank);

        auto index_list = std::array{static_cast<std::int64_t>(index)...};

        int block = -1;
        for (auto [i, _index] : enumerate(index_list)) {
            if (_index < 0) {
                index_list[i] = _dim + _index;
            }

            if (block == -1) {
                for (int j = 0; j < _ranges.size(); j++) {
                    if (_ranges[j][0] <= _index && _index < _ranges[j][1]) {
                        block = j;
                        break;
                    }
                }
            }

            if (_ranges[block][0] <= _index && _index < _ranges[block][1]) {
                // Remap the index to be in the block.
                index_list[i] -= _ranges[block][0];
            } else {
                return 0; // The indices point outside of all the blocks.
            }
        }
        size_t ordinal = std::inner_product(index_list.begin(), index_list.end(), _blocks.at(block).strides().begin(), size_t{0});
        return _blocks.at(block).data()[ordinal];
    }

    /**
     * @brief Subscripts into the tensor.
     *
     * This version works when all elements are explicit values into the tensor.
     * It does not work with the All or Range tags.
     *
     * @tparam MultiIndex Datatype of the indices. Must be castable to std::int64_t.
     * @param index The explicit desired index into the tensor. Elements must be castable to std::int64_t.
     * @return T&
     */
    template <typename... MultiIndex>
        requires requires {
            requires NoneOfType<all_t, MultiIndex...>;
            requires NoneOfType<range, MultiIndex...>;
        }
    auto operator()(MultiIndex... index) -> T & {

        static_assert(sizeof...(MultiIndex) == Rank);

        auto index_list = std::array{static_cast<std::int64_t>(index)...};

        int block = -1;
        for (auto [i, _index] : enumerate(index_list)) {
            if (_index < 0) {
                index_list[i] = _dim + _index;
            }

            if (block == -1) {
                for (int j = 0; j < _ranges.size(); j++) {
                    if (_ranges[j][0] <= _index && _index < _ranges[j][1]) {
                        block = j;
                        break;
                    }
                }
            }

            if (_ranges.at(block)[0] <= _index && _index < _ranges.at(block)[1]) {
                // Remap the index to be in the block.
                index_list[i] -= _ranges.at(block)[0];
            } else {
                if (_zero_value != T(0.0)) {
                    _zero_value = T(0.0);
                }
                return _zero_value;
            }
        }

        return std::apply(_blocks.at(block), index_list);
    }

    /**
     * @brief Return the block with the given index.
     */
    const TensorType &operator[](size_t index) const { return block(index); }

    /**
     * @brief Return the block with the given index.
     */
    TensorType &operator[](size_t index) { return block(index); }

    /**
     * @brief Return the block with the given name.
     */
    const TensorType &operator[](const std::string &name) const { return block(name); }

    /**
     * @brief Return the block with the given name.
     */
    TensorType &operator[](const std::string &name) { return block(name); }

    /**
     * @brief Copy assignment.
     */
    auto operator=(const block_tensor<T, Rank, TensorType> &other) -> block_tensor<T, Rank, TensorType> & {

        if (_blocks.size() != other._blocks.size()) {
            _blocks.resize(other._blocks.size());
        }

        _dims = other._dims;

        _dim = other._dim;

        _ranges = other._ranges;

        EINSUMS_ASSERT(_dims.size() > 0 || _dim == 0);
        EINSUMS_ASSERT(_dims.size() == _ranges.size());
        EINSUMS_ASSERT(_dims.size() == _blocks.size());

        EINSUMS_OMP_PARALLEL_FOR
        for (int i = 0; i < _blocks.size(); i++) {
            _blocks[i] = other._blocks[i];
        }

        update_dims();

        return *this;
    }

    /**
     * @brief Copy assignment with a cast.
     */
    template <typename TOther>
        requires(!std::same_as<T, TOther>)
    auto operator=(const block_tensor<TOther, Rank, TensorType> &other) -> block_tensor<T, Rank, TensorType> & {
        if (_blocks.size() != other._blocks.size()) {
            _blocks.resize(other._blocks.size());
        }

        _dims = other._dims;

        _dim = other._dim;

        _ranges = other._ranges;

        EINSUMS_OMP_PARALLEL_FOR
        for (int i = 0; i < _blocks.size(); i++) {
            _blocks[i] = other._blocks[i];
        }

        update_dims();

        return *this;
    }

#define OPERATOR(OP)                                                                                                                       \
    auto operator OP(const T &b)->block_tensor<T, Rank, TensorType> & {                                                                    \
        for (int i = 0; i < _blocks.size(); i++) {                                                                                         \
            if (block_dim(i) == 0) {                                                                                                       \
                continue;                                                                                                                  \
            }                                                                                                                              \
            _blocks[i] OP b;                                                                                                               \
        }                                                                                                                                  \
        return *this;                                                                                                                      \
    }                                                                                                                                      \
                                                                                                                                           \
    auto operator OP(const block_tensor<T, Rank, TensorType> &b)->block_tensor<T, Rank, TensorType> & {                                    \
        if (_blocks.size() != b._blocks.size()) {                                                                                          \
            EINSUMS_THROW_EXCEPTION(error::bad_parameter, "tensors differ in number of blocks : {} {}", _blocks.size(), b._blocks.size()); \
        }                                                                                                                                  \
        for (int i = 0; i < _blocks.size(); i++) {                                                                                         \
            if (_blocks[i].size() != b._blocks[i].size()) {                                                                                \
                EINSUMS_THROW_EXCEPTION(error::bad_parameter, "tensor blocks differ in size : {} {}", _blocks[i].size(),                   \
                                        b._blocks[i].size());                                                                              \
            }                                                                                                                              \
        }                                                                                                                                  \
        EINSUMS_OMP_PARALLEL_FOR                                                                                                           \
        for (int i = 0; i < _blocks.size(); i++) {                                                                                         \
            if (block_dim(i) == 0) {                                                                                                       \
                continue;                                                                                                                  \
            }                                                                                                                              \
            _blocks[i] OP b._blocks[i];                                                                                                    \
        }                                                                                                                                  \
        return *this;                                                                                                                      \
    }

    OPERATOR(*=)
    OPERATOR(/=)
    OPERATOR(+=)
    OPERATOR(-=)

#undef OPERATOR

    /**
     * @brief Convert block tensor into a normal tensor.
     */
    explicit operator TensorType() const {
        einsums::dim<Rank> block_dims;

        for (int i = 0; i < Rank; i++) {
            block_dims[i] = _dim;
        }

        TensorType out(block_dims);

        out.set_name(_name);

        out.zero();

        EINSUMS_OMP_PARALLEL_FOR
        for (int i = 0; i < _ranges.size(); i++) {
            if (block_dim(i) == 0) {
                continue;
            }
            std::array<range, Rank> ranges;
            ranges.fill(_ranges[i]);
            std::apply(out, ranges) = _blocks[i];
        }

        return out;
    }

    explicit operator TensorType() {
        update_dims();
        einsums::dim<Rank> block_dims;

        for (int i = 0; i < Rank; i++) {
            block_dims[i] = _dim;
        }

        TensorType out(block_dims);

        out.set_name(_name);

        out.zero();

        EINSUMS_OMP_PARALLEL_FOR
        for (int i = 0; i < _ranges.size(); i++) {
            if (block_dim(i) == 0) {
                continue;
            }
            std::array<range, Rank> ranges;
            ranges.fill(_ranges[i]);
            std::apply(out, ranges) = _blocks[i];
        }

        return out;
    }

    /**
     * @brief Return the number of blocks.
     */
    size_t num_blocks() const { return _blocks.size(); }

    /**
     * @brief Return the dimensions of each block.
     */
    [[nodiscard]] auto block_dims() const -> const std::vector<size_t> { return _dims; }

    /**
     * @brief Return a list containing the ranges for each block.
     */
    std::vector<range> ranges() const { return _ranges; }

    /**
     * @brief Return the range for a given block.
     */
    range block_range(int i) const { return _ranges.at(i); }

    /**
     * @brief Return the dimensions of the given block.
     */
    einsums::dim<Rank> block_dims(size_t block) const { return _blocks.at(block).dims(); }

    /**
     * @brief Return the dimension of a block on a given axis.
     *
     * Because the tensors are assumed to be square, changing the second parameter should not affect the output.
     * The second parameter is not ignored.
     */
    size_t block_dim(size_t block, int ind = 0) const { return _blocks.at(block).dim(ind); }

    /**
     * @brief Return the dimensions of a given block.
     */
    einsums::dim<Rank> block_dims(const std::string &name) const {
        for (auto tens : _blocks) {
            if (tens.name() == name) {
                return tens.block_dims();
            }
        }

        EINSUMS_THROW_EXCEPTION(error::bad_parameter, "Could not find block with the name {}", name);
    }

    /**
     * @brief Return the dimension of a block on a given axis.
     *
     * Because the tensors are assumed to be square, changing the second parameter should not affect the output.
     * The second parameter is not ignored.
     */
    size_t block_dim(const std::string &name, int ind = 0) const {
        for (auto tens : _blocks) {
            if (tens.name() == name) {
                return tens.block_dim(ind);
            }
        }

        EINSUMS_THROW_EXCEPTION(error::bad_parameter, "Could not find block with the name {}", name);
    }

    /**
     * @brief Return the dimensions of this tensor.
     */
    virtual einsums::dim<Rank> dims() const override {
        einsums::dim<Rank> out;
        out.fill(_dim);
        return out;
    }

    /**
     * @brief Return the dimension of this tensor along an axis.
     *
     * Because the tensor is square, the argument is ignored.
     */
    virtual size_t dim(int dim) const override { return _dim; }

    virtual size_t dim() const { return _dim; }

    /**
     * @brief Return the dimensions of each of the blocks.
     */
    std::vector<size_t> vector_dims() const {
        std::vector<size_t> out(num_blocks());

        for (int i = 0; i < out.size(); i++) {
            out[i] = _blocks[i].dim(0);
        }

        return out;
    }

    /**
     * @brief Returns the list of tensors.
     */
    auto vector_data() const -> const std::vector<TensorType> & { return _blocks; }

    /**
     * @brief Returns the list of tensors.
     */
    auto vector_data() -> std::vector<TensorType> & { return _blocks; }

    /**
     * @brief Gets the name of the tensor.
     */
    [[nodiscard]] auto name() const -> const std::string & override { return _name; }

    /**
     * @brief Sets the name of the tensor.
     */
    void set_name(const std::string &name) override { _name = name; }

    /**
     * @brief Gets the name of a block.
     */
    [[nodiscard]] auto name(int i) const -> const std::string & { return _blocks[i].name(); }

    /**
     * @brief Sets the name of a block.
     */
    void set_name(int i, const std::string &name) { _blocks[i].set_name(name); }

    /**
     * @brief Returns the strides of a given block.
     */
    auto strides(int i) const noexcept -> const auto & { return _blocks[i].strides(); }

    // Returns the linear size of the tensor
    [[nodiscard]] auto size() const {
        size_t sum = 0;
        for (auto tens : _blocks) {
            sum += tens.size();
        }

        return sum;
    }

    [[nodiscard]] auto full_view_of_underlying() const noexcept -> bool override { return true; }

    virtual void lock() const override { lockable_tensor::lock(); }

    virtual void unlock() const override { lockable_tensor::unlock(); }

    virtual bool try_lock() const override { return lockable_tensor::try_lock(); }

    /**
     * Lock the specific block.
     */
    virtual void lock(int block) const {
        if constexpr (is_lockable_tensor_v<TensorType>) {
            _blocks.at(block).lock();
        }
    }

    /**
     * Try to lock the specific block.
     */
    virtual bool try_lock(int block) const {
        if constexpr (is_lockable_tensor_v<TensorType>) {
            return _blocks.at(block).try_lock();
        } else {
            return true;
        }
    }

    /**
     * Unlock the specific block.
     */
    virtual void unlock(int block) const {
        if constexpr (is_lockable_tensor_v<TensorType>) {
            _blocks.at(block).unlock();
        }
    }
};
} // namespace tensor_base

/**
 * @struct BlockTensor
 *
 * Represents a block-diagonal tensor in core memory.
 *
 * @tparam T The type of data stored in the tensor.
 * @tparam Rank The rank of the tensor.
 */
template <typename T, size_t Rank>
struct block_tensor : public virtual tensor_base::block_tensor<T, Rank, tensor<T, Rank>>, virtual tensor_base::core_tensor {
  public:
    /**
     * @brief Construct a new block_tensor object. Default constructor.
     */
    block_tensor() = default;

    /**
     * @brief Construct a new block_tensor object. Default copy constructor
     */
    block_tensor(const block_tensor &) = default;

    /**
     * @brief Destroy the block_tensor object.
     */
    ~block_tensor() = default;

    /**
     * @brief Construct a new block_tensor object with the given name and dimensions.
     *
     * Constructs a new block_tensor object using the information provided in \p name and \p block_dims .
     *
     * @code
     * // Constructs a rank 4 tensor with two blocks, the first is 2x2x2x2, the second is 3x3x3x3.
     * auto A = block_tensor<double, 4>("A", 2, 3);
     * @endcode
     *
     * The newly constructed block_tensor is NOT zeroed out for you. If you start having NaN issues
     * in your code try calling Tensor.zero() or zero(Tensor) to see if that resolves it.
     *
     * @tparam Dims Variadic template arguments for the dimensions. Must be castable to size_t.
     * @param name Name of the new tensor.
     * @param block_dims The size of each block.
     */
    template <typename... Dims>
    explicit block_tensor(std::string name, Dims... block_dims)
        : tensor_base::block_tensor<T, Rank, tensor<T, Rank>>(name, block_dims...) {}

    /**
     * @brief Construct a new block_tensor object with the given name and dimensions.
     *
     * Constructs a new block_tensor object using the information provided in \p name and \p block_dims .
     *
     * @code
     * // Constructs a rank 4 tensor with two blocks, the first is 2x2x2x2, the second is 3x3x3x3.
     * auto A = block_tensor<double, 4>("A", 2, 3);
     * @endcode
     *
     * The newly constructed block_tensor is NOT zeroed out for you. If you start having NaN issues
     * in your code try calling Tensor.zero() or zero(Tensor) to see if that resolves it.
     *
     * @tparam Dims Variadic template arguments for the dimensions. Must be castable to size_t.
     * @param name Name of the new tensor.
     * @param block_dims The size of each block.
     */
    template <typename ArrayArg>
    explicit block_tensor(std::string name, const ArrayArg &block_dims)
        : tensor_base::block_tensor<T, Rank, tensor<T, Rank>>(name, block_dims) {}

    /**
     * @brief Construct a new block_tensor object using the dimensions given by Dim object.
     *
     * @param block_dims The dimensions of the new tensor in Dim form.
     */
    template <size_t Dims>
    explicit block_tensor(dim<Dims> block_dims) : tensor_base::block_tensor<T, Rank, tensor<T, Rank>>(block_dims) {}

    // size_t dim(int d) const override { return detail::block_tensor<T, Rank, Tensor>::dim(d); }
};

#if defined(EINSUMS_COMPUTE_CODE)
/**
 * @struct BlockDeviceTensor
 *
 * Represents a block-diagonal tensor stored on the device.
 *
 * @tparam T The type of data stored. Automatic conversion of complex types.
 * @tparam Rank The rank of the tensor.
 */
template <typename T, size_t Rank>
struct BlockDeviceTensor : public virtual tensor_base::block_tensor<T, Rank, DeviceTensor<T, Rank>>,
                           virtual tensor_base::DeviceTensorBase,
                           virtual tensor_base::DevTypedTensorBase<T> {
  public:
    using host_datatype = typename tensor_base::DevTypedTensorBase<T>::host_datatype;
    using dev_datatype  = typename tensor_base::DevTypedTensorBase<T>::dev_datatype;

    /**
     * @brief Construct a new BlockDeviceTensor object. Default constructor.
     */
    BlockDeviceTensor() = default;

    /**
     * @brief Construct a new BlockDeviceTensor object. Default copy constructor
     */
    BlockDeviceTensor(const BlockDeviceTensor &) = default;

    /**
     * @brief Destroy the BlockDeviceTensor object.
     */
    ~BlockDeviceTensor() = default;

    /**
     * @brief Construct a new BlockDeviceTensor object with the given name and dimensions.
     *
     * Constructs a new BlockDeviceTensor object using the information provided in \p name and \p block_dims .
     *
     * @code
     * // Constructs a rank 4 tensor with two blocks, the first is 2x2x2x2, the second is 3x3x3x3.
     * auto A = block_tensor<double, 4>("A", 2, 3);
     * @endcode
     *
     * The newly constructed Tensor is NOT zeroed out for you. If you start having NaN issues
     * in your code try calling Tensor.zero() or zero(Tensor) to see if that resolves it.
     *
     * @tparam Dims Variadic template arguments for the dimensions. Must be castable to size_t.
     * @param name Name of the new tensor.
     * @param mode The storage mode.
     * @param block_dims The size of each block.
     */
    template <typename... Dims>
    explicit BlockDeviceTensor(std::string name, detail::HostToDeviceMode mode, Dims... block_dims)
        : tensor_base::block_tensor<T, Rank, DeviceTensor<T, Rank>>(name) {

        auto dims = std::array<size_t, sizeof...(Dims)>{static_cast<size_t>(block_dims)...};

        for (int i = 0; i < sizeof...(Dims); i++) {

            Dim<Rank> pass_dims;

            pass_dims.fill(dims[i]);

            tensor_base::block_tensor<T, Rank, DeviceTensor<T, Rank>>::push_block(DeviceTensor<T, Rank>(pass_dims, mode));
        }
    }

    /**
     * @brief Construct a new BlockDeviceTensor object with the given name and dimensions.
     *
     * Constructs a new BlockDeviceTensor object using the information provided in \p name and \p block_dims .
     *
     * @code
     * // Constructs a rank 4 tensor with two blocks, the first is 2x2x2x2, the second is 3x3x3x3.
     * auto A = block_tensor<double, 4>("A", 2, 3);
     * @endcode
     *
     * The newly constructed Tensor is NOT zeroed out for you. If you start having NaN issues
     * in your code try calling Tensor.zero() or zero(Tensor) to see if that resolves it.
     *
     * @tparam Dims Variadic template arguments for the dimensions. Must be castable to size_t.
     * @param name Name of the new tensor.
     * @param mode The storage mode.
     * @param block_dims The size of each block.
     */
    template <typename ArrayArg>
    explicit BlockDeviceTensor(std::string name, detail::HostToDeviceMode mode, const ArrayArg &block_dims)
        : tensor_base::block_tensor<T, Rank, DeviceTensor<T, Rank>>(name) {
        for (int i = 0; i < block_dims.size(); i++) {

            Dim<Rank> pass_dims;

            pass_dims.fill(block_dims[i]);

            tensor_base::block_tensor<T, Rank, DeviceTensor<T, Rank>>::push_block(DeviceTensor<T, Rank>(pass_dims, mode));
        }
    }

    /**
     * @brief Construct a new block_tensor object using the dimensions given by Dim object.
     *
     * @param mode The storage mode.
     * @param block_dims The dimensions of the new tensor in Dim form.
     */
    template <size_t Dims>
    explicit BlockDeviceTensor(detail::HostToDeviceMode mode, Dim<Dims> block_dims)
        : tensor_base::block_tensor<T, Rank, DeviceTensor<T, Rank>>() {
        for (int i = 0; i < block_dims.size(); i++) {

            Dim<Rank> pass_dims;

            pass_dims.fill(block_dims[i]);

            push_block(DeviceTensor<T, Rank>(pass_dims, mode));
        }
    }

    /**
     * @brief Construct a new BlockDeviceTensor object with the given name and dimensions.
     *
     * Constructs a new BlockDeviceTensor object using the information provided in \p name and \p block_dims .
     *
     * @code
     * // Constructs a rank 4 tensor with two blocks, the first is 2x2x2x2, the second is 3x3x3x3.
     * auto A = block_tensor<double, 4>("A", 2, 3);
     * @endcode
     *
     * The newly constructed Tensor is NOT zeroed out for you. If you start having NaN issues
     * in your code try calling Tensor.zero() or zero(Tensor) to see if that resolves it.
     *
     * @tparam Dims Variadic template arguments for the dimensions. Must be castable to size_t.
     * @param name Name of the new tensor.
     * @param mode The storage mode.
     * @param block_dims The size of each block.
     */
    template <typename... Dims>
        requires(NoneOfType<detail::HostToDeviceMode, Dims...>)
    explicit BlockDeviceTensor(std::string name, Dims... block_dims) : tensor_base::block_tensor<T, Rank, DeviceTensor<T, Rank>>(name) {

        auto dims = std::array<size_t, sizeof...(Dims)>{static_cast<size_t>(block_dims)...};

        for (int i = 0; i < sizeof...(Dims); i++) {

            Dim<Rank> pass_dims;

            pass_dims.fill(dims[i]);

            tensor_base::block_tensor<T, Rank, DeviceTensor<T, Rank>>::push_block(DeviceTensor<T, Rank>(pass_dims, detail::DEV_ONLY));
        }
    }

    /**
     * @brief Construct a new BlockDeviceTensor object with the given name and dimensions.
     *
     * Constructs a new BlockDeviceTensor object using the information provided in \p name and \p block_dims .
     *
     * @code
     * // Constructs a rank 4 tensor with two blocks, the first is 2x2x2x2, the second is 3x3x3x3.
     * auto A = block_tensor<double, 4>("A", 2, 3);
     * @endcode
     *
     * The newly constructed Tensor is NOT zeroed out for you. If you start having NaN issues
     * in your code try calling Tensor.zero() or zero(Tensor) to see if that resolves it.
     *
     * @tparam Dims Variadic template arguments for the dimensions. Must be castable to size_t.
     * @param name Name of the new tensor.
     * @param mode The storage mode.
     * @param block_dims The size of each block.
     */
    template <typename ArrayArg>
    explicit BlockDeviceTensor(std::string name, const ArrayArg &block_dims)
        : tensor_base::block_tensor<T, Rank, DeviceTensor<T, Rank>>(name) {
        for (int i = 0; i < block_dims.size(); i++) {

            Dim<Rank> pass_dims;

            pass_dims.fill(block_dims[i]);

            tensor_base::block_tensor<T, Rank, DeviceTensor<T, Rank>>::push_block(DeviceTensor<T, Rank>(pass_dims, detail::DEV_ONLY));
        }
    }

    /**
     * @brief Construct a new block_tensor object using the dimensions given by Dim object.
     *
     * @param block_dims The dimensions of the new tensor in Dim form.
     */
    template <size_t Dims>
    explicit BlockDeviceTensor(Dim<Dims> block_dims) : tensor_base::block_tensor<T, Rank, DeviceTensor<T, Rank>>() {
        for (int i = 0; i < block_dims.size(); i++) {

            Dim<Rank> pass_dims;

            pass_dims.fill(block_dims[i]);

            push_block(DeviceTensor<T, Rank>(pass_dims, detail::DEV_ONLY));
        }
    }

    /**
     * Returns a pointer into the tensor at the given location.
     *
     *
     * @tparam MultiIndex The datatypes of the passed parameters. Must be castable to
     * @param index The explicit desired index into the tensor. Must be castable to std::int64_t.
     * @return A pointer into the tensor at the requested location.
     */
    template <typename... MultiIndex>
        requires requires {
            requires NoneOfType<AllT, MultiIndex...>;
            requires NoneOfType<Range, MultiIndex...>;
        }
    auto gpu_data(MultiIndex... index) -> T * {
#    if !defined(DOXYGEN_SHOULD_SKIP_THIS)
        assert(sizeof...(MultiIndex) <= Rank);

        auto index_list = std::array{static_cast<std::int64_t>(index)...};
        int  block      = -1;

        for (auto [i, _index] : enumerate(index_list)) {
            if (_index < 0) {
                index_list[i] = _dim + _index;
            }

            // Find the block.
            if (block == -1) {
                for (int j = 0; j < _ranges.size(); j++) {
                    if (_ranges[j][0] <= _index && _index < _ranges[j][1]) {
                        block = j;
                        break;
                    }
                }
            }

            if (_ranges[block][0] <= _index && _index < _ranges[block][1]) {
                // Remap the index to be in the block.
                index_list[i] -= _ranges[block][0];
            } else {
                return nullptr; // The indices point outside of all the blocks.
            }
        }

        size_t ordinal = std::inner_product(index_list.begin(), index_list.end(), _blocks[block].strides().begin(), size_t{0});
        return &(_blocks[block].gpu_data()[ordinal]);
#    endif
    }

    /**
     * @brief Subscripts into the tensor.
     *
     * This is different from the normal subscript since there needs to be a way to
     * access data on the device. C++ references do not handle data synchronization
     * in this way, so it needs to be done differently.
     */
    template <typename... MultiIndex>
        requires requires {
            requires NoneOfType<AllT, MultiIndex...>;
            requires NoneOfType<Range, MultiIndex...>;
        }
    auto operator()(MultiIndex... index) -> HostDevReference<T> {

        static_assert(sizeof...(MultiIndex) == Rank);

        auto index_list = std::array{static_cast<std::int64_t>(index)...};

        int block = -1;
        for (auto [i, _index] : enumerate(index_list)) {
            if (_index < 0) {
                index_list[i] = _dim + _index;
            }

            if (block == -1) {
                for (int j = 0; j < _ranges.size(); j++) {
                    if (_ranges[j][0] <= _index && _index < _ranges[j][1]) {
                        block = j;
                        break;
                    }
                }
            }

            if (_ranges.at(block)[0] <= _index && _index < _ranges.at(block)[1]) {
                // Remap the index to be in the block.
                index_list[i] -= _ranges.at(block)[0];
            } else {
                return HostDevReference<T>();
            }
        }

        return std::apply(_blocks.at(block), index_list);
    }

    size_t dim(int d) const override { return tensor_base::block_tensor<T, Rank, DeviceTensor<T, Rank>>::dim(d); }
};
#endif

template <typename Type = double, typename... Args>
auto create_block_tensor(const std::string name, Args... args) {
    return block_tensor<Type, sizeof...(Args)>{name, args...};
}

template <typename Type = double, std::integral... Args>
auto create_block_tensor(Args... args) {
    return block_tensor<Type, sizeof...(Args)>{"Temporary", args...};
}

template <FileOrOstream Output, BlockTensorConcept AType, typename... Args>
void fprintln(Output fp, const AType &A, tensor_print_options options = {}) {
    fprintln(fp, "Name: {}", A.name());
    {
        print::Indent const indent{};
        fprintln(fp, "Block Tensor");
        fprintln(fp, "Data Type: {}", util::type_name<typename AType::data_type>());

        for (int i = 0; i < A.num_blocks(); i++) {
            fprintln(fp, A[i], options);
        }
    }
}

template <BlockTensorConcept AType>
void println(const AType &A, tensor_print_options options = {}) {
    fprintf(stdout, A, options);
}

} // namespace einsums
