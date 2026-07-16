//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/*
  Copyright 2018 Paul Springer

  Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are
  met:

  1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this
  software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
  ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
  USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#pragma once

#include <Einsums/HPTT/HPTTTypes.hpp>
#include <Einsums/HPTT/Transpose.hpp>

// Each SIMD-dispatch rung compiles this implementation into its own
// namespace (see Einsums_AddSIMDDispatch.cmake). When the file is compiled
// outside the dispatch machinery, default to the single native namespace.
#if !defined(EINSUMS_SIMD_ARCH_NS)
#    define EINSUMS_SIMD_ARCH_NS arch_native
#endif
#include <Einsums/Logging.hpp>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <algorithm>
#include <list>
#include <memory>
#include <vector>
#ifdef _OPENMP
#    include <omp.h>
#endif

namespace hptt {

class Plan;

namespace EINSUMS_SIMD_ARCH_NS {

template <typename floatType>
class TransposeImpl final : public hptt::Transpose<floatType> {

  public:
    /***************************************************
     * Cons, Decons, Copy, ...
     ***************************************************/
    /**
     * \param[in] perm dim-dimensional array representing the permutation of the indices.
     *                 * For instance, perm[] = {1,0,2} denotes the following transposition: \f$B_{i1,i0,i2} \gets A_{i0,i1,i2}\f$.
     * \param[in] dim Dimensionality of the tensors
     * \param[in] alpha scaling factor for A
     * \param[in] A Pointer to the raw-data of the input tensor A
     * \param[in] sizeA dim-dimensional array that stores the sizes of each dimension of A
     * \param[in] outerSizeA dim-dimensional array that stores the outer-sizes of each dimension of A.
     *                       * This parameter may be NULL, indicating that the outer-size is equal to sizeA.
     *                       * If outerSizeA is not NULL, outerSizeA[i] >= sizeA[i] for all 0 <= i < dim must hold.
     *                       * This option enables HPTT to operate on sub-tensors.
     * \param[in] offsetA dim-dimensional array that stores the offsets in each dimension of A
     *                       * This parameter may be NULL, indicating that the offset is zero.
     *                       * If offsetA is not NULL, outerSizeA[i] >= offsetA[i] + sizeA[i] >= 0 for all 0 <= i < dim must hold.
     *                       * This option enables HPTT to operate on intermediate sub-tensors.
     * \param[in] innerStrideA integer storing a non-unitary stride for the innermost dimension of A.
     * \param[in] beta scaling factor for B
     * \param[inout] B Pointer to the raw-data of the output tensor B
     * \param[in] outerSizeB dim-dimensional array that stores the outer-sizes of each dimension of B.
     *                       * This parameter may be NULL, indicating that the outer-size is equal to the perm(sizeA).
     *                       * If outerSizeA is not NULL, outerSizeB[i] >= perm(sizeA)[i] for all 0 <= i < dim must hold.
     * \param[in] offsetB dim-dimensional array that stores the offsets in each dimension of B
     *                       * This parameter may be NULL, indicating that the offset is zero.
     *                       * If offsetB is not NULL, outerSizeB[i] >= offsetB[i] + sizeB[i] >= 0 for all 0 <= i < dim must hold.
     *                       * This option enables HPTT to operate on intermediate sub-tensors.
     *                       * This option enables HPTT to operate on sub-tensors.
     * \param[in] innerStrideB integer storing a non-unitary stride for the innermost dimension of B.
     * \param[in] selectionMethod Determines if auto-tuning should be used. See hptt::SelectionMethod for details.
     *                            ATTENTION: If you enable auto-tuning (e.g., hptt::MEASURE)
     *                            then the output data will be used during the
     *                            auto-tuning process. The original data (i.e., A and B), however, is preserved
     *                            after this function call completes -- unless your input
     *                            data (i.e. A) has invalid data (e.g., NaN, inf).
     * \param[in] numThreads number of threads that participate in this tensor transposition.
     * \param[in] threadIds Array of OpenMP threadIds that participate in this
     *            tensor transposition. This parameter is only important if you want to call
     *            HPTT from within a parallel region (i.e., via execute_expert()).
     * \param[in] useRowMajor This flag indicates whether a row-major memory layout should be used (default: off = column-major).
     *            Column-Major: indices are stored from left to right (leftmost = stride-1 index)
     *            Row-Major: indices are stored from right to left (right = stride-1 index)
     */
    TransposeImpl(size_t const *sizeA, int const *perm, size_t const *outerSizeA, size_t const *outerSizeB, size_t const *offsetA,
                  size_t const *offsetB, size_t const innerStrideA, size_t const innerStrideB, int const dim, floatType const *A,
                  floatType const alpha, floatType *B, floatType const beta, SelectionMethod const selectionMethod, int const numThreads,
                  int const *threadIds = nullptr, bool const useRowMajor = false);

    /**
     * Copy construct a Transpose object.
     */
    TransposeImpl(TransposeImpl const &other);

    ~TransposeImpl() override;

    /***************************************************
     * Getter & Setter
     ***************************************************/

    /**
     * Indicates whether to conjugate the input tensor.
     */
    bool get_conj_a() noexcept { return _conjA; }

    /**
     * Change whether to conjugate the input tensor.
     */
    void set_conj_a(bool conjA) noexcept override { _conjA = conjA; }

    /**
     * Get the number of threads to use in this operation.
     */
    [[nodiscard]] int get_num_threads() const noexcept { return _numThreads; }

    /**
     * Set the number of threads to use in this operation.
     */
    void set_num_threads(int numThreads) noexcept { _numThreads = numThreads; }

    /**
     * Get the scaling factor for A.
     */
    [[nodiscard]] floatType get_alpha() const noexcept { return _alpha; }

    /**
     * Get the scaling factor for B.
     */
    [[nodiscard]] floatType get_beta() const noexcept { return _beta; }
    /**
     * \brief set the scaling factor for A.
     */
    void set_alpha(floatType alpha) noexcept override { _alpha = alpha; }
    /**
     * \brief set the scaling factor for B.
     */
    void set_beta(floatType beta) noexcept override { _beta = beta; }
    /**
     * \brief Set the pointer for A
     *
     * This features is especially useful if one wants to reuse the
     * transposition over multiple invocations.
     */
    void set_input_ptr(floatType const *A) noexcept override { _A = A; }
    /**
     * \brief Set the pointer for B
     *
     * This features is especially useful if one wants to reuse the
     * transposition over multiple invocations.
     */
    void set_output_ptr(floatType *B) noexcept override { _B = B; }
    /**
     * \brief Get raw-data pointer to A
     */
    [[nodiscard]] floatType const *get_input_ptr() const noexcept { return _A; }
    /**
     * \brief Get raw-data pointer to B
     */
    [[nodiscard]] floatType *get_output_ptr() const noexcept { return _B; }

    /**
     * \brief Clears the array that stores the OpenMP threadIds. This function
     *        should only be used in conjuction with addThreadId().
     */
    void reset_thread_ids() noexcept { _threadIds.clear(); }

    /**
     * setMaxAutotuningCandidates() enables users to specify the number of
     * candidates that should be tested during the autotuning phase
     */
    void set_max_autotuning_candidates(int num) { _maxAutotuningCandidates = num; }

    /**
     * This thread-safe function adds an OpenMP threadId to the set of threads
     * that will participate in this tensor transposition. This function is
     * only required in conjunction with the execute_expert() interface where
     * the transposition is executed from within a parallel region (i.e.,~HPTT
     * does not spawn the threads). It is the programmers responsibility to
     * specify the correct thread IDs that participate in this call.
     *
     * \param[in] threadId An OpenMP threadId
     */
    void add_thread_id(int threadId) noexcept {
#ifdef _OPENMP
        omp_set_lock(&_writelock);
        _threadIds.push_back(threadId);
        std::ranges::sort(_threadIds);
        omp_unset_lock(&_writelock);
#endif
    }

    /**
     * Prints the thread ID's used in this operation.
     */
    void print_thread_ids() const noexcept { EINSUMS_LOG_DEBUG("HPTT: thread IDs: [{}]", fmt::join(_threadIds, ", ")); }

    /**
     * Gets the ID of the master thread for the operation.
     */
    [[nodiscard]] int get_master_thread_id() const noexcept { return _threadIds[0]; }

    /***************************************************
     * Public Methods
     ***************************************************/
    /**
     * \brief Creates the plan that encodes the execution of the tensor transposition.
     */
    void create_plan();

    /**
     * Executes the transposition. This functions requires that the plan has
     * already been created via the createPlan() function.
     * This function behaves similarly to the execute() function but it
     * offers additional template parameters to improve performance for very
     * small tensor transpositions. Moreover it adds more flexibility.
     *
     * \tparam useStreamingStores Iff this variable is set, HPTT will use
     *                         streaming stores which improves performance because they avoid the
     *                         write-allocate traffic incurred by the write to B. However, sometimes
     *                         the user might want to avoid streaming stores
     *                         because the packed data fits int cache and is
     *                         reused shortly (e.g., within BLAS packing
     *                         routines).
     * \tparam spawnThreads If the variable is set, the threads will be
     *                         spawned from within this call, otherwise it is
     *                         expected that this function call executes from
     *                         within a parallel region.
     * \tparam betaIsZero   Only set this variable if beta is zero.
     */
    template <bool useStreamingStores = true, bool spawnThreads = true, bool betaIsZero>
    void execute_expert() noexcept;

    /**
     * Executes the transposition. This functions requires that the plan has
     * already been created via the createPlan() function.
     */
    void execute() noexcept override;

    /**
     * Print information about this transpose operation.
     */
    void print() noexcept;

    void write_to_file(std::FILE *fp) const override;

    /**
     * Deep-copy this plan via the copy constructor.
     */
    [[nodiscard]] std::shared_ptr<Transpose<floatType>> clone() const override { return std::make_shared<TransposeImpl>(*this); }

    TransposeImpl(std::FILE *fp, floatType alpha, floatType const *A, floatType beta, floatType *B);

  private:
    /***************************************************
     * Private Methods
     ***************************************************/
    void                  create_plans(std::vector<std::shared_ptr<Plan>> &plans) const;
    std::shared_ptr<Plan> select_plan(std::vector<std::shared_ptr<Plan>> const &plans);
    void                  fuse_indices();
    void skip_indices(size_t const *_sizeA, int const *_perm, size_t const *_outerSizeA, size_t const *_outerSizeB, size_t const *_offsetA,
                      size_t const *_offsetB, int const dim);
    void compute_leading_dimensions();
    [[nodiscard]] double loop_cost_heuristic(std::vector<int> const &loopOrder) const;
    [[nodiscard]] double parallelism_cost_heuristic(std::vector<int> const &loopOrder) const;
    [[nodiscard]] int    get_local_thread_id(int myThreadId) const;
    template <bool spawnThreads>
    void get_start_end(size_t n, size_t &myStart, size_t &myEnd) const;
    void set_parallel_strategy(int id) noexcept { _selectedParallelStrategyId = id; }
    void set_loop_order(int id) noexcept { _selectedLoopOrderId = id; }

    /***************************************************
     * Helper Methods
     ***************************************************/
    // parallelizes the loops by changing the value of parallelismStrategy
    void                parallelize(std::vector<int> &parallelismStrategy, std::vector<int> &availableParallelismAtLoop, int &totalTasks,
                                    std::list<int> &primeFactors, float const minBalancing, std::vector<int> const &loopsAllowed) const;
    [[nodiscard]] float get_load_balance(std::vector<int> const &parallelismStrategy) const;
    float estimate_execution_time(std::shared_ptr<Plan> const plan); // execute just a few iterations and extrapolate the result
    void  verify_parameter(size_t const *size, int const *perm, size_t const *outerSizeA, size_t const *outerSizeB, size_t const *offsetA,
                           size_t const *offsetB, size_t const innerStrideA, size_t const innerStrideB, int const dim) const;
    void  get_best_parallelism_strategy(std::vector<int> &bestParallelismStrategy) const;
    void  get_best_loop_order(std::vector<int> &loopOrder) const; // innermost loop idx is stored at dim_-1
    void  get_loop_orders(std::vector<std::vector<int>> &loopOrders) const;
    void  get_parallelism_strategies(std::vector<std::vector<int>> &parallelismStrategies) const;
    void  get_all_parallelism_strategies(std::list<int> &primeFactorsToMatch, std::vector<int> &availableParallelismAtLoop,
                                         std::vector<int>              &achievedParallelismAtLoop,
                                         std::vector<std::vector<int>> &parallelismStrategies) const;
    void  get_available_parallelism(std::vector<int> &numTasksPerLoop) const;
    [[nodiscard]] size_t get_increment(int loopIdx) const;
    void
    execute_estimate(Plan const *plan) noexcept; // almost identical to execute, but it just executes few iterations and then extrapolates
    [[nodiscard]] double get_time_limit() const;

    floatType const    *_A;            //!< rawdata pointer for A
    floatType          *_B;            //!< rawdata pointer for B
    floatType           _alpha;        //!< scaling factor for A
    floatType           _beta;         //!< scaling factor for B
    int                 _dim;          //!< dimension of the tensor
    std::vector<size_t> _sizeA;        //!< size of A
    std::vector<int>    _perm;         //!< permutation
    std::vector<size_t> _outerSizeA;   //!< outer sizes of A
    std::vector<size_t> _outerSizeB;   //!< outer sizes of B
    std::vector<size_t> _offsetA;      //!< offsets of A
    std::vector<size_t> _offsetB;      //!< offsets of B
    size_t              _innerStrideA; //!< innerStride of A
    size_t              _innerStrideB; //!< innerStride of B
    std::vector<size_t> _lda;          //!< strides for all dimensions of A (first dimension has a stride of 1)
    std::vector<size_t> _ldb;          //!< strides for all dimensions of B (first dimension has a stride of 1)
    std::vector<int>    _threadIds;    //!< OpenMP threadIds of the threads involed in the transposition
    int                 _numThreads;
    int                 _selectedParallelStrategyId;
    int                 _selectedLoopOrderId;
    bool                _conjA;
#ifdef _OPENMP
    omp_lock_t _writelock;
#endif

    std::shared_ptr<Plan> _masterPlan;
    SelectionMethod       _selectionMethod;
    int                   _maxAutotuningCandidates;
    static constexpr int  blocking_micro_ = einsums::simd::native_bits / 8 / sizeof(floatType);
    static constexpr int  blocking_       = blocking_micro_ * 4;

    static constexpr int infoLevel_ = 0; // determines which auxiliary messages should be printed
};

extern template class TransposeImpl<float>;
extern template class TransposeImpl<double>;
extern template class TransposeImpl<FloatComplex>;
extern template class TransposeImpl<DoubleComplex>;

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) || defined(__AVX512FP16__)
extern template class TransposeImpl<einsums::simd::half_t>;
#endif

#if defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC) || defined(__AVX512BF16__)
extern template class TransposeImpl<einsums::simd::bfloat16_t>;
#endif

} // namespace EINSUMS_SIMD_ARCH_NS
} // namespace hptt
