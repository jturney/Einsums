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

/**
 * \file
 * Plan a tensor transposition: the planner, compiled once. The kernels it runs are in
 * TransposeKernels.cpp, compiled once per SIMD dispatch rung.
 */

#include <Einsums/Config/Namespace.hpp>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <list>
#include <memory>
#include <numeric>
#include <tuple>
#include <vector>

#ifdef _OPENMP
#    include <omp.h>
#endif

#include <Einsums/Assert.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/HPTT/ComputeNode.hpp>
#include <Einsums/HPTT/Files.hpp>
#include <Einsums/HPTT/HPTTTypes.hpp>
#include <Einsums/HPTT/Plan.hpp>
#include <Einsums/HPTT/Utils.hpp>
#include <Einsums/Logging.hpp>

#include <Stripes/RuntimeFeatures.hpp>

#include "TransposeImpl.hpp"
#include "TransposeKernels.hpp"

EINSUMS_NAMESPACE_BEGIN(hptt)

template <typename floatType>
TransposeImpl<floatType>::TransposeImpl(TransposeKernels<floatType> const &kernels, size_t const *sizeA, int const *perm,
                                        size_t const *outerSizeA, size_t const *outerSizeB, size_t const *offsetA, size_t const *offsetB,
                                        size_t const innerStrideA, size_t const innerStrideB, int const dim, floatType const *A,
                                        floatType const alpha, floatType *B, floatType const beta, SelectionMethod const selectionMethod,
                                        int const numThreads, int const *threadIds, bool const useRowMajor)
    : _kernels(&kernels), _A(A), _B(B), _alpha(alpha), _beta(beta), _dim(-1), _innerStrideA(0), _innerStrideB(0), _numThreads(numThreads),
      _masterPlan(nullptr), _selectionMethod(selectionMethod), _maxAutotuningCandidates(-1), _selectedParallelStrategyId(-1),
      _selectedLoopOrderId(-1), _conjA(false) {
    // The caller's permutation indexes the size and stride arrays below (account_for_row_major reads
    // sizeA[perm[i]]), so it is checked before anything uses it. A bad argument throws: this is a
    // library, and ending the caller's process, a Python interpreter included, is not its call.
    if (dim < 1) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "HPTT: dimensionality {} is too low", dim);
    }
    std::vector<int> seen(dim, 0);
    for (int i = 0; i < dim; ++i) {
        if (perm[i] < 0 || perm[i] >= dim || seen[perm[i]]++ != 0) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument, "HPTT: permutation invalid at position {} (value {})", i, perm[i]);
        }
        if (sizeA[i] == 0) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument, "HPTT: size at position {} is zero", i);
        }
    }

    std::vector<int>    tmpPerm(dim);
    std::vector<size_t> tmpSizeA(dim), tmpOuterSizeA(dim), tmpOuterSizeB(dim), tmpOffsetA(dim), tmpOffsetB(dim);

    account_for_row_major(sizeA, outerSizeA, outerSizeB, offsetA, offsetB, perm, tmpSizeA.data(), tmpOuterSizeA.data(),
                          tmpOuterSizeB.data(), tmpOffsetA.data(), tmpOffsetB.data(), tmpPerm.data(), dim, useRowMajor);

    _sizeA.resize(dim);
    _perm.resize(dim);
    _outerSizeA.resize(dim);
    _outerSizeB.resize(dim);
    _offsetA.resize(dim);
    _offsetB.resize(dim);
    _lda.resize(dim);
    _ldb.resize(dim);
    _threadIds.reserve(dim);
    _callerManagedThreads = (threadIds != nullptr);
    if (threadIds) {
        // compact threadIds. E.g., 1, 7, 5 -> local_id(1) = 0, local_id(7) = 2,
        // local_id(5) = 1
        for (int i = 0; i < numThreads; ++i)
            _threadIds.push_back(threadIds[i]);
        std::sort(_threadIds.begin(), _threadIds.end());
    } else {
        for (int i = 0; i < numThreads; ++i)
            _threadIds.push_back(i);
    }

    verify_parameter(tmpSizeA.data(), tmpPerm.data(), tmpOuterSizeA.data(), tmpOuterSizeB.data(), tmpOffsetA.data(), tmpOffsetB.data(),
                     innerStrideA, innerStrideB, dim);

    // Only once the arguments are known good: a constructor that throws never runs the destructor
    // that would release the lock.
#ifdef _OPENMP
    omp_init_lock(&_writelock);
#endif

    _innerStrideA = innerStrideA;
    _innerStrideB = innerStrideB;

    // initializes dim_, outerSizeA, outerSizeB, sizeA and perm
    skip_indices(tmpSizeA.data(), tmpPerm.data(), tmpOuterSizeA.data(), tmpOuterSizeB.data(), tmpOffsetA.data(), tmpOffsetB.data(), dim);
    fuse_indices();

    // initializes lda_ and ldb_
    compute_leading_dimensions();

    // create plan
    this->create_plan();
}

template <typename floatType>
TransposeImpl<floatType>::TransposeImpl(TransposeImpl<floatType> const &other)
    : _kernels(other._kernels), _A(other._A), _B(other._B), _alpha(other._alpha), _beta(other._beta), _dim(other._dim),
      _numThreads(other._numThreads), _callerManagedThreads(other._callerManagedThreads), _masterPlan(other._masterPlan),
      _selectionMethod(other._selectionMethod), _selectedParallelStrategyId(other._selectedParallelStrategyId),
      _selectedLoopOrderId(other._selectedLoopOrderId), _maxAutotuningCandidates(other._maxAutotuningCandidates), _sizeA(other._sizeA),
      _perm(other._perm), _outerSizeA(other._outerSizeA), _outerSizeB(other._outerSizeB), _offsetA(other._offsetA),
      _offsetB(other._offsetB), _innerStrideA(other._innerStrideA), _innerStrideB(other._innerStrideB), _lda(other._lda), _ldb(other._ldb),
      _threadIds(other._threadIds), _conjA(other._conjA) {
#ifdef _OPENMP
    omp_init_lock(&_writelock);
#endif
}

template <typename floatType>
TransposeImpl<floatType>::~TransposeImpl() {
#ifdef _OPENMP
    omp_destroy_lock(&_writelock);
#endif
}

template <typename floatType>
KernelArgs<floatType> TransposeImpl<floatType>::kernel_args(Plan const *plan) const noexcept {
    return KernelArgs<floatType>{.A                    = _A,
                                 .B                    = _B,
                                 .alpha                = _alpha,
                                 .beta                 = _beta,
                                 .dim                  = _dim,
                                 .perm0                = _perm[0],
                                 .conjA                = _conjA,
                                 .innerStrideA         = _innerStrideA,
                                 .innerStrideB         = _innerStrideB,
                                 .sizeA                = _sizeA.data(),
                                 .offsetA              = _offsetA.data(),
                                 .offsetB              = _offsetB.data(),
                                 .lda                  = _lda.data(),
                                 .ldb                  = _ldb.data(),
                                 .plan                 = plan,
                                 .numThreads           = _numThreads,
                                 .threadIds            = _threadIds.data(),
                                 .callerManagedThreads = _callerManagedThreads};
}

template <typename floatType>
void TransposeImpl<floatType>::execute_estimate(Plan const *plan) noexcept {
    EINSUMS_ASSERT_(plan != nullptr, "HPTT: plan has not yet been created.");
    _kernels->execute_estimate(kernel_args(plan));
}

template <typename floatType>
void TransposeImpl<floatType>::execute() noexcept {
    EINSUMS_ASSERT_(_masterPlan != nullptr, "HPTT: master plan has not yet been created.");
    _kernels->execute(kernel_args(_masterPlan.get()));
}

template <typename floatType>
void TransposeImpl<floatType>::print() noexcept {
    _masterPlan->print();
}

template <typename floatType>
size_t TransposeImpl<floatType>::get_increment(int loopIdx) const {
    size_t inc = 1;
    if (_perm[0] != 0) {
        if (loopIdx == 0 || loopIdx == _perm[0])
            inc = _kernels->blocking;
    }
    return inc;
}

template <typename floatType>
void TransposeImpl<floatType>::get_available_parallelism(std::vector<int> &numTasksPerLoop) const {
    numTasksPerLoop.resize(_dim);
    for (int loopIdx = 0; loopIdx < _dim; ++loopIdx) {
        size_t inc               = this->get_increment(loopIdx);
        numTasksPerLoop[loopIdx] = (_sizeA[loopIdx] + inc - 1) / inc;
    }
}

template <typename floatType>
void TransposeImpl<floatType>::get_all_parallelism_strategies(std::list<int>                &primeFactorsToMatch,
                                                              std::vector<int>              &availableParallelismAtLoop, // NOLINT
                                                              std::vector<int>              &achievedParallelismAtLoop,
                                                              std::vector<std::vector<int>> &parallelismStrategies) const {
    if (primeFactorsToMatch.size() > 0) {
        // match every primefactor ...
        for (auto p : primeFactorsToMatch) {
            // ... with every loop
            for (int i = 0; i < _dim; i++) {
                std::list<int>   primeFactorsToMatch_(primeFactorsToMatch);
                std::vector<int> availableParallelismAtLoop_(availableParallelismAtLoop);
                std::vector<int> achievedParallelismAtLoop_(achievedParallelismAtLoop);

                primeFactorsToMatch_.erase(std::find(primeFactorsToMatch_.begin(), primeFactorsToMatch_.end(), p));
                availableParallelismAtLoop_[i] = (availableParallelismAtLoop_[i] + p - 1) / p;
                achievedParallelismAtLoop_[i] *= p;

                this->get_all_parallelism_strategies(primeFactorsToMatch_, availableParallelismAtLoop_, achievedParallelismAtLoop_,
                                                     parallelismStrategies);
            }
        }
    } else {
        // avoid duplicates
        if (parallelismStrategies.end() == std::find(parallelismStrategies.begin(), parallelismStrategies.end(), achievedParallelismAtLoop))
            parallelismStrategies.push_back(achievedParallelismAtLoop);
    }
}

// balancing if one tries to parallelize avail many tasks with req many threads
// e.g., balancing(3,4) = 0.75
static float getBalancing(int avail, int req) {
    return ((float)(avail)) / (float)((int)((avail + req - 1) / req) * req);
}

template <typename floatType>
float TransposeImpl<floatType>::get_load_balance(std::vector<int> const &parallelismStrategy) const {
    float load_balance = 1.0;
    int   totalTasks   = 1;
    for (int i = 0; i < _dim; ++i) {

        size_t inc = this->get_increment(i);
        while (_sizeA[i] < inc)
            inc /= 2;
        size_t availableParallelism = (_sizeA[i] + inc - 1) / inc;

        if (i == 0 || _perm[i] == 0)
            // account for the load-imbalancing due to blocking
            load_balance *= getBalancing(_sizeA[i], inc);
        load_balance *= getBalancing(availableParallelism, parallelismStrategy[i]);
        totalTasks *= parallelismStrategy[i];
    }

    // how well can these tasks be distributed among _numThreads?
    //  e.g., totalTasks = 3, numThreads = 8 => 3./8
    //  e.g., totalTasks = 5, numThreads = 8 => 5./8
    //  e.g., totalTasks = 15, numThreads = 8 => 15./16
    //  e.g., totalTasks = 17, numThreads = 8 => 17./24
    float workDistribution = ((float)totalTasks) / (((totalTasks + _numThreads - 1) / _numThreads) * _numThreads);

    load_balance *= workDistribution;
    return load_balance;
}

template <typename floatType>
void TransposeImpl<floatType>::get_best_parallelism_strategy(std::vector<int> &bestParallelismStrategy) const {
    std::vector<int> availableParallelismAtLoop;
    this->get_available_parallelism(availableParallelismAtLoop);
    int totalAvailableParallelism =
        std::accumulate(availableParallelismAtLoop.begin(), availableParallelismAtLoop.end(), 1, std::multiplies<int>());

    // reduce the probability of parallelizing the stride-1 index
    // if this loops would be parallelized, these two statements ensure that each
    // thread would have at least two macro-kernels of work at this loop-level
    //
    // However, if the total available parallelism is too small, then we do not
    // artificially limit the available parallelism further
    int reduceParallelismB = 4; // avoid parallelization in stride-1 B more strongly
    int reduceParallelismA = 2;
    if (totalAvailableParallelism < 2 * _numThreads)
        reduceParallelismB = 1;
    else if (totalAvailableParallelism < 4 * _numThreads)
        reduceParallelismB = 2;
    totalAvailableParallelism =
        (totalAvailableParallelism / availableParallelismAtLoop[_perm[0]]) * (availableParallelismAtLoop[_perm[0]] / reduceParallelismB);
    if (totalAvailableParallelism < 2 * _numThreads)
        reduceParallelismA = 1;
    availableParallelismAtLoop[_perm[0]] = std::max(1, availableParallelismAtLoop[_perm[0]] / reduceParallelismB);
    availableParallelismAtLoop[0]        = std::max(1, availableParallelismAtLoop[0] / reduceParallelismA);

    // Objectives: 1) load-balancing
    //             2) avoid parallelizing stride-1 loops (rational: less
    //             consecutive memory accesses) 3) avoid false sharing

    std::vector<int> loopsAllowed;
    for (int i = _dim - 1; i >= 1; i--)
        if (_perm[i] != 0)
            loopsAllowed.push_back(_perm[i]);
    std::vector<int> loopsAllowedStride1{0, _perm[0]};

    int            totalTasks = 1; // goal: totalTasks should be a close multiple of numTasks_
    std::list<int> primeFactors;
    get_prime_factors(_numThreads, primeFactors);

    // 1. parallelize using 100% load balancing
    parallelize(bestParallelismStrategy, availableParallelismAtLoop, totalTasks, primeFactors, 1.0, loopsAllowed);

    if (totalTasks != _numThreads) { // no perfect match has been found

        // Option 1: keep parallelizing non-stride-1 loops only, but allowing
        // load-imbalance
        std::vector<int> strat1(bestParallelismStrategy);
        std::vector<int> avail1(availableParallelismAtLoop);
        std::list<int>   primes1(primeFactors);
        int              totalTasks1 = totalTasks;
        parallelize(strat1, avail1, totalTasks1, primes1, 0.92, loopsAllowed);
        if (get_load_balance(strat1) > 0.90) {
            std::copy(strat1.begin(), strat1.end(), bestParallelismStrategy.begin());
            return;
        }

        if (_perm[0] != 0) {
            // Option 2: also parallelize stride-1 loops, enforcing perfect loop
            // balancing
            std::vector<int> strat2(bestParallelismStrategy);
            std::vector<int> avail2(availableParallelismAtLoop);
            std::list<int>   primes2(primeFactors);
            int              totalTasks2 = totalTasks;
            parallelize(strat2, avail2, totalTasks2, primes2, 1.0, loopsAllowedStride1);
            if (get_load_balance(strat2) > 0.92) {
                std::copy(strat2.begin(), strat2.end(), bestParallelismStrategy.begin());
                return;
            }

            // keep on going based on strat1
            parallelize(strat1, avail1, totalTasks1, primes1, 1.0, loopsAllowedStride1);
            if (get_load_balance(strat1) > 0.90) {
                std::copy(strat1.begin(), strat1.end(), bestParallelismStrategy.begin());
                return;
            }

            // keep on going based on strat2
            parallelize(strat2, avail2, totalTasks2, primes2, 0.92, loopsAllowed);
            if (get_load_balance(strat2) > 0.92) {
                std::copy(strat2.begin(), strat2.end(), bestParallelismStrategy.begin());
                return;
            }

            if (get_load_balance(strat1) > 0.80) // reduced threshold
            {
                std::copy(strat1.begin(), strat1.end(), bestParallelismStrategy.begin());
                return;
            }
            if (get_load_balance(strat2) > 0.82) // reduced threshold
            {
                std::copy(strat2.begin(), strat2.end(), bestParallelismStrategy.begin());
                return;
            }

            parallelize(strat1, avail1, totalTasks1, primes1, 0.9, loopsAllowedStride1);
            parallelize(strat2, avail2, totalTasks2, primes2, 0.8, loopsAllowed);
            float lb1 = get_load_balance(strat1);
            float lb2 = get_load_balance(strat2);
            //         printVector(strat2,"strat2");
            //         printf("strat2: %f\n",getLoadBalance(strat2));
            if ((lb1 > 0.8 && lb2 < 0.85) || (lb1 > lb2 && lb1 > 0.75)) {
                std::copy(strat1.begin(), strat1.end(), bestParallelismStrategy.begin());
                return;
            }
            if (lb2 >= 0.85) {
                std::copy(strat2.begin(), strat2.end(), bestParallelismStrategy.begin());
                return;
            }

            // fallback
            std::vector<int> allLoops;
            for (int i = _dim - 1; i >= 1; i--)
                allLoops.push_back(_perm[i]);
            allLoops.push_back(0);
            allLoops.push_back(_perm[0]);
            parallelize(strat1, avail1, totalTasks1, primes1, 0., allLoops);
            std::copy(strat1.begin(), strat1.end(), bestParallelismStrategy.begin());

        } else {
            parallelize(strat1, avail1, totalTasks1, primes1, 0.0, loopsAllowed);
            std::copy(strat1.begin(), strat1.end(), bestParallelismStrategy.begin());
        }
    }
}

template <typename floatType>
void TransposeImpl<floatType>::parallelize(std::vector<int> &parallelismStrategy, std::vector<int> &availableParallelismAtLoop,
                                           int &totalTasks, std::list<int> &primeFactors, float const minBalancing,
                                           std::vector<int> const &loopsAllowed) const

{
    bool suboptimalParallelizationUsed = false;
    // find loop which minimizes load imbalance for the given prime factor
    for (auto it = primeFactors.begin(); it != primeFactors.end(); it++) {
        int   suitedLoop    = -1;
        float bestBalancing = 0;

        for (auto idx : loopsAllowed) {
            float balancing = getBalancing(availableParallelismAtLoop[idx], *it);
            if (balancing > bestBalancing) {
                bestBalancing = balancing;
                suitedLoop    = idx;
            }
        }
        // allow up to one slightly less optimal splitting to prefer parallelizing
        // idx=0 over idx=perm[0]
        if (suboptimalParallelizationUsed == false && suitedLoop == _perm[0] && getBalancing(availableParallelismAtLoop[0], *it) >= 0.949) {
            suitedLoop                    = 0;
            suboptimalParallelizationUsed = true;
        }
        if (suitedLoop != -1 && bestBalancing >= minBalancing) {
            availableParallelismAtLoop[suitedLoop] /= *it;
            parallelismStrategy[suitedLoop] *= *it;
            totalTasks *= *it;
            it = primeFactors.erase(it);
            it--;
        }
    }
}

template <typename floatType>
double TransposeImpl<floatType>::parallelism_cost_heuristic(std::vector<int> const &achievedParallelismAtLoop) const {
    std::vector<int> availableParallelismAtLoop;
    this->get_available_parallelism(availableParallelismAtLoop);

    double cost = 1;
    // penalize load-imbalance
    for (int loopIdx = 0; loopIdx < _dim; ++loopIdx) {
        if (achievedParallelismAtLoop[loopIdx] <= 1)
            continue;

        int const blocksPerThread =
            (availableParallelismAtLoop[loopIdx] + achievedParallelismAtLoop[loopIdx] - 1) / achievedParallelismAtLoop[loopIdx];
        int       inc           = this->get_increment(loopIdx);
        int const effectiveSize = blocksPerThread * inc * achievedParallelismAtLoop[loopIdx];
        cost *= ((double)(effectiveSize) / _sizeA[loopIdx]);
    }

    // penalize parallelization of stride-1 loops
    if (_perm[0] == 0)
        cost *= std::pow(1.01, achievedParallelismAtLoop[0] - 1); // strongly penalize this case

    cost *= std::pow(1.00010, std::min(16, achievedParallelismAtLoop[0] - 1));        // if at all, prefer ...
    cost *= std::pow(1.00015, std::min(16, achievedParallelismAtLoop[_perm[0]] - 1)); // parallelization in stride-1 of A

    int const workPerThread =
        (availableParallelismAtLoop[_perm[0]] + achievedParallelismAtLoop[_perm[0]] - 1) / achievedParallelismAtLoop[_perm[0]];
    if (workPerThread * sizeof(floatType) % 64 != 0 && achievedParallelismAtLoop[_perm[0]] > 1) { // avoid false-sharing
        cost *= std::pow(1.00015, std::min(16, achievedParallelismAtLoop[_perm[0]] - 1));         // penalize this parallelization again
    }
    return cost;
}

template <typename floatType>
void TransposeImpl<floatType>::get_parallelism_strategies(std::vector<std::vector<int>> &parallelismStrategies) const {
    parallelismStrategies.clear();
    if (_numThreads == 1) {
        parallelismStrategies.emplace_back(std::vector<int>(_dim, 1));
        return;
    }
    std::vector<int> bestParallelismStrategy(_dim, 1);
    get_best_parallelism_strategy(bestParallelismStrategy);
    if (this->infoLevel_ > 0)
        EINSUMS_LOG_INFO("HPTT: loadbalancing: {}", get_load_balance(bestParallelismStrategy));

    if (_selectionMethod == ESTIMATE) {
        parallelismStrategies.push_back(bestParallelismStrategy);
        return;
    }

    // ATTENTION: we don't care about the case where _numThreads is a large prime
    // number... (sorry, KNC)
    //
    // we factorize numThreads into its prime factors because we have to match
    // every one to a certain loop. In principle every loop could be used to
    // match every primefactor, but some choices are preferable over others.
    // E.g., we want to achieve good load-balancing _and_ try to avoid the
    // stride-1 index of B (due to false sharing)
    std::list<int> primeFactors;
    get_prime_factors(_numThreads, primeFactors);
    if (this->infoLevel_ > 0)
        print_vector(primeFactors, "primes");

    std::vector<int> availableParallelismAtLoop;
    this->get_available_parallelism(availableParallelismAtLoop);
    if (this->infoLevel_ > 0)
        print_vector(availableParallelismAtLoop, "available Parallelism");

    std::vector<int> achievedParallelismAtLoop(_dim, 1);

    this->get_all_parallelism_strategies(primeFactors, availableParallelismAtLoop, achievedParallelismAtLoop, parallelismStrategies);

    // sort according to loop heuristic
    std::sort(parallelismStrategies.begin(), parallelismStrategies.end(),
              [this](std::vector<int> const &loopOrder1, std::vector<int> const &loopOrder2) {
                  return this->parallelism_cost_heuristic(loopOrder1) < this->parallelism_cost_heuristic(loopOrder2);
              });

    parallelismStrategies.insert(parallelismStrategies.begin(), bestParallelismStrategy);

    if (this->infoLevel_ > 1)
        for (auto const &strat : parallelismStrategies) {
            print_vector(strat, "parallelization");
            EINSUMS_LOG_INFO("HPTT: cost: {}", this->parallelism_cost_heuristic(strat));
        }
}

template <typename floatType>
void TransposeImpl<floatType>::verify_parameter(size_t const *size, int const *perm, size_t const *outerSizeA, size_t const *outerSizeB,
                                                size_t const *offsetA, size_t const *offsetB, size_t const innerStrideA,
                                                size_t const innerStrideB, int const dim) const {
    if (dim < 1) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "HPTT: dimensionality too low.");
    }

    std::vector<int> found(dim, 0);

    for (int i = 0; i < dim; ++i) {
        if (size[i] <= 0) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument, "HPTT: size at position {} is invalid", i);
        }
        if (perm[i] < 0 || perm[i] >= dim) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument, "HPTT: permutation invalid at position {} (value {})", i, perm[i]);
        }
        found[perm[i]] = 1;
    }

    for (int i = 0; i < dim; ++i)
        if (found[i] <= 0) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument, "HPTT: permutation invalid");
        }

    if (outerSizeA != nullptr)
        for (int i = 0; i < dim; ++i)
            if (outerSizeA[i] < size[i]) {
                EINSUMS_THROW_EXCEPTION(std::invalid_argument, "HPTT: outerSizeA invalid");
            }

    if (outerSizeB != nullptr)
        for (int i = 0; i < dim; ++i)
            if (outerSizeB[i] < size[perm[i]]) {
                EINSUMS_THROW_EXCEPTION(std::invalid_argument, "HPTT: outerSizeB invalid");
            }

    if (offsetA != nullptr)
        for (int i = 0; i < dim; ++i)
            if (offsetA[i] + size[i] > outerSizeA[i]) {
                EINSUMS_THROW_EXCEPTION(std::invalid_argument, "HPTT: offsetA invalid");
            }

    if (offsetB != nullptr)
        for (int i = 0; i < dim; ++i)
            if (offsetB[i] + size[perm[i]] > outerSizeB[i]) {
                EINSUMS_THROW_EXCEPTION(std::invalid_argument, "HPTT: offsetB invalid");
            }

    if (innerStrideA < 0) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "HPTT: innerStrideA invalid");
    }

    if (innerStrideB < 0) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "HPTT: innerStrideB invalid");
    }
}

template <typename floatType>
void TransposeImpl<floatType>::compute_leading_dimensions() {
    _lda[0] = _innerStrideA;
    if (_outerSizeA[0] == -1)
        for (int i = 1; i < _dim; ++i)
            _lda[i] = _lda[i - 1] * _sizeA[i - 1];
    else
        for (int i = 1; i < _dim; ++i)
            _lda[i] = _outerSizeA[i - 1] * _lda[i - 1];

    _ldb[0] = _innerStrideB;
    if (_outerSizeB[0] == -1)
        for (int i = 1; i < _dim; ++i)
            _ldb[i] = _ldb[i - 1] * _sizeA[_perm[i - 1]];
    else
        for (int i = 1; i < _dim; ++i)
            _ldb[i] = _outerSizeB[i - 1] * _ldb[i - 1];
}

template <typename floatType>
void TransposeImpl<floatType>::skip_indices(size_t const *sizeA, int const *perm, size_t const *outerSizeA, size_t const *outerSizeB,
                                            size_t const *offsetA, size_t const *offsetB, int const dim) {
    for (int i = 0; i < dim; ++i) {
        _perm[i]  = perm[i];
        _sizeA[i] = sizeA[i];
        if (outerSizeA)
            _outerSizeA[i] = outerSizeA[i];
        else
            _outerSizeA[i] = sizeA[i];
        if (outerSizeB)
            _outerSizeB[i] = outerSizeB[i];
        else
            _outerSizeB[i] = sizeA[perm[i]];
        if (offsetA)
            _offsetA[i] = offsetA[i];
        else
            _offsetA[i] = 0;
        if (offsetB)
            _offsetB[i] = offsetB[i];
        else
            _offsetB[i] = 0;
    }

    size_t skipped = 0;
    for (int i = 0; i < dim; ++i) {
        int idxB = 0;
        for (; idxB < dim; ++idxB)
            if (perm[idxB] == i)
                break;
        if (sizeA[i] == 1 && (!outerSizeA || outerSizeA[i] == 1) && (!outerSizeB || outerSizeB[idxB] == 1)) {
            _sizeA[i]         = -1;
            _outerSizeA[i]    = -1;
            _outerSizeB[idxB] = -1;
            _offsetA[i]       = -1;
            _offsetB[idxB]    = -1;
            _perm[idxB]       = -1;
            skipped++;
        }
    }
    // compact arrays (remove -1)
    for (int i = 0; i < dim; ++i)
        if (_sizeA[i] == -1) {
            int j = i + 1;
            for (; j < dim; ++j)
                if (_sizeA[j] != -1)
                    break;
            if (j < dim)
                std::swap(_sizeA[i], _sizeA[j]);
        }
    for (int i = 0; i < dim; ++i)
        if (_outerSizeA[i] == -1) {
            int j = i + 1;
            for (; j < dim; ++j)
                if (_outerSizeA[j] != -1)
                    break;
            if (j < dim) {
                std::swap(_outerSizeA[i], _outerSizeA[j]);
                std::swap(_offsetA[i], _offsetA[j]);
            }
        }
    for (int i = 0; i < dim; ++i)
        if (_outerSizeB[i] == -1) {
            int j = i + 1;
            for (; j < dim; ++j)
                if (_outerSizeB[j] != -1)
                    break;
            if (j < dim) {
                std::swap(_outerSizeB[i], _outerSizeB[j]);
                std::swap(_offsetB[i], _offsetB[j]);
            }
        }
    for (int i = 0; i < dim; ++i)
        if (_perm[i] == -1) {
            int j = i + 1;
            for (; j < dim; ++j)
                if (_perm[j] != -1)
                    break;
            if (j < dim)
                std::swap(_perm[i], _perm[j]);
        }

    _dim = dim - skipped;
    if (_dim == 0) {
        _dim = 1;
        _perm.resize(_dim);
        _sizeA.resize(_dim);
        _outerSizeA.resize(_dim);
        _outerSizeB.resize(_dim);
        _perm[0]       = 0;
        _sizeA[0]      = 1;
        _outerSizeA[0] = 1;
        _outerSizeB[0] = 1;
        _offsetA[0]    = 0;
        _offsetB[0]    = 0;
    } else {
        _perm.resize(_dim);
        _sizeA.resize(_dim);
        _outerSizeA.resize(_dim);
        _outerSizeB.resize(_dim);
        _offsetA.resize(_dim);
        _offsetB.resize(_dim);

        // remove gaps in the perm, if requried (e.g., perm=3,1,0 -> 2,1,0)
        int currentValue = 0;
        for (int i = 0; i < _dim; ++i) {
            // find smallest element in perm_ and rename it to currentValue
            int minValue = std::numeric_limits<int>::max();
            int minPos   = -1;
            for (int pos = 0; pos < _dim; ++pos) {
                if (_perm[pos] >= currentValue && _perm[pos] < minValue) {
                    minValue = _perm[pos];
                    minPos   = pos;
                }
            }
            _perm[minPos] = currentValue; // minValue renamed to currentValue
            currentValue++;
        }
    }

    EINSUMS_LOG_DEBUG("HPTT: dim={}, innerStrideA={}, innerStrideB={}", _dim, _innerStrideA, _innerStrideB);
}

/**
 * \brief fuses indices whenever possible
 * \detailed For instance:
 *           perm=3,1,2,0 & size=10,11,12,13  becomes: perm=2,1,0 &
 * size=10,11*12,13 \return This function will initialize sizeA_, perm_,
 * outerSizeA_, outersize_ and dim_
 */
template <typename floatType>
void TransposeImpl<floatType>::fuse_indices() {
    std::list<std::tuple<int, int>> fusedIndices;

    std::vector<int> perm;
    // correct perm
    for (int i = 0; i < _dim; ++i) {
        // merge indices if the two consecutive entries are identical
        int toMerge = i;
        perm.push_back(_perm[i]);
        /* By definition if size == outerSize, then no offsets are present. However,
         *  by merging with the subsequent dimension the stride the offset depends upon
         *  is lost. Therefore, the offset of the next offset must be zero too! */
        while (i + 1 < _dim && _perm[i] + 1 == _perm[i + 1] && (_sizeA[_perm[i]] == _outerSizeA[_perm[i]]) &&
               (_sizeA[_perm[i]] == _outerSizeB[i]) && (_offsetA[_perm[i + 1]] == 0) && (_offsetB[i + 1] == 0)) {
            EINSUMS_LOG_DEBUG("HPTT: merging indices {} and {}", _perm[i], _perm[i + 1]);
            fusedIndices.emplace_back(std::make_tuple(_perm[toMerge], _perm[i + 1]));
            i++;
        }
    }

    // correct sizes and outer-sizes
    for (auto tup : fusedIndices) {
        _sizeA[std::get<0>(tup)] *= _sizeA[std::get<1>(tup)];
        _outerSizeA[std::get<0>(tup)] *= _outerSizeA[std::get<1>(tup)];
        _outerSizeA[std::get<1>(tup)] = -1;
        _offsetA[std::get<1>(tup)]    = -1;

        auto pos1 = std::find(_perm.begin(), _perm.end(), std::get<0>(tup)) - _perm.begin();
        auto pos2 = std::find(_perm.begin(), _perm.end(), std::get<1>(tup)) - _perm.begin();
        _outerSizeB[pos1] *= _outerSizeB[pos2];
        _outerSizeB[pos2] = -1;
        _offsetB[pos2]    = -1;
    }

    if (fusedIndices.size() > 0) {
        _perm = perm;
        // remove gaps in the perm, if requried (e.g., perm=3,1,0 -> 2,1,0)
        int currentValue = 0;
        for (int i = 0; i < _perm.size(); ++i) {
            // find smallest element in perm_ and rename it to currentValue
            int minValue = std::numeric_limits<int>::max();
            int minPos   = -1;
            for (int pos = 0; pos < _perm.size(); ++pos) {
                if (_perm[pos] >= currentValue && _perm[pos] < minValue) {
                    minValue = _perm[pos];
                    minPos   = pos;
                }
            }
            EINSUMS_LOG_DEBUG("HPTT: perm[{}]: {} -> {}", minPos, _perm[minPos], currentValue);
            _perm[minPos]        = currentValue; // minValue renamed to currentValue
            _sizeA[currentValue] = _sizeA[minValue];
            currentValue++;
        }

        // compact outer size (e.g.: outerSizeA_[] = {24,-1,5,-1,13} ->
        // {24,5,13,-1,-1} -> {24,5,13}
        for (int i = 0; i < _dim; ++i)
            if (_outerSizeA[i] == -1) {
                int j = i + 1;
                for (; j < _dim; ++j)
                    if (_outerSizeA[j] != -1)
                        break;
                if (j < _dim) {
                    std::swap(_outerSizeA[i], _outerSizeA[j]);
                    std::swap(_offsetA[i], _offsetA[j]);
                }
            }
        for (int i = 0; i < _dim; ++i)
            if (_outerSizeB[i] == -1) {
                int j = i + 1;
                for (; j < _dim; ++j)
                    if (_outerSizeB[j] != -1)
                        break;
                if (j < _dim) {
                    std::swap(_outerSizeB[i], _outerSizeB[j]);
                    std::swap(_offsetB[i], _offsetB[j]);
                }
            }
        _dim -= fusedIndices.size();
        _outerSizeA.resize(_dim);
        _outerSizeB.resize(_dim);
        _offsetA.resize(_dim);
        _offsetB.resize(_dim);
        _sizeA.resize(_dim);
        _perm.resize(_dim);

        EINSUMS_LOG_DEBUG("HPTT: after index fusion: dim={}", _dim);
    }
}

// returns the best loop order (same as the best one with exhaustive search)
template <typename floatType>
void TransposeImpl<floatType>::get_best_loop_order(std::vector<int> &loopOrder) const {
    auto totalOuterSizeA = std::accumulate(_outerSizeA.begin(), _outerSizeA.end(), 1, std::multiplies<size_t>()) * sizeof(floatType);
    auto totalOuterSizeB = std::accumulate(_outerSizeB.begin(), _outerSizeB.end(), 1, std::multiplies<size_t>()) * sizeof(floatType);
    if (totalOuterSizeA > totalOuterSizeB && totalOuterSizeB <= 22 * 1024. * 1024.) // B is likely to fit into L3 cache
    {
        // prefer accesses to A over those to B (Rationale: reduce TLB misses)
        for (int i = 0; i < _dim; ++i)
            loopOrder[_dim - 1 - i] = i; // innermost loop idx is stored at dim_-1
        return;
    } else if (totalOuterSizeB > totalOuterSizeA && totalOuterSizeA <= 22 * 1024. * 1024.) // B is likely to fit into L3 cache
    {
        // prefer accesses to B over those to A (Rationale: reduce TLB misses)
        for (int i = 0; i < _dim; ++i)
            loopOrder[_dim - 1 - i] = _dim - 1 - i; // innermost loop idx is stored at dim_-1
        return;
    }

    // create cost matrix; cost[i,idx] === cost for idx being at loop-level i
    std::vector<double> costs(_dim * _dim);
    for (int i = 0; i < _dim; ++i) {
        for (int idx = 0; idx < _dim; ++idx) { // idx is at loop i
            double cost = 0;
            if (i != 0) {
                int const        posB        = find_pos(idx, _perm);
                int const        importanceA = (1 << (_dim - idx));  // stride-1 has the most importance ...
                int const        importanceB = (1 << (_dim - posB)); // subsequent indices are half as important
                int const        penalty     = 10 * (1 << (i - 1));
                constexpr double bias        = 1.01;
                cost                         = (importanceA + importanceB * bias) * penalty;
            }
            costs[i + idx * _dim] = cost;
        }
    }
    std::list<int> availLoopLevels; // available rows
    std::list<int> availIndices;
    for (int i = 0; i < _dim; ++i) {
        availLoopLevels.push_back(i);
        availIndices.push_back(i);
    }

    // create best loop order constructively without generating all
    for (int i = 0; i < _dim; ++i) {
        // find column with maximum cost
        int    selectedIdx = 0;
        double maxValueAll = 0;
        for (auto c : availIndices) {
            double maxValue = 0;
            for (auto r : availLoopLevels) {
                double const val = costs[c * _dim + r];
                maxValue         = (val > maxValue) ? val : maxValue;
            }

            if (maxValue > maxValueAll) {
                maxValueAll = maxValue;
                selectedIdx = c;
            }
        }
        // find minimum in that column
        int    selectedLoopLevel = 0;
        double minValue          = 1e100;
        for (auto r : availLoopLevels) {
            double const val = costs[selectedIdx * _dim + r];
            if (val < minValue) {
                minValue          = val;
                selectedLoopLevel = r;
            }
        }
        // update loop order
        loopOrder[_dim - 1 - i] = selectedIdx; // innermost loop idx is stored at dim_-1
        // remove selected row
        for (auto it = availLoopLevels.begin(); it != availLoopLevels.end(); it++)
            if (*it == selectedLoopLevel) {
                availLoopLevels.erase(it);
                break;
            }
        // remove selected col
        for (auto it = availIndices.begin(); it != availIndices.end(); it++)
            if (*it == selectedIdx) {
                availIndices.erase(it);
                break;
            }
    }
}

template <typename floatType>
double TransposeImpl<floatType>::loop_cost_heuristic(std::vector<int> const &loopOrder) const {
    double loopCost = 0.0;
    for (int i = 1; i < _dim; ++i) {
        int const idx         = loopOrder[_dim - 1 - i];
        int const posB        = find_pos(idx, _perm);
        int const importanceA = (1 << (_dim - idx));  // stride-1 has the most importance ...
        int const importanceB = (1 << (_dim - posB)); // subsequent indices are half as important
        int const penalty     = 10 * (1 << (i - 1));
        double    bias        = 1.01;
        loopCost += (importanceA + importanceB * bias) * penalty;
    }

    return loopCost;
}

template <typename floatType>
void TransposeImpl<floatType>::get_loop_orders(std::vector<std::vector<int>> &loopOrders) const {
    loopOrders.clear();
    if (_selectionMethod == ESTIMATE) {
        loopOrders.emplace_back(std::vector<int>(_dim));
        get_best_loop_order(loopOrders[0]);
        return;
    }

    std::vector<int> loopOrder;
    for (int i = 0; i < _dim; i++)
        loopOrder.push_back(i); // NOLINT

    // create all loopOrders
    do {
        if (_perm[0] == 0 && loopOrder[_dim - 1] != 0)
            continue; // ATTENTION: we skip all loop-orders where the stride-1 index
                      // is not the inner-most loop iff perm[0] == 0 (both for perf &
                      // correctness)

        loopOrders.push_back(loopOrder);
    } while (std::next_permutation(loopOrder.begin(), loopOrder.end()));

    // sort according to loop heuristic
    std::sort(loopOrders.begin(), loopOrders.end(), [this](std::vector<int> const &loopOrder1, std::vector<int> const &loopOrder2) {
        return this->loop_cost_heuristic(loopOrder1) < this->loop_cost_heuristic(loopOrder2);
    });

    if (this->infoLevel_ > 1)
        for (auto const &loopOrder : loopOrders) {
            print_vector(loopOrder, "loop");
            EINSUMS_LOG_INFO("HPTT: penalty: {}", loop_cost_heuristic(loopOrder));
        }
}

template <typename floatType>
void TransposeImpl<floatType>::create_plan() {
//   printf("entering createPlan()\n");
#ifdef HPTT_TIMERS
    double timeStart = omp_get_wtime();
#endif

    std::vector<std::shared_ptr<Plan>> allPlans;
    create_plans(allPlans);

#ifdef HPTT_TIMERS
    EINSUMS_LOG_INFO("HPTT: createPlans() took {} ms", (omp_get_wtime() - timeStart) * 1000);
    timeStart = omp_get_wtime();
#endif
    _masterPlan = select_plan(allPlans);
    if (this->infoLevel_ > 0) {
        EINSUMS_LOG_INFO("HPTT: configuration of best plan:");
        _masterPlan->print();
    }
#ifdef HPTT_TIMERS
    EINSUMS_LOG_INFO("HPTT: SelectPlan() took {} ms", (omp_get_wtime() - timeStart) * 1000);
#endif
}

template <typename floatType>
void TransposeImpl<floatType>::create_plans(std::vector<std::shared_ptr<Plan>> &plans) const {
    if (_dim == 1 || (_dim == 2 && _perm[0] == 0)) {
        plans.emplace_back(new Plan); // create dummy plan
        return;                       // handled within execute()
    }
#ifdef HPTT_TIMERS
    double parallelStrategiesTime = omp_get_wtime();
#endif
    std::vector<std::vector<int>> parallelismStrategies;
    this->get_parallelism_strategies(parallelismStrategies);
#ifdef HPTT_TIMERS
    EINSUMS_LOG_INFO("HPTT: {} parallel strategies. Time: {} ms", parallelismStrategies.size(),
                     (omp_get_wtime() - parallelStrategiesTime) * 1000);

    double loopOrdersTime = omp_get_wtime();
#endif
    std::vector<std::vector<int>> loopOrders;
    this->get_loop_orders(loopOrders);
#ifdef HPTT_TIMERS
    EINSUMS_LOG_INFO("HPTT: {} loop orders. Time: {} ms", loopOrders.size(), (omp_get_wtime() - loopOrdersTime) * 1000);
#endif

    if (_selectedParallelStrategyId != -1) {
        int              selectedParallelStrategyId = std::min((int)parallelismStrategies.size() - 1, _selectedParallelStrategyId);
        std::vector<int> parStrategy(parallelismStrategies[selectedParallelStrategyId]);
        print_vector(parStrategy, "selected parallel: ");
        parallelismStrategies.clear();
        parallelismStrategies.push_back(parStrategy);
    }
    if (_selectedLoopOrderId != -1) {
        int              selectedLoopOrderId = std::min((int)loopOrders.size() - 1, _selectedLoopOrderId);
        std::vector<int> loopOrder(loopOrders[selectedLoopOrderId]);
        print_vector(loopOrder, "selected loopOrder: ");
        loopOrders.clear();
        loopOrders.push_back(loopOrder);
    }

    int const posStride1A_inB = find_pos(0, _perm);
    int const posStride1B_inA = _perm[0];

    // combine the loopOrder and parallelismStrategies according to their
    // heuristics, search the space with a growing rectangle (from best to worst,
    // see line marked with ***)
    bool done = false;
    for (int start = 0; start < std::max(parallelismStrategies.size(), loopOrders.size()) && !done; start++)
        for (int i = 0; i < parallelismStrategies.size() && !done; i++) {
            for (int j = 0; j < loopOrders.size() && !done; j++) {
                if (i > start || j > start || (i != start && j != start))
                    continue; // these are already done ***

                auto      numThreadsAtLoop = parallelismStrategies[i];
                auto      loopOrder        = loopOrders[j];
                auto      plan             = std::make_shared<Plan>(loopOrder, numThreadsAtLoop);
                int const numTasks         = plan->get_num_tasks();

                // Plan construction is serial: it fills numTasks (~= _numThreads)
                // ComputeNode chains of trivial integer arithmetic, so an OMP
                // team here (once per candidate plan, up to hundreds) costs more
                // in fork/join than it saves. Serial also removes the only
                // benign-but-noisy TSan reports HPTT emitted outside the actual
                // transpose kernels. (Execution - execute_expert/axpy/macro_kernel
                // - stays threaded; that is where the parallelism pays off.)
                for (int taskId = 0; taskId < numTasks; taskId++) {
                    ComputeNode *currentNode = plan->get_root_node(taskId);

                    int numThreadsPerComm = numTasks; // global communicator // e.g., 6
                    int taskIdComm        = taskId;   // e.g., 0,1,2,3,4,5
                    // divide each loop-level l, corresponding to index loopOrder[l], into
                    // numThreadsAtLoop[index] chunks
                    for (int l = 0; l < _dim; ++l) {
                        int const index  = loopOrder[l];
                        currentNode->inc = this->get_increment(index);

                        int const numTasksAtLevel         = numThreadsAtLoop[index];                                   //  e.g., 3
                        int const numParallelismAvailable = (_sizeA[index] + currentNode->inc - 1) / currentNode->inc; // e.g., 5
                        int const workPerThread = (numParallelismAvailable + numTasksAtLevel - 1) / numTasksAtLevel;   // ceil(5/3) = 2

                        numThreadsPerComm /= numTasksAtLevel;                // numThreads in next communicator // 6/3 = 2
                        int const commId = (taskIdComm / numThreadsPerComm); //  = 0,0,1,1,2,2
                        taskIdComm       = taskIdComm % numThreadsPerComm;   // local taskId in next
                                                                             // communicator // 0,1,0,1,0,1

                        if (index == 0)
                            currentNode->indexA = true;
                        if (find_pos(index, _perm) == 0)
                            currentNode->indexB = true;
                        currentNode->start = std::min(_sizeA[index] + _offsetB[find_pos(index, _perm)],
                                                      commId * workPerThread * currentNode->inc + _offsetB[find_pos(index, _perm)]);
                        currentNode->end   = std::min(_sizeA[index] + _offsetB[find_pos(index, _perm)],
                                                      (commId + 1) * workPerThread * currentNode->inc + _offsetB[find_pos(index, _perm)]);

                        currentNode->lda       = _lda[index];
                        currentNode->ldb       = _ldb[find_pos(index, _perm)];
                        currentNode->offDiffAB = (ptrdiff_t)_offsetA[index] - (ptrdiff_t)_offsetB[find_pos(index, _perm)];

                        if (_perm[0] != 0 || l != _dim - 1) {
                            currentNode->next = std::make_unique<ComputeNode>();
                            currentNode       = currentNode->next.get();
                        }
                    }

                    // macro-kernel
                    if (_perm[0] != 0) {
                        if (posStride1A_inB == 0)
                            currentNode->indexB = true;
                        currentNode->start     = -1;
                        currentNode->end       = -1;
                        currentNode->inc       = -1;
                        currentNode->lda       = _lda[posStride1B_inA];
                        currentNode->ldb       = _ldb[posStride1A_inB];
                        currentNode->offDiffAB = (ptrdiff_t)_offsetA[posStride1B_inA] - (ptrdiff_t)_offsetB[posStride1A_inB];
                        currentNode->next.reset();
                    }
                }
                plans.push_back(plan);
                if (_selectionMethod == ESTIMATE || (_selectionMethod == MEASURE && plans.size() > 200) ||
                    (_selectionMethod == PATIENT && plans.size() > 400) || (_selectionMethod == CRAZY && plans.size() > 800))
                    done = true;
            }
        }
}

/**
 * Estimates the time in seconds for the given computeTree
 */
template <typename floatType>
float TransposeImpl<floatType>::estimate_execution_time(std::shared_ptr<Plan> const plan) {
    auto startTime = std::chrono::high_resolution_clock::now();
    this->execute_estimate(plan.get());
    double elapsedTime =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(std::chrono::high_resolution_clock::now() - startTime)
            .count();

    double const minMeasurementTime = 0.1; // in seconds

    // do at least 3 repetitions or spent at least 'minMeasurementTime' seconds
    // for each candidate
    int nRepeat = std::min(3, (int)std::ceil(minMeasurementTime / elapsedTime));

    // execute just a few iterations and exterpolate the result
    startTime = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < nRepeat; ++i) // ATTENTION: we are not clearing the caches inbetween runs
        this->execute_estimate(plan.get());
    elapsedTime =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(std::chrono::high_resolution_clock::now() - startTime)
            .count();
    elapsedTime /= nRepeat;

    EINSUMS_LOG_DEBUG("HPTT: estimated time: {:.3e} ms.", elapsedTime);
    return elapsedTime;
}

template <typename floatType>
double TransposeImpl<floatType>::get_time_limit() const {
    if (_selectionMethod == ESTIMATE)
        return 0.0;
    else if (_selectionMethod == MEASURE)
        return 10.; // 10s
    else if (_selectionMethod == PATIENT)
        return 60.; // 1m
    else if (_selectionMethod == CRAZY)
        return 3600.; // 1h
    else {
        EINSUMS_ASSERT_(false, "HPTT: selectionMethod {} unknown.", static_cast<int>(_selectionMethod));
    }
    return -1;
}

template <typename floatType>
std::shared_ptr<Plan> TransposeImpl<floatType>::select_plan(std::vector<std::shared_ptr<Plan>> const &plans) {
    EINSUMS_ASSERT_(!plans.empty(), "HPTT: internal error: no plans generated.");
    if (_selectionMethod == ESTIMATE) // fast return
        return plans[0];

    double timeLimit               = this->get_time_limit() * 1000; // in ms
    int    maxAutotuningCandidates = plans.size();
    if (_maxAutotuningCandidates != -1) {
        maxAutotuningCandidates = _maxAutotuningCandidates;
        timeLimit               = 1e9;
    }

    float minTime     = std::numeric_limits<float>::max();
    int   bestPlan_id = 0;

    if (plans.size() > 1) {
        int  plansEvaluated = 0;
        auto startTime      = std::chrono::high_resolution_clock::now();
        for (int plan_id = 0; plan_id < maxAutotuningCandidates; plan_id++) {
            auto const &p = plans[plan_id];

            double elapsedTime =
                std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(std::chrono::high_resolution_clock::now() - startTime)
                    .count();
            if (elapsedTime >= timeLimit) // timelimit reached
                break;

            float estimatedTime = this->estimate_execution_time(p);
            plansEvaluated++;

            if (estimatedTime < minTime) {
                bestPlan_id = plan_id;
                minTime     = estimatedTime;
            }
            if (this->infoLevel_ > 1) {
                EINSUMS_LOG_INFO("HPTT: plan {} will take roughly {} ms.", plan_id, estimatedTime * 1000.);
                plans[plan_id]->print();
            }
        }
        if (this->infoLevel_ > 0)
            EINSUMS_LOG_INFO("HPTT: evaluated {}/{} candidates and selected candidate {}.", plansEvaluated, plans.size(), bestPlan_id);
    }
    return plans[bestPlan_id];
}

// The geometry this rung's plans are built for; see PlanTarget.
template <typename FloatType>
static PlanTarget plan_target(TransposeKernels<FloatType> const &kernels) {
    return PlanTarget{.vector_bits  = static_cast<uint16_t>(kernels.vector_bits),
                      .element_size = static_cast<uint8_t>(sizeof(FloatType)),
                      .rung         = static_cast<uint8_t>(kernels.rung),
                      .pad          = 0};
}

template <typename FloatType>
void TransposeImpl<FloatType>::write_to_file(std::FILE *fp) const {
    setup_file(fp);

    // Get the file header.
    FileHeader header;
    uint32_t   check;
    size_t     error2;

    TransposeConstants constants;
    PlanTarget         target;

    int error1 = fseek(fp, 0, SEEK_SET);

    if (error1 != 0) {
        goto write_to_file_error;
    }

    error2 = fread(&header, sizeof(FileHeader), 1, fp);

    if (error2 < 1) {
        goto write_to_file_error;
    }

    if (strncmp(header.magic, "HPTT", 4) != 0) {
        EINSUMS_THROW_EXCEPTION(std::runtime_error, "Trying to write to a file that is not a HPTT transpose file!");
    }

    error1 = fseek(fp, sizeof(FileHeader), SEEK_SET);

    if (error1 != 0) {
        goto write_to_file_error;
    }

    target = plan_target(*_kernels);

    error2 = fwrite(&target, sizeof(PlanTarget), 1, fp);

    if (error2 < 1) {
        goto write_to_file_error;
    }

    constants = {.dim                      = _dim,
                 .numThreads               = _numThreads,
                 .innerStrideA             = _innerStrideA,
                 .innerStrideB             = _innerStrideB,
                 .selectedParallelStrategy = _selectedParallelStrategyId,
                 .selectedLoopOrderId      = _selectedLoopOrderId,
                 .conjA                    = _conjA,
                 .pad                      = 0};

    error2 = fwrite(&constants, sizeof(TransposeConstants), 1, fp);

    if (error2 < 1) {
        goto write_to_file_error;
    }

    std::fflush(fp);

    error2 = fwrite(_sizeA.data(), sizeof(size_t), _dim, fp);

    if (error2 < _dim) {
        goto write_to_file_error;
    }

    std::fflush(fp);

    error2 = fwrite(_outerSizeA.data(), sizeof(size_t), _dim, fp);

    if (error2 < _dim) {
        goto write_to_file_error;
    }

    std::fflush(fp);

    error2 = fwrite(_outerSizeB.data(), sizeof(size_t), _dim, fp);

    if (error2 < _dim) {
        goto write_to_file_error;
    }

    std::fflush(fp);

    error2 = fwrite(_offsetA.data(), sizeof(size_t), _dim, fp);

    if (error2 < _dim) {
        goto write_to_file_error;
    }

    std::fflush(fp);

    error2 = fwrite(_offsetB.data(), sizeof(size_t), _dim, fp);

    if (error2 < _dim) {
        goto write_to_file_error;
    }

    std::fflush(fp);

    error2 = fwrite(_lda.data(), sizeof(size_t), _dim, fp);

    if (error2 < _dim) {
        goto write_to_file_error;
    }

    std::fflush(fp);

    error2 = fwrite(_ldb.data(), sizeof(size_t), _dim, fp);

    if (error2 < _dim) {
        goto write_to_file_error;
    }

    std::fflush(fp);

    error2 = fwrite(_perm.data(), sizeof(int), _dim, fp);

    if (error2 < _dim) {
        goto write_to_file_error;
    }

    std::fflush(fp);

    _masterPlan->write_to_file(fp);

    check = compute_checksum(fp);

    error1 = fseek(fp, offsetof(FileHeader, checksum), SEEK_SET);

    if (error1 != 0) {
        goto write_to_file_error;
    }

    error2 = fwrite(&check, sizeof(uint32_t), 1, fp);

    std::fflush(fp);

    if (error2 < 1) {
        goto write_to_file_error;
    }

    return;

write_to_file_error:
    EINSUMS_LOG_ERROR("HPTT: error writing to file: {}", std::strerror(errno));
    EINSUMS_THROW_EXCEPTION(std::runtime_error, "IO error");
}

template <typename FloatType>
TransposeImpl<FloatType>::TransposeImpl(TransposeKernels<FloatType> const &kernels, std::FILE *fp, FloatType alpha, FloatType const *A,
                                        FloatType beta, FloatType *B)
    : _kernels(&kernels) {
#ifdef _OPENMP
    omp_init_lock(&_writelock);
#endif
    // Get the file header.
    FileHeader         header;
    size_t             error2;
    TransposeConstants constants;
    PlanTarget         target;
    uint32_t           check;

    int error1 = fseek(fp, 0, SEEK_SET);

    if (error1 != 0) {
        goto read_from_file_error;
    }

    error2 = fread(&header, sizeof(FileHeader), 1, fp);

    if (error2 < 1) {
        goto read_from_file_error;
    }

    if (strncmp(header.magic, "HPTT", 4) != 0) {
        EINSUMS_THROW_EXCEPTION(std::runtime_error, "Trying to read from a file that is not a HPTT transpose file!");
    }

    if (header.version[2] != plan_file_format) {
        EINSUMS_THROW_EXCEPTION(std::runtime_error,
                                "HPTT plan file has format version {}, but this build reads version {}; the file predates the "
                                "recorded vector width and cannot be trusted. Recreate the plan.",
                                static_cast<int>(header.version[2]), static_cast<int>(plan_file_format));
    }

    error1 = fseek(fp, sizeof(FileHeader), SEEK_SET);

    if (error1 != 0) {
        goto read_from_file_error;
    }

    error2 = fread(&target, sizeof(PlanTarget), 1, fp);

    if (error2 < 1) {
        goto read_from_file_error;
    }

    {
        PlanTarget const mine = plan_target(*_kernels);
        if (endian_char() != header.version[3]) {
            target.vector_bits = byteswap(target.vector_bits);
        }
        if (target.vector_bits != mine.vector_bits || target.element_size != mine.element_size) {
            EINSUMS_THROW_EXCEPTION(
                std::runtime_error,
                "HPTT plan file was written by the {} rung for {}-bit vectors and {}-byte elements, but this process reads it with the {} "
                "rung, {}-bit vectors and {}-byte elements. Its loop increments are wrong here; recreate the plan.",
                stripes::to_string(static_cast<stripes::InstructionSet>(target.rung)), target.vector_bits,
                static_cast<int>(target.element_size), stripes::to_string(static_cast<stripes::InstructionSet>(mine.rung)),
                mine.vector_bits, static_cast<int>(mine.element_size));
        }
    }

    error2 = fread(&constants, sizeof(TransposeConstants), 1, fp);

    if (error2 < 1) {
        goto read_from_file_error;
    }

    _A     = A;
    _B     = B;
    _alpha = alpha;
    _beta  = beta;
    _dim   = constants.dim;

    if (endian_char() != header.version[3]) {
        _dim = byteswap(_dim);
    }

    _sizeA.resize(_dim);
    _perm.resize(_dim);
    _outerSizeA.resize(_dim);
    _outerSizeB.resize(_dim);
    _offsetA.resize(_dim);
    _offsetB.resize(_dim);
    _innerStrideA = constants.innerStrideA;
    _innerStrideB = constants.innerStrideB;
    _lda.resize(_dim);
    _ldb.resize(_dim);
    _threadIds.reserve(_dim);
    _numThreads                 = constants.numThreads;
    _selectedParallelStrategyId = constants.selectedParallelStrategy;
    _selectedLoopOrderId        = constants.selectedLoopOrderId;
    _conjA                      = constants.conjA;

    if (endian_char() != header.version[3]) {
        _innerStrideA               = byteswap(constants.innerStrideA);
        _innerStrideB               = byteswap(constants.innerStrideB);
        _numThreads                 = byteswap(constants.numThreads);
        _selectedParallelStrategyId = byteswap(constants.selectedParallelStrategy);
        _selectedLoopOrderId        = byteswap(constants.selectedLoopOrderId);
    }

    for (int i = 0; i < _numThreads; ++i)
        _threadIds.push_back(i);

    error2 = fread(_sizeA.data(), sizeof(size_t), _dim, fp);

    if (error2 < _dim) {
        goto read_from_file_error;
    }

    error2 = fread(_outerSizeA.data(), sizeof(size_t), _dim, fp);

    if (error2 < _dim) {
        goto read_from_file_error;
    }

    error2 = fread(_outerSizeB.data(), sizeof(size_t), _dim, fp);

    if (error2 < _dim) {
        goto read_from_file_error;
    }

    error2 = fread(_offsetA.data(), sizeof(size_t), _dim, fp);

    if (error2 < _dim) {
        goto read_from_file_error;
    }

    error2 = fread(_offsetB.data(), sizeof(size_t), _dim, fp);

    if (error2 < _dim) {
        goto read_from_file_error;
    }

    error2 = fread(_lda.data(), sizeof(size_t), _dim, fp);

    if (error2 < _dim) {
        goto read_from_file_error;
    }

    error2 = fread(_ldb.data(), sizeof(size_t), _dim, fp);

    if (error2 < _dim) {
        goto read_from_file_error;
    }

    error2 = fread(_perm.data(), sizeof(int), _dim, fp);

    if (error2 < _dim) {
        goto read_from_file_error;
    }

    if (endian_char() != header.version[3]) {
        for (int i = 0; i < _dim; i++) {
            _sizeA[i]      = byteswap(_sizeA[i]);
            _perm[i]       = byteswap(_perm[i]);
            _outerSizeA[i] = byteswap(_outerSizeA[i]);
            _outerSizeB[i] = byteswap(_outerSizeB[i]);
            _offsetA[i]    = byteswap(_offsetA[i]);
            _offsetB[i]    = byteswap(_offsetB[i]);
            _lda[i]        = byteswap(_lda[i]);
            _ldb[i]        = byteswap(_ldb[i]);
        }
    }

    _masterPlan = std::make_shared<Plan>(fp, endian_char() != header.version[3]);

    return;

read_from_file_error:
    EINSUMS_LOG_ERROR("HPTT: error reading from file: {}", std::strerror(errno));
    EINSUMS_THROW_EXCEPTION(std::runtime_error, "IO error");
}

template class TransposeImpl<float>;
template class TransposeImpl<double>;
template class TransposeImpl<FloatComplex>;
template class TransposeImpl<DoubleComplex>;

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) || defined(__AVX512FP16__)
template class TransposeImpl<stripes::half_t>;

#endif

#if defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC) || defined(__AVX512BF16__)
template class TransposeImpl<stripes::bfloat16_t>;

#endif

EINSUMS_NAMESPACE_END(hptt)
