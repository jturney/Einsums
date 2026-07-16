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

#include <Einsums/Logging.hpp>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <algorithm>
#include <list>
#include <memory>
#include <vector>

#include "HPTTTypes.hpp"
#ifdef _OPENMP
#    include <omp.h>
#endif

namespace hptt {

/**
 * @class Transpose
 *
 * \brief Arch-neutral interface to a compiled tensor-transposition plan.
 *
 * A plan is created once via hptt::create_plan() (or Transpose::create())
 * and can be executed many times via execute(). To reuse a plan on new
 * buffers, repoint it with set_input_ptr()/set_output_ptr(); scaling
 * factors and conjugation can likewise be updated between executions.
 *
 * This class is a pure interface: the concrete implementation (selected at
 * plan-creation time) lives inside the HPTT module. Copying a plan is done
 * through clone(); plans round-trip through files via write_to_file() and
 * read_from_file().
 *
 * @tparam floatType The element type this plan transposes.
 */
template <typename floatType>
class Transpose {
  public:
    virtual ~Transpose() = default;

    /**
     * Executes the transposition.
     */
    virtual void execute() noexcept = 0;

    /**
     * \brief Set the pointer for A.
     *
     * Useful to reuse the compiled transposition over multiple invocations.
     */
    virtual void set_input_ptr(floatType const *A) noexcept = 0;

    /**
     * \brief Set the pointer for B.
     *
     * Useful to reuse the compiled transposition over multiple invocations.
     */
    virtual void set_output_ptr(floatType *B) noexcept = 0;

    /**
     * \brief Set the scaling factor for A.
     */
    virtual void set_alpha(floatType alpha) noexcept = 0;

    /**
     * \brief Set the scaling factor for B.
     */
    virtual void set_beta(floatType beta) noexcept = 0;

    /**
     * Change whether to conjugate the input tensor.
     */
    virtual void set_conj_a(bool conjA) noexcept = 0;

    /**
     * Serialize this plan (parameters and compiled loop structure) to a file
     * previously prepared with hptt::setup_file(). Restore with
     * read_from_file().
     */
    virtual void write_to_file(std::FILE *fp) const = 0;

    /**
     * \brief Deep-copy this plan.
     *
     * Replaces the copy constructor of earlier versions: the concrete plan
     * type is hidden behind this interface, so copying must go through a
     * virtual call. Used to clone a cached template plan before repointing
     * it at new buffers.
     */
    [[nodiscard]] virtual std::shared_ptr<Transpose> clone() const = 0;

    /**
     * \brief Create a transposition plan.
     *
     * This is the factory behind every hptt::create_plan() overload; the
     * overloads normalize their arguments into this superset signature.
     * The concrete plan implementation is chosen here, at creation time.
     *
     * \param[in] sizeA dim-dimensional array with the sizes of each dimension of A.
     * \param[in] perm dim-dimensional array representing the permutation of the indices.
     *                 For instance, perm[] = {1,0,2} denotes: \f$B_{i1,i0,i2} \gets A_{i0,i1,i2}\f$.
     * \param[in] outerSizeA outer sizes of A, or NULL (meaning equal to sizeA); enables operating on sub-tensors.
     * \param[in] outerSizeB outer sizes of B, or NULL (meaning equal to perm(sizeA)).
     * \param[in] offsetA per-dimension offsets into A, or NULL (meaning zero).
     * \param[in] offsetB per-dimension offsets into B, or NULL (meaning zero).
     * \param[in] innerStrideA non-unit stride of A's innermost dimension.
     * \param[in] innerStrideB non-unit stride of B's innermost dimension.
     * \param[in] dim Dimensionality of the tensors.
     * \param[in] A Pointer to the raw data of the input tensor A.
     * \param[in] alpha Scaling factor for A.
     * \param[inout] B Pointer to the raw data of the output tensor B.
     * \param[in] beta Scaling factor for B.
     * \param[in] selectionMethod Whether/how to auto-tune (see hptt::SelectionMethod).
     *                            ATTENTION: auto-tuning (e.g. hptt::MEASURE) writes to B during tuning.
     * \param[in] numThreads Number of threads participating in the transposition.
     * \param[in] threadIds OpenMP thread IDs participating, for execution inside an existing parallel region.
     * \param[in] useRowMajor Row-major memory layout when set (default: column-major).
     */
    static std::shared_ptr<Transpose> create(size_t const *sizeA, int const *perm, size_t const *outerSizeA, size_t const *outerSizeB,
                                             size_t const *offsetA, size_t const *offsetB, size_t const innerStrideA,
                                             size_t const innerStrideB, int const dim, floatType const *A, floatType const alpha,
                                             floatType *B, floatType const beta, SelectionMethod const selectionMethod,
                                             int const numThreads, int const *threadIds = nullptr, bool const useRowMajor = false);

    /**
     * \brief Reconstruct a plan previously saved with write_to_file().
     *
     * The data pointers, alpha, and beta are not part of the serialized
     * form and must be supplied afresh.
     */
    static std::shared_ptr<Transpose> read_from_file(std::FILE *fp, floatType alpha, floatType const *A, floatType beta, floatType *B);
};

extern template class EINSUMS_EXPORT Transpose<float>;
extern template class EINSUMS_EXPORT Transpose<double>;
extern template class EINSUMS_EXPORT Transpose<FloatComplex>;
extern template class EINSUMS_EXPORT Transpose<DoubleComplex>;

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) || defined(__AVX512FP16__)
extern template class EINSUMS_EXPORT Transpose<einsums::simd::half_t>;
#endif

#if defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC) || defined(__AVX512BF16__)
extern template class EINSUMS_EXPORT Transpose<einsums::simd::bfloat16_t>;
#endif

} // namespace hptt
