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
 * @author: Paul Springer (springer@aices.rwth-aachen.de)
 */

#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/HPTT/Utils.hpp>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <list>

#include "Primes.hpp"

namespace hptt {

template <>
void get_prime_factors(std::uint8_t n, std::list<std::uint8_t> &primeFactors) {
    primeFactors.clear();

    std::uint64_t factor_list = detail::char_factors[n];

    constexpr std::uint64_t mask_0 = 0xffUL, mask_1 = 0xff00UL, mask_2 = 0xff0000UL, mask_3 = 0xff000000UL, mask_4 = 0xff00000000UL,
                            mask_5 = 0xff0000000000UL, mask_6 = 0xff000000000000UL, mask_7 = 0xff00000000000000UL;

    // Extract the factors.
    std::uint8_t factor = factor_list & mask_0;

    if (!factor) {
        return;
    }

    primeFactors.push_back(factor);

    factor = (factor_list & mask_1) >> 8;

    if (!factor) {
        return;
    }

    primeFactors.push_back(factor);

    factor = (factor_list & mask_2) >> 16;

    if (!factor) {
        return;
    }

    primeFactors.push_back(factor);

    factor = (factor_list & mask_3) >> 24;

    if (!factor) {
        return;
    }

    primeFactors.push_back(factor);

    factor = (factor_list & mask_4) >> 32;

    if (!factor) {
        return;
    }

    primeFactors.push_back(factor);

    factor = (factor_list & mask_5) >> 40;

    if (!factor) {
        return;
    }

    primeFactors.push_back(factor);

    factor = (factor_list & mask_6) >> 48;

    if (!factor) {
        return;
    }

    primeFactors.push_back(factor);

    // There will never be a seventh factor.
}

template <>
void get_prime_factors(std::uint16_t n, std::list<std::uint16_t> &primeFactors) {
    primeFactors.clear();

    /*
     * 1 is not a prime and zero is not prime.
     */
    if (n <= 1) {
        return;
    }

    uint16_t quotient = n;

    for (unsigned short prime : detail::short_primes) {
        while (quotient % prime == 0) {
            quotient /= prime;
            primeFactors.push_back(prime);
        }

        /*
         * To test if a number is prime, we only need to
         * check up to its square root. If there is some prime p that is greater than
         * the square root of n, then p * p > n. The only way for n to be divisible by p
         * is if there is some other prime p' less than p such that p * p' = n. Since p' < p,
         * we have already done this check, so there are no more primes to test. We can use the
         * same logic when doing the factorization. At this point, we have removed all primes less than
         * p, so we know quotient is not divisible by those. Thus, if p is greater than the square root
         * of the quotient, we know there are no more primes to check, and either the quotient is 1 or
         * the quotient is prime.
         */
        if (prime * prime > quotient) {
            break;
        }
    }

    if (quotient != 1) {
        primeFactors.push_back(quotient);
    }
}

template <>
void get_prime_factors(std::uint32_t n, std::list<std::uint32_t> &primeFactors) {
    primeFactors.clear();

    /*
     * 1 is not a prime and zero is not prime.
     */
    if (n <= 1) {
        return;
    }

    uint32_t quotient = n;

    for (unsigned int prime : detail::int_primes) {
        while (quotient % prime == 0) {
            quotient /= prime;
            primeFactors.push_back(prime);
        }

        /*
         * To test if a number is prime, we only need to
         * check up to its square root. If there is some prime p that is greater than
         * the square root of n, then p * p > n. The only way for n to be divisible by p
         * is if there is some other prime p' less than p such that p * p' = n. Since p' < p,
         * we have already done this check, so there are no more primes to test. We can use the
         * same logic when doing the factorization. At this point, we have removed all primes less than
         * p, so we know quotient is not divisible by those. Thus, if p is greater than the square root
         * of the quotient, we know there are no more primes to check, and either the quotient is 1 or
         * the quotient is prime.
         */
        if (prime * prime > quotient) {
            break;
        }
    }

    if (quotient != 1) {
        primeFactors.push_back(quotient);
    }
}

template <>
void get_prime_factors(std::int8_t n, std::list<std::int8_t> &primeFactors) {
    primeFactors.clear();

    if (n <= 0) {
        return;
    }

    std::uint64_t factor_list = detail::char_factors[n];

    constexpr std::uint64_t mask_0 = 0xffUL, mask_1 = 0xff00UL, mask_2 = 0xff0000UL, mask_3 = 0xff000000UL, mask_4 = 0xff00000000UL,
                            mask_5 = 0xff0000000000UL, mask_6 = 0xff000000000000UL, mask_7 = 0xff00000000000000UL;

    // Extract the factors.
    std::uint8_t factor = factor_list & mask_0;

    if (!factor) {
        return;
    }

    primeFactors.push_back(factor);

    factor = (factor_list & mask_1) >> 8;

    if (!factor) {
        return;
    }

    primeFactors.push_back(factor);

    factor = (factor_list & mask_2) >> 16;

    if (!factor) {
        return;
    }

    primeFactors.push_back(factor);

    factor = (factor_list & mask_3) >> 24;

    if (!factor) {
        return;
    }

    primeFactors.push_back(factor);

    factor = (factor_list & mask_4) >> 32;

    if (!factor) {
        return;
    }

    primeFactors.push_back(factor);

    factor = (factor_list & mask_5) >> 40;

    if (!factor) {
        return;
    }

    primeFactors.push_back(factor);

    // There will never be a sixth or seventh factor.
}

template <>
void get_prime_factors(std::int16_t n, std::list<std::int16_t> &primeFactors) {
    primeFactors.clear();

    /*
     * 1 is not a prime and zero is not prime.
     */
    if (n <= 1) {
        return;
    }

    int16_t quotient = n;

    for (short prime : detail::short_primes) {
        while (quotient % prime == 0) {
            quotient /= prime;
            primeFactors.push_back(prime);
        }

        /*
         * To test if a number is prime, we only need to
         * check up to its square root. If there is some prime p that is greater than
         * the square root of n, then p * p > n. The only way for n to be divisible by p
         * is if there is some other prime p' less than p such that p * p' = n. Since p' < p,
         * we have already done this check, so there are no more primes to test. We can use the
         * same logic when doing the factorization. At this point, we have removed all primes less than
         * p, so we know quotient is not divisible by those. Thus, if p is greater than the square root
         * of the quotient, we know there are no more primes to check, and either the quotient is 1 or
         * the quotient is prime.
         */
        if (prime * prime > quotient) {
            break;
        }
    }

    if (quotient != 1) {
        primeFactors.push_back(quotient);
    }
}

template <>
void get_prime_factors(std::int32_t n, std::list<std::int32_t> &primeFactors) {
    primeFactors.clear();

    /*
     * 1 is not a prime and zero is not prime.
     */
    if (n <= 1) {
        return;
    }

    int32_t quotient = n;

    for (int prime : detail::int_primes) {
        while (quotient % prime == 0) {
            quotient /= prime;
            primeFactors.push_back(prime);
        }

        /*
         * To test if a number is prime, we only need to
         * check up to its square root. If there is some prime p that is greater than
         * the square root of n, then p * p > n. The only way for n to be divisible by p
         * is if there is some other prime p' less than p such that p * p' = n. Since p' < p,
         * we have already done this check, so there are no more primes to test. We can use the
         * same logic when doing the factorization. At this point, we have removed all primes less than
         * p, so we know quotient is not divisible by those. Thus, if p is greater than the square root
         * of the quotient, we know there are no more primes to check, and either the quotient is 1 or
         * the quotient is prime.
         */
        if (prime * prime > quotient) {
            break;
        }
    }

    if (quotient != 1) {
        primeFactors.push_back(quotient);
    }
}

int find_pos(int value, int const *array, int n) {
    for (int i = 0; i < n; ++i)
        if (array[i] == value)
            return i;
    return -1;
}

void trash_cache(double *A, double *B, int n) {
#ifdef _OPENMP
#    pragma omp parallel
#endif
    for (int i = 0; i < n; i++)
        A[i] += 0.999 * B[i];
}

static constexpr uint8_t  num_factorials = 21;
static constexpr uint64_t factorials[]   = {1UL,
                                            1UL,
                                            2UL,
                                            6UL,
                                            24UL,
                                            120UL,
                                            720UL,
                                            5040UL,
                                            40320UL,
                                            362880UL,
                                            3628800UL,
                                            39916800UL,
                                            479001600UL,
                                            6227020800UL,
                                            87178291200UL,
                                            1307674368000UL,
                                            20922789888000UL,
                                            355687428096000UL,
                                            6402373705728000UL,
                                            121645100408832000UL,
                                            2432902008176640000UL};

uint64_t factorial(uint8_t n) {
    if (n >= num_factorials) {
        EINSUMS_THROW_EXCEPTION(std::overflow_error, "Can not take a factorial that large!");
    } else {
        return factorials[n];
    }
}

void account_for_row_major(size_t const *sizeA, size_t const *outerSizeA, size_t const *outerSizeB, size_t const *offsetA,
                           size_t const *offsetB, int const *perm, size_t *tmpSizeA, size_t *tmpOuterSizeA, size_t *tmpOuterSizeB,
                           size_t *tmpOffsetA, size_t *tmpOffsetB, int *tmpPerm, int const dim, bool const useRowMajor) {
    for (int i = 0; i < dim; ++i) {
        int idx = i;
        if (useRowMajor) {
            idx        = dim - 1 - i; // reverse order
            tmpPerm[i] = dim - perm[idx] - 1;
        } else
            tmpPerm[i] = perm[i];
        tmpSizeA[i] = sizeA[idx];

        if (outerSizeA == nullptr)
            tmpOuterSizeA[i] = sizeA[idx];
        else
            tmpOuterSizeA[i] = outerSizeA[idx];
        if (outerSizeB == nullptr)
            tmpOuterSizeB[i] = sizeA[perm[idx]];
        else
            tmpOuterSizeB[i] = outerSizeB[idx];
        if (offsetA == nullptr)
            tmpOffsetA[i] = 0;
        else
            tmpOffsetA[i] = offsetA[idx];
        if (offsetB == nullptr)
            tmpOffsetB[i] = 0;
        else
            tmpOffsetB[i] = offsetB[idx];
    }
}

} // namespace hptt

extern "C" void random_numa_aware_init(float *data, long const *size, int dim) {
    long totalSize = 1;
    for (int i = 0; i < dim; i++)
        totalSize *= size[i];
#pragma omp parallel for
    for (int i = 0; i < totalSize; ++i)
        data[i] = (i + 1) % 1000 - 500;
}
