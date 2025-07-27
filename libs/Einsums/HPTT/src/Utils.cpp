//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

/**
 * @author: Paul Springer (springer@aices.rwth-aachen.de)
 */

#include <Einsums/HPTT/Utils.hpp>

#include <bit>
#include <cstdint>
#include <list>
#include <stdexcept>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "Primes.hpp"

namespace hptt {

template <>
void getPrimeFactors(std::uint8_t n, std::list<std::uint8_t> &primeFactors) {
    primeFactors.clear();

    std::uint64_t factor_list = detail::char_factors[n];

    constexpr std::uint64_t mask_0 = 0xffUL, mask_1 = 0xff00UL, mask_2 = 0xff0000UL, mask_3 = 0xff000000UL, mask_4 = 0xff00000000UL,
                            mask_5 = 0xff0000000000UL, mask_6 = 0xff000000000000UL;

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
void getPrimeFactors(std::uint16_t n, std::list<std::uint16_t> &primeFactors) {
    primeFactors.clear();

    /*
     * 1 is not a prime and zero is not prime.
     */
    if (n <= 1) {
        return;
    }

    uint16_t quotient = n;

    for (uint8_t index = 0; index < SHORT_PRIMES; index++) {
        std::uint16_t prime = detail::short_primes[index];
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
void getPrimeFactors(std::uint32_t n, std::list<std::uint32_t> &primeFactors) {
    primeFactors.clear();

    /*
     * 1 is not a prime and zero is not prime.
     */
    if (n <= 1) {
        return;
    }

    uint32_t quotient = n;

    for (uint16_t index = 0; index < INT_PRIMES; index++) {
        std::uint32_t prime = detail::int_primes[index];
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
void getPrimeFactors(std::int8_t n, std::list<std::int8_t> &primeFactors) {
    primeFactors.clear();

    if (n <= 0) {
        return;
    }

    std::uint64_t factor_list = detail::char_factors[n];

    constexpr std::uint64_t mask_0 = 0xffUL, mask_1 = 0xff00UL, mask_2 = 0xff0000UL, mask_3 = 0xff000000UL, mask_4 = 0xff00000000UL,
                            mask_5 = 0xff0000000000UL;

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
void getPrimeFactors(std::int16_t n, std::list<std::int16_t> &primeFactors) {
    primeFactors.clear();

    /*
     * 1 is not a prime and zero is not prime.
     */
    if (n <= 1) {
        return;
    }

    int16_t quotient = n;

    for (uint8_t index = 0; index < SHORT_PRIMES; index++) {
        std::int16_t prime = detail::short_primes[index];
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
void getPrimeFactors(std::int32_t n, std::list<std::int32_t> &primeFactors) {
    primeFactors.clear();

    /*
     * 1 is not a prime and zero is not prime.
     */
    if (n <= 1) {
        return;
    }

    int32_t quotient = n;

    for (uint16_t index = 0; index < INT_PRIMES; index++) {
        std::int32_t prime = detail::int_primes[index];
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

int findPos(int value, int const *array, int n) {
    for (int i = 0; i < n; ++i)
        if (array[i] == value)
            return i;
    return -1;
}

void trashCache(double *A, double *B, int n) {
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
        throw std::overflow_error("Can not take a factorial that large!");
    } else {
        return factorials[n];
    }
}

void accountForRowMajor(size_t const *sizeA, size_t const *outerSizeA, size_t const *outerSizeB, size_t const *offsetA,
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

extern "C" void randomNumaAwareInit(float *data, long const *size, int dim) {
    long totalSize = 1;
    for (int i = 0; i < dim; i++)
        totalSize *= size[i];
#pragma omp parallel for
    for (int i = 0; i < totalSize; ++i)
        data[i] = (i + 1) % 1000 - 500;
}
