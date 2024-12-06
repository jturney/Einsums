//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

/**
 * @author: Paul Springer (springer@aices.rwth-aachen.de)
 */

#include <Einsums/HPTT/Primes.hpp>
#include <Einsums/HPTT/Utils.hpp>

#include <bit>
#include <list>
#include <stdexcept>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <cstdint>

namespace hptt {

template <>
void getPrimeFactors(std::uint8_t n, std::list<std::uint8_t> &primeFactors) {
    primeFactors.clear();

    /*
     * 1 is not a prime and zero is not prime.
     */
    if (n <= 1) {
        return;
    }

    std::uint8_t quotient = n;

    for (size_t index = 0; index < CHAR_PRIMES; index++) {
        std::uint8_t prime = detail::char_primes[index];
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
void getPrimeFactors(std::uint16_t n, std::list<std::uint16_t> &primeFactors) {
    primeFactors.clear();

    /*
     * 1 is not a prime and zero is not prime.
     */
    if (n <= 1) {
        return;
    }

    int quotient = n;

    for (size_t index = 0; index < SHORT_PRIMES; index++) {
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

    int quotient = n;

    for (size_t index = 0; index < INT_PRIMES; index++) {
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

    /*
     * 1 is not a prime and zero is not prime.
     */
    if (n <= 1) {
        return;
    }

    std::int8_t quotient = n;

    for (size_t index = 0; index < CHAR_PRIMES; index++) {
        std::int8_t prime = detail::char_primes[index];
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
void getPrimeFactors(std::int16_t n, std::list<std::int16_t> &primeFactors) {
    primeFactors.clear();

    /*
     * 1 is not a prime and zero is not prime.
     */
    if (n <= 1) {
        return;
    }

    int quotient = n;

    for (size_t index = 0; index < SHORT_PRIMES; index++) {
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

    int quotient = n;

    for (size_t index = 0; index < INT_PRIMES; index++) {
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

void accountForRowMajor(int const *sizeA, int const *outerSizeA, int const *outerSizeB, int const *perm, int *tmpSizeA, int *tmpOuterSizeA,
                        int *tmpOuterSizeB, int *tmpPerm, int const dim, bool const useRowMajor) {
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
