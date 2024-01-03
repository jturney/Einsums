/**
 * @author: Paul Springer (springer@aices.rwth-aachen.de)
 */

#include <list>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

namespace hptt {

void getPrimeFactors(int n, std::list<int> &primeFactors) {
    primeFactors.clear();
    for (int i = 2; i <= n; ++i) {
        while (n % i == 0) {
            primeFactors.push_back(i);
            n /= i;
        }
    }
    if (primeFactors.size() <= 0) {
        fprintf(stderr, "[HPTT] Internal error: primefactorization for %d did not work.\n", n);
        exit(-1);
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

int factorial(int n) {
    if (n == 1)
        return 1;
    return n * factorial(n - 1);
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
