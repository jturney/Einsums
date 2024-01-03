/**
 * @author: Paul Springer (springer@aices.rwth-aachen.de)
 */

#pragma once

#include <iostream>
#include <list>
#include <vector>

#include "hptt_types.hpp"

namespace hptt {

template <typename floatType>
static floatType conj(floatType x) {
    return std::conj(x);
}
template <>
float conj(float x) {
    return x;
}
template <>
double conj(double x) {
    return x;
}

template <typename floatType>
static double getZeroThreshold();
template <>
double getZeroThreshold<double>() {
    return 1e-16;
}
template <>
double getZeroThreshold<DoubleComplex>() {
    return 1e-16;
}
template <>
double getZeroThreshold<float>() {
    return 1e-6;
}
template <>
double getZeroThreshold<FloatComplex>() {
    return 1e-6;
}

void trashCache(double *A, double *B, int n);

template <typename t>
int hasItem(std::vector<t> const &vec, t value) {
    return (std::find(vec.begin(), vec.end(), value) != vec.end());
}

template <typename t>
void printVector(std::vector<t> const &vec, char const *label) {
    std::cout << label << ": ";
    for (auto a : vec)
        std::cout << a << ", ";
    std::cout << "\n";
}

template <typename t>
void printVector(std::list<t> const &vec, char const *label) {
    std::cout << label << ": ";
    for (auto a : vec)
        std::cout << a << ", ";
    std::cout << "\n";
}

void getPrimeFactors(int n, std::list<int> &primeFactors);

template <typename t>
int findPos(t value, std::vector<t> const &array) {
    for (int i = 0; i < array.size(); ++i)
        if (array[i] == value)
            return i;
    return -1;
}

int findPos(int value, int const *array, int n);

int factorial(int n);

void accountForRowMajor(int const *sizeA, int const *outerSizeA, int const *outerSizeB, int const *perm, int *tmpSizeA, int *tmpOuterSizeA,
                        int *tmpouterSizeB, int *tmpPerm, int const dim, bool const useRowMajor);
} // namespace hptt
