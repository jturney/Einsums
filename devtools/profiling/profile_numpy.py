# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

import numpy as np
import argparse
import timeit
import math

def mean(iterable) :
    return sum(iterable) / len(iterable)

def stdev(iterable, mean) :
    return math.sqrt(sum((x - mean) ** 2 for x in iterable) / (len(iterable) - 1))

def build_J(D, TEI, J, path) :
    np.einsum("ls,mnls->mn", D, TEI, out = J, optimize = path)

def build_K(D, TEI, K, path) :
    np.einsum("ls,mlns->mn", D, TEI, out = K, optimize = path)

def profile_einsums(**kwargs) :
    orbs = kwargs['n']
    trials = kwargs['t']

    print(f"Profiling NumPy: {trials} trials with {orbs} orbitals.")

    D = np.random.random([orbs, orbs])
    TEI = np.random.random([orbs, orbs, orbs, orbs])
    J = np.zeros([orbs, orbs])
    K = np.zeros([orbs, orbs])
    G = np.zeros([orbs, orbs])

    path_J = np.einsum_path("ls,mnls->mn", D, TEI, optimize = "optimal")[0]
    path_K, printable = np.einsum_path("ls,mlns->mn", D, TEI, optimize = "optimal")

    print(printable)

    all_vars = globals().copy()
    all_vars.update(locals())

    times_J = timeit.repeat(stmt = "build_J(D, TEI, J, path_J)", repeat = trials, number = 1, globals = all_vars)
    times_K = timeit.repeat(stmt = "build_K(D, TEI, K, path_K)", repeat = trials, number = 1, globals = all_vars)
    times_G = timeit.repeat(stmt = "G = J + K", repeat = trials, number = 1, globals = all_vars)
    times_total = [tj + tk + tg for tj, tk, tg in zip(times_J, times_K, times_G)]

    mean_J = mean(times_J)
    mean_K = mean(times_K)
    mean_G = mean(times_G)
    mean_total = mean(times_total)


    stdev_J = stdev(times_J, mean_J)
    stdev_K = stdev(times_K, mean_K)
    stdev_G = stdev(times_G, mean_G)
    stdev_total = stdev(times_total, mean_total)

    print(f"Build J: {mean_J} sec, stdev = {stdev_J} sec")
    print(f"Build K: {mean_K} sec, stdev = {stdev_K} sec")
    print(f"Build G: {mean_G} sec, stdev = {stdev_G} sec")
    print(f"Total: {mean_total} sec, stdev = {stdev_total} sec")

def main() :
    parser = argparse.ArgumentParser()

    parser.add_argument("-n", help = "The number of orbitals.", default = 20, type = int)
    parser.add_argument("-t", help = "The number of trials.", default = 20, type = int)

    args = vars(parser.parse_args())

    profile_einsums(**args)

if __name__ == "__main__" :
    main()
