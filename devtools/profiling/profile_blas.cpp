//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/BLAS.hpp>
#include <Einsums/Options.hpp>
#include <Einsums/Profile/Profile.hpp>
#include <Einsums/Runtime.hpp>

#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace std;
using namespace einsums;
using namespace einsums::blas;

void create_J(double *J, double const *D, double const *TEI, size_t norbs) {
    gemv('N', norbs * norbs, norbs * norbs, 2.0, TEI, norbs * norbs, D, 1, 0.0, J, 1);
}

void create_K(double *K, double const *D, double const *TEI, size_t norbs) {
    for (size_t i = 0; i < norbs; i++) {
        for (size_t j = 0; j < norbs; j++) {
            K[i * norbs + j] = 0.0;
            for (size_t k = 0; k < norbs; k++) {
                for (size_t l = 0; l < norbs; l++) {
                    K[i * norbs + j] -= D[k * norbs + l] * TEI[norbs * (norbs * (norbs * i + k) + j) + l];
                }
            }
        }
    }
}

void create_K_sorted(double *K, double const *D, double const *TEI, double *sorted_TEI, size_t norbs) {
    for (size_t i = 0; i < norbs; i++) {
        for (size_t j = 0; j < norbs; j++) {
            for (size_t k = 0; k < norbs; k++) {
                for (size_t l = 0; l < norbs; l++) {
                    sorted_TEI[norbs * (norbs * (norbs * i + j) + k) + l] = TEI[norbs * (norbs * (norbs * i + k) + j) + l];
                }
            }
        }
    }

    gemv('N', norbs * norbs, norbs * norbs, -1.0, sorted_TEI, norbs * norbs, D, 1, 0.0, K, 1);
}

void create_G(double *G, double const *J, double const *K, size_t norbs) {
    memcpy((void *)G, (void const *)J, norbs * norbs * sizeof(double));
    axpy(norbs * norbs, 1.0, K, 1, G, 1);
}

double mean(std::vector<double> const &values) {
    double out        = 0;
    size_t num_values = values.size();

    for (size_t i = 0; i < num_values; i++) {
        out += values[i];
    }

    return out / num_values;
}

double variance(std::vector<double> const &values, double mean) {
    double out        = 0;
    size_t num_values = values.size();

    for (size_t i = 0; i < num_values; i++) {
        out += (values[i] - mean) * (values[i] - mean);
    }

    return out / (num_values - 1);
}

double stdev(std::vector<double> const &values, double mean) {
    return sqrt(variance(values, mean));
}

namespace option {

/*
 * One descriptor per option: the name, help, and default in one place, and a
 * typed read at the call site.
 */
inline constinit einsums::cl::ConfigOption<std::int64_t> Start =
    einsums::cl::config_opt<std::int64_t>("start", "The starting number of orbitals for the calculation.", "Profiling", 20);
inline constinit einsums::cl::ConfigOption<std::int64_t> Step =
    einsums::cl::config_opt<std::int64_t>("step", "The step value for the range of orbitals.", "Profiling", 10);
inline constinit einsums::cl::ConfigOption<std::int64_t> End =
    einsums::cl::config_opt<std::int64_t>("end", "The ending number of orbitals for the calculation.", "Profiling", -1);
inline constinit einsums::cl::ConfigOption<std::int64_t> Trials =
    einsums::cl::config_opt<std::int64_t>("trials", "The number of trials for each step in the calculation.", "Profiling", 20);
inline constinit einsums::cl::ConfigOption<bool> ColMajor =
    einsums::cl::config_flag("col-major", "Whether the tensors should be column major or not.", "Profiling", false);
inline constinit einsums::cl::ConfigOption<bool> Csv =
    einsums::cl::config_flag("csv", "When present, output in comma-separated values.", "Profiling", false);

} // namespace option

void register_args() {
    einsums::cl::register_option(option::Start);
    einsums::cl::register_option(option::Step);
    einsums::cl::register_option(option::End);
    einsums::cl::register_option(option::Trials);
    einsums::cl::register_option(option::ColMajor);
    einsums::cl::register_option(option::Csv);
}

template <class Generator>
void fill_random(std::vector<double> &buffer, Generator &generator) {
    LabeledSection0();
    std::uniform_real_distribution random_gen(-1.0, 1.0);
#pragma omp parallel for
    for (size_t i = 0; i < buffer.size(); i++) {
        buffer[i] = random_gen(generator);
    }
}

int         main(int argc, char **argv) {
#pragma omp parallel
    {
#pragma omp single
        {
            einsums::register_arguments(register_args);

            einsums::initialize(argc, argv);
            // Initialize random number generator.
            std::default_random_engine engine(clock());

            int  start, end, step, trials;
            bool csv;

            {
                start     = static_cast<int>(einsums::config::get(option::Start));
                end       = static_cast<int>(einsums::config::get(option::End));
                step      = static_cast<int>(einsums::config::get(option::Step));
                trials    = static_cast<int>(einsums::config::get(option::Trials));
                csv       = einsums::config::get(option::Csv);
                row_major = !einsums::config::get(option::ColMajor);
            }

            if (end < start) {
                end = start;
            }

            for (int norbs = start; norbs <= end; norbs += step) {

                if (!csv) {
                    printf("Running %d trials with %d orbitals.\n", trials, norbs);
                }

                std::vector<double> times_J(trials), times_K(trials), times_G(trials), times_tot(trials);

                std::vector<double> J(norbs * norbs), K(norbs * norbs), G(norbs * norbs), D(norbs * norbs),
                    TEI(norbs * norbs * norbs * norbs);

                fill_random(D, engine);
                fill_random(TEI, engine);

                // Calculate the times.
                for (int i = 0; i < trials; i++) {
                    clock_t start = clock();

                    create_J(J.data(), D.data(), TEI.data(), norbs);

                    clock_t J_time = clock();

                    create_K(K.data(), D.data(), TEI.data(), norbs);

                    clock_t K_time = clock();

                    create_G(G.data(), J.data(), K.data(), norbs);

                    clock_t G_time = clock();

                    times_J[i]   = (J_time - start) / (double)CLOCKS_PER_SEC;
                    times_K[i]   = (K_time - J_time) / (double)CLOCKS_PER_SEC;
                    times_tot[i] = (G_time - start) / (double)CLOCKS_PER_SEC;
                    times_G[i]   = (G_time - K_time) / (double)CLOCKS_PER_SEC;
                }

                // Print the timing info.
                double J_mean   = mean(times_J);
                double K_mean   = mean(times_K);
                double G_mean   = mean(times_G);
                double tot_mean = mean(times_tot);
                if (csv) {
                    printf("%d,%lf,%lf,", norbs, tot_mean, stdev(times_tot, tot_mean));
                } else {
                    printf(
                        "einsums times:\nform J: %lg s, stdev %lg s\nform K: %lg s, stdev %lg s\nform G: %lg s, stdev %lg s\ntotal: %lg s, "
                        "stdev %lg s\n",
                        J_mean, stdev(times_J, J_mean), K_mean, stdev(times_K, K_mean), G_mean, stdev(times_G, G_mean), tot_mean,
                        stdev(times_tot, tot_mean));
                }

                std::vector<double> TEI_sorted(norbs * norbs * norbs * norbs);

                // Calculate the linear algebra times.
                for (int i = 0; i < trials; i++) {
                    clock_t start = clock();

                    create_J(J.data(), D.data(), TEI.data(), norbs);

                    clock_t J_time = clock();

                    create_K_sorted(K.data(), D.data(), TEI.data(), TEI_sorted.data(), norbs);

                    clock_t K_time = clock();

                    create_G(G.data(), J.data(), K.data(), norbs);

                    clock_t G_time = clock();

                    times_J[i]   = (J_time - start) / (double)CLOCKS_PER_SEC;
                    times_K[i]   = (K_time - J_time) / (double)CLOCKS_PER_SEC;
                    times_tot[i] = (G_time - start) / (double)CLOCKS_PER_SEC;
                    times_G[i]   = (G_time - K_time) / (double)CLOCKS_PER_SEC;
                }

                // Print the timing info.
                J_mean   = mean(times_J);
                K_mean   = mean(times_K);
                G_mean   = mean(times_G);
                tot_mean = mean(times_tot);

                if (csv) {
                    printf("%lf,%lf\n", tot_mean, stdev(times_tot, tot_mean));
                } else {
                    printf(
                        "sorted times:\nform J: %lg s, stdev %lg s\nform K: %lg s, stdev %lg s\nform G: %lg s, stdev %lg s\ntotal: %lg s, "
                        "stdev "
                        "%lg s\n",
                        J_mean, stdev(times_J, J_mean), K_mean, stdev(times_K, K_mean), G_mean, stdev(times_G, G_mean), tot_mean,
                        stdev(times_tot, tot_mean));
                }
            }

            einsums::finalize();
        }
    }
    return 0;
}