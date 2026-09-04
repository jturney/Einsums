//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/HPTT/Files.hpp>
#include <Einsums/HPTT/HPTTTypes.hpp>
#include <Einsums/LinearAlgebra.hpp>
#include <Einsums/Options.hpp>
#include <Einsums/Options/Options.hpp>
#include <Einsums/Runtime/InitRuntime.hpp>
#include <Einsums/Tensor.hpp>
#include <Einsums/TensorAlgebra.hpp>
#include <Einsums/TensorUtilities.hpp>

#include <filesystem>
#include <vector>

using namespace einsums;
using namespace einsums::tensor_algebra;
using namespace std;

void create_J(Tensor<double, 2> &J, Tensor<double, 2> const &D, Tensor<double, 4> const &TEI) {
    LabeledSection0();
    einsum(0.0, Indices{index::mu, index::nu}, &J, 2.0, Indices{index::mu, index::nu, index::lambda, index::sigma}, TEI,
           Indices{index::lambda, index::sigma}, D);
}

void create_K(Tensor<double, 2> &K, Tensor<double, 2> const &D, Tensor<double, 4> const &TEI) {
    LabeledSection0();
    einsum(0.0, Indices{index::mu, index::nu}, &K, -1.0, Indices{index::mu, index::lambda, index::nu, index::sigma}, TEI,
           Indices{index::lambda, index::sigma}, D);
}

void create_K_sorted(Tensor<double, 2> &K, Tensor<double, 2> const &D, Tensor<double, 4> const &TEI, Tensor<double, 4> &sorted_TEI) {
    LabeledSection0();
    einsum(0.0, Indices{index::mu, index::nu}, &K, -1.0, Indices{index::mu, index::nu, index::lambda, index::sigma}, sorted_TEI,
           Indices{index::lambda, index::sigma}, D);
}

void create_G(Tensor<double, 2> &G, Tensor<double, 2> const &J, Tensor<double, 2> const &K) {
    LabeledSection0();
    G = J;
    G += K;
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

int main(int argc, char **argv) {
#pragma omp parallel
    {
#pragma omp single
        {
            einsums::register_arguments(register_args);

            // Initialize einsums.
            einsums::initialize(argc, argv);

            int  start, end, step, trials;
            bool csv, row_major;

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

                std::vector<double> times_J(trials), times_K(trials), times_G(trials), times_tot(trials), times_sort(trials);

                auto D   = create_random_tensor(row_major, "D", norbs, norbs);
                auto TEI = create_random_tensor(row_major, "TEI", norbs, norbs, norbs, norbs);

                Tensor<double, 2> J{row_major, "J", norbs, norbs}, K{row_major, "K", norbs, norbs}, G{row_major, "G", norbs, norbs};

                double J_mean;
                double K_mean;
                double G_mean;
                double tot_mean;

                // Calculate the times.
                {
                    LabeledSection("Unsorted");
                    for (int i = 0; i < trials; i++) {
                        clock_t const start = clock();

                        create_J(J, D, TEI);

                        clock_t const J_time = clock();

                        create_K(K, D, TEI);

                        clock_t const K_time = clock();

                        create_G(G, J, K);

                        clock_t const G_time = clock();

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
                        printf("%d,%lf,%lf,", norbs, tot_mean, stdev(times_tot, tot_mean));
                    } else {
                        printf("einsums times:\nform J: %lg s, stdev %lg s\nform K: %lg s, stdev %lg s\nform G: %lg s, stdev %lg s\ntotal: "
                               "%lg s, "
                               "stdev %lg s\n",
                               J_mean, stdev(times_J, J_mean), K_mean, stdev(times_K, K_mean), G_mean, stdev(times_G, G_mean), tot_mean,
                               stdev(times_tot, tot_mean));
                    }
                }

                Tensor<double, 4> sorted_TEI{row_major, "sorted TEI", norbs, norbs, norbs, norbs};

                std::shared_ptr<hptt::Transpose<double>> plan;

                auto permute_path = std::filesystem::current_path();

                if (row_major) {
                    permute_path.append(fmt::format("permute_{}.hptt", norbs));
                } else {
                    permute_path.append(fmt::format("permute_{}_column.hptt", norbs));
                }

                if (std::filesystem::exists(permute_path)) {
                    auto fp = std::fopen(permute_path.c_str(), "r");
                    plan    = hptt::Transpose<double>::read_from_file(fp, 1.0, TEI.data(), 0.0, sorted_TEI.data());
                    std::fclose(fp);
                } else {
                    plan = einsums::tensor_algebra::compile_permute(Indices{index::mu, index::nu, index::lambda, index::sigma}, &sorted_TEI,
                                                                    Indices{index::mu, index::lambda, index::nu, index::sigma}, TEI,
                                                                    hptt::PATIENT);
                    auto fp = std::fopen(permute_path.c_str(), "w+");
                    hptt::setup_file(fp);
                    plan->write_to_file(fp);
                    std::fclose(fp);
                }

                // Calculate the linear algebra times.
                {
                    LabeledSection("Sorted");
                    for (int i = 0; i < trials; i++) {
                        clock_t start = clock();

                        create_J(J, D, TEI);

                        clock_t J_time = clock();

                        tensor_algebra::permute(&sorted_TEI, TEI, plan);

                        clock_t K_sort_time = clock();

                        create_K_sorted(K, D, TEI, sorted_TEI);

                        clock_t K_time = clock();

                        create_G(G, J, K);

                        clock_t G_time = clock();

                        times_J[i]    = (J_time - start) / (double)CLOCKS_PER_SEC;
                        times_K[i]    = (K_time - J_time) / (double)CLOCKS_PER_SEC;
                        times_tot[i]  = (G_time - start) / (double)CLOCKS_PER_SEC;
                        times_G[i]    = (G_time - K_time) / (double)CLOCKS_PER_SEC;
                        times_sort[i] = (K_sort_time - J_time) / (double)CLOCKS_PER_SEC;
                    }
                }

                // Print the timing info.
                J_mean           = mean(times_J);
                K_mean           = mean(times_K);
                G_mean           = mean(times_G);
                tot_mean         = mean(times_tot);
                double sort_mean = mean(times_sort);
                if (csv) {
                    printf("%lf,%lf\n", tot_mean, stdev(times_tot, tot_mean));
                } else {
                    printf("sorted times:\nform J: %lg s, stdev %lg s\nform K: %lg s, stdev %lg s\nform G: %lg s, "
                           "stdev %lg "
                           "s\ntotal: %lg s, stdev %lg s\n",
                           J_mean, stdev(times_J, J_mean), K_mean, stdev(times_K, K_mean), G_mean, stdev(times_G, G_mean), tot_mean,
                           stdev(times_tot, tot_mean));
                }
            }

            einsums::finalize();
        }
    }

    return 0;
}