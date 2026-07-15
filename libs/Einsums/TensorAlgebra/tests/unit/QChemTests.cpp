//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// Needed for M_SQRT1_2 on MSVC/clang-cl
#define _USE_MATH_DEFINES

#include <Einsums/LinearAlgebra.hpp>
#include <Einsums/Print.hpp>
#include <Einsums/Profile.hpp>
#include <Einsums/Tensor/BlockTensor.hpp>
#include <Einsums/Tensor/FunctionTensor.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/Tensor/TiledTensor.hpp>
#include <Einsums/TensorAlgebra.hpp>
#include <Einsums/TensorBase/TensorBase.hpp>

#include <cstring>

#include <Einsums/Testing.hpp>

class ScaleFunctionTensor : public einsums::tensor_base::FunctionTensor<double, 4> {
  private:
    einsums::Tensor<double, 1> const *Evals;

  public:
    ScaleFunctionTensor(std::string name, einsums::Tensor<double, 1> const *evals)
        : Evals{evals}, einsums::tensor_base::FunctionTensor<double, 4>(name, evals->dim(0), evals->dim(0), evals->dim(0), evals->dim(0)) {}

    virtual double call(std::array<ptrdiff_t, 4> const &inds) const override {
        return 1.0 / ((*Evals)(inds[0]) + (*Evals)(inds[2]) - (*Evals)(inds[1]) - (*Evals)(inds[3]));
    }

    size_t dim(int d) const override { return einsums::tensor_base::FunctionTensor<double, 4>::dim(d); }
};

template <size_t Rank>
static void read_tensor(std::string fname, einsums::Tensor<double, Rank> *out) {
    std::FILE *input = std::fopen(fname.c_str(), "r");

    char buffer[1024] = {0};
    int  line_num     = 0;

    while (!std::feof(input)) {
        line_num++;
        std::memset(buffer, 0, 1024);
        std::fgets(buffer, 1023, input);
        std::array<int, Rank> indices;

        char *next = std::strtok(buffer, " \t");

        if (next == NULL) {
            continue;
        }

        indices[0] = std::atoi(next) - 1;

        for (int i = 1; i < Rank; i++) {
            next = std::strtok(NULL, " \t");

            if (next == NULL) {
                EINSUMS_THROW_EXCEPTION(std::runtime_error, "Line {} in file {} not formatted correctly!", line_num, fname);
            }

            indices[i] = std::atoi(next) - 1;
        }

        next = std::strtok(NULL, " \t");

        if (next == NULL) {
            EINSUMS_THROW_EXCEPTION(std::runtime_error, "Line {} in file {} not formatted correctly!", line_num, fname);
        }

        if constexpr (Rank == 2) {
            double val                = std::atof(next);
            std::apply(*out, indices) = val;
            std::swap(indices[0], indices[1]);
            std::apply(*out, indices) = val;
        } else if constexpr (Rank == 4) {
            double val = std::atof(next);

            int i = indices[0], j = indices[1], k = indices[2], l = indices[3];

            (*out)(i, j, k, l) = val;
            (*out)(i, j, l, k) = val;
            (*out)(j, i, k, l) = val;
            (*out)(j, i, l, k) = val;
            (*out)(k, l, i, j) = val;
            (*out)(l, k, i, j) = val;
            (*out)(k, l, j, i) = val;
            (*out)(l, k, j, i) = val;
        }
    }

    std::fclose(input);
}

static void update_Cocc(einsums::Tensor<double, 1> const &energies, einsums::BlockTensor<double, 2> *Cocc,
                        einsums::BlockTensor<double, 2> const &C, std::array<int, 4> &occ_per_irrep) {
    // Update occupation.

    std::array<int, 4> irrep_sizes{4, 0, 1, 2};

    for (int i = 0; i < 4; i++) {
        occ_per_irrep[i] = 0;
    }

    for (int i = 0; i < 5; i++) {
        double curr_min  = INFINITY;
        int    irrep_occ = -1;

        for (int j = 0; j < 4; j++) {
            if (occ_per_irrep[j] >= irrep_sizes[j]) {
                continue;
            }

            double energy = energies(C.block_range(j)[0] + occ_per_irrep[j]);

            if (energy < curr_min) {
                curr_min  = energy;
                irrep_occ = j;
            }
        }

        occ_per_irrep[irrep_occ]++;
    }

    (*Cocc).zero();
    // #pragma omp parallel for
    for (int i = 0; i < 4; i++) {
        if (occ_per_irrep[i] == 0) {
            continue;
        }
        einsums::TensorView<double, 2>       view1 = (*Cocc)[i](einsums::All, einsums::Range{0, occ_per_irrep[i]});
        einsums::TensorView<double, 2> const view2 = C[i](einsums::All, einsums::Range{0, occ_per_irrep[i]});
        view1                                      = view2;
    }
}

static void compute_diis_coefs(std::vector<einsums::BlockTensor<double, 2>> const &errors, std::vector<double> *out) {
    einsums::Tensor<double, 2> *B_mat = new einsums::Tensor<double, 2>("DIIS error matrix", errors.size() + 1, errors.size() + 1);

    B_mat->zero();
    (*B_mat)(einsums::Range{errors.size(), errors.size() + 1}, einsums::Range{0, errors.size()}) = 1.0;
    (*B_mat)(einsums::Range{0, errors.size()}, einsums::Range{errors.size(), errors.size() + 1}) = 1.0;

#pragma omp parallel for
    for (int i = 0; i < errors.size(); i++) {
#pragma omp parallel for
        for (int j = 0; j <= i; j++) {
            (*B_mat)(i, j) = einsums::linear_algebra::dot(errors[i], errors[j]);
            (*B_mat)(j, i) = (*B_mat)(i, j);
        }
    }

    einsums::Tensor<double, 1> res_mat("DIIS result matrix", errors.size() + 1);

    res_mat.zero();
    res_mat(errors.size()) = 1.0;

    auto info = einsums::linear_algebra::gesv(B_mat, &res_mat);

    if (info < 0) {
        EINSUMS_THROW_EXCEPTION(std::runtime_error, "The {} parameter to gesv was invalid!", einsums::print::ordinal(-info));
    } else if (info > 0) {
        EINSUMS_THROW_EXCEPTION(std::runtime_error, "The {} diagonal element went to zero when solving the system!",
                                einsums::print::ordinal(info));
    }

    out->resize(errors.size());

    for (int i = 0; i < errors.size(); i++) {
        out->at(i) = res_mat(i);
    }

    delete B_mat;
}

static void compute_diis_fock(std::vector<double> const &coefs, std::vector<einsums::BlockTensor<double, 2>> const &focks,
                              einsums::BlockTensor<double, 2> *out) {

    out->zero();

    for (int i = 0; i < coefs.size(); i++) {
        einsums::linear_algebra::axpy(coefs[i], focks[i], out);
    }
}

TEST_CASE("RHF No symmetry", "[qchem]") {
    using namespace einsums;
    using namespace einsums::tensor_algebra;
    using namespace einsums::linear_algebra;

    Tensor<double, 2> S("Overlap", 7, 7), T("Kinetic Energy", 7, 7), V("Potential Energy", 7, 7), H("Core Hamiltonian", 7, 7),
        temp1("temp1", 7, 7), temp2("temp2", 7, 7), X("Unitary transform", 7, 7), C("MO Coefs", 7, 7), Ct("Transformed MO Coefs", 7, 7),
        D("Density Matrix", 7, 7), D_prev("Previous Density Matrix", 7, 7), F("Fock Matrix", 7, 7), Ft("Transformed Fock Matrix", 7, 7);

    Tensor<double, 4> TEI("Two-electron Integrals", 7, 7, 7, 7), MP2_temp1("MP2_temp1", 7, 7, 7, 7), MP2_temp2("MP2_temp2", 7, 7, 7, 7);

    Tensor<double, 1> Evals("Eigenvalues", 7);

    S.zero();
    T.zero();
    V.zero();
    H.zero();
    temp1.zero();
    temp2.zero();
    X.zero();
    C.zero();
    Ct.zero();
    D.zero();
    D_prev.zero();
    F.zero();
    Ft.zero();
    Evals.zero();
    TEI.zero();

    {
        LabeledSection("Reading tensors");
        REQUIRE_NOTHROW(read_tensor("data/water_sto3g/S.dat", &S));
        REQUIRE_NOTHROW(read_tensor("data/water_sto3g/T.dat", &T));
        REQUIRE_NOTHROW(read_tensor("data/water_sto3g/V.dat", &V));
        REQUIRE_NOTHROW(read_tensor("data/water_sto3g/TEI.dat", &TEI));
    }

    // Make sure that the tensors are formatted correctly.
    {
        LabeledSection("Checking tensors");
        for (int i = 0; i < 7; i++) {
            for (int j = 0; j < 7; j++) {
                REQUIRE_THAT(S(i, j), Catch::Matchers::WithinAbs(S(j, i), EINSUMS_ZERO));
                REQUIRE_THAT(T(i, j), Catch::Matchers::WithinAbs(T(j, i), EINSUMS_ZERO));
                REQUIRE_THAT(V(i, j), Catch::Matchers::WithinAbs(V(j, i), EINSUMS_ZERO));
            }
        }

        for (int i = 0; i < 7; i++) {
            for (int j = 0; j < 7; j++) {
                for (int k = 0; k < 7; k++) {
                    for (int l = 0; l < 7; l++) {
                        REQUIRE_THAT(TEI(i, j, l, k), Catch::Matchers::WithinAbs(TEI(i, j, k, l), EINSUMS_ZERO));
                        REQUIRE_THAT(TEI(j, i, k, l), Catch::Matchers::WithinAbs(TEI(i, j, k, l), EINSUMS_ZERO));
                        REQUIRE_THAT(TEI(j, i, l, k), Catch::Matchers::WithinAbs(TEI(i, j, k, l), EINSUMS_ZERO));
                        REQUIRE_THAT(TEI(k, l, i, j), Catch::Matchers::WithinAbs(TEI(i, j, k, l), EINSUMS_ZERO));
                        REQUIRE_THAT(TEI(k, l, j, i), Catch::Matchers::WithinAbs(TEI(i, j, k, l), EINSUMS_ZERO));
                        REQUIRE_THAT(TEI(l, k, i, j), Catch::Matchers::WithinAbs(TEI(i, j, k, l), EINSUMS_ZERO));
                        REQUIRE_THAT(TEI(l, k, j, i), Catch::Matchers::WithinAbs(TEI(i, j, k, l), EINSUMS_ZERO));
                    }
                }
            }
        }
    }

    H = T;
    H += V;

    X = pow(S, -0.5);

    double enuc = 8.002367061810450;

    // Set up initial Fock matrix.
    F = H;

    // Transform.
    REQUIRE_NOTHROW(einsum(Indices{index::i, index::j}, &temp1, Indices{index::i, index::k}, F, Indices{index::k, index::j}, X));
    REQUIRE_NOTHROW(einsum(Indices{index::i, index::j}, &Ft, Indices{index::k, index::j}, temp1, Indices{index::k, index::i}, X));

    // Diagonalize.
    Ct = Ft;
    syev(&Ct, &Evals);

    for (int i = 0; i < 6; i++) {
        REQUIRE(Evals(i) <= Evals(i + 1));
    }

    REQUIRE_NOTHROW(einsum(Indices{index::i, index::j}, &C, Indices{index::k, index::j}, Ct, Indices{index::i, index::k}, X));

    // Form the density matrix.
    auto Cocc = C(All, Range{0, 5});

    REQUIRE_NOTHROW(einsum(Indices{index::i, index::j}, &D, Indices{index::i, index::m}, Cocc, Indices{index::j, index::m}, Cocc));

    // Compute the energy.

    Tensor<double, 0> Elec;

    Elec = 0;

    REQUIRE_NOTHROW(einsum(Indices{}, &Elec, Indices{index::i, index::j}, D, Indices{index::i, index::j}, F));
    REQUIRE_NOTHROW(einsum(1.0, Indices{}, &Elec, 1.0, Indices{index::i, index::j}, D, Indices{index::i, index::j}, H));

    double e0 = (double)Elec + enuc, e1 = 0;

    int cycles = 0;

    double dRMS = 1;

    while (std::abs(e0 - e1) > 1e-10 && dRMS > 1e-6 && cycles < 50) {
        cycles++;
        e1     = e0;
        D_prev = D;
        // Form the new Fock matrix.
        F = H;

        REQUIRE_NOTHROW(einsum(1.0, Indices{index::i, index::j}, &F, 2.0, Indices{index::k, index::l}, D,
                               Indices{index::i, index::j, index::k, index::l}, TEI));
        REQUIRE_NOTHROW(einsum(1.0, Indices{index::i, index::j}, &F, -1.0, Indices{index::k, index::l}, D,
                               Indices{index::i, index::k, index::j, index::l}, TEI));

        // Transform.
        REQUIRE_NOTHROW(einsum(Indices{index::i, index::j}, &temp1, Indices{index::i, index::k}, F, Indices{index::k, index::j}, X));
        REQUIRE_NOTHROW(einsum(Indices{index::i, index::j}, &Ft, Indices{index::k, index::j}, temp1, Indices{index::k, index::i}, X));

        // Diagonalize.
        Ct = Ft;
        syev(&Ct, &Evals);

        REQUIRE_NOTHROW(einsum(Indices{index::i, index::j}, &C, Indices{index::k, index::j}, Ct, Indices{index::k, index::i}, X));

        // Form the density matrix.
        Cocc = C(All, Range{0, 5});

        REQUIRE_NOTHROW(einsum(Indices{index::i, index::j}, &D, Indices{index::i, index::m}, Cocc, Indices{index::j, index::m}, Cocc));

        // Compute the energy.
        Elec = 0;

        REQUIRE_NOTHROW(einsum(Indices{}, &Elec, Indices{index::i, index::j}, D, Indices{index::i, index::j}, F));
        REQUIRE_NOTHROW(einsum(1.0, Indices{}, &Elec, 1.0, Indices{index::i, index::j}, D, Indices{index::i, index::j}, H));

        e0 = Elec + enuc;

        Tensor<double, 0> rms;

        rms   = 0;
        temp1 = D;
        temp1 -= D_prev;

        REQUIRE_NOTHROW(einsum(Indices{}, &rms, Indices{index::i, index::j}, temp1, Indices{index::i, index::j}, temp1));

        dRMS = std::sqrt((double)rms) / 7.0;
    }

    REQUIRE(cycles < 50);
    REQUIRE_THAT(e0, Catch::Matchers::WithinAbs(-74.942079928192, 1e-6));

    // MP2

    // Transform the two-electron integrals.

    REQUIRE_NOTHROW(einsum(Indices{index::p, index::q, index::r, index::s}, &MP2_temp1, Indices{index::m, index::q, index::r, index::s},
                           TEI, Indices{index::m, index::p}, C));
    REQUIRE_NOTHROW(einsum(Indices{index::p, index::q, index::r, index::s}, &MP2_temp2, Indices{index::p, index::m, index::r, index::s},
                           MP2_temp1, Indices{index::m, index::q}, C));
    REQUIRE_NOTHROW(einsum(Indices{index::p, index::q, index::r, index::s}, &MP2_temp1, Indices{index::p, index::q, index::m, index::s},
                           MP2_temp2, Indices{index::m, index::r}, C));
    REQUIRE_NOTHROW(einsum(Indices{index::p, index::q, index::r, index::s}, &TEI, Indices{index::p, index::q, index::r, index::m},
                           MP2_temp1, Indices{index::m, index::s}, C));

    // Set up the scales.
    ScaleFunctionTensor MP2_scale("MP2 scale", &Evals);

    auto MP2_scale_view = MP2_scale(Range{0, 5}, Range{5, 7}, Range{0, 5}, Range{5, 7});
    auto TEI_iajb       = TEI(Range{0, 5}, Range{5, 7}, Range{0, 5}, Range{5, 7});
    auto MP2_amps       = Tensor<double, 4>("MP2 Scaled", 5, 2, 5, 2);
    auto MP2_amps_2     = Tensor<double, 4>("MP2 Scaled", 5, 2, 5, 2);

    Tensor<double, 4> TEI_iajb_tens = TEI_iajb;

    for (int i = 0; i < 5; i++) {
        for (int a = 0; a < 2; a++) {
            for (int j = 0; j < 5; j++) {
                for (int b = 0; b < 2; b++) {
                    REQUIRE_THAT(TEI_iajb(i, a, j, b), Catch::Matchers::WithinRelMatcher(TEI_iajb_tens(i, a, j, b), 1e-6));
                    REQUIRE_THAT(TEI_iajb(i, a, j, b), Catch::Matchers::WithinRelMatcher(TEI(i, a + 5, j, b + 5), 1e-6));
                }
            }
        }
    }

    MP2_amps.zero();

    REQUIRE_NOTHROW(einsum(Indices{index::i, index::a, index::j, index::b}, &MP2_amps, Indices{index::i, index::a, index::j, index::b},
                           TEI_iajb, Indices{index::i, index::a, index::j, index::b}, MP2_scale_view));

    REQUIRE_NOTHROW(einsum(Indices{index::i, index::a, index::j, index::b}, &MP2_amps_2, Indices{index::i, index::a, index::j, index::b},
                           TEI_iajb_tens, Indices{index::i, index::a, index::j, index::b}, MP2_scale_view));

    for (int i = 0; i < 5; i++) {
        for (int a = 0; a < 2; a++) {
            for (int j = 0; j < 5; j++) {
                for (int b = 0; b < 2; b++) {
                    REQUIRE_THAT(MP2_amps(i, a, j, b), Catch::Matchers::WithinRelMatcher(MP2_amps_2(i, a, j, b), 1e-6));
                }
            }
        }
    }

    Tensor<double, 0> EMP2_0, EMP2_1, EMP2_2, EMP2_3;

    REQUIRE_NOTHROW(einsum(0.0, Indices{}, &EMP2_0, 2.0, Indices{index::i, index::a, index::j, index::b}, MP2_amps,
                           Indices{index::i, index::a, index::j, index::b}, TEI_iajb));
    REQUIRE_NOTHROW(einsum(0.0, Indices{}, &EMP2_1, -1.0, Indices{index::i, index::a, index::j, index::b}, MP2_amps,
                           Indices{index::i, index::b, index::j, index::a}, TEI_iajb));

    REQUIRE_NOTHROW(einsum(0.0, Indices{}, &EMP2_2, 2.0, Indices{index::i, index::a, index::j, index::b}, MP2_amps,
                           Indices{index::i, index::a, index::j, index::b}, TEI_iajb_tens));
    REQUIRE_NOTHROW(einsum(0.0, Indices{}, &EMP2_3, -1.0, Indices{index::i, index::a, index::j, index::b}, MP2_amps,
                           Indices{index::i, index::b, index::j, index::a}, TEI_iajb_tens));

    CHECK_THAT((double)EMP2_0, Catch::Matchers::WithinRel((double)EMP2_2, 1e-6));
    CHECK_THAT((double)EMP2_1, Catch::Matchers::WithinRel((double)EMP2_3, 1e-6));

    double eMP2  = (double)EMP2_0 + (double)EMP2_1;
    double e_tot = e0 + eMP2;

    REQUIRE_THAT(eMP2, Catch::Matchers::WithinAbs(-0.049149636120, 1e-6));
    REQUIRE_THAT(e_tot, Catch::Matchers::WithinAbs(-74.991229564312, 1e-6));
}

TEST_CASE("RHF symmetry") {
    using namespace einsums;
    using namespace einsums::tensor_algebra;
    using namespace einsums::linear_algebra;

    Tensor<double, 2> S("Overlap", 7, 7), T("Kinetic Energy", 7, 7), V("Potential Energy", 7, 7), H("Core Hamiltonian", 7, 7),
        Symm("Symmetrizer", 7, 7), symm_temp1("Symmetrizing temp1", 7, 7), symm_temp2("Symmetrizing temp1", 7, 7),
        X("Unitary Transform", 7, 7);
    Tensor<double, 4> TEI("Two-electron integrals", 7, 7, 7, 7), TEI_temp1("Two-electron symmetrize temp1", 7, 7, 7, 7),
        TEI_temp2("Two-electron symmetrize temp2", 7, 7, 7, 7);

    BlockTensor<double, 2> S_sym("Overlap", 4, 0, 1, 2), H_sym("Hamiltonian", 4, 0, 1, 2), D("Density", 4, 0, 1, 2),
        D_prev("Previous density", 4, 0, 1, 2), F("Fock Matrix", 4, 0, 1, 2), Ft("Transformed Fock", 4, 0, 1, 2),
        X_sym("Unitary Transform", 4, 0, 1, 2), C("MO coefs", 4, 0, 1, 2), Cocc("Occupied MO coefs", 4, 0, 1, 2),
        Ct("Transformed MO coefficients", 4, 0, 1, 2), temp1("Temp1", 4, 0, 1, 2), temp2("Temp2", 4, 0, 1, 2), FDS("FDS", 4, 0, 1, 2),
        SDF("SDF", 4, 0, 1, 2);
    std::vector<BlockTensor<double, 2>> DIIS_errors, DIIS_focks;
    std::vector<double>                 DIIS_coefs;

    TiledTensor<double, 4> TEI_sym("Two-electron integrals", std::array{4, 0, 1, 2}), MP2_temp1("MP2 temp1", std::array{4, 0, 1, 2}),
        MP2_temp2("MP2 temp2", std::array{4, 0, 1, 2}),
        MP2_amps("MP2 amplitudes", std::array{3, 0, 1, 1}, std::array{1, 0, 0, 1}, std::array{3, 0, 1, 1}, std::array{1, 0, 0, 1}),
        MP2_amps_den("MP2 amplitudes with denominator", std::array{3, 0, 1, 1}, std::array{1, 0, 0, 1}, std::array{3, 0, 1, 1},
                     std::array{1, 0, 0, 1});

    Tensor<double, 1> Evals("Eigenvalues", 7);

    std::array<int, 4> occ_per_irrep, irrep_sizes{4, 0, 1, 2}, irrep_offs{0, 4, 4, 5};

    int cycles = 0;

    double e0 = -1, e1 = 0;
    double dRMS = 1;

    S.zero();
    T.zero();
    V.zero();
    H.zero();
    Symm.zero();
    symm_temp1.zero();
    symm_temp2.zero();
    X.zero();
    TEI.zero();
    TEI_temp1.zero();
    TEI_temp2.zero();
    S_sym.zero();
    H_sym.zero();
    D.zero();
    D_prev.zero();
    F.zero();
    Ft.zero();
    X_sym.zero();
    C.zero();
    Ct.zero();
    Cocc.zero();
    temp1.zero();
    temp2.zero();
    SDF.zero();
    FDS.zero();
    TEI_sym.zero();
    Evals.zero();
    MP2_temp1.zero();
    MP2_temp2.zero();

    // Read in the values.
    REQUIRE_NOTHROW(read_tensor("data/water_sto3g/S.dat", &S));
    REQUIRE_NOTHROW(read_tensor("data/water_sto3g/T.dat", &T));
    REQUIRE_NOTHROW(read_tensor("data/water_sto3g/V.dat", &V));
    REQUIRE_NOTHROW(read_tensor("data/water_sto3g/TEI.dat", &TEI));

    // Make sure that the tensors are formatted correctly.
    for (int i = 0; i < 7; i++) {
        for (int j = 0; j < 7; j++) {
            REQUIRE_THAT(S(i, j), Catch::Matchers::WithinAbs(S(j, i), EINSUMS_ZERO));
            REQUIRE_THAT(T(i, j), Catch::Matchers::WithinAbs(T(j, i), EINSUMS_ZERO));
            REQUIRE_THAT(V(i, j), Catch::Matchers::WithinAbs(V(j, i), EINSUMS_ZERO));
        }
    }

    for (int i = 0; i < 7; i++) {
        for (int j = 0; j < 7; j++) {
            for (int k = 0; k < 7; k++) {
                for (int l = 0; l < 7; l++) {
                    REQUIRE_THAT(TEI(i, j, l, k), Catch::Matchers::WithinAbs(TEI(i, j, k, l), EINSUMS_ZERO));
                    REQUIRE_THAT(TEI(j, i, k, l), Catch::Matchers::WithinAbs(TEI(i, j, k, l), EINSUMS_ZERO));
                    REQUIRE_THAT(TEI(j, i, l, k), Catch::Matchers::WithinAbs(TEI(i, j, k, l), EINSUMS_ZERO));
                    REQUIRE_THAT(TEI(k, l, i, j), Catch::Matchers::WithinAbs(TEI(i, j, k, l), EINSUMS_ZERO));
                    REQUIRE_THAT(TEI(k, l, j, i), Catch::Matchers::WithinAbs(TEI(i, j, k, l), EINSUMS_ZERO));
                    REQUIRE_THAT(TEI(l, k, i, j), Catch::Matchers::WithinAbs(TEI(i, j, k, l), EINSUMS_ZERO));
                    REQUIRE_THAT(TEI(l, k, j, i), Catch::Matchers::WithinAbs(TEI(i, j, k, l), EINSUMS_ZERO));
                }
            }
        }
    }

    double enuc = 8.002367061810450;

    // Matrix for forming salcs.
    std::vector<double> symm_values{1, 0, 0, 0, 0,         0, 0, 0,          1, 0, 0, 0,         0, 0, 0,        0, 0,
                                    0, 0, 1, 0, 0,         0, 1, 0,          0, 0, 0, 0,         0, 0, 0,        1, 0,
                                    0, 0, 0, 0, M_SQRT1_2, 0, 0, -M_SQRT1_2, 0, 0, 0, M_SQRT1_2, 0, 0, M_SQRT1_2};

    for (int i = 0; i < 7; i++) {
        for (int j = 0; j < 7; j++) {
            Symm(i, j) = symm_values[i * 7 + j];
        }
    }

    // Compute the Hamiltonian.
    H = T;
    H += V;
    X = pow(S, -0.5);

    // Symmetrize
    REQUIRE_NOTHROW(einsum(Indices{index::i, index::j}, &symm_temp1, Indices{index::i, index::k}, S, Indices{index::k, index::j}, Symm));
    REQUIRE_NOTHROW(
        einsum(Indices{index::i, index::j}, &symm_temp2, Indices{index::k, index::i}, Symm, Indices{index::k, index::j}, symm_temp1));

    S_sym[0]       = symm_temp2(Range{0, 4}, Range{0, 4});
    S_sym[2](0, 0) = symm_temp2(4, 4);
    S_sym[3]       = symm_temp2(Range{5, 7}, Range{5, 7});

    for (int i = 0; i < 7; i++) {
        for (int j = 0; j < 7; j++) {
            CHECK_THAT(S_sym(i, j), Catch::Matchers::WithinAbs(symm_temp2(i, j), EINSUMS_ZERO));
        }
    }

    REQUIRE_NOTHROW(einsum(Indices{index::i, index::j}, &symm_temp1, Indices{index::i, index::k}, H, Indices{index::k, index::j}, Symm));
    REQUIRE_NOTHROW(
        einsum(Indices{index::i, index::j}, &symm_temp2, Indices{index::k, index::i}, Symm, Indices{index::k, index::j}, symm_temp1));

    H_sym[0]       = symm_temp2(Range{0, 4}, Range{0, 4});
    H_sym[2](0, 0) = symm_temp2(4, 4);
    H_sym[3]       = symm_temp2(Range{5, 7}, Range{5, 7});

    REQUIRE_NOTHROW(einsum(Indices{index::i, index::j}, &symm_temp1, Indices{index::i, index::k}, X, Indices{index::k, index::j}, Symm));
    REQUIRE_NOTHROW(
        einsum(Indices{index::i, index::j}, &symm_temp2, Indices{index::k, index::i}, symm_temp1, Indices{index::k, index::j}, Symm));

    X_sym[0]       = symm_temp2(Range{0, 4}, Range{0, 4});
    X_sym[2](0, 0) = symm_temp2(4, 4);
    X_sym[3]       = symm_temp2(Range{5, 7}, Range{5, 7});

    for (int i = 0; i < 7; i++) {
        for (int j = 0; j < 7; j++) {
            CHECK_THAT(X_sym(i, j), Catch::Matchers::WithinAbs(symm_temp2(i, j), EINSUMS_ZERO));
        }
    }

    REQUIRE_NOTHROW(einsum(Indices{index::i, index::j, index::k, index::l}, &TEI_temp1, Indices{index::m, index::j, index::k, index::l},
                           TEI, Indices{index::m, index::i}, Symm));
    REQUIRE_NOTHROW(einsum(Indices{index::i, index::j, index::k, index::l}, &TEI_temp2, Indices{index::i, index::m, index::k, index::l},
                           TEI_temp1, Indices{index::m, index::j}, Symm));
    REQUIRE_NOTHROW(einsum(Indices{index::i, index::j, index::k, index::l}, &TEI_temp1, Indices{index::i, index::j, index::m, index::l},
                           TEI_temp2, Indices{index::m, index::k}, Symm));
    REQUIRE_NOTHROW(einsum(Indices{index::i, index::j, index::k, index::l}, &TEI_temp2, Indices{index::i, index::j, index::k, index::m},
                           TEI_temp1, Indices{index::m, index::l}, Symm));

    TEI_sym.tile(0, 0, 0, 0) = TEI_temp2(Range{0, 4}, Range{0, 4}, Range{0, 4}, Range{0, 4});
    TEI_sym.tile(2, 2, 2, 2) = TEI_temp2(4, 4, 4, 4);
    TEI_sym.tile(3, 3, 3, 3) = TEI_temp2(Range{5, 7}, Range{5, 7}, Range{5, 7}, Range{5, 7});

    TEI_sym.tile(0, 0, 2, 2) = TEI_temp2(Range{0, 4}, Range{0, 4}, Range{4, 5}, Range{4, 5});
    TEI_sym.tile(0, 2, 0, 2) = TEI_temp2(Range{0, 4}, Range{4, 5}, Range{0, 4}, Range{4, 5});
    TEI_sym.tile(0, 2, 2, 0) = TEI_temp2(Range{0, 4}, Range{4, 5}, Range{4, 5}, Range{0, 4});
    TEI_sym.tile(2, 0, 0, 2) = TEI_temp2(Range{4, 5}, Range{0, 4}, Range{0, 4}, Range{4, 5});
    TEI_sym.tile(2, 0, 2, 0) = TEI_temp2(Range{4, 5}, Range{0, 4}, Range{4, 5}, Range{0, 4});
    TEI_sym.tile(2, 2, 0, 0) = TEI_temp2(Range{4, 5}, Range{4, 5}, Range{0, 4}, Range{0, 4});

    TEI_sym.tile(0, 0, 3, 3) = TEI_temp2(Range{0, 4}, Range{0, 4}, Range{5, 7}, Range{5, 7});
    TEI_sym.tile(0, 3, 0, 3) = TEI_temp2(Range{0, 4}, Range{5, 7}, Range{0, 4}, Range{5, 7});
    TEI_sym.tile(0, 3, 3, 0) = TEI_temp2(Range{0, 4}, Range{5, 7}, Range{5, 7}, Range{0, 4});
    TEI_sym.tile(3, 0, 0, 3) = TEI_temp2(Range{5, 7}, Range{0, 4}, Range{0, 4}, Range{5, 7});
    TEI_sym.tile(3, 0, 3, 0) = TEI_temp2(Range{5, 7}, Range{0, 4}, Range{5, 7}, Range{0, 4});
    TEI_sym.tile(3, 3, 0, 0) = TEI_temp2(Range{5, 7}, Range{5, 7}, Range{0, 4}, Range{0, 4});

    TEI_sym.tile(2, 2, 3, 3) = TEI_temp2(Range{4, 5}, Range{4, 5}, Range{5, 7}, Range{5, 7});
    TEI_sym.tile(2, 3, 2, 3) = TEI_temp2(Range{4, 5}, Range{5, 7}, Range{4, 5}, Range{5, 7});
    TEI_sym.tile(2, 3, 3, 2) = TEI_temp2(Range{4, 5}, Range{5, 7}, Range{5, 7}, Range{4, 5});
    TEI_sym.tile(3, 2, 2, 3) = TEI_temp2(Range{5, 7}, Range{4, 5}, Range{4, 5}, Range{5, 7});
    TEI_sym.tile(3, 2, 3, 2) = TEI_temp2(Range{5, 7}, Range{4, 5}, Range{5, 7}, Range{4, 5});
    TEI_sym.tile(3, 3, 2, 2) = TEI_temp2(Range{5, 7}, Range{5, 7}, Range{4, 5}, Range{4, 5});

    auto const &TEI_sym_ref = *&TEI_sym;

    for (int i = 0; i < 7; i++) {
        for (int j = 0; j < 7; j++) {
            for (int k = 0; k < 7; k++) {
                for (int l = 0; l < 7; l++) {
                    CHECK_THAT(TEI_sym_ref(i, j, k, l), Catch::Matchers::WithinAbs(TEI_temp2(i, j, k, l), EINSUMS_ZERO));
                }
            }
        }
    }

    // Compute the unitary transform.
    temp1 = einsums::linear_algebra::pow(S_sym, -0.5);

    for (int i = 0; i < 7; i++) {
        for (int j = 0; j < 7; j++) {
            CHECK_THAT(temp1(i, j), Catch::Matchers::WithinAbs(X_sym(i, j), EINSUMS_ZERO));
        }
    }

    // Compute the guess Fock matrix.
    F = H_sym;

    // Transform.
    REQUIRE_NOTHROW(einsum(Indices{index::i, index::j}, &temp1, Indices{index::i, index::k}, F, Indices{index::j, index::k}, X_sym));
    REQUIRE_NOTHROW(einsum(Indices{index::i, index::j}, &Ft, Indices{index::k, index::j}, temp1, Indices{index::i, index::k}, X_sym));

    // Compute the coefficients.
    Ct = Ft;
    syev(&Ct, &Evals);

    // Transform back.
    REQUIRE_NOTHROW(einsum(Indices{index::i, index::j}, &C, Indices{index::k, index::j}, Ct, Indices{index::i, index::k}, X_sym));

    // Compute the occupied orbitals.
    update_Cocc(Evals, &Cocc, C, occ_per_irrep);

    // Compute the density matrix.
    einsum(Indices{index::i, index::j}, &D, Indices{index::i, index::m}, Cocc, Indices{index::j, index::m}, Cocc);

    e0 = enuc;

    Tensor<double, 0> Elec;

    REQUIRE_NOTHROW(einsum(Indices{}, &Elec, Indices{index::i, index::j}, F, Indices{index::i, index::j}, D));
    REQUIRE_NOTHROW(einsum(1.0, Indices{}, &Elec, 1.0, Indices{index::i, index::j}, H_sym, Indices{index::i, index::j}, D));

    e0 += (double)Elec;

    while (std::fabs(e0 - e1) > 1e-10 && cycles < 50) {
        D_prev = D;
        e1     = e0;

        // Compute the new Fock matrix.
        F = H_sym;

        REQUIRE_NOTHROW(einsum(1.0, Indices{index::i, index::j}, &F, 2.0, Indices{index::i, index::j, index::k, index::l}, TEI_sym,
                               Indices{index::k, index::l}, D));
        REQUIRE_NOTHROW(einsum(1.0, Indices{index::i, index::j}, &F, -1.0, Indices{index::i, index::k, index::j, index::l}, TEI_sym,
                               Indices{index::k, index::l}, D));

        // Transform.
        REQUIRE_NOTHROW(einsum(Indices{index::i, index::j}, &temp1, Indices{index::i, index::k}, F, Indices{index::k, index::j}, X_sym));
        REQUIRE_NOTHROW(einsum(Indices{index::i, index::j}, &Ft, Indices{index::k, index::j}, temp1, Indices{index::k, index::i}, X_sym));

        // Compute the coefficients.
        Ct = Ft;
        syev(&Ct, &Evals);

        // Transform back.
        REQUIRE_NOTHROW(einsum(Indices{index::i, index::j}, &C, Indices{index::k, index::j}, Ct, Indices{index::i, index::k}, X_sym));

        // Compute the occupied orbitals.
        update_Cocc(Evals, &Cocc, C, occ_per_irrep);

        // Compute the density matrix.
        einsum(Indices{index::i, index::j}, &D, Indices{index::i, index::m}, Cocc, Indices{index::j, index::m}, Cocc);

        // Compute the new energy.
        e0 = enuc;

        REQUIRE_NOTHROW(einsum(Indices{}, &Elec, Indices{index::i, index::j}, F, Indices{index::i, index::j}, D));
        REQUIRE_NOTHROW(einsum(1.0, Indices{}, &Elec, 1.0, Indices{index::i, index::j}, H_sym, Indices{index::i, index::j}, D));

        e0 += (double)Elec;

        Tensor<double, 0> rms;

        temp1 = D;
        temp1 -= D_prev;

        REQUIRE_NOTHROW(einsum(0.0, Indices{}, &rms, 1.0 / 49.0, Indices{index::i, index::j}, temp1, Indices{index::i, index::j}, temp2));

        dRMS = std::sqrt((double)rms);

        REQUIRE_NOTHROW(einsum(Indices{index::i, index::j}, &temp1, Indices{index::i, index::k}, S_sym, Indices{index::k, index::j}, D));
        REQUIRE_NOTHROW(einsum(Indices{index::i, index::j}, &SDF, Indices{index::i, index::k}, temp1, Indices{index::k, index::j}, F));
        REQUIRE_NOTHROW(einsum(Indices{index::i, index::j}, &temp1, Indices{index::i, index::k}, F, Indices{index::k, index::j}, D));
        REQUIRE_NOTHROW(einsum(Indices{index::i, index::j}, &FDS, Indices{index::i, index::k}, temp1, Indices{index::k, index::j}, S_sym));

        temp1 = FDS;
        temp1 -= SDF;

        if (DIIS_errors.size() == 6) {
            double max_error = -INFINITY;
            int    max_index = -1;

            Tensor<double, 0> dot_temp;

            for (int i = 0; i < DIIS_errors.size(); i++) {
                einsum(Indices{}, &dot_temp, Indices{index::i, index::j}, temp1, Indices{index::i, index::j}, DIIS_errors[i]);

                if ((double)dot_temp > max_error) {
                    max_error = (double)dot_temp;
                    max_index = i;
                }
            }

            DIIS_focks[max_index]  = F;
            DIIS_errors[max_index] = temp1;
        } else {
            DIIS_errors.push_back(temp1);
            DIIS_focks.push_back(F);
            DIIS_coefs.push_back(0.0);
        }

        compute_diis_coefs(DIIS_errors, &DIIS_coefs);
        compute_diis_fock(DIIS_coefs, DIIS_focks, &F);

        cycles++;
    }

    REQUIRE(cycles < 50);
    REQUIRE_THAT(e0, Catch::Matchers::WithinAbs(-74.942079928192, 1e-6));

    BlockTensor<double, 2> C2;

    C2 = C;

    for (int i = 0; i < 7; i++) {
        for (int j = 0; j < 7; j++) {
            REQUIRE_THAT(C2(i, j), Catch::Matchers::WithinAbs(C(i, j), 1e-6));
        }
    }

    // MP2 now.
    // Compute the new two electron integrals.
    MP2_temp1.zero();
    MP2_temp2.zero();
    MP2_amps.zero();
    MP2_amps_den.zero();
    REQUIRE_NOTHROW(einsum(Indices{index::i, index::j, index::k, index::l}, &MP2_temp1, Indices{index::m, index::j, index::k, index::l},
                           TEI_sym, Indices{index::m, index::i}, C));
    REQUIRE_NOTHROW(einsum(Indices{index::i, index::j, index::k, index::l}, &MP2_temp2, Indices{index::i, index::m, index::k, index::l},
                           MP2_temp1, Indices{index::m, index::j}, C));
    REQUIRE_NOTHROW(einsum(Indices{index::i, index::j, index::k, index::l}, &MP2_temp1, Indices{index::i, index::j, index::m, index::l},
                           MP2_temp2, Indices{index::m, index::k}, C));
    REQUIRE_NOTHROW(einsum(Indices{index::i, index::j, index::k, index::l}, &MP2_temp2, Indices{index::i, index::j, index::k, index::m},
                           MP2_temp1, Indices{index::m, index::l}, C));

    // Create the new tensor.
    ScaleFunctionTensor MP2_factors("MP2 factors", &Evals);

    // Create the view.

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            for (int k = 0; k < 4; k++) {
                for (int l = 0; l < 4; l++) {
                    if (!MP2_temp2.has_tile(i, j, k, l) || MP2_temp2.has_zero_size(i, j, k, l) || MP2_amps.has_zero_size(i, j, k, l)) {
                        continue;
                    }

                    auto &out_tile     = MP2_amps.tile(i, j, k, l);
                    auto &out_tile_den = MP2_amps_den.tile(i, j, k, l);
                    auto &in_tile      = MP2_temp2.tile(i, j, k, l);

                    for (int I = 0; I < occ_per_irrep[i]; I++) {
                        for (int A = occ_per_irrep[j]; A < irrep_sizes[j]; A++) {
                            for (int J = 0; J < occ_per_irrep[k]; J++) {
                                for (int B = occ_per_irrep[l]; B < irrep_sizes[l]; B++) {
                                    double den = Evals(I + irrep_offs[i]) + Evals(J + irrep_offs[k]) - Evals(A + irrep_offs[j]) -
                                                 Evals(B + irrep_offs[l]);
                                    out_tile_den(I, A - occ_per_irrep[j], J, B - occ_per_irrep[l]) = in_tile(I, A, J, B) / den;
                                    out_tile(I, A - occ_per_irrep[j], J, B - occ_per_irrep[l])     = in_tile(I, A, J, B);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Compute the MP2 energy.
    Tensor<double, 0> EMP2_0, EMP2_1, EMP2_2, EMP2_3;
    Tensor<double, 4> MP2_amps_den_tens = MP2_amps_den, MP2_amps_tens = MP2_amps;

    REQUIRE_NOTHROW(einsum(0.0, Indices{}, &EMP2_0, 2.0, Indices{index::i, index::a, index::j, index::b}, MP2_amps_den,
                           Indices{index::i, index::a, index::j, index::b}, MP2_amps));
    REQUIRE_NOTHROW(einsum(1.0, Indices{}, &EMP2_1, -1.0, Indices{index::i, index::a, index::j, index::b}, MP2_amps_den,
                           Indices{index::i, index::b, index::j, index::a}, MP2_amps));

    REQUIRE_NOTHROW(einsum(0.0, Indices{}, &EMP2_2, 2.0, Indices{index::i, index::a, index::j, index::b}, MP2_amps_den_tens,
                           Indices{index::i, index::a, index::j, index::b}, MP2_amps_tens));
    REQUIRE_NOTHROW(einsum(1.0, Indices{}, &EMP2_3, -1.0, Indices{index::i, index::a, index::j, index::b}, MP2_amps_den_tens,
                           Indices{index::i, index::b, index::j, index::a}, MP2_amps_tens));

    CHECK_THAT((double)EMP2_0, Catch::Matchers::WithinRel((double)EMP2_2, 1e-6));
    REQUIRE_THAT((double)EMP2_1, Catch::Matchers::WithinRel((double)EMP2_3, 1e-6));

    double eMP2  = (double)EMP2_0 + (double)EMP2_1;
    double e_tot = e0 + eMP2;

    REQUIRE_THAT(eMP2, Catch::Matchers::WithinAbs(-0.049149636120, 1e-6));
    REQUIRE_THAT(e_tot, Catch::Matchers::WithinAbs(-74.991229564312, 1e-6));
}
