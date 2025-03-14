#include "einsums/_Common.hpp"

#include <catch2/catch_all.hpp>

#include "einsums.hpp"

using namespace einsums;

TEST_CASE("Runtime Tensor Assignment") {
    RuntimeTensor<double> A{"A", std::vector<size_t>{10, 10}};
    A = create_random_tensor("A", 10, 10);
    RuntimeTensor<double> C{std::vector<size_t>{10, 10}};
    C                       = create_random_tensor("C", 10, 10);
    RuntimeTensor<double> D = create_random_tensor("D", 20, 20);
    RuntimeTensor<double> E;
    RuntimeTensor<double> F = C;
    E                       = A;
    auto D_view             = D(Range{0, 10}, Range{0, 10});

    REQUIRE(A.rank() == 2);

    Tensor<double, 4> B_base = create_random_tensor("B", 10, 10, 10, 10);

    RuntimeTensor<double> B = (RuntimeTensor<double>)B_base(Range{0, 5}, Range{1, 6}, Range{2, 7}, Range{3, 8});

    REQUIRE(B.rank() == 4);

    REQUIRE(A.data() != nullptr);
    REQUIRE(B.data() != nullptr);
    REQUIRE(C.data() != nullptr);
    REQUIRE(D.data() != nullptr);
    REQUIRE((E.data() != nullptr && E.data() != A.data()));
    REQUIRE((F.data() != nullptr && F.data() != C.data()));

    REQUIRE(A.data(std::array<ptrdiff_t, 2>{-1, 1}) != nullptr);

    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            for (int k = 0; k < 5; k++) {
                for (int l = 0; l < 5; l++) {
                    REQUIRE(B(std::vector<ptrdiff_t>{i, j, k, l}) == B_base(i, j + 1, k + 2, l + 3));
                }
            }
        }
    }

    A = C;

    const auto &C_const = *&C;

    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            REQUIRE(A(i, j) == std::get<double>(C_const(i, j)));
        }
    }

    A                   = D_view;
    const auto &D_const = *&D;

    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            REQUIRE(A(std::array<int, 2>{i, j}) == D_const(std::array<int, 2>{i, j}));
        }
    }

    D_view.zero();

    D_view = 1.0;

    for (int i = 0; i < 20; i++) {
        for (int j = 0; j < 20; j++) {
            if (i < 10 && j < 10) {
                REQUIRE(D(i, j) == 1.0);
            }
        }
    }

    // initializer list constructors
    REQUIRE_NOTHROW(RuntimeTensor<double>("test_tensor", {}));
    REQUIRE_NOTHROW(RuntimeTensor<double>("test_tensor", {3, 4, 5}));
    REQUIRE_NOTHROW(RuntimeTensor<double>({3, 4, 5}));
}

TEST_CASE("Runtime Tensor View Creation") {
    using namespace einsums;

    RuntimeTensor<double>        Base       = create_random_tensor("Base", 10, 10, 10);
    const RuntimeTensor<double> &const_base = Base;

    RuntimeTensorView<double>        base_view{Base};
    const RuntimeTensorView<double> &const_base_view = base_view;

    Tensor<double, 3>        rank_base       = create_random_tensor("rank_base", 10, 10, 10);
    const Tensor<double, 3> &const_rank_base = rank_base;

    TensorView<double, 3>        rank_view       = rank_base(All, All, All);
    const TensorView<double, 3> &const_rank_view = rank_view;

    RuntimeTensorView<double> A{Base, std::vector<size_t>{10, 100}}, B{A, std::vector<size_t>{100, 10}},
        C{Base, std::vector<size_t>{5, 5, 5}, std::vector<size_t>{100, 10, 1}, std::vector<size_t>{1, 2, 3}},
        D{RuntimeTensorView<double>(Base), std::vector<size_t>{5, 5, 5}, std::vector<size_t>{100, 10, 1}, std::vector<size_t>{1, 2, 3}},
        E{rank_view}, F{rank_base};

    const RuntimeTensorView<double> G{const_base, std::vector<size_t>{10, 100}}, H{const_base_view, std::vector<size_t>{100, 10}},
        I{const_rank_view}, J{const_rank_base};

    RuntimeTensorView<double> K = A(All, Range{0, 10});

    const RuntimeTensorView<double> L = G(All, Range{0, 10});

    REQUIRE(A.rank() == 2);
    REQUIRE(B.rank() == 2);
    REQUIRE(C.rank() == 3);
    REQUIRE(D.rank() == 3);
    REQUIRE(E.rank() == 3);
    REQUIRE(F.rank() == 3);
    REQUIRE(G.rank() == 2);
    REQUIRE(H.rank() == 2);
    REQUIRE(I.rank() == 3);
    REQUIRE(J.rank() == 3);

    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            for (int k = 0; k < 10; k++) {
                REQUIRE(A(i, j * 10 + k) == Base(i, j, k));
                REQUIRE(B(i * 10 + j, k) == Base(i, j, k));
                REQUIRE(E(i, j, k) == rank_base(i, j, k));
                REQUIRE(F(i, j, k) == rank_base(i, j, k));
                REQUIRE(std::get<double>(G(i, j * 10 + k)) == Base(i, j, k));
                REQUIRE(std::get<double>(H(i * 10 + j, k)) == Base(i, j, k));
                REQUIRE(std::get<double>(I(i, j, k)) == rank_base(i, j, k));
                REQUIRE(std::get<double>(J(i, j, k)) == rank_base(i, j, k));
            }
        }
    }

    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            for (int k = 0; k < 5; k++) {
                REQUIRE(C(i, j, k) == Base(i + 1, j + 2, k + 3));
                REQUIRE(D(i, j, k) == Base(i + 1, j + 2, k + 3));
            }
        }
    }

    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            REQUIRE(K(i, j) == A(i, j));
            REQUIRE(std::get<double>(L(i, j)) == std::get<double>(G(i, j)));
        }
    }
}

TEST_CASE("Runtime Tensor View Assignment") {
    using namespace einsums;

    RuntimeTensor<double> Base = create_random_tensor("Base", 10, 10, 10);
    RuntimeTensor<double> Base_copy = Base;

    RuntimeTensorView<double> A = Base(Range{5, 10}, Range{5, 10}, Range{5, 10});
    const RuntimeTensorView<double> &const_A = A;

    RuntimeTensor<double> B = create_random_tensor("B", 5, 5, 5);
    const RuntimeTensor<double> &const_B = B;

    RuntimeTensor<double> Base2 = create_random_tensor("Base2", 10, 10, 10);
    RuntimeTensorView<double> C = Base2(Range{0, 5}, Range{0, 5}, Range{0, 5});
    const RuntimeTensorView<double> &const_C = C;

    Tensor<double, 3> D = create_random_tensor("D", 5, 5, 5);
    TensorView<double, 3> E = D(All, All, All);

    const Tensor<double, 3> &const_D = D;
    const TensorView<double, 3> const_E = E;



    A.zero();

    for(int i = 0; i < 10; i++) {
        for(int j = 0; j < 10; j++) {
            for(int k = 0; k < 10; k++) {
                if(i >= 5 && j >= 5 && k >= 5) {
                    REQUIRE(Base(i, j, k) == 0);
                } else {
                    REQUIRE(Base(i, j, k) == Base_copy(i, j, k));
                }
            }
        }
    }

    A = 1.0;

    for(int i = 0; i < 10; i++) {
        for(int j = 0; j < 10; j++) {
            for(int k = 0; k < 10; k++) {
                if(i >= 5 && j >= 5 && k >= 5) {
                    REQUIRE(Base(i, j, k) == 1);
                } else {
                    REQUIRE(Base(i, j, k) == Base_copy(i, j, k));
                }
            }
        }
    }

    A = B;

    for(int i = 0; i < 10; i++) {
        for(int j = 0; j < 10; j++) {
            for(int k = 0; k < 10; k++) {
                if(i >= 5 && j >= 5 && k >= 5) {
                    REQUIRE(Base(i, j, k) == B(i - 5, j - 5, k - 5));
                } else {
                    REQUIRE(Base(i, j, k) == Base_copy(i, j, k));
                }
            }
        }
    }

    A = C;

    for(int i = 0; i < 10; i++) {
        for(int j = 0; j < 10; j++) {
            for(int k = 0; k < 10; k++) {
                if(i >= 5 && j >= 5 && k >= 5) {
                    REQUIRE(Base(i, j, k) == C(i - 5, j - 5, k - 5));
                } else {
                    REQUIRE(Base(i, j, k) == Base_copy(i, j, k));
                }
            }
        }
    }

    A = const_B;

    for(int i = 0; i < 10; i++) {
        for(int j = 0; j < 10; j++) {
            for(int k = 0; k < 10; k++) {
                if(i >= 5 && j >= 5 && k >= 5) {
                    REQUIRE(Base(i, j, k) == B(i - 5, j - 5, k - 5));
                } else {
                    REQUIRE(Base(i, j, k) == Base_copy(i, j, k));
                }
            }
        }
    }

    A = const_C;

    for(int i = 0; i < 10; i++) {
        for(int j = 0; j < 10; j++) {
            for(int k = 0; k < 10; k++) {
                if(i >= 5 && j >= 5 && k >= 5) {
                    REQUIRE(Base(i, j, k) == C(i - 5, j - 5, k - 5));
                } else {
                    REQUIRE(Base(i, j, k) == Base_copy(i, j, k));
                }
            }
        }
    }

    A = D;

    for(int i = 0; i < 10; i++) {
        for(int j = 0; j < 10; j++) {
            for(int k = 0; k < 10; k++) {
                if(i >= 5 && j >= 5 && k >= 5) {
                    REQUIRE(Base(i, j, k) == D(i - 5, j - 5, k - 5));
                } else {
                    REQUIRE(Base(i, j, k) == Base_copy(i, j, k));
                }
            }
        }
    }

    A = E;

    for(int i = 0; i < 10; i++) {
        for(int j = 0; j < 10; j++) {
            for(int k = 0; k < 10; k++) {
                if(i >= 5 && j >= 5 && k >= 5) {
                    REQUIRE(Base(i, j, k) == E(i - 5, j - 5, k - 5));
                } else {
                    REQUIRE(Base(i, j, k) == Base_copy(i, j, k));
                }
            }
        }
    }

    A = const_D;

    for(int i = 0; i < 10; i++) {
        for(int j = 0; j < 10; j++) {
            for(int k = 0; k < 10; k++) {
                if(i >= 5 && j >= 5 && k >= 5) {
                    REQUIRE(Base(i, j, k) == D(i - 5, j - 5, k - 5));
                } else {
                    REQUIRE(Base(i, j, k) == Base_copy(i, j, k));
                }
            }
        }
    }

    A = const_E;

    for(int i = 0; i < 10; i++) {
        for(int j = 0; j < 10; j++) {
            for(int k = 0; k < 10; k++) {
                if(i >= 5 && j >= 5 && k >= 5) {
                    REQUIRE(Base(i, j, k) == E(i - 5, j - 5, k - 5));
                } else {
                    REQUIRE(Base(i, j, k) == Base_copy(i, j, k));
                }
            }
        }
    }
}