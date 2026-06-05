//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config/Types.hpp>
#include <Einsums/Tensor/DiskTensor.hpp>
#include <Einsums/Tensor/InitModule.hpp>
#include <Einsums/Tensor/ModuleVars.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>

#include <H5Tpublic.h>
#include <algorithm>
#include <filesystem>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>

#include <catch2/catch_all.hpp>

TEST_CASE("File opening and closing") {
    using namespace einsums;

    auto &singleton = detail::Einsums_Tensor_vars::get_singleton();

    { DiskTensor<double, 2> A("/open-test", 3, 3); }

    H5Fclose(singleton.hdf5_file);

    if (singleton.link_property_list != H5I_INVALID_HID) {
        H5Pclose(singleton.link_property_list);
    }

    if (singleton.double_complex_type != H5I_INVALID_HID) {
        H5Tclose(singleton.double_complex_type);
    }

    if (singleton.float_complex_type != H5I_INVALID_HID) {
        H5Tclose(singleton.float_complex_type);
    }

    auto &global_config = GlobalConfigMap::get_singleton();

    auto fname = std::filesystem::path(global_config.get_string("scratch-dir"));
    fname /= global_config.get_string("hdf5-file-name");

    open_hdf5_file(fname.string());

    REQUIRE(H5Tget_size(singleton.double_complex_type) == sizeof(std::complex<double>));
    REQUIRE(H5Tget_size(singleton.float_complex_type) == sizeof(std::complex<float>));

    REQUIRE_NOTHROW([]() { DiskTensor<double, 2> A("/open-test", 3, 3); }());
}

TEMPLATE_TEST_CASE("disktensor-creation", "[disktensor]", float, double, std::complex<float>, std::complex<double>) {
    using namespace einsums;

    DiskTensor<TestType, 2> A(fmt::format("/A0/{}", type_name<TestType>()), 3, 3);
    DiskTensor<TestType, 2> B(fmt::format("/B0/{}", type_name<TestType>()), 3, 3);
    DiskTensor<TestType, 2> C(fmt::format("/C0/{}", type_name<TestType>()), 3, 3);

    REQUIRE((A.dim(0) == 3 && A.dim(1) == 3));
    REQUIRE((B.dim(0) == 3 && B.dim(1) == 3));
    REQUIRE((C.dim(0) == 3 && C.dim(1) == 3));
}

TEMPLATE_TEST_CASE("Write/Read", "[disktensor]", float, double, std::complex<float>, std::complex<double>) {
    using namespace einsums;

    DiskTensor<TestType, 2> A(fmt::format("/A1/{}", type_name<TestType>()), 3, 3);

    // Data must exist on disk before it can be read in.
    Tensor Ad = create_random_tensor<TestType>("A", 3, 3);

    { A(All, All) = Ad; }

    auto suba = A(0, All);
    // println(suba);
    REQUIRE(suba(0) == Ad(0, 0));
    REQUIRE(suba(1) == Ad(0, 1));
    REQUIRE(suba(2) == Ad(0, 2));

    auto subb = A(1, All);
    // println(subb);
    REQUIRE(subb(0) == Ad(1, 0));
    REQUIRE(subb(1) == Ad(1, 1));
    REQUIRE(subb(2) == Ad(1, 2));

    auto subc = A(All, 1);
    // println(subc);
    REQUIRE(subc(0) == Ad(0, 1));
    REQUIRE(subc(1) == Ad(1, 1));
    REQUIRE(subc(2) == Ad(2, 1));

    auto subd = A(Range{0, 2}, All);
    // println(subd);
    REQUIRE((subd.dim(0) == 2 && subd.dim(1) == 3));
    REQUIRE(subd(0, 0) == Ad(0, 0));
    REQUIRE(subd(0, 1) == Ad(0, 1));
    REQUIRE(subd(0, 2) == Ad(0, 2));
    REQUIRE(subd(1, 0) == Ad(1, 0));
    REQUIRE(subd(1, 1) == Ad(1, 1));
    REQUIRE(subd(1, 2) == Ad(1, 2));

    auto  tempe = A(All, Range{1, 3});
    auto &sube  = tempe.get();
    REQUIRE((sube.dim(0) == 3 && sube.dim(1) == 2));
    REQUIRE(sube(0, 0) == Ad(0, 1));
    REQUIRE(sube(0, 1) == Ad(0, 2));
    REQUIRE(sube(1, 0) == Ad(1, 1));
    REQUIRE(sube(1, 1) == Ad(1, 2));
    REQUIRE(sube(2, 0) == Ad(2, 1));
    REQUIRE(sube(2, 1) == Ad(2, 2));
}

TEMPLATE_TEST_CASE("Write/Read compressed", "[disktensor]", float, double, std::complex<float>, std::complex<double>) {
    using namespace einsums;

    DiskTensor<TestType, 2> A(fmt::format("/A2/{}", type_name<TestType>()), Dim{3, 3}, 9);

    // Data must exist on disk before it can be read in.
    Tensor Ad = create_random_tensor<TestType>("A", 3, 3);

    { A(All, All) = Ad; }

    auto suba = A(0, All);
    // println(suba);
    REQUIRE(suba(0) == Ad(0, 0));
    REQUIRE(suba(1) == Ad(0, 1));
    REQUIRE(suba(2) == Ad(0, 2));

    auto subb = A(1, All);
    // println(subb);
    REQUIRE(subb(0) == Ad(1, 0));
    REQUIRE(subb(1) == Ad(1, 1));
    REQUIRE(subb(2) == Ad(1, 2));

    auto subc = A(All, 1);
    // println(subc);
    REQUIRE(subc(0) == Ad(0, 1));
    REQUIRE(subc(1) == Ad(1, 1));
    REQUIRE(subc(2) == Ad(2, 1));

    auto subd = A(Range{0, 2}, All);
    // println(subd);
    REQUIRE((subd.dim(0) == 2 && subd.dim(1) == 3));
    REQUIRE(subd(0, 0) == Ad(0, 0));
    REQUIRE(subd(0, 1) == Ad(0, 1));
    REQUIRE(subd(0, 2) == Ad(0, 2));
    REQUIRE(subd(1, 0) == Ad(1, 0));
    REQUIRE(subd(1, 1) == Ad(1, 1));
    REQUIRE(subd(1, 2) == Ad(1, 2));

    auto  tempe = A(All, Range{1, 3});
    auto &sube  = tempe.get();
    REQUIRE((sube.dim(0) == 3 && sube.dim(1) == 2));
    REQUIRE(sube(0, 0) == Ad(0, 1));
    REQUIRE(sube(0, 1) == Ad(0, 2));
    REQUIRE(sube(1, 0) == Ad(1, 1));
    REQUIRE(sube(1, 1) == Ad(1, 2));
    REQUIRE(sube(2, 0) == Ad(2, 1));
    REQUIRE(sube(2, 1) == Ad(2, 2));
}

TEMPLATE_TEST_CASE("DiskView 3x3", "[disktensor]", float, double, std::complex<float>, std::complex<double>) {
    using namespace einsums;

    DiskTensor<TestType, 2> A(fmt::format("/A3/{}", type_name<TestType>()), 3, 3);

    // Data must exist on disk before it can be read in.
    Tensor Ad = create_random_tensor<TestType>("A", 3, 3);
    // println(Ad);
    A.write(Ad);

    {
        // Obtaining a DiskView does not automatically allocate memory.
        auto suba = A(2, All);

        Tensor<TestType, 1> tempa{"tempa", 3};
        tempa(0) = 1.0;
        tempa(1) = 2.0;
        tempa(2) = 3.0;

        // Perform a write setting the data on disk to tempa
        suba = tempa;

        // Perform a read
        auto &tempb = suba.get();

        // Test
        REQUIRE(tempb(0) == TestType{1.0});
        REQUIRE(tempb(1) == TestType{2.0});
        REQUIRE(tempb(2) == TestType{3.0});
    }

    {
        auto suba = A(2, Range{1, 3});

        Tensor<TestType, 1> tempa{"tempb", 2};
        tempa(0) = 4.0;
        tempa(1) = 5.0;

        suba = tempa;

        // Perform a read
        auto &tempb = suba.get();

        REQUIRE(tempb(0) == TestType{4.0});
        REQUIRE(tempb(1) == TestType{5.0});
    }

    {
        auto suba = A(Range{1, 3}, Range{0, 2});

        Tensor<TestType, 2> tempa{"tempa", 2, 2};
        tempa(0, 0) = 10.0;
        tempa(0, 1) = 11.0;
        tempa(1, 0) = 12.0;
        tempa(1, 1) = 13.0;

        suba = tempa;

        auto &tempb = suba.get();

        REQUIRE(tempb(0, 0) == TestType{10.0});
        REQUIRE(tempb(0, 1) == TestType{11.0});
        REQUIRE(tempb(1, 0) == TestType{12.0});
        REQUIRE(tempb(1, 1) == TestType{13.0});
    }

    {
        auto suba1 = A(Range{1, 3}, All);
        auto suba  = suba1(All, Range{0, 2});

        Tensor<TestType, 2> tempa{"tempa", 2, 2};
        tempa(0, 0) = 10.0;
        tempa(0, 1) = 11.0;
        tempa(1, 0) = 12.0;
        tempa(1, 1) = 13.0;

        suba = tempa;

        auto &tempb = suba.get();

        REQUIRE(tempb(0, 0) == TestType{10.0});
        REQUIRE(tempb(0, 1) == TestType{11.0});
        REQUIRE(tempb(1, 0) == TestType{12.0});
        REQUIRE(tempb(1, 1) == TestType{13.0});
    }
}

TEMPLATE_TEST_CASE("DiskView 7x7x7x7", "[disktensor]", float, double, std::complex<float>, std::complex<double>) {
    using namespace einsums;

    SECTION("Write [7,7] data to [:,2,4,:]") {
        DiskTensor<TestType, 4> g(fmt::format("/g0/{}", type_name<TestType>()), 7, 7, 7, 7);
        Tensor                  data = create_random_tensor<TestType>("data", 7, 7);
        g(All, 2, 4, All)            = data;
    }

    SECTION("Write [7,2,7] data to [:,4-5,2,:]") {
        DiskTensor<TestType, 4> g(fmt::format("/g1/{}", type_name<TestType>()), 7, 7, 7, 7);
        Tensor                  data2 = create_random_tensor<TestType>("data", 7, 2, 7);
        g(All, Range{4, 6}, 2, All)   = data2;
    }

    SECTION("Write/Read [7,7] data to/from [2,2,:,:]") {
        DiskTensor<TestType, 4> g(fmt::format("/g2/{}", type_name<TestType>()), 3, 3, 3, 3);
        Tensor                  data3 = create_random_tensor<TestType>("data", 3, 3);
        TestType                value = 0.0;

        for (size_t i = 0; i < 3; i++) {
            for (size_t j = 0; j < 3; j++) {

                for (size_t k = 0; k < 3; k++) {
                    for (size_t l = 0; l < 3; l++) {
                        data3(k, l) = value;
                        value += 1.0;
                    }
                }

                auto  diskView = g(i, j, All, All);
                auto &tensor   = diskView.get();

                tensor = data3;

                diskView.put();
            }
        }
    }
}
