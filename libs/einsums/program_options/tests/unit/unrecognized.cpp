// Copyright Sascha Ochsenknecht 2009.
//  SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <einsums/program_options/cmdline.hpp>
#include <einsums/program_options/detail/cmdline.hpp>
#include <einsums/program_options/option.hpp>
#include <einsums/program_options/options_description.hpp>
#include <einsums/program_options/parsers.hpp>
#include <einsums/testing.hpp>

#include <cstddef>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace einsums::program_options;
using namespace std;

// Test free function collect_unrecognized()
//
//  it collects the tokens of all not registered options. It can be used
//  to pass them to an own parser implementation

void test_unrecognize_cmdline() {
    options_description desc;

    string         content = "prg --input input.txt --optimization 4 --opt option";
    vector<string> tokens  = split_unix(content);

    einsums::program_options::detail::cmdline cmd(tokens);
    cmd.set_options_description(desc);
    cmd.allow_unregistered();

    vector<option> opts   = cmd.run();
    vector<string> result = collect_unrecognized(opts, include_positional);

    EINSUMS_TEST_EQ(result.size(), std::size_t(7));
    EINSUMS_TEST_EQ(result[0], "prg");
    EINSUMS_TEST_EQ(result[1], "--input");
    EINSUMS_TEST_EQ(result[2], "input.txt");
    EINSUMS_TEST_EQ(result[3], "--optimization");
    EINSUMS_TEST_EQ(result[4], "4");
    EINSUMS_TEST_EQ(result[5], "--opt");
    EINSUMS_TEST_EQ(result[6], "option");
}

void test_unrecognize_config() {
    options_description desc;

    string content = " input = input.txt\n"
                     " optimization = 4\n"
                     " opt = option\n";

    stringstream   ss(content);
    vector<option> opts   = parse_config_file(ss, desc, true).options;
    vector<string> result = collect_unrecognized(opts, include_positional);

    EINSUMS_TEST_EQ(result.size(), std::size_t(6));
    EINSUMS_TEST_EQ(result[0], "input");
    EINSUMS_TEST_EQ(result[1], "input.txt");
    EINSUMS_TEST_EQ(result[2], "optimization");
    EINSUMS_TEST_EQ(result[3], "4");
    EINSUMS_TEST_EQ(result[4], "opt");
    EINSUMS_TEST_EQ(result[5], "option");
}

int main(int /*ac*/, char ** /*av*/) {
    test_unrecognize_cmdline();
    test_unrecognize_config();

    return 0;
}
