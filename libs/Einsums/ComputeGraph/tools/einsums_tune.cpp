//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// einsums-tune: measure this machine and write its hardware profile, or report on the one in
// effect. The work is compute_graph::tune, which `einsums tune` and `python -m einsums.tune` call
// too; this only starts the runtime around it.

#include <Einsums/ComputeGraph/Tune.hpp>
#include <Einsums/Runtime.hpp>

#include <functional>
#include <string>
#include <string_view>
#include <vector>

int main(int argc, char **argv) {
    std::vector<std::string> const args(argv + 1, argv + argc);

    // The runtime answers --help with its own options and exits before the tune runs, so the
    // tune's usage is printed first and the runtime's follows.
    for (auto const &arg : args) {
        if (arg == "--help" || arg == "-h") {
            einsums::compute_graph::tune({"--help"});
            break;
        }
    }
    std::function<int()> const run = [&args] { return einsums::compute_graph::tune(args); };
    return einsums::start(run, argc, argv);
}
