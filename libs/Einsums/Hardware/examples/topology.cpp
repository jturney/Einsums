//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Hardware/Topology.hpp>
#include <Einsums/Runtime.hpp>

#include <sstream>

int einsums_main() {
    using namespace einsums;

    std::stringstream ostr;

    hardware::Topology &topo = hardware::Topology::get_singleton();
    topo.print_hwloc(ostr);

    std::cout << ostr.str() << std::endl;

    return 0;
}

int main(int argc, char **argv) {
    return einsums::start(einsums_main, argc, argv);
}