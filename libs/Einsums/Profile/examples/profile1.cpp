//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Print.hpp>
#include <Einsums/Profile.hpp>
#include <Einsums/Runtime.hpp>

using namespace waggle;

namespace {
void microkernel() {
    WAGGLE_ZONE("microkernel");
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
}

void pack() {
    WAGGLE_ZONE("pack");
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
}

void contract() {
    WAGGLE_ZONE("contract");
    ScopedZone const z("contract");
    pack();
    microkernel();
    microkernel();
}

int einsums_main() {
    {
        WAGGLE_ZONE("main");
        std::thread t([] {
            WAGGLE_ZONE("worker thread");
            contract();
        });

        contract();
        t.join();
    }

    // Print human-readable report
    print_report();

    return 0;
}
} // namespace

int main(int argc, char **argv) {
    return einsums::start(einsums_main, argc, argv);
}