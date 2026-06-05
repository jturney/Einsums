//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Comm/ProcessGrid.hpp>
#include <Einsums/Comm/Runtime.hpp>

#include <cassert>

namespace einsums::comm {

namespace {

/// Find the near-square factorization of n: Pr × Pc = n with |Pr - Pc| minimized.
/// Returns (Pr, Pc) with Pr >= Pc.
std::pair<int, int> near_square_factorization(int n) {
    int best_r = n, best_c = 1;

    for (int c = 1; c * c <= n; c++) {
        if (n % c == 0) {
            int r = n / c;
            if (r - c < best_r - best_c) {
                best_r = r;
                best_c = c;
            }
        }
    }

    return {best_r, best_c};
}

} // namespace

void ProcessGrid::init(int pr, int pc, Communicator const &parent) {
    _pr = pr;
    _pc = pc;

    int rank = parent.rank();
    _my_row  = rank / _pc;
    _my_col  = rank % _pc;

    // Row communicator: all ranks with the same my_row_ (color = my_row_, key = my_col_)
    _row_comm = parent.split(_my_row, _my_col);

    // Column communicator: all ranks with the same my_col_ (color = my_col_, key = my_row_)
    _col_comm = parent.split(_my_col, _my_row);
}

ProcessGrid::ProcessGrid(Communicator const &parent) {
    auto [pr, pc] = near_square_factorization(parent.size());
    init(pr, pc, parent);
}

ProcessGrid::ProcessGrid(int pr, int pc, Communicator const &parent) {
    assert(pr * pc == parent.size() && "ProcessGrid dimensions must multiply to communicator size");
    init(pr, pc, parent);
}

ProcessGrid &ProcessGrid::default_grid() {
    static ProcessGrid grid;
    return grid;
}

} // namespace einsums::comm
