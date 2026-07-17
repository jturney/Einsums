//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Comm/Communicator.hpp>

#include <utility>

namespace einsums::comm {

/**
 * @brief 2D process grid for distributed tensor operations.
 *
 * Arranges P MPI ranks into a Pr × Pc grid. Each rank has a (row, col) coordinate.
 * Row and column sub-communicators enable efficient collective communication
 * patterns like SUMMA broadcasts.
 *
 * For P=4: default grid is 2×2. For P=6: 2×3 or 3×2 (whichever is more square).
 * For prime P (e.g., 7): falls back to P×1 (1D grid).
 *
 * @code
 * auto& grid = comm::ProcessGrid::default_grid();
 * fmt::print("Rank {} is at ({}, {}) in {}x{} grid\n",
 *            world_rank(), grid.my_row(), grid.my_col(), grid.rows(), grid.cols());
 * @endcode
 */
class EINSUMS_EXPORT ProcessGrid {
  public:
    /// Auto-compute near-square Pr × Pc from parent communicator size.
    explicit ProcessGrid(Communicator const &parent = Communicator::world());

    /// Explicit grid dimensions. Requires pr * pc == parent.size().
    ProcessGrid(int pr, int pc, Communicator const &parent = Communicator::world());

    /// Grid dimensions.
    [[nodiscard]] int rows() const { return _pr; }
    [[nodiscard]] int cols() const { return _pc; }
    [[nodiscard]] int size() const { return _pr * _pc; }

    /// This rank's coordinates in the grid.
    [[nodiscard]] int my_row() const { return _my_row; }
    [[nodiscard]] int my_col() const { return _my_col; }

    /// Map grid coordinates to MPI rank (row-major: rank = row * Pc + col).
    [[nodiscard]] int rank_at(int row, int col) const { return row * _pc + col; }

    /// Map MPI rank to grid coordinates.
    [[nodiscard]] std::pair<int, int> coords(int rank) const { return {rank / _pc, rank % _pc}; }

    /// Sub-communicator for all ranks in the same row (used for SUMMA B-broadcasts).
    [[nodiscard]] Communicator const &row_comm() const { return _row_comm; }

    /// Sub-communicator for all ranks in the same column (used for SUMMA A-broadcasts).
    [[nodiscard]] Communicator const &col_comm() const { return _col_comm; }

    /// Default grid constructed from Communicator::world() on first access.
    [[nodiscard]] static ProcessGrid const &default_grid();

  private:
    int          _pr;
    int          _pc;
    int          _my_row;
    int          _my_col;
    Communicator _row_comm; ///< Ranks in the same row share this communicator
    Communicator _col_comm; ///< Ranks in the same column share this communicator

    void init(int pr, int pc, Communicator const &parent);
};

} // namespace einsums::comm
