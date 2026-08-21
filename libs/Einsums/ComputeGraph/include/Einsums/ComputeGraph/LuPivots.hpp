//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/BLAS/Types.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Python/Annotations.hpp>

#include <cstddef>
#include <memory>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

/**
 * @brief The row interchanges @ref getrf produces, as something a graph can outlive.
 *
 * A pivot array is a list of BLAS integers, which is not one of the dtypes a tensor operand may carry, so it cannot be a graph
 * TENSOR and cannot be a slot. It is instead a handle whose buffer is shared: @ref getrf and @ref getrs both bake a copy of the
 * @c shared_ptr into their executors at capture time, so the buffer survives however long the graph does even if the handle the
 * caller held is dropped first, and both see the same array on every replay.
 *
 * Because the pivots ride outside the dataflow, they carry no ordering of their own. What orders a factorization against its
 * solves is the FACTORIZATION TENSOR: @ref getrf writes it, @ref getrs reads it, and the hazard scan serializes the pair on that
 * edge. A @ref getrs against a handle whose matching @ref getrf writes a different tensor has nothing holding the two together
 * and is a bug in the caller.
 *
 * The buffer is sized by @ref getrf on the first execution and reused afterwards.
 */
class APIARY_EXPOSE APIARY_MODULE("linalg") LuPivots {
  public:
    /// An empty pivot array. ``getrf`` sizes it to the order of the matrix it factors.
    APIARY_EXPOSE LuPivots() : _pivots(std::make_shared<std::vector<blas::int_t>>()) {}

    /// The number of interchanges recorded, which is zero until a ``getrf`` node has executed.
    APIARY_EXPOSE APIARY_GETTER("size") [[nodiscard]] size_t size() const { return _pivots->size(); }

    /// The shared buffer, for an executor to bake in at capture time.
    [[nodiscard]] std::shared_ptr<std::vector<blas::int_t>> const &buffer() const { return _pivots; }

  private:
    std::shared_ptr<std::vector<blas::int_t>> _pivots;
};

EINSUMS_NAMESPACE_END(compute_graph)
