//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/Python/Annotations.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

/**
 * @brief An array of conditional gates a graph reads without calling back into its caller.
 *
 * @ref Graph::add_conditional takes a @c std::function<bool()>, which is the general form and the
 * right one when the answer has to be computed. It is the wrong one when the answer is already
 * known before the replay starts and the graph holds hundreds of conditionals: a predicate bound to
 * a Python callable takes the GIL on every evaluation, so a graph whose conditionals are executed
 * from several threads serializes on it, and a caller that already knows every answer pays that
 * serialization for nothing.
 *
 * This is the other form. The caller writes the whole array once per replay, through @ref fill or
 * @ref assign, and each conditional reads one slot of it. The buffer is shared: @ref
 * Graph::add_conditional_flag bakes a copy of the @c shared_ptr into the node at capture time, so
 * the flags survive however long the graph does even if the handle the caller held is dropped
 * first, and every node sees the same array on every replay. Evaluation is a load and a compare.
 *
 * The flags ride outside the dataflow and carry no ordering of their own, exactly as @ref LuPivots
 * does. Nothing serializes a write to a flag against a replay that reads it; the caller must finish
 * writing before it calls @c execute().
 *
 * @code
 * cg::GateFlags live(blocks.size(), true);
 * for (size_t b = 0; b < blocks.size(); b++) {
 *     auto [branch, _] = graph.add_conditional_flag(fmt::format("block [{}]", b), live, b);
 *     CaptureGuard const guard(branch);
 *     emit(blocks[b]);
 * }
 * live.set(3, false);   // block 3 is skipped by the next replay
 * graph.execute();
 * @endcode
 */
class APIARY_EXPOSE APIARY_MODULE("graph") GateFlags {
  public:
    /// @p count flags, all @p value.
    APIARY_EXPOSE explicit GateFlags(size_t count = 0, bool value = false)
        : _flags(std::make_shared<std::vector<std::uint8_t>>(count, static_cast<std::uint8_t>(value ? 1 : 0))) {}

    /// How many flags the array holds.
    APIARY_EXPOSE APIARY_GETTER("size") [[nodiscard]] size_t size() const { return _flags->size(); }

    /// Grow or shrink the array, giving any new flag @p value.
    ///
    /// Resizing while a graph holds indices into the array is the caller's problem: an index past
    /// the end reads false, which silently skips a branch rather than reading out of bounds.
    APIARY_EXPOSE void resize(size_t count, bool value = false) { _flags->resize(count, static_cast<std::uint8_t>(value ? 1 : 0)); }

    /// The flag at @p index.
    APIARY_EXPOSE [[nodiscard]] bool get(size_t index) const {
        if (index >= _flags->size()) {
            EINSUMS_THROW_EXCEPTION(std::out_of_range, "GateFlags: index {} is past the end of {} flags", index, _flags->size());
        }
        return (*_flags)[index] != 0;
    }

    /// Set the flag at @p index.
    APIARY_EXPOSE void set(size_t index, bool value) {
        if (index >= _flags->size()) {
            EINSUMS_THROW_EXCEPTION(std::out_of_range, "GateFlags: index {} is past the end of {} flags", index, _flags->size());
        }
        (*_flags)[index] = static_cast<std::uint8_t>(value ? 1 : 0);
    }

    /// Set every flag to @p value.
    APIARY_EXPOSE void fill(bool value) { std::fill(_flags->begin(), _flags->end(), static_cast<std::uint8_t>(value ? 1 : 0)); }

    /// Replace the whole array in one call, which is the point of the type: a caller that knows
    /// every answer writes them all at once instead of once per conditional.
    APIARY_EXPOSE void assign(std::vector<std::uint8_t> const &values) {
        _flags->assign(values.begin(), values.end());
        for (auto &flag : *_flags) {
            flag = static_cast<std::uint8_t>(flag != 0 ? 1 : 0);
        }
    }

    /// The shared buffer, for a node to bake in at capture time.
    [[nodiscard]] std::shared_ptr<std::vector<std::uint8_t>> const &buffer() const { return _flags; }

    /**
     * @brief Wrap an existing shared buffer rather than allocating one.
     *
     * @param[in] buffer The array to share. Null is treated as an empty array,
     *            so the object is always usable.
     * @return A handle onto @p buffer.
     *
     * For a caller that was handed the buffer rather than the handle:
     * @ref Graph::gate_flags recreates a loaded graph's named arrays this way,
     * so a caller can set the gates of a graph it did not capture. Sharing
     * rather than copying is the whole point - the nodes already hold this
     * ``shared_ptr``, so a write through the returned handle is what the next
     * replay reads.
     * @versionadded{2.0.0}
     */
    [[nodiscard]] static GateFlags adopt(std::shared_ptr<std::vector<std::uint8_t>> buffer) {
        GateFlags out;
        if (buffer != nullptr) {
            out._flags = std::move(buffer);
        }
        return out;
    }

  private:
    std::shared_ptr<std::vector<std::uint8_t>> _flags;
};

EINSUMS_NAMESPACE_END(compute_graph)
