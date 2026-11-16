//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/// @file PlacementPolicy.hpp
/// @brief Where a node runs, chosen as the node is dispatched.

#include <Einsums/ComputeGraph/CostModel.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraphTypes/Ids.hpp>
#include <Einsums/Config/Namespace.hpp>

#include <cstddef>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

/**
 * @brief Which of a tensor's copies hold its current value when a node is about to run.
 *
 * The host-and-device bookkeeping of a replay, reduced to what a policy may weigh. A tensor the
 * replay does not track, and every tensor on a build whose device shares the host's memory, is
 * @c Host: there is no second copy to be stale.
 *
 * @versionadded{2.0.0}
 */
enum class Validity : std::uint8_t {
    Host,   ///< Only the host copy is current.
    Both,   ///< The host and device copies agree.
    Device, ///< Only the device copy is current; the host copy is stale.
};

/**
 * @brief One operand of the node a policy is asked about.
 * @versionadded{2.0.0}
 */
struct PlacementOperand {
    TensorId    id{0};                 ///< The operand's tensor id in the graph being replayed.
    std::size_t bytes{0};              ///< Its size in bytes.
    Validity    valid{Validity::Host}; ///< Where its current value is.
};

/**
 * @brief Everything a @ref PlacementPolicy may read about one node, at the moment it is dispatched.
 *
 * Valid only for the duration of the call it is passed to. There is deliberately nothing about
 * how busy a device or the host is: nothing measures that yet.
 *
 * @versionadded{2.0.0}
 */
struct PlacementQuery {
    Node const &node; ///< The node.
    /// Where the graph's plan (@ref Node::target) puts the node.
    Place planned;
    /// The places the node can run: @ref Node::eligible, plus the planned device when the plan
    /// names one, less any device whose operands could not be given device storage for this replay.
    EligiblePlaces eligible;
    /// The node's inputs, in the order of @ref Node::inputs.
    std::span<PlacementOperand const> inputs;
    /// The node's outputs, in the order of @ref Node::outputs.
    std::span<PlacementOperand const> outputs;

    /// @name Estimates
    /// Filled only when the policy returns a model from @ref PlacementPolicy::cost_model, and empty
    /// otherwise, or for a node the model cannot price.
    /// @{

    /// Estimated time of the node on the host, in microseconds, operands already in place, on the
    /// threads the graph's thread plan gave it (@ref Node::thread_width), or on one thread when it
    /// has no plan. An executor that ignores the plan may give the node more.
    std::optional<double> host_us;
    /// Estimated time of the node on the device, in microseconds, operands already in place. Empty
    /// when the node cannot run there or the model has no device.
    std::optional<double> device_us;
    /// The run the node belongs to is the largest set of device-eligible nodes of this graph
    /// connected through the tensors they exchange. This is the sum over its members of the host
    /// time less the device time, for the members with both estimates.
    std::optional<double> run_gain_us;
    /// Bytes that cross between host and device if the whole run is placed on the device: tensors
    /// its members read that none of them produces, and tensors a member writes that a node outside
    /// the run reads or that are results the replay copies back.
    std::optional<std::size_t> run_boundary_bytes;
    /// @}
};

/**
 * @brief Chooses where a node runs, when it runs.
 *
 * Three things are kept apart. Eligibility (@ref Node::eligible, written by the DeviceEligibility
 * pass) is what a node CAN do. The plan (@ref Node::target, written by GPUPlacement) is what the
 * graph's optimizer decided SHOULD happen. The choice is made here, as each node is dispatched,
 * and is never written to the graph: the same graph replays under one policy and then another.
 *
 * @par Concurrency
 * A policy is shared by every node of a replay, including the nodes of loop bodies and branches,
 * and @ref choose is called from whichever worker thread dispatches the node, several at once.
 * It must be const and thread-safe: no unsynchronized mutable state, and it must not call back
 * into the graph being replayed.
 *
 * @par The answer
 * Must be a place in @ref PlacementQuery::eligible or the node's planned place; anything else is
 * a @c std::logic_error naming the policy and the node. A device answer for a node the dispatcher
 * then declines still runs on the host, as a planned device node does.
 *
 * @par Which nodes are asked
 * Not the transfer nodes (@c HostToDevice, @c DeviceToHost), which are coherence requests and run
 * as they always have. Not, under a policy other than @ref PlannedPlacement, a node that touches a
 * view or a tensor a view aliases: residency is tracked per tensor id, so a view and its parent
 * would not agree on which copy is current, and such a node runs at its planned place.
 *
 * @versionadded{2.0.0}
 */
class EINSUMS_EXPORT PlacementPolicy {
  public:
    PlacementPolicy()                                   = default;
    PlacementPolicy(PlacementPolicy const &)            = default;
    PlacementPolicy &operator=(PlacementPolicy const &) = default;
    PlacementPolicy(PlacementPolicy &&)                 = default;
    PlacementPolicy &operator=(PlacementPolicy &&)      = default;
    virtual ~PlacementPolicy();

    /// @brief Where to run the node described by @p query.
    [[nodiscard]] virtual Place choose(PlacementQuery const &query) const = 0;

    /// @brief The policy's name, for the diagnostic when its answer is refused.
    [[nodiscard]] virtual std::string name() const = 0;

    /// @brief Whether this policy always answers with the planned place.
    ///
    /// True only of @ref PlannedPlacement. The node runner then does not build queries and keeps
    /// the shadow set the plan alone needs, so a graph under it runs exactly as it did before
    /// policies existed. A policy that merely happens to agree with the plan returns false.
    [[nodiscard]] virtual bool follows_plan() const noexcept { return false; }

    /// @brief The cost model this policy prices nodes with, or null when it prices none.
    ///
    /// When non-null the node runner estimates every node it offers, and the runs of nodes they
    /// form, once per replay, and fills the estimate fields of @ref PlacementQuery. When null it
    /// does none of that work and the fields are empty.
    [[nodiscard]] virtual CostModel const *cost_model() const noexcept { return nullptr; }
};

/// @brief Runs each node where the graph's plan put it. The default.
class EINSUMS_EXPORT PlannedPlacement final : public PlacementPolicy {
  public:
    [[nodiscard]] Place       choose(PlacementQuery const &query) const override { return query.planned; }
    [[nodiscard]] std::string name() const override { return "planned"; }
    [[nodiscard]] bool        follows_plan() const noexcept override { return true; }
};

/// @brief Runs every node on the host, whatever the plan says.
class EINSUMS_EXPORT HostPlacement final : public PlacementPolicy {
  public:
    [[nodiscard]] Place       choose(PlacementQuery const &) const override { return Place::host(); }
    [[nodiscard]] std::string name() const override { return "host"; }
};

/// @brief Runs a node on device 0 whenever its eligible set has one, whatever the plan says.
class EINSUMS_EXPORT EligibleDevicePlacement final : public PlacementPolicy {
  public:
    [[nodiscard]] Place choose(PlacementQuery const &query) const override {
        return query.eligible.device(0) ? Place::device(0) : Place::host();
    }
    [[nodiscard]] std::string name() const override { return "device"; }
};

/**
 * @brief Runs each node where its estimated time is lowest, transfers included.
 *
 * With @c bw the model's host-device transfer bandwidth and @c launch the device's launch latency,
 * a node costs @c host_us plus the time to bring its device-only inputs to the host on the host,
 * and @c device_us plus @c launch plus the time to send its host-only inputs up on the device.
 * Outputs are never charged: a destination the node reads is already its trailing input.
 *
 * - Without a device place, or when either estimate is missing, the node runs where it is planned.
 * - The device wins when its total is lower.
 * - A node whose device total is not lower still enters the device when the run it belongs to
 *   gains more than the run's boundary traffic and one launch cost: the run as a whole pays for
 *   the entry, and its other members then find their inputs already there.
 * - Otherwise the host wins, except that an exact tie keeps the planned place.
 *
 * The estimates come from the same pricing the thread planner uses, on the model's CPU profile
 * and, for the device, on its GPU profile. They are dtype-blind and a device profile is a table
 * entry, not a measurement of the device in use, so a close call is not a reliable one.
 *
 * @versionadded{2.0.0}
 */
class EINSUMS_EXPORT CheapestPlacement final : public PlacementPolicy {
  public:
    /// @param[in] model The model whose CPU and GPU profiles price the node, kept by value.
    explicit CheapestPlacement(CostModel model) : _model(std::move(model)) {}

    [[nodiscard]] Place            choose(PlacementQuery const &query) const override;
    [[nodiscard]] std::string      name() const override { return "cheapest"; }
    [[nodiscard]] CostModel const *cost_model() const noexcept override { return &_model; }

  private:
    CostModel _model;
};

/**
 * @brief The built-in policy a value of ``einsums:graph:placement`` names.
 *
 * @param[in] value One of ``planned``, ``host``, ``device`` or ``cheapest``. The last is built on first use
 *                  from the machine's detected cost model.
 * @return A policy that lives for the whole process.
 * @throws std::invalid_argument If @p value names none; the message lists the valid ones.
 * @versionadded{2.0.0}
 */
[[nodiscard]] EINSUMS_EXPORT PlacementPolicy const &builtin_placement_policy(std::string_view value);

/// @brief The values @ref builtin_placement_policy accepts, in the order they are documented.
[[nodiscard]] EINSUMS_EXPORT std::vector<std::string> const &builtin_placement_policy_names();

EINSUMS_NAMESPACE_END(compute_graph)
