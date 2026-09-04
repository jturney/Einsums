//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/**
 * @file Spaces.hpp
 * @brief Index-space annotations and the registry that holds them.
 *
 * An index space is the *meaning* of a tensor index: occupied orbitals, virtual orbitals, an
 * auxiliary fitting basis, a grid. Slot annotations (added by a later task) carry a @ref SpaceId
 * per index, and the algebraic passes read the registry to answer the questions they need:
 *
 * - which of two extents is the small one (scale order, so a cost polynomial can be minimized
 *   symbolically before any tensor is bound),
 * - whether two spaces can share an element (disjointness, which is what makes a cross-space
 *   contraction provably zero),
 * - whether one space lives inside another (containment, e.g. a PNO subspace within the virtual
 *   space).
 *
 * @par What deliberately is NOT here
 * @ref IndexSpace carries semantics and nothing else. Distribution and storage preferences are
 * machine policy, they live in @ref SpacePolicy, and the registry keeps them in a separate table
 * keyed by space NAME. A graph saved to disk must never carry the policy of the machine that
 * captured it, and the only way to guarantee that is for the annotation type to have nowhere to
 * put it.
 *
 * @par Ids versus names
 * @ref SpaceId is an opaque registry-local handle. It is stable for the life of the registry that
 * issued it and meaningless outside that process. In-memory annotations hold ids; serialization
 * writes names and resolves them back to ids on load.
 *
 * @par Spaces are assumed non-empty
 * Every query below reads "space" as a non-empty set of indices. That is what makes a space not
 * disjoint from itself, and what makes a space known to be disjoint from another one also known
 * not to be contained in it.
 */

#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Python/Annotations.hpp>

#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

/**
 * @brief How a space's extent grows with the size of the system.
 *
 * Stored as an exponent so a user-defined space is expressible without extending an enum: the
 * extent is taken to scale as @c N^exponent for system size @c N. The two cases that matter for
 * electronic structure have names, @ref constant and @ref linear, and everything else goes through
 * @ref power.
 *
 * The value is advisory. It is what lets a pass rank two candidate contraction orders whose cost
 * polynomials differ only in which symbol carries the higher power, before any tensor with real
 * extents has been bound.
 */
struct APIARY_EXPOSE APIARY_MODULE("graph") GrowthClass {
    /// Exponent of the system size that the extent follows. Defaults to linear growth.
    ///
    /// Read-only from Python: a growth class reached through a registered space is a view of what
    /// the registry holds, and a space's semantics are settled at registration.
    APIARY_EXPOSE APIARY_READONLY double exponent{1.0};

    /// @brief A space whose extent does not grow with the system (exponent 0).
    /// @return The constant growth class.
    APIARY_EXPOSE [[nodiscard]] static constexpr GrowthClass constant() noexcept { return GrowthClass{0.0}; }

    /// @brief A space whose extent grows proportionally with the system (exponent 1).
    /// @return The linear growth class.
    APIARY_EXPOSE [[nodiscard]] static constexpr GrowthClass linear() noexcept { return GrowthClass{1.0}; }

    /// @brief A space whose extent grows as a user-chosen power of the system size.
    /// @param[in] exponent The exponent to use.
    /// @return The requested growth class.
    APIARY_EXPOSE [[nodiscard]] static constexpr GrowthClass power(double exponent) noexcept { return GrowthClass{exponent}; }

    /// @brief Compare two growth classes.
    /// @param[in] lhs Left operand.
    /// @param[in] rhs Right operand.
    /// @return True when the exponents are identical.
    [[nodiscard]] friend constexpr bool operator==(GrowthClass lhs, GrowthClass rhs) noexcept { return lhs.exponent == rhs.exponent; }
};

/**
 * @brief The semantics of one index space, and nothing else.
 *
 * Value type: two of these compare equal exactly when every field matches, which is what lets
 * @ref SpaceRegistry::register_space be idempotent for a repeated identical declaration and an
 * error for a conflicting one.
 *
 * An aggregate, deliberately: every construction site in the library uses designated initializers,
 * which is what keeps a five-field declaration readable. Python has no aggregate initialization, so
 * it builds one through @ref make_index_space (spelled ``cg.index_space`` there) instead, and reads
 * the fields back.
 */
struct APIARY_EXPOSE APIARY_MODULE("graph") IndexSpace {
    /// Human-readable name, unique within a registry ("occ", "virt", "aux", "grid", ...).
    APIARY_EXPOSE APIARY_READONLY std::string name;

    /// Letter this space contributes to a symbolic cost polynomial ("o", "v", "x", "g", ...).
    APIARY_EXPOSE APIARY_READONLY std::string scale_symbol;

    /// Name a SYMBOLIC EXTENT over this space goes by ("no", "nv", ...). Empty when the space has
    /// none, and then an axis over it cannot be given a symbolic extent from the space alone.
    ///
    /// Distinct from @ref scale_symbol, which names the space in a cost POLYNOMIAL ("o"), and
    /// distinct from @ref name, which identifies the space itself. Spelling a dim symbol as the
    /// space's name would make the ``(symbol, space)`` tie a tautology and, worse, would have a
    /// plain symbol claim a single extent for a space that may be ragged - which is exactly what
    /// ``"ragged:<space>"`` exists to say instead.
    APIARY_EXPOSE APIARY_READONLY std::string dim_symbol;

    /// Advisory extent, used only to break ties when no tensor instance is bound. Zero means unset.
    APIARY_EXPOSE APIARY_READONLY double typical_extent{0};

    /// How the extent grows with system size.
    APIARY_EXPOSE APIARY_READONLY GrowthClass growth{GrowthClass::linear()};

    /// @brief Compare two index spaces field by field.
    /// @param[in] lhs Left operand.
    /// @param[in] rhs Right operand.
    /// @return True when name, scale symbol, dim symbol, typical extent and growth all match.
    [[nodiscard]] friend bool operator==(IndexSpace const &lhs, IndexSpace const &rhs) noexcept {
        return lhs.name == rhs.name && lhs.scale_symbol == rhs.scale_symbol && lhs.dim_symbol == rhs.dim_symbol &&
               lhs.typical_extent == rhs.typical_extent && lhs.growth == rhs.growth;
    }
};

/**
 * @brief Build an @ref IndexSpace from its parts.
 * @param[in] name Human-readable name, unique within a registry.
 * @param[in] scale_symbol Letter this space contributes to a symbolic cost polynomial.
 * @param[in] typical_extent Advisory extent. Zero, the default, means unset.
 * @param[in] growth How the extent grows with system size. Linear by default.
 * @return The space.
 *
 * A named constructor rather than a real one: @ref IndexSpace stays an aggregate so the library's
 * designated-initializer construction sites keep working, and this is what a caller with no
 * aggregate initialization (Python) builds one through.
 */
[[nodiscard]] APIARY_EXPOSE APIARY_MODULE("graph") APIARY_RENAME("index_space") inline IndexSpace
make_index_space(std::string name, std::string scale_symbol, double typical_extent = 0.0, GrowthClass growth = GrowthClass::linear(),
                 std::string dim_symbol = std::string{}) {
    return IndexSpace{.name           = std::move(name),
                      .scale_symbol   = std::move(scale_symbol),
                      .dim_symbol     = std::move(dim_symbol),
                      .typical_extent = typical_extent,
                      .growth         = growth};
}

/**
 * @brief Opaque handle to a space held by a @ref SpaceRegistry.
 *
 * Stable for the life of the registry that issued it, meaningless anywhere else. A
 * default-constructed id is invalid and every registry query rejects it.
 */
class APIARY_EXPOSE APIARY_MODULE("graph") SpaceId {
  public:
    /// @brief Construct an invalid id.
    APIARY_EXPOSE constexpr SpaceId() noexcept = default;

    /// @brief Whether this id refers to a registered space.
    /// @return True unless the id is default-constructed.
    APIARY_EXPOSE APIARY_GETTER("valid") [[nodiscard]] constexpr bool valid() const noexcept { return _value != invalid_value; }

    /// @brief The underlying registry-local index.
    /// @return The index, or @ref invalid_value for an invalid id. For debugging and hashing only.
    APIARY_EXPOSE APIARY_GETTER("value") [[nodiscard]] constexpr std::uint32_t value() const noexcept { return _value; }

    /**
     * @brief Whether two ids name the same space of the same registry.
     * @param[in] other The id to compare against.
     * @return True when both wrap the same registry-local index.
     *
     * The named form of @c operator==, bound as Python's @c __eq__. Comparison has to reach Python
     * (a caller that looked a space up twice must be able to tell the two answers are the same
     * space), and a named method is what the binding generator renders reliably.
     */
    APIARY_EXPOSE APIARY_OPERATOR("__eq__") [[nodiscard]] constexpr bool equals(SpaceId other) const noexcept {
        return _value == other._value;
    }

    /// Sentinel stored by an invalid id.
    static constexpr std::uint32_t invalid_value = static_cast<std::uint32_t>(-1);

    /// @brief Compare two ids.
    /// @param[in] lhs Left operand.
    /// @param[in] rhs Right operand.
    /// @return True when both name the same space of the same registry.
    [[nodiscard]] friend constexpr bool operator==(SpaceId lhs, SpaceId rhs) noexcept { return lhs._value == rhs._value; }

    /// @brief Order two ids so they can key an ordered container.
    /// @param[in] rhs Right operand.
    /// @return The ordering of the underlying indices, which is registration order.
    ///
    /// A member rather than a hidden friend, which every other spaceship in the
    /// tree already is: an id converts from nothing, so the two forms behave
    /// identically here, and the member form is the one the documentation
    /// extractor renders without mangling the operator's name.
    [[nodiscard]] constexpr std::strong_ordering operator<=>(SpaceId rhs) const noexcept { return _value <=> rhs._value; }

  private:
    friend class SpaceRegistry;

    /// @brief Construct an id for a registry slot.
    /// @param[in] value The registry-local index.
    explicit constexpr SpaceId(std::uint32_t value) noexcept : _value{value} {}

    std::uint32_t _value{invalid_value};
};

/**
 * @brief Answer to a relation query that may simply not be known.
 *
 * "Unknown" is a first-class answer, not a failure: the registry only ever holds what was declared
 * and what follows from it, and a pass that would rewrite a graph on the strength of a relation
 * must treat Unknown exactly as it treats No.
 */
// With the annotation macros in front, clang-format reads the enum base as a braced initializer
// and glues the brace to it.
// clang-format off
enum class APIARY_EXPOSE APIARY_MODULE("graph") Tristate : std::uint8_t {
    No,      ///< The relation is known not to hold.
    Yes,     ///< The relation is known to hold.
    Unknown, ///< Nothing declared or derivable settles the question.
};
// clang-format on

/**
 * @brief Name of a @ref Tristate value.
 * @param[in] value The value to name.
 * @return "No", "Yes" or "Unknown".
 */
[[nodiscard]] inline std::string_view tristate_name(Tristate value) noexcept {
    switch (value) {
    case Tristate::No:
        return "No";
    case Tristate::Yes:
        return "Yes";
    case Tristate::Unknown:
        return "Unknown";
    }
    return "Unknown";
}

/// How a space's axis should be laid out across ranks. Machine policy, never part of @ref IndexSpace.
enum class DistributionHint : std::uint8_t {
    None,      ///< Undecided. The planner is free to choose.
    Replicate, ///< Keep a full copy of this axis on every rank.
    Tile,      ///< Split this axis across ranks.
};

/// Where a space's data should preferentially live. Machine policy, never part of @ref IndexSpace.
enum class StorageHint : std::uint8_t {
    None,         ///< Undecided. The planner is free to choose.
    PreferMemory, ///< Keep resident in memory if at all possible.
    SpillOk,      ///< Spilling to disk is acceptable.
};

/**
 * @brief Machine policy attached to a space by name.
 *
 * Deliberately a separate type from @ref IndexSpace, held in a separate table. Semantics travel
 * with a saved graph; policy does not.
 */
struct SpacePolicy {
    /// Distribution preference for axes over this space.
    DistributionHint dist{DistributionHint::None};

    /// Storage preference for data indexed by this space.
    StorageHint storage{StorageHint::None};

    /// @brief Compare two policies.
    /// @param[in] lhs Left operand.
    /// @param[in] rhs Right operand.
    /// @return True when both hints match.
    [[nodiscard]] friend constexpr bool operator==(SpacePolicy lhs, SpacePolicy rhs) noexcept {
        return lhs.dist == rhs.dist && lhs.storage == rhs.storage;
    }
};

/**
 * @brief Spaces, the relations declared between them, and the policy table keyed by their names.
 *
 * @par Declared versus derived
 * A declaration is a fact the caller asserts and the registry stores. A query answers from the
 * closure of those facts, and NOTHING is inferred at declaration time:
 *
 * - Scale order is stored as declared edges and queried through their transitive closure, so
 *   @c o < v and @c v < x answer @c o < x.
 * - Containment is stored as declared edges and queried through their transitive closure, extended
 *   with the reflexive case: a space is contained in itself.
 * - Disjointness is stored symmetrically as declared, and queries additionally derive it through
 *   containment on either side: if @c a is inside @c b and @c b is disjoint from @c c, then @c a is
 *   disjoint from @c c, and so is everything inside @c c.
 *
 * Scale order and containment are kept independent on purpose. A subspace being smaller than its
 * parent is a plausible reading of @c a within @c b, but it is not a consequence of it, and the
 * caller who wants that ordering can declare it.
 *
 * @par Consistency
 * A declaration inconsistent with what the registry already holds is rejected with
 * @c std::invalid_argument, in either declaration order. The governing invariant for the set
 * relations is that no space may be contained in two spaces declared disjoint, which rules out
 * "a inside b" together with "a disjoint from b" and also the transitive versions of it. Scale
 * order rejects cycles, including cycles closed through transitivity. Self-relations are rejected
 * as well: @c a < a is false, @c a disjoint from @c a is false for a non-empty space, and @c a
 * within @c a is trivially true and therefore never worth asserting.
 *
 * @par Thread safety
 * A single mutex guards everything. The expectation is that registration and declaration happen
 * during startup or setup, from one thread, and that queries then run from anywhere; the locking
 * exists so a late registration cannot corrupt a concurrent reader, not because this is a hot path.
 * References returned by @ref space stay valid for the life of the registry.
 */
class APIARY_EXPOSE APIARY_MODULE("graph") APIARY_NOCOPY APIARY_NOMOVE SpaceRegistry {
  public:
    /// @brief Construct an empty registry.
    APIARY_EXPOSE SpaceRegistry() = default;

    /// The mutex makes a registry neither copyable nor movable. Pass it by reference.
    SpaceRegistry(SpaceRegistry const &)            = delete;
    SpaceRegistry &operator=(SpaceRegistry const &) = delete;

    /**
     * @brief Register a space, or recover the id of an identical one already registered.
     * @param[in] space The space to register. Its name must not be empty.
     * @return The id of the registered space.
     * @throws std::invalid_argument if the name is empty, or if a space of that name is already
     *         registered with different content.
     */
    APIARY_EXPOSE SpaceId register_space(IndexSpace space) {
        std::scoped_lock const guard(_mutex);

        if (space.name.empty()) {
            throw std::invalid_argument("SpaceRegistry::register_space: a space name must not be empty");
        }

        auto const existing = _by_name.find(space.name);
        if (existing != _by_name.end()) {
            IndexSpace const &held = _spaces[existing->second];
            if (!(held == space)) {
                throw std::invalid_argument("SpaceRegistry::register_space: space '" + space.name +
                                            "' is already registered with different content");
            }
            return SpaceId{existing->second};
        }

        auto const index = static_cast<std::uint32_t>(_spaces.size());
        _by_name.emplace(space.name, index);
        _spaces.push_back(std::move(space));
        grow_relations(_spaces.size());
        return SpaceId{index};
    }

    /**
     * @brief Look a space up by name.
     * @param[in] name The name to look for.
     * @return The id, or an empty optional when no space of that name is registered.
     */
    APIARY_EXPOSE [[nodiscard]] std::optional<SpaceId> find(std::string_view name) const {
        std::scoped_lock const guard(_mutex);
        auto const             it = _by_name.find(std::string(name));
        if (it == _by_name.end()) {
            return std::nullopt;
        }
        return SpaceId{it->second};
    }

    /**
     * @brief The space an id names.
     * @param[in] id The id to resolve. Must be valid and must come from this registry.
     * @return The registered space. The reference stays valid for the life of the registry.
     * @throws std::invalid_argument if the id is invalid or out of range.
     */
    APIARY_EXPOSE APIARY_RVP(reference_internal) [[nodiscard]] IndexSpace const &space(SpaceId id) const {
        std::scoped_lock const guard(_mutex);
        return _spaces[check(id, "space")];
    }

    /**
     * @brief Number of registered spaces.
     * @return The count.
     */
    APIARY_EXPOSE APIARY_GETTER("size") [[nodiscard]] std::size_t size() const {
        std::scoped_lock const guard(_mutex);
        return _spaces.size();
    }

    /**
     * @brief Every registered id, in registration order.
     * @return The ids.
     */
    APIARY_EXPOSE APIARY_GETTER("ids") [[nodiscard]] std::vector<SpaceId> ids() const {
        std::scoped_lock const guard(_mutex);
        std::vector<SpaceId>   out;
        out.reserve(_spaces.size());
        for (std::uint32_t i = 0; i < static_cast<std::uint32_t>(_spaces.size()); ++i) {
            out.push_back(SpaceId{i});
        }
        return out;
    }

    /**
     * @brief Declare that the extent of @p smaller is below the extent of @p larger.
     * @param[in] smaller The space with the smaller scale.
     * @param[in] larger The space with the larger scale.
     * @throws std::invalid_argument if either id is invalid, if the two are the same space, or if
     *         the ordering would close a cycle with what is already declared.
     */
    APIARY_EXPOSE void declare_less(SpaceId smaller, SpaceId larger) {
        std::scoped_lock const guard(_mutex);
        std::size_t const      a = check(smaller, "declare_less");
        std::size_t const      b = check(larger, "declare_less");

        if (a == b) {
            throw std::invalid_argument("SpaceRegistry::declare_less: '" + _spaces[a].name + "' cannot have a smaller scale than itself");
        }
        if (reaches(_less, b, a)) {
            throw std::invalid_argument("SpaceRegistry::declare_less: '" + _spaces[a].name + "' < '" + _spaces[b].name +
                                        "' contradicts the declared ordering '" + _spaces[b].name + "' < '" + _spaces[a].name + "'");
        }
        close_edge(_less, a, b);
    }

    /**
     * @brief Declare that two spaces share no element.
     * @param[in] first One space.
     * @param[in] second The other space. The relation is symmetric.
     * @throws std::invalid_argument if either id is invalid, if the two are the same space, or if
     *         some space is already known to sit inside both of them (which includes one of them
     *         sitting inside the other).
     */
    APIARY_EXPOSE void declare_disjoint(SpaceId first, SpaceId second) {
        std::scoped_lock const guard(_mutex);
        std::size_t const      a = check(first, "declare_disjoint");
        std::size_t const      b = check(second, "declare_disjoint");

        if (a == b) {
            throw std::invalid_argument("SpaceRegistry::declare_disjoint: '" + _spaces[a].name + "' cannot be disjoint from itself");
        }

        Matrix trial = _disjoint;
        trial[a][b]  = 1;
        trial[b][a]  = 1;
        validate(_contained, trial, "declare_disjoint");
        _disjoint = std::move(trial);
    }

    /**
     * @brief Declare that every element of @p inner is an element of @p outer.
     * @param[in] inner The contained space.
     * @param[in] outer The containing space.
     * @throws std::invalid_argument if either id is invalid, if the two are the same space
     *         (self-containment is trivially true and asserting it signals a mistake), if the
     *         reverse containment is already declared, or if the containment would place some
     *         space inside two spaces declared disjoint.
     */
    APIARY_EXPOSE void declare_contained(SpaceId inner, SpaceId outer) {
        std::scoped_lock const guard(_mutex);
        std::size_t const      a = check(inner, "declare_contained");
        std::size_t const      b = check(outer, "declare_contained");

        if (a == b) {
            throw std::invalid_argument("SpaceRegistry::declare_contained: '" + _spaces[a].name +
                                        "' is trivially contained in itself; declaring it is not allowed");
        }
        if (reaches(_contained, b, a)) {
            throw std::invalid_argument("SpaceRegistry::declare_contained: '" + _spaces[a].name + "' within '" + _spaces[b].name +
                                        "' contradicts the declared containment '" + _spaces[b].name + "' within '" + _spaces[a].name +
                                        "'");
        }

        Matrix trial = _contained;
        close_edge(trial, a, b);
        validate(trial, _disjoint, "declare_contained");
        _contained = std::move(trial);
    }

    /**
     * @brief Whether @p smaller has a smaller scale than @p larger.
     * @param[in] smaller The candidate smaller space.
     * @param[in] larger The candidate larger space.
     * @return Yes when the closure of the declared orderings puts @p smaller below @p larger, No
     *         when it puts @p larger below @p smaller or the two are the same space, Unknown when
     *         no declared ordering relates them.
     * @throws std::invalid_argument if either id is invalid.
     */
    APIARY_EXPOSE [[nodiscard]] Tristate is_less(SpaceId smaller, SpaceId larger) const {
        std::scoped_lock const guard(_mutex);
        std::size_t const      a = check(smaller, "is_less");
        std::size_t const      b = check(larger, "is_less");

        if (a == b) {
            return Tristate::No;
        }
        if (_less[a][b] != 0) {
            return Tristate::Yes;
        }
        if (_less[b][a] != 0) {
            return Tristate::No;
        }
        return Tristate::Unknown;
    }

    /**
     * @brief Whether two spaces share no element.
     * @param[in] first One space.
     * @param[in] second The other space.
     * @return Yes when disjointness was declared for the pair or follows from containment on either
     *         side, No when the two are the same space or one is known to contain the other,
     *         Unknown otherwise.
     * @throws std::invalid_argument if either id is invalid.
     */
    APIARY_EXPOSE [[nodiscard]] Tristate is_disjoint(SpaceId first, SpaceId second) const {
        std::scoped_lock const guard(_mutex);
        std::size_t const      a = check(first, "is_disjoint");
        std::size_t const      b = check(second, "is_disjoint");

        if (a == b) {
            return Tristate::No;
        }
        if (derived_disjoint(a, b)) {
            return Tristate::Yes;
        }
        if (_contained[a][b] != 0 || _contained[b][a] != 0) {
            return Tristate::No;
        }
        return Tristate::Unknown;
    }

    /**
     * @brief Whether every element of @p inner is an element of @p outer.
     * @param[in] inner The candidate contained space.
     * @param[in] outer The candidate containing space.
     * @return Yes when the two are the same space or the closure of the declared containments
     *         relates them, No when @p outer is instead contained in @p inner or the two are known
     *         disjoint, Unknown otherwise.
     * @throws std::invalid_argument if either id is invalid.
     */
    APIARY_EXPOSE [[nodiscard]] Tristate is_contained(SpaceId inner, SpaceId outer) const {
        std::scoped_lock const guard(_mutex);
        std::size_t const      a = check(inner, "is_contained");
        std::size_t const      b = check(outer, "is_contained");

        if (a == b) {
            return Tristate::Yes;
        }
        if (_contained[a][b] != 0) {
            return Tristate::Yes;
        }
        if (_contained[b][a] != 0 || derived_disjoint(a, b)) {
            return Tristate::No;
        }
        return Tristate::Unknown;
    }

    /**
     * @brief Attach machine policy to a space name.
     * @param[in] name The space name. It need not be registered: policy is configuration and may be
     *            set before the graph that names the space is built.
     * @param[in] policy The policy to store, replacing any previous entry.
     * @throws std::invalid_argument if the name is empty.
     */
    void set_policy(std::string_view name, SpacePolicy policy) {
        std::scoped_lock const guard(_mutex);
        if (name.empty()) {
            throw std::invalid_argument("SpaceRegistry::set_policy: a space name must not be empty");
        }
        _policies[std::string(name)] = policy;
    }

    /**
     * @brief The policy attached to a space name.
     * @param[in] name The space name.
     * @return The stored policy, or a default-constructed one (both hints undecided) when none was
     *         set.
     */
    [[nodiscard]] SpacePolicy policy(std::string_view name) const {
        std::scoped_lock const guard(_mutex);
        auto const             it = _policies.find(std::string(name));
        if (it == _policies.end()) {
            return SpacePolicy{};
        }
        return it->second;
    }

  private:
    /// Square boolean relation matrix, one row and column per registered space.
    using Matrix = std::vector<std::vector<std::uint8_t>>;

    /**
     * @brief Validate an id and turn it into a storage index.
     * @param[in] id The id to check.
     * @param[in] what Name of the calling operation, for the message.
     * @return The storage index.
     * @throws std::invalid_argument if the id is invalid or out of range.
     */
    [[nodiscard]] std::size_t check(SpaceId id, char const *what) const {
        if (!id.valid() || id.value() >= _spaces.size()) {
            throw std::invalid_argument(std::string("SpaceRegistry::") + what + ": the space id is not valid for this registry");
        }
        return static_cast<std::size_t>(id.value());
    }

    /// @brief Resize every relation matrix to hold @p n spaces.
    /// @param[in] n The new number of spaces.
    void grow_relations(std::size_t n) {
        auto grow = [n](Matrix &m) {
            m.resize(n);
            for (auto &row : m) {
                row.resize(n, 0);
            }
        };
        grow(_less);
        grow(_contained);
        grow(_disjoint);
    }

    /**
     * @brief Reflexive-transitive reachability in a closed relation matrix.
     * @param[in] m The closure matrix.
     * @param[in] from Source index.
     * @param[in] to Target index.
     * @return True when @p from equals @p to or the closure holds between them.
     */
    [[nodiscard]] static bool reaches(Matrix const &m, std::size_t from, std::size_t to) noexcept { return from == to || m[from][to] != 0; }

    /**
     * @brief Add one edge to a closure matrix and re-close it.
     * @param[in,out] m The closure matrix.
     * @param[in] a Source index.
     * @param[in] b Target index.
     *
     * The caller must already have rejected the case where @p b reaches @p a, so the diagonal stays
     * clear and the stored relation stays strict.
     */
    static void close_edge(Matrix &m, std::size_t a, std::size_t b) {
        std::size_t const n = m.size();
        for (std::size_t i = 0; i < n; ++i) {
            if (!reaches(m, i, a)) {
                continue;
            }
            for (std::size_t j = 0; j < n; ++j) {
                if (reaches(m, b, j)) {
                    m[i][j] = 1;
                }
            }
        }
    }

    /**
     * @brief Whether disjointness follows from the declared pairs plus containment on either side.
     * @param[in] a One index.
     * @param[in] b The other index.
     * @return True when some space containing @p a is declared disjoint from some space containing
     *         @p b, where "containing" includes the space itself.
     *
     * Caller holds the mutex.
     */
    [[nodiscard]] bool derived_disjoint(std::size_t a, std::size_t b) const {
        std::size_t const n = _spaces.size();
        for (std::size_t x = 0; x < n; ++x) {
            if (!reaches(_contained, a, x)) {
                continue;
            }
            for (std::size_t y = 0; y < n; ++y) {
                if (_disjoint[x][y] != 0 && reaches(_contained, b, y)) {
                    return true;
                }
            }
        }
        return false;
    }

    /**
     * @brief Reject a candidate pair of relation tables that would contradict itself.
     * @param[in] contained Candidate containment closure.
     * @param[in] disjoint Candidate declared-disjointness table.
     * @param[in] what Name of the calling operation, for the message.
     * @throws std::invalid_argument when some space is contained in both members of a disjoint
     *         pair, which is the single invariant tying the two relations together.
     */
    void validate(Matrix const &contained, Matrix const &disjoint, char const *what) const {
        std::size_t const n = _spaces.size();
        for (std::size_t x = 0; x < n; ++x) {
            for (std::size_t y = x + 1; y < n; ++y) {
                if (disjoint[x][y] == 0) {
                    continue;
                }
                for (std::size_t z = 0; z < n; ++z) {
                    if (reaches(contained, z, x) && reaches(contained, z, y)) {
                        throw std::invalid_argument(std::string("SpaceRegistry::") + what + ": '" + _spaces[z].name +
                                                    "' would sit inside both '" + _spaces[x].name + "' and '" + _spaces[y].name +
                                                    "', which are disjoint");
                    }
                }
            }
        }
    }

    mutable std::mutex                             _mutex;
    std::deque<IndexSpace>                         _spaces;
    std::unordered_map<std::string, std::uint32_t> _by_name;
    std::unordered_map<std::string, SpacePolicy>   _policies;

    /// Transitive closure of the declared scale orderings. @c _less[i][j] means scale(i) < scale(j).
    Matrix _less;

    /// Transitive closure of the declared containments. @c _contained[i][j] means i is strictly inside j.
    Matrix _contained;

    /// Declared disjointness, symmetric. Derived disjointness is computed by @ref derived_disjoint.
    Matrix _disjoint;
};

EINSUMS_NAMESPACE_END(compute_graph)

namespace std {

/// @brief Hash support so a @ref einsums::compute_graph::SpaceId can key an unordered container.
template <>
struct hash<::einsums::compute_graph::SpaceId> {
    /// @brief Hash a space id.
    /// @param[in] id The id to hash.
    /// @return The hash of its underlying index.
    [[nodiscard]] std::size_t operator()(::einsums::compute_graph::SpaceId id) const noexcept {
        return std::hash<std::uint32_t>{}(id.value());
    }
};

} // namespace std
