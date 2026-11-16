//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/BLAS/Types.hpp>
#include <Einsums/Comm/Collectives.hpp>
#include <Einsums/ComputeGraph/Detail/MaintainedViews.hpp>
#include <Einsums/ComputeGraph/TensorBacking.hpp>
#include <Einsums/ComputeGraph/TensorRank.hpp>
#include <Einsums/ComputeGraphTypes/Enums.hpp>
#include <Einsums/ComputeGraphTypes/Ids.hpp>
#include <Einsums/ComputeGraphTypes/Spaces.hpp>
#include <Einsums/Concepts/Complex.hpp>
#include <Einsums/Concepts/TensorConcepts.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/PackedGemm/ContractionKey.hpp>
#include <Einsums/Python/Annotations.hpp>
#include <Einsums/Tensor/PendingInit.hpp>
#include <Einsums/Tensor/StorageBlock.hpp>
#include <Einsums/TensorBase/SymmetryDescriptor.hpp>
#include <Einsums/TensorBase/TensorBase.hpp>

#include <any>
#include <array>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <memory>
#include <numeric>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

/**
 * @brief What a tensor is, for a pass that has to recognize it rather than merely size it.
 *
 * A @ref name from an open vocabulary, plus optional key/value attributes for anything the name
 * alone cannot carry (which basis a set of integrals is over, which order a denominator is).
 * Attributes are kept SORTED by key, so two tags built in different orders compare equal and a
 * saved graph's bytes do not depend on the order a caller happened to set them in.
 *
 * @see TensorHandle::tag
 */
struct APIARY_EXPOSE APIARY_MODULE("graph") ProvenanceTag {
    /// The vocabulary name, or empty for untagged. Compared exactly, including case.
    APIARY_EXPOSE APIARY_READONLY std::string name;

    /// Sorted key/value attributes. Empty for a tag that needs no qualification.
    APIARY_EXPOSE APIARY_READONLY std::vector<std::pair<std::string, std::string>> attributes;

    /// @brief Build a tag carrying just a vocabulary name.
    ///
    /// A factory rather than a constructor, and the reason is that this type is an AGGREGATE:
    /// C++ code writes ``ProvenanceTag{.name = "eri"}``, and any user-declared constructor
    /// would end that. Python needs some way to make one, since the binding generator gives an
    /// exposed aggregate no constructor at all and the type was reachable but unbuildable.
    ///
    /// @param[in] name The vocabulary name.
    /// @return The tag.
    APIARY_EXPOSE [[nodiscard]] static ProvenanceTag make(std::string name) { return ProvenanceTag{.name = std::move(name)}; }

    /// @brief Build a tag with attributes.
    ///
    /// A separate name rather than a defaulted argument on @ref make, because the generator
    /// copies a default argument through verbatim and a braced empty one does not compile.
    ///
    /// @param[in] name The vocabulary name.
    /// @param[in] attributes Key/value qualifications.
    /// @return The tag.
    APIARY_EXPOSE [[nodiscard]] static ProvenanceTag make_with_attributes(std::string                                      name,
                                                                          std::vector<std::pair<std::string, std::string>> attributes) {
        return ProvenanceTag{.name = std::move(name), .attributes = std::move(attributes)};
    }

    /// @brief Whether this tag says anything at all.
    /// @return True when @ref name is non-empty.
    APIARY_EXPOSE APIARY_GETTER("valid") [[nodiscard]] bool valid() const noexcept { return !name.empty(); }

    /// @brief The value of one attribute.
    /// @param[in] key The attribute to look up.
    /// @return The value, or an empty optional when the tag does not carry that key.
    APIARY_EXPOSE [[nodiscard]] std::optional<std::string> attribute(std::string_view key) const {
        for (auto const &entry : attributes) {
            if (entry.first == key) {
                return entry.second;
            }
        }
        return std::nullopt;
    }

    /// @brief Structural equality, which the sorted attribute order makes meaningful.
    /// @param[in] lhs Left operand.
    /// @param[in] rhs Right operand.
    /// @return True when the names and the attribute lists match exactly.
    [[nodiscard]] friend bool operator==(ProvenanceTag const &lhs, ProvenanceTag const &rhs) = default;
};

/**
 * @brief Build a @ref ProvenanceTag.
 *
 * A named constructor rather than a real one, for the same reason @ref make_index_space is one:
 * @ref ProvenanceTag stays an aggregate so the library's designated-initializer construction
 * sites keep working, and this is what a caller with no aggregate initialization builds one
 * through.
 *
 * @param[in] name       The vocabulary name.
 * @param[in] attributes Key/value pairs, possibly empty. Sorted by @ref Graph::annotate_tag on
 *                       the way in, so the order given here does not matter.
 * @return The tag.
 *
 * Both parameters are required rather than the second defaulted, which is a binding constraint
 * rather than a preference: the codegen renders a ``= {}`` default as a bare brace-init that
 * pybind's ``py::arg`` cannot be assigned from. The Python-side helper in ``einsums.graph``
 * supplies the empty list, so a caller there still writes just the name.
 * @versionadded{2.0.0}
 */
[[nodiscard]] APIARY_EXPOSE APIARY_MODULE("graph") APIARY_RENAME("provenance_tag") inline ProvenanceTag
make_provenance_tag(std::string name, std::vector<std::pair<std::string, std::string>> attributes) {
    return ProvenanceTag{.name = std::move(name), .attributes = std::move(attributes)};
}

/**
 * @brief The tag name for a Kronecker delta, the identity over one index space.
 *
 * Spelled once here rather than as a string literal at each site, because a pass matching a
 * misspelled name silently does nothing and the tag vocabulary is deliberately unvalidated.
 */
inline constexpr std::string_view provenance_identity = "identity";

namespace detail {

/// True when @p token names no control block at all, as opposed to naming one
/// that has since expired. The distinction matters: an untracked handle always
/// matches, while an expired one names a tensor that is gone.
inline bool untracked(std::weak_ptr<void> const &token) noexcept {
    static std::weak_ptr<void> const none;
    return !token.owner_before(none) && !none.owner_before(token);
}

/// Whether two liveness tokens name the same tensor. An untracked token on
/// either side matches anything, which keeps tensor types that expose no token
/// (and handles registered before capture) on the pre-existing behaviour.
inline bool same_tensor(std::weak_ptr<void> const &a, std::weak_ptr<void> const &b) noexcept {
    if (untracked(a) || untracked(b)) {
        return true;
    }
    return !a.owner_before(b) && !b.owner_before(a);
}

/// @brief The liveness token @p tensor publishes, or an untracked token when its type
///        publishes none.
///
/// Every by-address lookup pairs a pointer with one of these, because an address does not
/// identify a tensor across a destruction. The types that expose no token keep the
/// pre-existing behaviour through @ref same_tensor, which matches an untracked token
/// against anything.
///
/// A view a graph's View node maintains answers with its HOLDER's token instead of its own: the
/// node re-emplaces the view on every replay, so the instance's token expires while the view
/// lives on, in this graph and in any loop body that names it (Detail/MaintainedViews.hpp).
template <typename TensorType>
[[nodiscard]] std::weak_ptr<void> liveness_token_of(TensorType const &tensor) noexcept {
    if constexpr (requires { tensor.liveness_token(); }) {
        try {
            if (auto life = maintained_view_life(static_cast<void const *>(&tensor))) {
                return *life;
            }
        } catch (...) { // NOLINT(bugprone-empty-catch): a failed lookup falls back to the instance's own token
        }
        return tensor.liveness_token();
    } else {
        return {};
    }
}

/**
 * @brief Half-open BYTE span of a strided buffer.
 *
 * The one place the extent of a dense operand's storage is computed. Strides are
 * unsigned, so every element the operand addresses lies at an offset in
 * ``[0, sum (dim - 1) * stride]`` whatever its axis order and however its own axes
 * overlap each other; the span is that range plus one element, in bytes.
 *
 * Shared on purpose. ``Graph::link_alias_storage`` reasons about containment between
 * two such spans and ``Graph::bind`` rejects an undeclared OVERLAP between two of
 * them, and the bug history of this module is two derivations of one relation
 * disagreeing, so there is exactly one of these.
 *
 * @param[in]  data         Base address, or null when the operand has none (a deferred
 *                          shell, a tile-wise sparse tensor).
 * @param[in]  dims         Extents, one per axis.
 * @param[in]  strides      Strides in ELEMENTS, one per axis, parallel to @p dims.
 * @param[in]  element_size Size of one element in bytes.
 * @param[out] lo           First byte of the span.
 * @param[out] hi           One past the last byte.
 * @return False, leaving @p lo and @p hi untouched, when the operand has no span that
 *         can be reasoned about: no address, no axes, a zero extent (which addresses
 *         nothing at all), a zero element size, or a strides/dims length disagreement.
 *
 * @versionadded{2.0.0}
 */
inline bool strided_byte_span(void const *data, std::span<std::size_t const> dims, std::span<std::size_t const> strides,
                              std::size_t element_size, char const *&lo, char const *&hi) noexcept {
    if (data == nullptr || element_size == 0 || dims.empty() || strides.size() != dims.size()) {
        return false;
    }
    std::size_t last = 0;
    for (std::size_t d = 0; d < dims.size(); ++d) {
        if (dims[d] == 0) {
            return false;
        }
        last += (dims[d] - 1) * strides[d];
    }
    lo = static_cast<char const *>(data);
    hi = lo + (last + 1) * element_size;
    return true;
}

} // namespace detail

/**
 * @brief A tensor the graph can capture into a handle.
 *
 * Either a dense, in-core basic tensor (GeneralTensor, TensorView,
 * RuntimeTensor, ...) or a tile-wise sparse in-core tensor
 * (TiledRuntimeTensor). The latter does not satisfy BasicTensorConcept (a
 * tiled tensor isn't a single-buffer tensor), but it still exposes the metadata
 * surface make_handle needs, name/rank/dim/stride/data (with data() null for
 * the multi-tile case), so we admit it explicitly here.
 */
template <typename D>
concept GraphCapturableTensor =
    CoreBasicTensorConcept<D> || (IsIncoreTensorV<std::remove_cvref_t<D>> && TiledTensorConcept<D> && requires(D t) {
        t.data();
        t.stride(0);
        t.strides();
        t.rank();
        t.name();
    });

/**
 * @brief Unique identifier for a tensor within a computation graph.
 *
 * Each tensor registered with a Graph receives a unique TensorId.
 * These IDs are used in Node::inputs and Node::outputs to express
 * data dependencies between operations.
 */
// TensorId, AllocState, InitKind, Residency are defined in ComputeGraphTypes module
// (included via Einsums/ComputeGraphTypes/Enums.hpp and Ids.hpp above)

/**
 * @brief What one tensor type can do, as plain functions over a pointer to the tensor object.
 *
 * One table per tensor type, built at compile time by ``tensor_ops_for``, and shared by every
 * handle naming a tensor of that type. Its entries are called on @ref TensorHandle::live_ptr, read
 * at call time, so repointing a handle at another tensor is repointing two pointers, with nothing
 * bound to the old object left behind.
 *
 * An entry is null when the type lacks the method behind it, decided once per type at compile
 * time. The object pointer is
 * always the type the table was built for; nothing checks it.
 *
 * Tables are per shared library (each one that instantiates a type has its own copy), so compare
 * the entries, never the table addresses.
 */
struct TensorOps {
    /// The tensor's rank-erased ``detail::TensorImpl<T>``, type-erased; @ref TensorHandle::impl calls it.
    void *(*impl)(void *object){nullptr};
    /// Whether the tensor has storage now. Null means "always materialized".
    bool (*is_materialized)(void const *object){nullptr};
    void (*materialize)(void *object){nullptr};
    /// Attach caller-provided storage, typed as the tensor's element.
    void (*materialize_into)(void *object, void *storage){nullptr};
    void (*release)(void *object){nullptr};
    /// The tensor's own storage block. Set only for owning types with a releasable block.
    einsums::detail::StorageBase *(*storage)(void *object){nullptr};
    /// Materialize, then zero.
    void (*fill_zero)(void *object){nullptr};
    /// Materialize, then fill uniformly on ``[-1, 1)`` through ``std::rand``, which is what every
    /// ``declare_random_*`` has always given: scratch whose values nothing reads for their distribution.
    void (*fill_random)(void *object){nullptr};
    /// Give a deferred tensor new extents before it is allocated.
    void (*resize_deferred)(void *object, std::span<std::size_t const> dims){nullptr};
    /// Record a distributed tensor's global extents and this rank's offsets.
    void (*set_distribution)(void *object, std::span<std::size_t const> global_dims, std::span<std::size_t const> offsets){nullptr};
    void (*allreduce_sum)(void *object){nullptr};
    comm::Request (*iallreduce_sum)(void *object){nullptr};
    /// The symmetry the tensor declares now, or null when it declares none.
    std::shared_ptr<SymmetryDescriptor const> (*declared_symmetry)(void const *object){nullptr};
    void (*set_symmetry)(void *object, SymmetryDescriptor descriptor){nullptr};
    /// Restrict the tensor to a slice; the returned state is the caller's, to hand to
    /// ``end_local_view``.
    std::any (*begin_local_view)(void *object, std::size_t dim, std::size_t start, std::size_t count){nullptr};
    void (*end_local_view)(void *object, std::any &state){nullptr};
    std::string const &(*name)(void const *object){nullptr};
};

namespace detail {

/// The table a default-constructed handle points at: every entry null.
inline constexpr TensorOps no_tensor_ops{};

template <typename TensorType>
consteval TensorOps make_tensor_ops() {
    using Clean = std::remove_cvref_t<TensorType>;
    using Value = typename Clean::ValueType;
    TensorOps ops;

    if constexpr (requires(Clean &t) { t.impl(); }) {
        ops.impl = [](void *object) -> void * { return static_cast<void *>(&static_cast<Clean *>(object)->impl()); };
    }
    if constexpr (requires(Clean const &t) { t.is_materialized(); }) {
        ops.is_materialized = [](void const *object) -> bool { return static_cast<Clean const *>(object)->is_materialized(); };
    }
    if constexpr (requires(Clean &t) { t.materialize(); }) {
        ops.materialize = [](void *object) { static_cast<Clean *>(object)->materialize(); };
    }
    if constexpr (requires(Clean &t, Value *p) { t.materialize_into(p); }) {
        ops.materialize_into = [](void *object, void *storage) {
            static_cast<Clean *>(object)->materialize_into(static_cast<Value *>(storage));
        };
    }
    if constexpr (requires(Clean &t) { t.release(); }) {
        ops.release = [](void *object) { static_cast<Clean *>(object)->release(); };
        if constexpr (requires(Clean &t) { t.storage().get(); }) {
            ops.storage = [](void *object) -> einsums::detail::StorageBase * { return static_cast<Clean *>(object)->storage().get(); };
        }
    }
    if constexpr (requires(Clean &t) {
                      t.materialize();
                      t.zero();
                  }) {
        ops.fill_zero = [](void *object) {
            auto *tensor = static_cast<Clean *>(object);
            tensor->materialize();
            tensor->zero();
        };
    }
    if constexpr (requires(Clean &t) {
                      t.materialize();
                      t.data();
                      t.size();
                  }) {
        ops.fill_random = [](void *object) {
            auto *tensor = static_cast<Clean *>(object);
            tensor->materialize();
            auto        *data = tensor->data();
            size_t const size = tensor->size();
            for (size_t idx = 0; idx < size; idx++) {
                // NOLINTNEXTLINE(misc-predictable-rand)
                data[idx] = static_cast<Value>(static_cast<double>(std::rand()) / RAND_MAX * 2.0 - 1.0);
            }
        };
    }
    if constexpr (requires(Clean &t, std::vector<std::size_t> const &dims) { t.resize_deferred(dims); }) {
        ops.resize_deferred = [](void *object, std::span<std::size_t const> dims) {
            static_cast<Clean *>(object)->resize_deferred(std::vector<std::size_t>(dims.begin(), dims.end()));
        };
    } else if constexpr (requires(Clean &t) {
                             Clean::Rank;
                             t.resize_deferred(::einsums::Dim<Clean::Rank>{});
                         }) {
        ops.resize_deferred = [](void *object, std::span<std::size_t const> dims) {
            ::einsums::Dim<Clean::Rank> d;
            for (size_t i = 0; i < Clean::Rank && i < dims.size(); i++) {
                d[i] = dims[i];
            }
            static_cast<Clean *>(object)->resize_deferred(d);
        };
    }
    if constexpr (requires(Clean &t, std::array<std::size_t, Clean::Rank> a) { t.set_distribution(a, a); }) {
        ops.set_distribution = [](void *object, std::span<std::size_t const> global_dims, std::span<std::size_t const> offsets) {
            std::array<std::size_t, Clean::Rank> gd{}, off{};
            for (size_t i = 0; i < Clean::Rank && i < global_dims.size(); i++) {
                gd[i] = global_dims[i];
            }
            for (size_t i = 0; i < Clean::Rank && i < offsets.size(); i++) {
                off[i] = offsets[i];
            }
            static_cast<Clean *>(object)->set_distribution(gd, off);
        };
    }
    ops.allreduce_sum = [](void *object) {
        auto *tensor = static_cast<Clean *>(object);
        auto  span   = std::span<Value>(tensor->data(), tensor->size());
        (void)comm::allreduce_inplace<Value>(span, comm::ReduceOp::Sum);
    };
    ops.iallreduce_sum = [](void *object) -> comm::Request {
        auto *tensor = static_cast<Clean *>(object);
        auto  span   = std::span<Value>(tensor->data(), tensor->size());
        auto  result = comm::iallreduce_inplace<Value>(span, comm::ReduceOp::Sum);
        if (result.has_value()) {
            return std::move(result.value());
        }
        return comm::Request{};
    };
    if constexpr (requires(Clean const &t) { t.symmetry(); }) {
        ops.declared_symmetry = [](void const *object) -> std::shared_ptr<SymmetryDescriptor const> {
            auto const *desc = static_cast<Clean const *>(object)->symmetry();
            return desc != nullptr ? std::make_shared<SymmetryDescriptor const>(*desc) : nullptr;
        };
    }
    if constexpr (requires(Clean &t, SymmetryDescriptor d) { t.set_symmetry(std::move(d)); }) {
        ops.set_symmetry = [](void *object, SymmetryDescriptor descriptor) {
            static_cast<Clean *>(object)->set_symmetry(std::move(descriptor));
        };
    }
    if constexpr (requires(Clean &t) {
                      typename Clean::LocalViewState;
                      t.begin_local_view(size_t{}, size_t{}, size_t{});
                  }) {
        ops.begin_local_view = [](void *object, std::size_t dim, std::size_t start, std::size_t count) -> std::any {
            return std::any(static_cast<Clean *>(object)->begin_local_view(dim, start, count));
        };
        ops.end_local_view = [](void *object, std::any &state) {
            static_cast<Clean *>(object)->end_local_view(std::any_cast<typename Clean::LocalViewState &>(state));
        };
    }
    if constexpr (requires(Clean const &t) {
                      { t.name() } -> std::same_as<std::string const &>;
                  }) {
        ops.name = [](void const *object) -> std::string const & { return static_cast<Clean const *>(object)->name(); };
    }
    return ops;
}

template <typename T>
consteval TensorOps make_scalar_ops() {
    TensorOps ops;
    ops.allreduce_sum = [](void *object) {
        auto span = std::span<T>(static_cast<T *>(object), 1);
        (void)comm::allreduce_inplace<T>(span, comm::ReduceOp::Sum);
    };
    return ops;
}

} // namespace detail

/// The operations table of tensor type @p TensorType. @see TensorOps
template <typename TensorType>
inline constexpr TensorOps tensor_ops_for = detail::make_tensor_ops<TensorType>();

/// The operations table of a bare element of type @p T, which a returning-form op writes through.
/// Only the allreduce is defined. @see make_scalar_handle
template <typename T>
inline constexpr TensorOps scalar_ops_for = detail::make_scalar_ops<T>();

/// The operations table of an LU pivot array. Every entry is null: the array has no
/// ``TensorImpl``, the graph neither allocates nor frees it (its buffer is shared with the
/// ``LuPivots`` that named it, or owned by a loaded graph), and it is never reduced across ranks.
/// A distinct table rather than the empty default so that a pivot handle is told from a handle
/// nothing was ever registered for. @see make_pivot_handle
inline constexpr TensorOps lu_pivot_ops{};

/**
 * @brief Type-erased handle to a tensor, storing pointer and metadata.
 *
 * TensorHandle provides a uniform representation for tensors of any type
 * and rank within the computation graph. It stores:
 * - A void pointer to the actual Tensor object (not the raw data pointer)
 * - Dimensional metadata (rank, dims, strides, element size, dtype)
 * - A validation function to detect use-after-free at runtime
 *
 * @note TensorHandle does NOT own the tensor by default. The user is
 *       responsible for ensuring tensors outlive the graph, or should
 *       use Graph::create_tensor() for graph-owned intermediates.
 *
 * @see make_handle() to construct a TensorHandle from a typed tensor
 * @see Graph::create_tensor() for graph-owned tensor creation
 */
struct TensorHandle {
    /**
     * @brief The tensor object this handle names, and the handle's identity.
     *
     * Two handles name the same tensor exactly when this is equal, in one graph or across a
     * graph and its loop bodies, which is what every pass that compares tensors compares. It is
     * also the object it is legal to DEREFERENCE: it lives at least as long as the handle.
     *
     * For an operand capture adopted (see ``Graph::adopt_operand``) this is the graph's STAND-IN,
     * not the caller's tensor. The caller may destroy its tensor before the graph runs, and from
     * then on the caller's address can hold any later allocation; the stand-in's cannot while a
     * handle holds it. The caller's address is kept in @ref caller_ptr for lookups by the
     * caller's object. Two exceptions keep the caller's address here, both because it cannot be
     * reused while the handle could be confused with another: a tensor some graph owns (it lives as
     * long as that graph and its loop bodies), and a tensor type that publishes no liveness token
     * (nothing could tell a reuse apart).
     *
     * Not the tensor's ``data()`` pointer; that is @ref data_ptr.
     */
    void *tensor_ptr{nullptr};
    /**
     * @brief The caller's tensor, when it differs from @ref tensor_ptr; null otherwise.
     *
     * A lookup key, never an identity and never dereferenced: the caller may have destroyed it.
     * The graph's pointer index files the handle under it as well, checked against
     * @ref caller_token, so asking the graph about the caller's own object still finds the
     * handle while that object lives. @see lookup_ptr
     */
    void                   *caller_ptr{nullptr};
    void                   *data_ptr{nullptr};                       ///< Pointer to the raw data buffer (tensor.data())
    TensorId                id{0};                                   ///< Unique identifier assigned by Graph::register_tensor()
    std::string             name;                                    ///< Human-readable tensor name (copied from tensor at registration)
    size_t                  rank{0};                                 ///< Number of dimensions (e.g., 2 for a matrix)
    size_t                  element_size{0};                         ///< Size of one element in bytes (sizeof(ValueType))
    std::vector<size_t>     dims;                                    ///< Size of each dimension, in order
    std::vector<size_t>     strides;                                 ///< Stride of each dimension, in elements
    packed_gemm::ScalarType dtype{packed_gemm::ScalarType::Unknown}; ///< Element type enum for runtime dispatch
    bool                    is_intermediate{false};                  ///< True if this tensor is owned by the graph (from create_tensor())
    bool                    is_tiled{false}; ///< True if a tile-wise sparse tensor (no single contiguous data() buffer)
    /// True when @ref tensor_ptr points at a bare element (the ``T *`` a returning-form op such as
    /// a dot writes through) rather than at a tensor object. Rank 0 either way, and the two are not
    /// interchangeable: an executor built for one writes through the pointer as the other.
    bool raw_scalar{false};
    /// True when @ref tensor_ptr points at the ``std::vector<blas::int_t>`` a ``LuPivots`` shares:
    /// the row interchanges a getrf writes and a getrs reads. It is an operand so the schedulers
    /// see that edge through the node lists, and it is not a tensor: its elements are BLAS
    /// integers, which no ``packed_gemm::ScalarType`` names, so @ref dtype is Unknown, it has no
    /// ``TensorImpl``, and nothing binds, frees or allocates it. It is rank 1 with the order of the
    /// factored matrix as its one extent. See @ref make_pivot_handle.
    bool lu_pivots{false};
    bool is_runtime{false}; ///< True if tensor_ptr points at a GeneralRuntimeTensor<T> (passes that cast tensor_ptr to a
                            ///< runtime-tensor type MUST gate on this: statically-typed Tensor<T, Rank> captures pass through
                            ///< the same handles, and a blind cast is type confusion)
    /// True when @ref tensor_ptr points at a @c RuntimeTensorView<T> rather than at an owning
    /// @c GeneralRuntimeTensor<T>.
    ///
    /// @ref is_runtime says the object is rank-erased and nothing more, and the two rank-erased
    /// types have different layouts, so a pass that has to name the STATIC type in order to call
    /// a templated capture entry point needs this second bit as well. Reading a tensor's data
    /// needs neither: @ref impl answers that for both without asking.
    bool is_tensor_view{false};
    bool is_distributed{false}; ///< True if this tensor is distributed across ranks
    /// True for a deferred intermediate a library helper declared on its caller's behalf, such as a
    /// partial product of an einsum over three or more operands. The caller never sees it, so a
    /// graph run without the Materialization pass gives it storage at its first execute when a node
    /// still reads or writes it and no Materialize does; a tensor the caller declared is refused
    /// there instead. See @ref Graph::declare_internal_tensor.
    bool materialize_on_execute{false};
    bool is_replicated{true}; ///< True if distributed tensor is replicated on all ranks

    /**
     * @brief Which scope owns this tensor's storage.
     *
     * @ref TensorOwnership::Graph, the default, means "supplied to (or created by) this
     * graph and scoped to it". @ref TensorOwnership::Pipeline means the tensor carries
     * state between a pipeline's stages, and @ref TensorOwnership::Workspace means it
     * outlives every pipeline over it. Written at the declaring site -
     * ``Workspace::declare_*``, ``Pipeline::declare_*``, ``Graph::declare_tensor`` - and
     * carried to a capturing graph's own handle through @ref Graph::add_scope_map.
     *
     * Orthogonal to @ref is_intermediate, but only in one direction: an intermediate is
     * always graph-owned by construction, so ``is_intermediate == true`` implies
     * ``ownership == Graph``, and such a handle never appears in @ref InterfaceManifest.
     * The reverse does not hold - a graph-scoped operand a caller supplied is
     * ``Graph``-owned and IS a manifest entry.
     *
     * Handles minted fresh inside a Loop body or a Conditional branch keep the default,
     * as every other piece of handle metadata does at that boundary. See the
     * MetadataBoundary contract.
     */
    TensorOwnership ownership{TensorOwnership::Graph};

    /**
     * @brief Set by a rebind, or a bind by name: the slot now holds the caller's tensor, as it is.
     *
     * The bound tensor's contents are its value, so the graph's lifecycle for the slot lapses. A
     * Materialize, Initialize or Free node built before the rebind does nothing when it runs, since
     * allocating, filling and freeing belong to storage the graph owns, and a Free acting on the
     * bound tensor would release the caller's. A rebind also resets @ref init_kind and
     * @ref alloc_state from the bound tensor, so the passes give it no new lifecycle nodes.
     */
    bool caller_bound{false};

    /**
     * @brief How the slot was declared, before a rebind handed it to the caller's tensor.
     *
     * Meaningful only while @ref caller_bound is set, and recorded by the first rebind that
     * redefines the slot, so a second one does not overwrite it with the first binding. A saved
     * graph writes these rather than @ref init_kind and @ref alloc_state: the file describes the
     * graph, and a loaded graph holds no binding, so a result declared at zero must come back
     * deferred and zero-initialized whatever this process bound it to before saving.
     */
    struct Declaration {
        AllocState alloc_state{AllocState::Materialized};
        InitKind   init_kind{InitKind::None};
    } declared;

    AllocState alloc_state{AllocState::Materialized}; ///< Whether data is allocated (Materialized) or deferred
    InitKind   init_kind{InitKind::None};             ///< How to initialize after materialization
    Residency  residency{Residency::Host};            ///< Where the tensor data currently lives (updated by GPU passes)

    /// Whether @ref Graph::execute has leased this intermediate's storage, so it does it once.
    bool scratch_leased{false};

    /// Whether this graph may give the deferred tensor new extents (a bind's derived extents, a
    /// relayout, a distribution's local shape). Set by the declarations that give the graph the
    /// tensor's shape: ``Graph::declare_tensor``, the deferred runtime declarations, and the
    /// loader. Never set for a tensor another scope declared, whose storage other stages share:
    /// that the tensor's type could be resized does not make it this graph's to resize.
    bool resizable{false};

    /// The caller's fill for an @ref InitKind::Custom tensor, called on @ref live_ptr once the
    /// tensor is materialized; it materializes first. The one initializer that is code rather than
    /// a property of the tensor's type, so a graph holding one cannot be saved.
    /// @see Graph::declare_tensor_filled
    std::function<void(void *)> custom_init;

    /// Distribution metadata (set by DistributionPlanningPass, read by MaterializationPass).
    /// Stores the index of the dimension to block-distribute as shared_ptr<size_t>.
    std::shared_ptr<void> distribution_info;

    /// Declared or inferred tensor symmetry.
    ///
    /// At registration time ``make_handle()`` reads the backing tensor's
    /// ``.symmetry()`` and stores a copy here, and every pass pipeline reads it
    /// again before its first pass (@ref Graph::refresh_declared_symmetry), so a
    /// symmetry declared on a tensor the graph already holds is seen. The
    /// ``SymmetryPropagation`` pass may later infer additional symmetry and
    /// update this hint plus the backing tensor (through @ref TensorOps::set_symmetry).
    /// ``nullptr`` means "no declared symmetry", so downstream dispatch falls
    /// through.
    std::shared_ptr<SymmetryDescriptor const> symmetry_hint;

    /**
     * @brief Per-axis index-space annotation, parallel to @ref dims.
     *
     * ``spaces[i]`` names the set axis ``i`` ranges over: occupied orbitals, virtuals, an
     * auxiliary basis, a grid. It is what lets a pass reason about how a family of problems
     * GROWS, where @ref dims only says how big this one instance is.
     *
     * Two states are legal and nothing else: EMPTY, meaning unannotated, which is the default
     * and is exactly the behaviour that predates annotations; or exactly @ref rank entries.
     * ``Graph::annotate_spaces`` is the only supported writer and it enforces that, so nothing
     * on a hot path has to re-check the size.
     *
     * ``Graph::annotate_spaces`` rejects an invalid @ref SpaceId outright, so a populated
     * annotation resolves end to end. A reader that encounters one anyway is to treat that axis
     * as unannotated rather than to fail.
     *
     * @see Graph::annotate_spaces
     * @see Graph::tensor_spaces
     */
    std::vector<SpaceId> spaces;

    /**
     * @brief Whether @ref spaces was inferred rather than declared.
     *
     * False for an annotation a user wrote with ``Graph::annotate_spaces``, and for the default
     * (empty) state. True when the annotation was derived: by capture, from the letters of the
     * contraction that first writes this intermediate, or by the ``SpacePropagation`` pass, from
     * the annotations on a producing node's operands.
     *
     * A declaration is authoritative and an inference is only as good as the declarations it came
     * from, which is exactly the distinction ``CrossSpaceValidation`` needs in order to report a
     * weaker verdict on an inferred slot rather than an authoritative wrong one. It is also what
     * lets propagation refine its own earlier guess while never overwriting a declaration.
     *
     * @see Graph::annotate_spaces
     */
    bool spaces_inferred{false};

    /**
     * @brief What this tensor IS, as a name from an open vocabulary plus free-form attributes.
     *
     * Index spaces say how big a tensor's axes are and what they range over, which is enough to
     * cost a contraction and not nearly enough to recognize one. A pass that wants to know a
     * tensor is a Kronecker delta, an electron-repulsion integral, a Coulomb metric or an energy
     * denominator is asking a question about PROVENANCE, and no amount of extent information
     * answers it. This field is that answer.
     *
     * @par Declared, never derived from values
     * A tag is a statement by whoever built the graph, and the alternative was considered and
     * rejected rather than overlooked: a pass could look at a tensor's CONTENTS and notice it
     * happens to be an identity matrix. That would be wrong here in a way that is easy to miss.
     * A structural-algebraic rewrite's output is what a saved graph keeps, and a later bind may
     * supply a different tensor under the same manifest name, so a rewrite justified by today's
     * contents would be baked into a file and wrong on the next problem. The whole reason the
     * phase rule exists is to stop exactly that, and reading values would route around it.
     *
     * @par The vocabulary is open
     * Names are strings and nothing validates them against a list, because the set of things
     * worth recognizing grows with the passes that recognize them, and a closed enum would mean
     * a library change for every new one. The cost is that a typo is a tag nobody matches; a
     * pass looking for a name it never finds reports it through its skip tally rather than
     * silently doing nothing.
     *
     * @par Propagation
     * A tag travels only through operations that preserve IDENTITY - a view, a permute, a copy -
     * and is never inferred through a contraction. The permute of a delta is still a delta; the
     * contraction of two integrals is not an integral. ``ProvenancePropagation`` implements
     * exactly that rule and nothing more permissive.
     *
     * Empty @ref ProvenanceTag::name means untagged, which is the default and the state of every
     * tensor in a program that has not been annotated.
     *
     * @see Graph::annotate_tag
     * @see Graph::tensor_tag
     */
    ProvenanceTag tag;

    /**
     * @brief Where the tensor's data lives when the graph reads it rather than holding it.
     *
     * Null for every tensor not annotated through @ref Graph::annotate_backing. Set, the tensor
     * is deferred, never written by the graph, and filled from the backing's reader: whole by the
     * Initialize node Materialization emits, or a slab at a time by a pass that streams it.
     *
     * Kept apart from @ref tag because the two answer different questions and change
     * differently. A tag says what the tensor IS and follows it through a view, a permute or a
     * copy; a backing says where it LIVES, and a copy of a backed tensor lives in memory. It is
     * therefore never propagated, and like every other piece of handle metadata it does not cross
     * into a loop body's fresh handle: a pass that needs it inside a body resolves it by
     * @ref tensor_ptr through the enclosing graphs.
     *
     * @see Graph::annotate_backing
     */
    std::shared_ptr<TensorBacking const> backing;

    /**
     * @brief Per-axis symbolic extent declaration, parallel to @ref dims.
     *
     * What makes a captured graph valid for a FAMILY of problem sizes rather than for the
     * one geometry it was captured at. @ref dims says how big this instance is;
     * ``dim_symbols[i]`` says whether axis @c i is allowed to be a different size in the
     * next problem, and which other axes have to move with it.
     *
     * Two states are legal and nothing else: EMPTY, meaning every axis is literal, which
     * is the default and is exactly the behaviour that predates symbolic extents; or
     * exactly @ref rank entries. ``Graph::annotate_dims`` is the only supported writer and
     * it enforces that.
     *
     * Three spellings per entry, all NAME-based so they survive a save unchanged:
     *
     * - The empty string: a LITERAL axis. Its extent is part of the contract and
     *   ``Graph::bind`` requires an exact match, as ``Graph::rebind`` always has.
     * - Any other ordinary string: a SYMBOL. Every axis in the graph carrying that symbol
     *   has one extent, solved from whatever a bind supplies and checked for agreement
     *   across every slot that names it.
     * - ``"ragged:<space>"``: a RAGGED axis, whose extent differs per instance of
     *   @c \<space\> (a PNO domain has a different virtual extent per pair, so no single
     *   symbol describes it). Such an axis constrains nothing across slots; the per-instance
     *   extents arrive separately through ``Graph::bind_ragged_extents``. Use
     *   @ref make_ragged_symbol rather than spelling the prefix by hand.
     *
     * A symbol may also be TIED to an index space, when the same axis carries a @ref spaces
     * annotation. The tie is graph-level state, not handle state, because its purpose is to
     * catch one symbol meaning two different spaces in two places; see
     * ``Graph::symbol_spaces``.
     *
     * @see Graph::annotate_dims
     * @see Graph::annotate_ragged_dim
     * @see Graph::bind
     */
    std::vector<std::string> dim_symbols;

    /**
     * @brief Hash of the tensor name at registration time.
     *
     * The canary @ref alive compares against the tensor's current name for a type that
     * publishes no liveness token.
     */
    size_t name_hash{0};

    /**
     * @brief Aliasing parent, set when this handle represents a non-owning
     *        view of another tensor.
     *
     * Populated by ``cg::view()`` (the @c View op): the slice's TensorHandle
     * has ``aliases == parent_id``. Storage allocation is the parent's
     * responsibility: the Alloc/Free passes skip handles with ``aliases``
     * set, the Lifetime pass extends the parent's live range to cover all
     * uses of any of its aliases, and the InplaceOptimization /
     * scheduling passes treat reads/writes through an alias as touching
     * the parent.
     *
     * ``0`` means "not an alias" (no parent, own your own storage).
     */
    TensorId aliases{0};

    /**
     * @brief The region this alias covers, in the parent's axis space.
     *
     * One ``[lo, hi)`` interval per parent axis. Empty means "unknown", which
     * the hazard scan reads as the whole parent (conservative).
     *
     * ``cg::view()`` does not need this: its View node carries a
     * ViewDescriptor the scheduler reads the box from directly. It is for
     * views that reach the graph WITHOUT a View node, i.e. sliced outside a
     * capture and registered on first use, where the box is recovered from
     * the data pointer offset and strides (see Graph::link_alias_storage).
     * Without it every such view would conflict with every other slice of its
     * parent, which is correct but serializes work that is provably disjoint.
     */
    std::vector<std::pair<std::int64_t, std::int64_t>> alias_box;

    /**
     * @brief Liveness token of the CALLER's wrapper at registration time.
     *
     * An address does not identify a tensor across a capture that frees them.
     * A destroyed wrapper's address is immediately reusable, so a tensor
     * allocated on top of a dead one would inherit its TensorId from the
     * address-keyed caches and every node referring to it would silently
     * operate on the wrong operand. This token distinguishes "same tensor" from
     * "same address": a recycled address arrives with a different token.
     *
     * Empty for handles registered outside capture (``create_tensor`` /
     * ``declare_tensor``), whose wrapper the graph owns and cannot recycle, and
     * for tensor types that expose no token. Both are treated as always
     * matching, which is the behaviour that predates the check.
     *
     * @see einsums::compute_graph::detail::untracked
     */
    std::weak_ptr<void> caller_token;

    /**
     * @brief The graph's stand-in for this operand, when it adopted one.
     *
     * Set by capture to a wrapper sharing the caller's storage that the graph
     * keeps alive (see ``Graph::adopt_operand``). @ref live_ptr names this
     * object, and the operand's TensorSlot points at it, which is
     * what lets the caller's own wrapper be destroyed before ``execute()``.
     *
     * Empty when the graph adopted nothing: a tensor it already owns, a
     * deferred tensor it will relocate, or a type with no storage block to
     * share. Those keep the older "operands must outlive the graph" contract.
     *
     * Cleared by a rebind: the handle then names the tensor it was rebound to,
     * and the slot's own reference keeps the stand-in alive for whatever was
     * built over it before.
     */
    std::shared_ptr<void> owner;

    /**
     * @brief The tensor object it is legal to DEREFERENCE at execute time.
     *
     * Usually @ref tensor_ptr itself. It differs where the identity stays the
     * caller's address although capture adopted the operand (a tensor another
     * graph owns, a type with no liveness token): the stand-in in @ref owner is
     * then the object the graph keeps alive, and the one to reach the storage
     * through. ``Graph::live_tensor_ptr`` is the by-id spelling of this and
     * defers to it.
     *
     * @return The stand-in when the handle holds one, otherwise @ref tensor_ptr.
     */
    [[nodiscard]] void *live_ptr() const noexcept { return owner ? owner.get() : tensor_ptr; }

    /**
     * @brief What this tensor's type can do; each entry is called on @ref live_ptr.
     *
     * Set by @ref make_handle and @ref make_scalar_handle for the type they were given, and by a
     * rebind for the type of the tensor it binds. A default handle points at a table of nulls.
     */
    TensorOps const *ops{&detail::no_tensor_ops};

    /// @brief Whether the tensor's type has a single rank-erased ``detail::TensorImpl<T>`` (a
    ///        tile-wise sparse tensor has none).
    [[nodiscard]] bool has_impl() const noexcept { return ops->impl != nullptr; }

    /// @brief The tensor's CURRENT ``detail::TensorImpl<T>``, type-erased, or null when
    ///        @ref has_impl is false. Cast it once the element type is known from @ref dtype.
    [[nodiscard]] void *impl() const { return ops->impl != nullptr ? ops->impl(live_ptr()) : nullptr; }

    /// @brief Whether the tensor has storage now, as its type reports it; empty for a type that
    ///        cannot say. @ref alloc_state is the graph's record, which this is the live answer to.
    [[nodiscard]] std::optional<bool> materialized_now() const {
        if (ops->is_materialized == nullptr) {
            return std::nullopt;
        }
        return ops->is_materialized(live_ptr());
    }

    /// @brief The address the caller knows this tensor by: @ref caller_ptr when set, else @ref tensor_ptr.
    [[nodiscard]] void *lookup_ptr() const noexcept { return caller_ptr != nullptr ? caller_ptr : tensor_ptr; }

    /**
     * @brief The lifetime of the object @ref live_ptr names, which @ref alive reads.
     *
     * Taken by @ref make_handle from the tensor it is given and by a rebind from the tensor it
     * binds. A graph-owned view's handle holds its holder's token instead, because the view
     * instance inside the holder is re-made on every replay. Untracked for a type that publishes
     * no token.
     *
     * Not @ref caller_token: that one watches the caller's wrapper, which may legitimately be
     * gone while the graph's stand-in keeps the storage alive.
     */
    std::weak_ptr<void> live_token;

    /**
     * @brief Whether the tensor this handle reads at execute time still exists.
     *
     * Definitive when @ref live_token is tracked: an expired token is a destroyed tensor, and
     * a live one is the tensor itself, whatever it has since been renamed to. A type with no
     * token gets a best-effort check instead, the tensor's current name against @ref name_hash,
     * which reads the object and so cannot be certain; a handle whose type has no name says yes.
     *
     * @return False when the tensor has been destroyed (or, untracked, appears to have been).
     */
    [[nodiscard]] bool alive() const noexcept {
        if (!detail::untracked(live_token)) {
            return !live_token.expired();
        }
        if (ops->name == nullptr) {
            return true;
        }
        try {
            return std::hash<std::string>{}(ops->name(live_ptr())) == name_hash;
        } catch (...) {
            return false;
        }
    }

    /**
     * @brief Compute the total element count of this tensor.
     * @return product(dims), or 0 if dims is empty.
     */
    [[nodiscard]] size_t total_elems() const {
        if (dims.empty())
            return 0;
        return std::accumulate(dims.begin(), dims.end(), size_t{1}, std::multiplies<>{});
    }

    /**
     * @brief Compute the total memory footprint of this tensor in bytes.
     * @return element_size * product(dims), or 0 if dims is empty.
     */
    [[nodiscard]] size_t total_bytes() const { return element_size * total_elems(); }
};

/**
 * @brief Construct a TensorHandle from a typed tensor.
 *
 * Extracts metadata (name, rank, dims, strides, dtype) from the tensor, points the handle at
 * its type's @ref TensorOps table, and takes the liveness token @ref TensorHandle::alive reads.
 *
 * @tparam TensorType Any type satisfying CoreBasicTensorConcept.
 * @param[in] tensor The tensor to create a handle for. The tensor must outlive the graph.
 * @param[in] id Initial ID (will be reassigned by Graph::register_tensor()).
 * @param[in] identity Address to record as @ref TensorHandle::tensor_ptr, when
 *            that must differ from @p tensor's own. Capture passes the caller's
 *            wrapper here while handing @p tensor the graph's stand-in (see
 *            ``Graph::adopt_operand``): the liveness token is then the stand-in's,
 *            an object the graph keeps alive, while ``tensor_ptr`` keeps naming the tensor
 *            the caller knows, which is what every identity comparison against a
 *            user-held tensor relies on. Null means "use @p tensor's address".
 * @return A fully populated TensorHandle.
 *
 * @code
 * auto A = create_random_tensor<double>("A", 10, 10);
 * auto handle = make_handle(A, 0);
 * // handle.name == "A", handle.rank == 2, handle.dims == {10, 10}
 * @endcode
 */
template <GraphCapturableTensor TensorType>
TensorHandle make_handle(TensorType const &tensor, TensorId id, void const *identity = nullptr) {
    TensorHandle h;
    h.tensor_ptr = const_cast<void *>(identity != nullptr ? identity : static_cast<void const *>(&tensor));
    h.ops        = &tensor_ops_for<TensorType>;

    // data_ptr may be nullptr for deferred (shell) tensors, that's expected.
    if constexpr (requires { tensor.is_materialized(); }) {
        h.data_ptr    = tensor.is_materialized() ? const_cast<void *>(static_cast<void const *>(tensor.data())) : nullptr;
        h.alloc_state = tensor.is_materialized() ? AllocState::Materialized : AllocState::Deferred;
    } else {
        h.data_ptr = const_cast<void *>(static_cast<void const *>(tensor.data()));
    }
    using CleanTensorEarly = std::remove_cvref_t<TensorType>;
    h.id                   = id;
    h.is_runtime           = std::is_base_of_v<tensor_base::RuntimeTensorNoType, CleanTensorEarly>;
    h.is_tensor_view       = std::is_base_of_v<tensor_base::RuntimeTensorViewNoType, CleanTensorEarly>;
    h.name                 = tensor.name();
    h.rank                 = detail::tensor_rank(tensor);
    h.element_size         = sizeof(typename std::remove_cvref_t<TensorType>::ValueType);
    h.dtype                = packed_gemm::get_scalar_type<typename std::remove_cvref_t<TensorType>::ValueType>();
    h.dims                 = detail::tensor_dims(tensor);
    h.strides              = detail::tensor_strides(tensor);

    // Tile-wise sparse tensors (TiledRuntimeTensor) advertise themselves so the
    // graph can flag the handle and keep them out of dense buffer-level passes;
    // their data_ptr is null (multi-tile) since there is no single buffer.
    if constexpr (requires { tensor.is_tiled_tensor(); }) {
        h.is_tiled = tensor.is_tiled_tensor();
    }

    // Post-materialize init: if the tensor was tagged with a pending-init
    // policy at declaration time (e.g. by ``Workspace::declare_zero_tensor``),
    // propagate that to the handle so the Materialization pass knows to emit
    // an Initialize node alongside the Materialize. ``pending_init()`` lives
    // on the tensor itself, not on workspace's _handles vector, so the
    // information survives capture into bodies the workspace doesn't own.
    using CleanTensor = std::remove_cvref_t<TensorType>;
    if constexpr (requires(CleanTensor const &t) { t.pending_init(); }) {
        switch (tensor.pending_init()) {
        case PendingInit::Zero:
            h.init_kind = InitKind::Zero;
            break;
        case PendingInit::Random:
            h.init_kind = InitKind::Random;
            break;
        case PendingInit::None:
            break;
        }
    }

    // Seed the symmetry hint from the backing tensor's current descriptor, if any; a pipeline
    // reads it again before its first pass (Graph::refresh_declared_symmetry).
    if constexpr (requires { tensor.symmetry(); }) {
        if (auto const *desc = tensor.symmetry())
            h.symmetry_hint = std::make_shared<SymmetryDescriptor>(*desc);
    }

    h.name_hash  = std::hash<std::string>{}(h.name);
    h.live_token = detail::liveness_token_of(tensor);

    return h;
}

/**
 * @brief Construct the handle a DEFERRED declaration registers.
 *
 * @tparam TensorType The tensor's type.
 * @param[in] tensor The shell tensor. Must outlive the handle.
 * @return The handle, ready for whatever the declaring site adds to it (its ownership scope,
 *         its intermediate flag, its @ref TensorHandle::init_kind, whether it is
 *         @ref TensorHandle::resizable).
 *
 * @ref make_handle, with the state DEFERRED because the declaration says so, not because the
 * tensor currently reads as unmaterialized. A zero-extent tensor reports itself materialized from
 * construction, and taking that answer would leave it outside the lifecycle its declaration
 * asked for. How the tensor is allocated, filled and freed is its type's @ref TensorOps table.
 *
 * @see make_handle
 * @versionadded{2.0.0}
 */
template <GraphCapturableTensor TensorType>
[[nodiscard]] TensorHandle make_deferred_handle(TensorType *tensor) {
    TensorHandle h = make_handle(*tensor, 0);
    h.alloc_state  = AllocState::Deferred;
    return h;
}

/**
 * @brief Construct a TensorHandle for a scalar value (rank-0 tensor).
 *
 * Scalars don't have dimensions or strides, but can participate in the graph
 * as inputs/outputs for operations like dot() or det() that return scalars.
 *
 * @tparam T An arithmetic type (double, float, int, etc.).
 * @param[in] scalar Pointer to the scalar value.
 * @param[in] id Initial ID (will be reassigned by Graph::register_tensor()).
 * @param[in] name Human-readable name for the scalar.
 * @return A TensorHandle with rank=0 and no dims/strides.
 */
template <typename T>
    requires(std::is_arithmetic_v<T> || IsComplexV<T>)
TensorHandle make_scalar_handle(T *scalar, TensorId id, std::string name = "scalar") {
    TensorHandle h;
    h.tensor_ptr   = scalar;
    h.id           = id;
    h.name         = std::move(name);
    h.rank         = 0;
    h.element_size = sizeof(T);
    h.dtype        = packed_gemm::get_scalar_type<T>();
    h.raw_scalar   = true;
    h.ops          = &scalar_ops_for<T>;

    return h;
}

/**
 * @brief Create a TensorHandle for an LU pivot array.
 *
 * The handle holds @p buffer in @ref TensorHandle::owner, so the array lives as long as any graph
 * that names it, whatever happens to the ``LuPivots`` it came from, and @ref TensorHandle::tensor_ptr
 * is the array's address: the identity a second capture against the same ``LuPivots`` finds.
 *
 * @param[in] buffer The shared pivot array.
 * @param[in] order The order of the matrix the pivots factor, ``min(m, n)``: the array's one extent.
 * @param[in] id Initial ID (will be reassigned by Graph::register_tensor()).
 * @param[in] name Human-readable name.
 * @return A rank-1 handle with @ref TensorHandle::lu_pivots set and an Unknown dtype.
 */
inline TensorHandle make_pivot_handle(std::shared_ptr<std::vector<blas::int_t>> buffer, std::size_t order, TensorId id,
                                      std::string name = "lu_pivots") {
    TensorHandle h;
    h.tensor_ptr = buffer.get();
    // No data_ptr: the alias geometry reads it as the start of a byte span over float storage, and
    // a vector's buffer moves whenever a getrf resizes it, so a snapshot would go stale and could
    // come to overlap some unrelated tensor's storage.
    h.owner        = std::move(buffer);
    h.id           = id;
    h.name         = std::move(name);
    h.rank         = 1;
    h.dims         = {order};
    h.strides      = {1};
    h.element_size = sizeof(blas::int_t);
    h.lu_pivots    = true;
    h.ops          = &lu_pivot_ops;
    return h;
}

EINSUMS_NAMESPACE_END(compute_graph)
