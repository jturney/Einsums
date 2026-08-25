//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/**
 * @file ElementOps.hpp
 * @brief Named element-wise kernels, so an @ref OpKind::ElementTransform node is data.
 *
 * @par Why this exists
 * ``cg::element_transform(C, unary_op)`` takes an arbitrary callable, and the
 * captured node holds it inside its executor. That is fine for a graph which is
 * only ever replayed in the process that captured it and impossible for one
 * that is written to a file: a closure has no name, no address that survives a
 * restart, and no content a reader could reconstruct.
 *
 * A registry answers it. A host registers the kernel ONCE under a name, capture
 * records the NAME, and @ref build_executor looks the name up. The saved graph
 * then holds a string, and a load that cannot resolve that string fails naming
 * the op rather than failing in the middle of a replay.
 *
 * Anonymous lambdas stay fully legal. They simply report as per-node blockers
 * in @ref Graph::serializability_report, with the fix named in the message.
 *
 * @par One registration, four dtypes
 * The kernels that matter here are dtype-generic: a reciprocal is
 * ``T{1} / x`` whether ``T`` is ``float`` or ``std::complex<double>``. A
 * registration therefore takes ONE generic callable -- a generic lambda, or a
 * functor with a template ``operator()`` -- and the registry instantiates it
 * for each of the four BLAS element types it will actually compile for. An op
 * that is meaningful only for real types is written as a CONSTRAINED generic
 * lambda, which makes "not defined for complex" a property the registry can
 * see rather than a comment:
 *
 * @code
 * element_ops::register_op("sqrt_or_zero",
 *                          []<std::floating_point T>(T x) { return x > T{0} ? std::sqrt(x) : T{0}; },
 *                          ElementOpSignature{.domain = ElementOpDomain::RealOnly});
 * @endcode
 *
 * Per-dtype behavior inside one op is spelled with ``if constexpr`` in the
 * kernel, which is where it belongs; the registry deliberately has no
 * four-callables-in-a-struct entry point for it.
 *
 * @par Named `Custom` is deferred, on purpose
 * @ref OpKind::Custom records around forty distinct internal operations -- a
 * reshape, a gather, a scatter-add, several LAPACK-adjacent wrappers -- whose
 * signatures differ in operand count, in operand roles and in whether the
 * operands share a dtype at all. A named registry for that surface is a
 * signature language, not a table, and nothing in this milestone needs one:
 * every one of those sites is internal and stays anonymous either way. It is
 * therefore left to the milestone that has a client for it, and the five
 * existing ``cg::custom`` overloads are untouched. The mechanism proved here is
 * the one that generalizes when that client appears.
 *
 * @see DESIGN-algebraic-optimizer.md, Part 3.3 and Part 9 item 2
 */

#include <Einsums/Config.hpp>

#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/TensorImpl/TensorImpl.hpp>

#include <complex>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <typeindex>
#include <typeinfo>
#include <utility>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph::element_ops)

/// Which element types an op declares itself defined for.
enum class ElementOpDomain : std::uint8_t {
    AllDtypes, ///< float, double, complex<float> and complex<double>.
    RealOnly,  ///< float and double. A complex lookup is refused, naming the op.
};

/**
 * @brief What a registered op promises about its shape.
 *
 * Held beside the kernels so a caller can ask what an op is before calling it,
 * and so a conflicting re-registration is detected on the declaration as well
 * as on the callable.
 */
struct ElementOpSignature {
    /// How many operands the kernel takes. Only 1 (unary) exists today; the
    /// field is here so a binary op does not have to change the entry type.
    std::size_t arity{1};

    /// Which element types the op is defined for.
    ElementOpDomain domain{ElementOpDomain::AllDtypes};

    /// One-line human description, shown in a listing. Not part of identity.
    std::string description;

    /// @brief Compare two signatures for the identity check.
    /// @param[in] lhs Left operand.
    /// @param[in] rhs Right operand.
    /// @return True when arity and domain match. The description is prose and
    ///         deliberately not compared.
    [[nodiscard]] friend bool operator==(ElementOpSignature const &lhs, ElementOpSignature const &rhs) noexcept {
        return lhs.arity == rhs.arity && lhs.domain == rhs.domain;
    }
};

namespace detail {

/**
 * @brief Apply @p kernel to every element of @p impl, in place.
 *
 * @param[in]     kernel The unary map.
 * @param[in,out] impl   The tensor to transform. Must not be null.
 * @throws std::invalid_argument When @p impl is null.
 *
 * Rank-erased, so one dtype dispatch covers every rank, and SERIAL, matching
 * ``element_transform_python``: the kernel reaches the loop through a
 * ``std::function``, an indirect call per element that a parallel region would
 * only make contended. Element order does not affect an element-wise map, so
 * the serial walk is a cost question rather than a numerical one.
 *
 * The contiguous case is lifted out because it is the common one; the general
 * case walks the index odometer with the last axis fastest, which visits every
 * element of any strided layout exactly once.
 */
template <typename T>
void apply_element_op(std::function<T(T)> const &kernel, ::einsums::detail::TensorImpl<T> *impl) {
    if (impl == nullptr) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "element_transform: the destination tensor has no storage");
    }
    std::size_t const total = impl->size();
    T                *data  = impl->data();
    if (data == nullptr || total == 0) {
        return;
    }

    if (std::size_t incx = 0; impl->is_totally_vectorable(&incx)) {
        for (std::size_t i = 0; i < total; ++i) {
            data[i * incx] = kernel(data[i * incx]);
        }
        return;
    }

    std::size_t const        rank = impl->rank();
    std::vector<std::size_t> index(rank, 0);
    for (std::size_t item = 0; item < total; ++item) {
        std::size_t offset = 0;
        for (std::size_t axis = 0; axis < rank; ++axis) {
            offset += index[axis] * impl->stride(axis);
        }
        data[offset] = kernel(data[offset]);
        for (std::size_t axis = rank; axis-- > 0;) {
            if (++index[axis] < impl->dim(axis)) {
                break;
            }
            index[axis] = 0;
        }
    }
}

} // namespace detail

/**
 * @brief Element-wise kernels looked up by name.
 *
 * @par Identity, and what a conflict is
 * Re-registering a name is idempotent when the registration is the SAME one and
 * an error otherwise, which is @ref SpaceRegistry's rule and for the same
 * reason: two translation units seeding the same op must not race into a
 * failure, and two DIFFERENT kernels sharing a name must not silently resolve
 * to whichever ran first.
 *
 * "The same one" is the callable's C++ TYPE together with the signature. A
 * lambda expression has a unique closure type, so registering the same
 * expression twice -- from a header-scope initializer included in several
 * translation units, say -- is idempotent, and any other lambda under that name
 * conflicts. A plain function POINTER carries no such identity (two unrelated
 * functions share one type), but a function pointer cannot be dtype-generic in
 * the first place, so it is not a shape this registry accepts.
 *
 * @par Thread safety
 * One mutex guards the table. Registration is expected during startup or setup
 * and lookup from anywhere; the lock exists so a late registration cannot
 * corrupt a concurrent reader, not because this is a hot path. A built executor
 * copies the kernel it needs ONCE, at build time, so a replay never touches the
 * registry at all.
 */
class ElementOpRegistry {
  public:
    ElementOpRegistry() = default;

    /// The mutex makes a registry neither copyable nor movable. Pass it by reference.
    ElementOpRegistry(ElementOpRegistry const &)            = delete;
    ElementOpRegistry &operator=(ElementOpRegistry const &) = delete;
    ElementOpRegistry(ElementOpRegistry &&)                 = delete;
    ElementOpRegistry &operator=(ElementOpRegistry &&)      = delete;
    ~ElementOpRegistry()                                    = default;

    /**
     * @brief Register a dtype-generic element-wise kernel under @p name.
     *
     * @tparam Kernel A callable invocable as ``kernel(T{})`` for at least the
     *         element types @p signature declares.
     * @param[in] name      The lookup name. Must not be empty.
     * @param[in] kernel    The kernel.
     * @param[in] signature What the op promises. Arity must be 1 today.
     * @throws std::invalid_argument When the name is empty, when the arity is
     *         not 1, when the kernel does not compile for a dtype the declared
     *         domain requires, or when @p name is already registered with
     *         different content.
     *
     * The four per-dtype instantiations are made here, once, so a lookup is a
     * map probe and a ``std::function`` copy rather than a dispatch.
     */
    template <typename Kernel>
    void register_op(std::string name, Kernel kernel, ElementOpSignature signature = {}) {
        if (name.empty()) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument, "element_ops::register_op: an op name must not be empty");
        }
        if (signature.arity != 1) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                    "element_ops::register_op('{}'): arity {} is not supported; element ops are unary", name,
                                    signature.arity);
        }

        Entry entry{.signature = std::move(signature), .content = std::type_index(typeid(Kernel))};

        // Install exactly the arms the kernel compiles for. A constrained
        // generic lambda is what makes "not defined for complex" visible here
        // instead of being a hard error at instantiation.
        if constexpr (std::is_invocable_r_v<float, Kernel const &, float>) {
            entry.f32 = kernel;
        }
        if constexpr (std::is_invocable_r_v<double, Kernel const &, double>) {
            entry.f64 = kernel;
        }
        if constexpr (std::is_invocable_r_v<std::complex<float>, Kernel const &, std::complex<float>>) {
            entry.c64 = kernel;
        }
        if constexpr (std::is_invocable_r_v<std::complex<double>, Kernel const &, std::complex<double>>) {
            entry.c128 = kernel;
        }

        // The declared domain is a MASK as well as a promise: an op declared
        // real-only stays real-only even when its kernel happens to compile for
        // complex, because the declaration is what a saved graph carries.
        if (entry.signature.domain == ElementOpDomain::RealOnly) {
            entry.c64  = nullptr;
            entry.c128 = nullptr;
        }

        require_arm(name, entry.f32 != nullptr, "float");
        require_arm(name, entry.f64 != nullptr, "double");
        if (entry.signature.domain == ElementOpDomain::AllDtypes) {
            require_arm(name, entry.c64 != nullptr, "complex<float>");
            require_arm(name, entry.c128 != nullptr, "complex<double>");
        }

        std::scoped_lock const guard(_mutex);
        auto const             existing = _entries.find(name);
        if (existing != _entries.end()) {
            if (existing->second.content == entry.content && existing->second.signature == entry.signature) {
                return; // the same registration, made twice
            }
            EINSUMS_THROW_EXCEPTION(std::invalid_argument, "element_ops::register_op: op '{}' is already registered with different content",
                                    name);
        }
        _entries.emplace(std::move(name), std::move(entry));
    }

    /**
     * @brief Whether @p name is registered.
     * @param[in] name The name to look for.
     * @return True when an op of that name exists.
     */
    [[nodiscard]] bool contains(std::string_view name) const {
        std::scoped_lock const guard(_mutex);
        return _entries.find(name) != _entries.end();
    }

    /**
     * @brief The declared signature of @p name.
     * @param[in] name The op.
     * @return Its signature.
     * @throws std::invalid_argument When no op of that name is registered.
     */
    [[nodiscard]] ElementOpSignature signature(std::string_view name) const {
        std::scoped_lock const guard(_mutex);
        return find_locked(name).signature;
    }

    /**
     * @brief The kernel registered under @p name, for element type @p T.
     *
     * @tparam T One of the four BLAS element types.
     * @param[in] name The op.
     * @return A copy of the kernel, so the caller may hold it after the
     *         registry's lock is released.
     * @throws std::invalid_argument When no op of that name is registered, or
     *         when the op is not defined for @p T. Both messages name the op.
     */
    template <typename T>
    [[nodiscard]] std::function<T(T)> kernel(std::string_view name) const {
        std::scoped_lock const     guard(_mutex);
        Entry const               &entry = find_locked(name);
        std::function<T(T)> const *slot  = nullptr;
        char const                *dtype = nullptr;
        if constexpr (std::is_same_v<T, float>) {
            slot  = &entry.f32;
            dtype = "float";
        } else if constexpr (std::is_same_v<T, double>) {
            slot  = &entry.f64;
            dtype = "double";
        } else if constexpr (std::is_same_v<T, std::complex<float>>) {
            slot  = &entry.c64;
            dtype = "complex<float>";
        } else {
            static_assert(std::is_same_v<T, std::complex<double>>, "element_ops: unsupported element type");
            slot  = &entry.c128;
            dtype = "complex<double>";
        }
        if (!*slot) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument, "element_ops: op '{}' is not defined for {}", name, dtype);
        }
        return *slot;
    }

    /**
     * @brief Every registered name, sorted.
     * @return The names, for a listing or a diagnostic.
     */
    [[nodiscard]] std::vector<std::string> names() const {
        std::scoped_lock const   guard(_mutex);
        std::vector<std::string> out;
        out.reserve(_entries.size());
        for (auto const &[name, entry] : _entries) {
            out.push_back(name);
        }
        return out;
    }

    /// How many ops are registered.
    [[nodiscard]] std::size_t size() const {
        std::scoped_lock const guard(_mutex);
        return _entries.size();
    }

  private:
    struct Entry {
        ElementOpSignature signature;
        /// The registered callable's C++ type; see the identity note on the class.
        std::type_index content{typeid(void)};

        std::function<float(float)>                               f32;
        std::function<double(double)>                             f64;
        std::function<std::complex<float>(std::complex<float>)>   c64;
        std::function<std::complex<double>(std::complex<double>)> c128;
    };

    /// Complain that a kernel does not compile for a dtype its domain requires.
    static void require_arm(std::string const &name, bool present, char const *dtype) {
        if (!present) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                    "element_ops::register_op('{}'): the kernel is not invocable for {}, which its declared domain "
                                    "requires; constrain the kernel and declare a narrower domain if that is intended",
                                    name, dtype);
        }
    }

    /// The entry for @p name. The caller must hold @ref _mutex.
    [[nodiscard]] Entry const &find_locked(std::string_view name) const {
        auto const it = _entries.find(name);
        if (it == _entries.end()) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument, "element_ops: no op named '{}' is registered in this process", name);
        }
        return it->second;
    }

    mutable std::mutex                        _mutex;
    std::map<std::string, Entry, std::less<>> _entries;
};

/**
 * @brief The process-wide element-op registry.
 * @return The single registry every graph in this process looks names up in.
 *
 * Defined in the ComputeGraph library rather than inline, for the reason
 * @ref global_space_registry gives: a function-local static in a header gets one
 * instance per binary that includes it, so libEinsums and the Python ``_core``
 * extension would each hold their own and an op registered on one side would be
 * missing on the other.
 *
 * The starter ops of @ref register_builtin_element_ops are already in it.
 */
[[nodiscard]] EINSUMS_EXPORT ElementOpRegistry &global_element_op_registry();

/**
 * @brief Seed @p registry with the ops this library ships.
 * @param[in,out] registry The registry to seed.
 *
 * Deliberately a tiny set, and deliberately obvious: it exists to prove the
 * mechanism and to serve the shapes that recur in the tree (a reciprocal for
 * MP2-style denominators, a guarded square root for a metric power, a square,
 * a negation). Hosts register their own; that is what the registry is for, and
 * the library taking a position on which chemistry kernels are "standard" is
 * exactly the layering the design's non-goals rule out.
 */
EINSUMS_EXPORT void register_builtin_element_ops(ElementOpRegistry &registry);

/**
 * @brief Register an op in the process-wide registry.
 * @tparam Kernel A dtype-generic callable; see @ref ElementOpRegistry::register_op.
 * @param[in] name      The lookup name.
 * @param[in] kernel    The kernel.
 * @param[in] signature What the op promises.
 */
template <typename Kernel>
void register_op(std::string name, Kernel kernel, ElementOpSignature signature = {}) {
    global_element_op_registry().register_op(std::move(name), std::move(kernel), std::move(signature));
}

EINSUMS_NAMESPACE_END(compute_graph::element_ops)
