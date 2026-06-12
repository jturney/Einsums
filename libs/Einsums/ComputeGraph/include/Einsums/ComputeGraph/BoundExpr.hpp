//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <variant>

namespace einsums::compute_graph {

class ParamTable;

/// @brief A scalar integer expression that can be resolved at execute time.
///
/// Used to specify slice bounds on @ref ViewDesc and similar runtime-resolvable
/// scalars. A ``BoundExpr`` is one of:
///
///   - **Const**:    a known integer literal (e.g. ``0``, ``5``).
///   - **Param**:    a named lookup into a @ref ParamTable. Resolved by walking the
///                     active pipeline's parameter table at execute time. Codegen
///                     emits ``params.<name>``.
///   - **Callback**: a ``std::function<int64_t()>`` evaluated at execute time.
///                     Most flexible but interpreted-mode only, since codegen has
///                     to either fall back to a thunk or refuse the graph.
///
/// Implicit conversions are provided so callers can write
/// ``cg::Range{0, 5}``, ``cg::Range{0, "n_occ"}``, or
/// ``cg::Range{0, [&]{ return n_occ; }}`` interchangeably.
class BoundExpr {
  public:
    struct Const {
        std::int64_t value{0};
    };
    struct Param {
        std::string name;
    };
    struct Callback {
        std::function<std::int64_t()> fn;
    };

    using Storage = std::variant<Const, Param, Callback>;

    BoundExpr() = default;

    template <std::integral I>
    BoundExpr(I v) : _storage(Const{static_cast<std::int64_t>(v)}) {}

    BoundExpr(char const *param_name) : _storage(Param{std::string(param_name)}) {}
    BoundExpr(std::string param_name) : _storage(Param{std::move(param_name)}) {}
    BoundExpr(std::function<std::int64_t()> fn) : _storage(Callback{std::move(fn)}) {}

    [[nodiscard]] Storage const &storage() const noexcept { return _storage; }

    /// True if this is a compile-time literal. Optimization passes may
    /// special-case ``is_const()`` bounds (e.g., fuse two views with
    /// statically-known overlapping ranges).
    [[nodiscard]] bool is_const() const noexcept { return std::holds_alternative<Const>(_storage); }
    [[nodiscard]] bool is_param() const noexcept { return std::holds_alternative<Param>(_storage); }
    [[nodiscard]] bool is_callback() const noexcept { return std::holds_alternative<Callback>(_storage); }

    /// Read the constant value. Precondition: ``is_const()`` is true.
    [[nodiscard]] std::int64_t const_value() const { return std::get<Const>(_storage).value; }

    /// Read the referenced parameter name. Precondition: ``is_param()``.
    [[nodiscard]] std::string const &param_name() const { return std::get<Param>(_storage).name; }

    /// Resolve the expression to a concrete integer.
    ///
    /// - ``Const``:    returns the literal.
    /// - ``Param``:    looks up the name in @p params; throws if absent.
    /// - ``Callback``: invokes the captured lambda.
    [[nodiscard]] std::int64_t resolve(ParamTable const &params) const;

  private:
    Storage _storage{Const{0}};
};

/// @brief Mutable name-keyed table of integer parameters owned by a Pipeline.
///
/// Parameters are runtime values that ``BoundExpr::Param`` references resolve to.
/// They are read every time a node executes, so updates between iterations
/// (e.g., from a ``LoopCondition`` callback or a @c WriteParam node) take
/// effect on the next read.
///
/// Optimization passes treat parameter values as opaque. They may inspect
/// the structure of a ``BoundExpr`` (Const vs. Param) but must not branch
/// on the runtime value of a ``Param``. See ``BoundExpr`` documentation.
///
/// Held as a ``std::shared_ptr`` so executor lambdas can capture and read
/// the same table that mutator code (loop conditions, ``WriteParam``
/// executors) writes to.
class ParamTable {
  public:
    /// Set or overwrite a parameter. Allowed at any time: during capture, between
    /// iterations, and between Pipeline::execute() invocations.
    void set(std::string name, std::int64_t value) { _values[std::move(name)] = value; }

    /// Read a parameter. Throws ``std::runtime_error`` if @p name is unset.
    [[nodiscard]] std::int64_t get(std::string const &name) const {
        auto it = _values.find(name);
        if (it == _values.end())
            throw std::runtime_error("ParamTable: parameter '" + name + "' is not set");
        return it->second;
    }

    /// Read with a fallback if the parameter is unset.
    [[nodiscard]] std::int64_t get_or(std::string const &name, std::int64_t fallback) const {
        auto it = _values.find(name);
        return it == _values.end() ? fallback : it->second;
    }

    [[nodiscard]] bool   contains(std::string const &name) const { return _values.contains(name); }
    [[nodiscard]] size_t size() const noexcept { return _values.size(); }

    /// Read-only view of all entries, used by codegen to emit a Params POD.
    [[nodiscard]] std::unordered_map<std::string, std::int64_t> const &entries() const noexcept { return _values; }

  private:
    std::unordered_map<std::string, std::int64_t> _values;
};

inline std::int64_t BoundExpr::resolve(ParamTable const &params) const {
    return std::visit(
        [&](auto const &kind) -> std::int64_t {
            using T = std::decay_t<decltype(kind)>;
            if constexpr (std::is_same_v<T, Const>)
                return kind.value;
            else if constexpr (std::is_same_v<T, Param>)
                return params.get(kind.name);
            else if constexpr (std::is_same_v<T, Callback>)
                return kind.fn();
            else
                static_assert(sizeof(T) == 0, "Unhandled BoundExpr alternative");
        },
        _storage);
}

} // namespace einsums::compute_graph
