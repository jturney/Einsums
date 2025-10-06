//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/TypeSupport/TypeTraits.hpp>

#include <cassert>
#include <memory>
#include <optional>
#include <type_traits>

namespace einsums {

namespace detail {

// ------------------------------------------------------------------------------------------ //
// SimplifyType
// ------------------------------------------------------------------------------------------ //
template <typename From>
struct SimplifyType {
    using SimpleType = From;

    static SimpleType &get_simplified_value(From &val) { return val; }
};

template <typename From>
struct SimplifyType<From const> {
    using NonConstSimpleType = SimplifyType<From>::SimpleType;
    using SimpleType         = AddConstPastPointer<NonConstSimpleType>::type;
    using ReturnType         = AddLValueReferenceIfNotPointer<SimpleType>::type;

    static ReturnType get_simplified_value(From const &val) { return SimplifyType<From>::get_simplified_value(const_cast<From &>(val)); }
};

// ------------------------------------------------------------------------------------------ //
// IsaImpl
// ------------------------------------------------------------------------------------------ //

template <typename To, typename From, typename = void>
struct IsaImpl {
    static bool doit(From const &val) { return To::classof(&val); }
};

template <typename To, typename From>
struct IsaImpl<To, From, std::enable_if_t<std::is_base_of_v<To, From>>> {
    static bool doit(From const &) { return true; }
};

template <typename To, typename From>
struct IsaImplCl {
    static bool doit(From const &val) { return IsaImpl<To, From>::doit(val); }
};

template <typename To, typename From>
struct IsaImplCl<To, From const> {
    static bool doit(From const &val) { return IsaImpl<To, From>::doit(val); }
};

template <typename To, typename From>
struct IsaImplCl<To, std::unique_ptr<From> const> {
    static bool doit(std::unique_ptr<From> const &val) {
        assert(val && "isa<> used on a null pointer");
        return IsaImplCl<To, From>::doit(*val);
    }
};

template <typename To, typename From>
struct IsaImplCl<To, From *> {
    static bool doit(From const *val) {
        assert(val && "isa<> used on a null pointer");
        return IsaImpl<To, From>::doit(*val);
    }
};

template <typename To, typename From>
struct IsaImplCl<To, From *const> {
    static bool doit(From const *val) {
        assert(val && "isa<> used on a null pointer");
        return IsaImpl<To, From>::doit(*val);
    }
};

template <typename To, typename From>
struct IsaImplCl<To, From const *> {
    static bool doit(From const *val) {
        assert(val && "isa<> used on a null pointer");
        return IsaImpl<To, From>::doit(*val);
    }
};

template <typename To, typename From>
struct IsaImplCl<To, From const *const> {
    static bool doit(From const *val) {
        assert(val && "isa<> used on a null pointer");
        return IsaImpl<To, From>::doit(*val);
    }
};

template <typename To, typename From, typename SimpleFrom>
struct IsaImplWrap {
    // When From != SimplifiedType, we can simplify the type some more by using
    // the SimplifyType template
    static bool doit(From const &val) {
        return IsaImplWrap<To, SimpleFrom, typename SimplifyType<SimpleFrom>::SimpleType>::doit(
            SimplifyType<From const>::get_simplified_value(val));
    }
};

template <typename To, typename From>
struct IsaImplWrap<To, From, From> {
    // When From == SimplifyType, we are as simple as we are going to get.
    static bool doit(From const &val) { return IsaImplCl<To, From>::doit(val); }
};

// ------------------------------------------------------------------------------------------ //
// CastReturnType + CastReturnTypeImpl
// ------------------------------------------------------------------------------------------ //

template <typename To, typename From>
struct CastReturnType;

/// Calculate what type the 'cast' function should return, based on a requested
/// type of To and a source type of From.
template <typename To, typename From>
struct CastReturnTypeImpl {
    using ReturnType = To &; // Normal case, return To&;
};

template <typename To, typename From>
struct CastReturnTypeImpl<To, From const> {
    using ReturnType = To const &;
};

template <typename To, typename From>
struct CastReturnTypeImpl<To, From *> {
    using ReturnType = To *;
};

template <typename To, typename From>
struct CastReturnTypeImpl<To, From const *> {
    using ReturnType = To const *;
};

template <typename To, typename From>
struct CastReturnTypeImpl<To, From const *const> {
    using ReturnType = To const *;
};

template <typename To, typename From>
struct CastReturnTypeImpl<To, std::unique_ptr<From>> {
  private:
    using PointerType = CastReturnTypeImpl<To, From *>::ReturnType;
    using ResultType  = std::remove_pointer_t<PointerType>;

  public:
    using ReturnType = std::unique_ptr<ResultType>;
};

template <typename To, typename From, typename SimpleFrom>
struct CastReturnTypeWrap {
    using ReturnType = CastReturnType<To, SimpleFrom>::ReturnType;
};

template <typename To, typename From>
struct CastReturnTypeWrap<To, From, From> {
    using ReturnType = CastReturnTypeImpl<To, From>::ReturnType;
};

template <typename To, typename From>
struct CastReturnType {
    using ReturnType = CastReturnTypeWrap<To, From, typename SimplifyType<From>::SimpleType>::ReturnType;
};

// ------------------------------------------------------------------------------------------ //
// CastConvertValue
// ------------------------------------------------------------------------------------------ //

template <typename To, typename From, typename SimpleFrom>
struct CastConvertValue {
    static CastReturnType<To, From>::ReturnType doit(From const &val) {
        return CastConvertValue<To, SimpleFrom, typename SimplifyType<SimpleFrom>::SimpleType>::doit(
            SimplifyType<From>::get_simplified_value(const_cast<From &>(val)));
    }
};

template <typename To, typename From>
struct CastConvertValue<To, From, From> {
    static CastReturnType<To, From>::ReturnType doit(From const &val) {
        return *(std::remove_reference_t<typename CastReturnType<To, From>::ReturnType> *)&const_cast<From &>(val);
    }
};

template <typename To, typename From>
struct CastConvertValue<To, From *, From *> {
    static CastReturnType<To, From *>::ReturnType doit(From const *val) {
        return (typename CastReturnType<To, From *>::ReturnType) const_cast<From *>(val);
    }
};

// ------------------------------------------------------------------------------------------ //
// CastConvertValue
// ------------------------------------------------------------------------------------------ //

template <typename X>
struct IsSimpleType {
    static bool const Value = std::is_same_v<X, typename SimplifyType<X>::SimpleType>;
};

} // namespace detail

// ------------------------------------------------------------------------------------------ //
// CastIsPossible
// ------------------------------------------------------------------------------------------ //

/// This struct provides a way to check if a given cast is possible. It provides
/// a static function called isPossible that is used to check if a cast can be
/// performed. It should be overridden like this:
///
/// template<> struct CastIsPossible<foo, bar> {
///   static inline bool is_possible(const bar &b) {
///     return bar.isFoo();
///   }
/// };
template <typename To, typename From, typename = void>
struct CastIsPossible {
    static bool is_possible(From const &f) {
        return detail::IsaImplWrap<To, From const, typename detail::SimplifyType<From const>::SimpleType>::doit(f);
    }
};

// Needed for optional unwrapping. This could be implemented with IsaImpl, but
// we want to implement things in the new method and move old implementations
// over. In fact, some of the Is templates should be moved over to
// CastIsPossible.
template <typename To, typename From>
struct CastIsPossible<To, std::optional<From>> {
    static bool is_possible(std::optional<From> const &f) {
        assert(f && "CastIsPossible::is_possible called on a nullopt!");
        return detail::IsaImplWrap<To, From const, typename detail::SimplifyType<From const>::SimpleType>::doit(*f);
    }
};

/// Upcasting (from derived to base) and casting from a type to itself should
/// always be possible.
template <typename To, typename From>
struct CastIsPossible<To, From, std::enable_if_t<std::is_base_of_v<To, From>>> {
    static bool is_possible(From const &) { return true; }
};

// ------------------------------------------------------------------------------------------ //
// Cast traits
// ------------------------------------------------------------------------------------------ //

/// All of these cast traits are meant to be implementations for useful casts
/// that users may want to use that are outside the standard behavior. An
/// example of how to use a special cast called `CastTrait` is:
///
/// template<> struct CastInfo<foo, bar> : public CastTrait<foo, bar> {};
///
/// Essentially, if your use case falls directly into one of the use cases
/// supported by a given cast trait, simply inherit your special CastInfo
/// directly from one of these to avoid having to reimplement the boilerplate
/// `is_possible/cast_failed/do_cast/do_cast_if_possible`. A cast trait can also
/// provide a subset of those functions.

/// This cast trait just provides cast_failed for the specified `To` type to make
/// CastInfo specializations more declarative. In order to use this, the target
/// result type must be `To` and `To` must be constructible from `nullptr`.
template <typename To>
struct NullableValueCastFailed {
    static To cast_failed() { return To(nullptr); }
};

/// This cast trait just provides the default implementation of do-cast_if_possible
/// to make CastInfo specializations more declarative. The `Derived` template
/// parameter *must* be provided for forwarding cast_failed and do_cast.
template <typename To, typename From, typename Derived>
struct Defaultdo_castIfPossible {
    static To do_cast_if_possible(From f) {
        if (!Derived::is_possible(f))
            return Derived::cast_failed();
        return Derived::do_cast(f);
    }
};

namespace detail {
template <typename OptionalDerived, typename Default>
using SelfType = std::conditional_t<std::is_same_v<OptionalDerived, void>, Default, OptionalDerived>;
}

/// This cast trait provides casting for the specific case of casting to a
/// value-typed object from a pointer-typed object. Note that `To` must be
/// nullable/constructible from a pointer to `From` to use this cast.
template <typename To, typename From, typename Derived = void>
struct ValueFromPointerCast : CastIsPossible<To, From *>,
                              NullableValueCastFailed<To>,
                              Defaultdo_castIfPossible<To, From *, detail::SelfType<Derived, ValueFromPointerCast<To, From>>> {
    static To do_cast(From *f) { return To(f); }
};

/// This cast trait provides std::unique_ptr casting. It has the semantics of
/// moving the contents of the input unique_ptr into the output unique_ptr
/// during the cast. It's also a good example of how to implement a move-only
/// cast.
template <typename To, typename From, typename Derived = void>
struct UniquePtrCast : CastIsPossible<To, From *> {
    using Self           = detail::SelfType<Derived, UniquePtrCast<To, From>>;
    using CastResultType = std::unique_ptr<std::remove_reference_t<typename detail::CastReturnType<To, From>::ReturnType>>;

    static CastResultType do_cast(std::unique_ptr<From> &&f) {
        return CastResultType((typename CastResultType::element_type *)f.release());
    }

    static CastResultType cast_failed() { return CastResultType(nullptr); }

    static CastResultType do_cast_if_possible(std::unique_ptr<From> &f) {
        if (!Self::is_possible(f.get()))
            return cast_failed();
        return do_cast(std::move(f));
    }
};

/// This cast trait provides std::optional<T> casting. This means that if you
/// have a value type, you can cast it to another value type and have dyn_cast
/// return an std::optional<T>.
template <typename To, typename From, typename Derived = void>
struct OptionalValueCast : CastIsPossible<To, From>,
                           Defaultdo_castIfPossible<std::optional<To>, From, detail::SelfType<Derived, OptionalValueCast<To, From>>> {
    static std::optional<To> cast_failed() { return std::optional<To>{}; }

    static std::optional<To> do_cast(From const &f) { return To(f); }
};

/// Provides a cast trait that strips `const` from types to make it easier to
/// implement a const-version of a non-const cast. It just removes boilerplate
/// and reduces the amount of code you as the user need to implement. You can
/// use it like this:
///
/// template<> struct CastInfo<foo, bar> {
///   ...verbose implementation...
/// };
///
/// template<> struct CastInfo<foo, const bar> : public
///        ConstStrippingForwardingCast<foo, const bar, CastInfo<foo, bar>> {};
///
template <typename To, typename From, typename ForwardTo>
struct ConstStrippingForwardingCast {
    // Remove the pointer if it exists
    using DecayedFrom  = std::remove_cv_t<std::remove_pointer_t<From>>;
    using NonConstFrom = std::conditional_t<std::is_pointer_v<From>, DecayedFrom *, DecayedFrom &>;

    static bool is_possible(From const &f) { return ForwardTo::is_possible(const_cast<NonConstFrom>(f)); }

    static decltype(auto) cast_failed() { return ForwardTo::cast_failed(); }

    static decltype(auto) do_cast(From const &f) { return ForwardTo::do_cast(const_cast<NonConstFrom>(f)); }

    static decltype(auto) do_cast_if_possible(From const &f) { return ForwardTo::do_cast_if_possible(const_cast<NonConstFrom>(f)); }
};

/// Provides a cast trait that uses a defined pointer to pointer cast as a base
/// for reference-to-reference casts. Note that it does not provide cast_failed
/// and do_cast_if_possible because a pointer-to-pointer cast would likely just
/// return `nullptr` which could cause nullptr dereference. You can use it like
/// this:
///
///   template <> struct CastInfo<foo, bar *> { ... verbose implementation... };
///
///   template <>
///   struct CastInfo<foo, bar>
///       : public ForwardToPointerCast<foo, bar, CastInfo<foo, bar *>> {};
///
template <typename To, typename From, typename ForwardTo>
struct ForwardToPointerCast {
    static bool is_possible(From const &f) { return ForwardTo::is_possible(&f); }

    static decltype(auto) do_cast(From const &f) { return *ForwardTo::do_cast(&f); }
};

// ------------------------------------------------------------------------------------------ //
// Cast traits
// ------------------------------------------------------------------------------------------ //

/// This struct provides a method for customizing the way a cast is performed.
/// It inherits from CastIsPossible, to support the case of declaring many
/// CastIsPossible specializations without having to specialize the full
/// CastInfo.
///
/// In order to specialize different behaviors, specify different functions in
/// your CastInfo specialization.
/// For isa<> customization, provide:
///
///   `static bool is_possible(const From &f)`
///
/// For cast<> customization, provide:
///
///  `static To do_cast(const From &f)`
///
/// For dyn_cast<> and the *_if_present<> variants' customization, provide:
///
///  `static To cast_failed()` and `static To do_cast_if_possible(const From &f)`
///
/// Your specialization might look something like this:
///
///  template<> struct CastInfo<foo, bar> : public CastIsPossible<foo, bar> {
///    static inline foo do_cast(const bar &b) {
///      return foo(const_cast<bar &>(b));
///    }
///    static inline foo cast_failed() { return foo(); }
///    static inline foo do_cast_if_possible(const bar &b) {
///      if (!CastInfo<foo, bar>::is_possible(b))
///        return cast_failed();
///      return do_cast(b);
///    }
///  };

// The default implementations of CastInfo don't use cast traits for now because
// we need to specify types all over the place due to the current expected
// casting behavior and the way CastReturnType works. New use cases can and should
// take advantage of the cast traits whenever possible!

template <typename To, typename From, typename Enable = void>
struct CastInfo : CastIsPossible<To, From> {
    using Self = CastInfo<To, From, Enable>;

    using CastReturnType = typename detail::CastReturnType<To, From>::ReturnType;

    static CastReturnType do_cast(From const &f) {
        return detail::CastConvertValue<To, From, typename detail::SimplifyType<From>::SimpleType>::doit(const_cast<From &>(f));
    }

    // This assumes that you can construct the cast return type from `nullptr`.
    // This is largely to support legacy use cases - if you don't want this
    // behavior you should specialize CastInfo for your use case.
    static CastReturnType cast_failed() { return CastReturnType(nullptr); }

    static CastReturnType do_cast_if_possible(From const &f) {
        if (!Self::is_possible(f))
            return cast_failed();
        return do_cast(f);
    }
};

/// This struct provides an overload for CastInfo where From has detail::SimplifyType
/// defined. This simply forwards to the appropriate CastInfo with the
/// simplified type/value, so you don't have to implement both.
template <typename To, typename From>
struct CastInfo<To, From, std::enable_if_t<!detail::IsSimpleType<From>::value>> {
    using Self           = CastInfo<To, From>;
    using SimpleFrom     = detail::SimplifyType<From>::SimpleType;
    using SimplifiedSelf = CastInfo<To, SimpleFrom>;

    static bool is_possible(From &f) { return SimplifiedSelf::is_possible(detail::SimplifyType<From>::get_simplified_value(f)); }

    static decltype(auto) do_cast(From &f) { return SimplifiedSelf::do_cast(detail::SimplifyType<From>::get_simplified_value(f)); }

    static decltype(auto) cast_failed() { return SimplifiedSelf::cast_failed(); }

    static decltype(auto) do_cast_if_possible(From &f) {
        return SimplifiedSelf::do_cast_if_possible(detail::SimplifyType<From>::get_simplified_value(f));
    }
};

//===----------------------------------------------------------------------===//
// Pre-specialized CastInfo
//===----------------------------------------------------------------------===//

/// Provide a CastInfo specialized for std::unique_ptr.
template <typename To, typename From>
struct CastInfo<To, std::unique_ptr<From>> : UniquePtrCast<To, From> {};

/// Provide a CastInfo specialized for std::optional<From>. It's assumed that if
/// the input is std::optional<From> that the output can be std::optional<To>.
/// If that's not the case, specialize CastInfo for your use case.
template <typename To, typename From>
struct CastInfo<To, std::optional<From>> : OptionalValueCast<To, From> {};

/// isa<X> - Return true if the parameter to the template is an instance of one
/// of the template type arguments.  Used like this:
///
///  if (isa<Type>(myVal)) { ... }
///  if (isa<Type0, Type1, Type2>(myVal)) { ... }
template <typename... To, typename From>
[[nodiscard]] bool isa(From const &Val) {
    return (CastInfo<To, From const>::is_possible(Val) || ...);
}

/// cast<X> - Return the argument parameter cast to the specified type.  This
/// casting operator asserts that the type is correct, so it does not return
/// null on failure.  It does not allow a null argument (use cast_if_present for
/// that). It is typically used like this:
///
///  cast<Instruction>(myVal)->getParent()

template <typename To, typename From>
[[nodiscard]] decltype(auto) cast(From const &Val) {
    assert(isa<To>(Val) && "cast<Ty>() argument of incompatible type!");
    return CastInfo<To, From const>::do_cast(Val);
}

template <typename To, typename From>
[[nodiscard]] decltype(auto) cast(From &Val) {
    assert(isa<To>(Val) && "cast<Ty>() argument of incompatible type!");
    return CastInfo<To, From>::do_cast(Val);
}

template <typename To, typename From>
[[nodiscard]] decltype(auto) cast(From *Val) {
    assert(isa<To>(Val) && "cast<Ty>() argument of incompatible type!");
    return CastInfo<To, From *>::do_cast(Val);
}

template <typename To, typename From>
[[nodiscard]] decltype(auto) cast(std::unique_ptr<From> &&Val) {
    assert(isa<To>(Val) && "cast<Ty>() argument of incompatible type!");
    return CastInfo<To, std::unique_ptr<From>>::do_cast(std::move(Val));
}

//===----------------------------------------------------------------------===//
// ValueIsPresent
//===----------------------------------------------------------------------===//

template <typename T>
constexpr bool IsNullable = std::is_pointer_v<T> || std::is_constructible_v<T, std::nullptr_t>;

/// ValueIsPresent provides a way to check if a value is, well, present. For
/// pointers, this is the equivalent of checking against nullptr, for Optionals
/// this is the equivalent of checking hasValue(). It also provides a method for
/// unwrapping a value (think calling .value() on an optional).

// Generic values can't *not* be present.
template <typename T, typename Enable = void>
struct ValueIsPresent {
    using UnwrappedType = T;
    static bool           is_present(T const &t) { return true; }
    static decltype(auto) unwrap_value(T &t) { return t; }
};

// Optional provides its own way to check if something is present.
template <typename T>
struct ValueIsPresent<std::optional<T>> {
    using UnwrappedType = T;
    static bool           is_present(std::optional<T> const &t) { return t.has_value(); }
    static decltype(auto) unwrap_value(std::optional<T> &t) { return *t; }
};

// If something is "nullable" then we just compare it to nullptr to see if it
// exists.
template <typename T>
struct ValueIsPresent<T, std::enable_if_t<IsNullable<T>>> {
    using UnwrappedType = T;
    static bool           is_present(T const &t) { return t != T(nullptr); }
    static decltype(auto) unwrap_value(T &t) { return t; }
};

namespace detail {
// Convenience function we can use to check if a value is present. Because of
// detail::SimplifyType, we have to call it on the simplified type for now.
template <typename T>
bool is_present(T const &t) {
    return ValueIsPresent<typename SimplifyType<T>::SimpleType>::is_present(SimplifyType<T>::get_simplified_value(const_cast<T &>(t)));
}

// Convenience function we can use to unwrap a value.
template <typename T>
decltype(auto) unwrap_value(T &t) {
    return ValueIsPresent<T>::unwrap_value(t);
}
} // namespace detail

/// dyn_cast<X> - Return the argument parameter cast to the specified type. This
/// casting operator returns null if the argument is of the wrong type, so it
/// can be used to test for a type as well as cast if successful. The value
/// passed in must be present, if not, use dyn_cast_if_present. This should be
/// used in the context of an if statement like this:
///
///  if (const Instruction *I = dyn_cast<Instruction>(myVal)) { ... }

template <typename To, typename From>
[[nodiscard]] decltype(auto) dyn_cast(From const &Val) {
    assert(detail::is_present(Val) && "dyn_cast on a non-existent value");
    return CastInfo<To, From const>::do_cast_if_possible(Val);
}

template <typename To, typename From>
[[nodiscard]] decltype(auto) dyn_cast(From &Val) {
    assert(detail::is_present(Val) && "dyn_cast on a non-existent value");
    return CastInfo<To, From>::do_cast_if_possible(Val);
}

template <typename To, typename From>
[[nodiscard]] decltype(auto) dyn_cast(From *Val) {
    assert(detail::is_present(Val) && "dyn_cast on a non-existent value");
    return CastInfo<To, From *>::do_cast_if_possible(Val);
}

template <typename To, typename From>
[[nodiscard]] decltype(auto) dyn_cast(std::unique_ptr<From> &Val) {
    assert(detail::is_present(Val) && "dyn_cast on a non-existent value");
    return CastInfo<To, std::unique_ptr<From>>::do_cast_if_possible(Val);
}

/// isa_and_present<X> - Functionally identical to isa, except that a null value
/// is accepted.
template <typename... X, class Y>
[[nodiscard]] bool isa_and_present(Y const &Val) {
    if (!detail::is_present(Val))
        return false;
    return isa<X...>(Val);
}

template <typename... X, class Y>
[[nodiscard]] bool isa_and_nonnull(Y const &Val) {
    return isa_and_present<X...>(Val);
}

/// cast_if_present<X> - Functionally identical to cast, except that a null
/// value is accepted.
template <class X, class Y>
[[nodiscard]] auto cast_if_present(Y const &Val) {
    if (!detail::is_present(Val))
        return CastInfo<X, Y const>::cast_failed();
    assert(isa<X>(Val) && "cast_if_present<Ty>() argument of incompatible type!");
    return cast<X>(detail::unwrap_value(Val));
}

template <class X, class Y>
[[nodiscard]] inline auto cast_if_present(Y &Val) {
    if (!detail::is_present(Val))
        return CastInfo<X, Y>::cast_failed();
    assert(isa<X>(Val) && "cast_if_present<Ty>() argument of incompatible type!");
    return cast<X>(detail::unwrap_value(Val));
}

template <class X, class Y>
[[nodiscard]] inline auto cast_if_present(Y *Val) {
    if (!detail::is_present(Val))
        return CastInfo<X, Y *>::cast_failed();
    assert(isa<X>(Val) && "cast_if_present<Ty>() argument of incompatible type!");
    return cast<X>(detail::unwrap_value(Val));
}

template <class X, class Y>
[[nodiscard]] inline auto cast_if_present(std::unique_ptr<Y> &&Val) {
    if (!detail::is_present(Val))
        return UniquePtrCast<X, Y>::cast_failed();
    return UniquePtrCast<X, Y>::do_cast(std::move(Val));
}

// Provide a forwarding from cast_or_null to cast_if_present for current
// users. This is deprecated and will be removed in a future patch, use
// cast_if_present instead.
template <class X, class Y>
auto cast_or_null(Y const &Val) {
    return cast_if_present<X>(Val);
}

template <class X, class Y>
auto cast_or_null(Y &Val) {
    return cast_if_present<X>(Val);
}

template <class X, class Y>
auto cast_or_null(Y *Val) {
    return cast_if_present<X>(Val);
}

template <class X, class Y>
auto cast_or_null(std::unique_ptr<Y> &&Val) {
    return cast_if_present<X>(std::move(Val));
}

/// dyn_cast_if_present<X> - Functionally identical to dyn_cast, except that a
/// null (or none in the case of optionals) value is accepted.
template <class X, class Y>
auto dyn_cast_if_present(Y const &Val) {
    if (!detail::is_present(Val))
        return CastInfo<X, Y const>::cast_failed();
    return CastInfo<X, Y const>::do_cast_if_possible(detail::unwrap_value(Val));
}

template <class X, class Y>
auto dyn_cast_if_present(Y &Val) {
    if (!detail::is_present(Val))
        return CastInfo<X, Y>::cast_failed();
    return CastInfo<X, Y>::do_cast_if_possible(detail::unwrap_value(Val));
}

template <class X, class Y>
auto dyn_cast_if_present(Y *Val) {
    if (!detail::is_present(Val))
        return CastInfo<X, Y *>::cast_failed();
    return CastInfo<X, Y *>::do_cast_if_possible(detail::unwrap_value(Val));
}

// Forwards to dyn_cast_if_present to avoid breaking current users. This is
// deprecated and will be removed in a future patch, use
// dyn_cast_if_present instead.
template <class X, class Y>
auto dyn_cast_or_null(Y const &Val) {
    return dyn_cast_if_present<X>(Val);
}

template <class X, class Y>
auto dyn_cast_or_null(Y &Val) {
    return dyn_cast_if_present<X>(Val);
}

template <class X, class Y>
auto dyn_cast_or_null(Y *Val) {
    return dyn_cast_if_present<X>(Val);
}

/// unique_dyn_cast<X> - Given a unique_ptr<Y>, try to return a unique_ptr<X>,
/// taking ownership of the input pointer iff isa<X>(Val) is true.  If the
/// cast is successful, From refers to nullptr on exit and the casted value
/// is returned.  If the cast is unsuccessful, the function returns nullptr
/// and From is unchanged.
template <class X, class Y>
[[nodiscard]] CastInfo<X, std::unique_ptr<Y>>::CastResultType unique_dyn_cast(std::unique_ptr<Y> &Val) {
    if (!isa<X>(Val))
        return nullptr;
    return cast<X>(std::move(Val));
}

template <class X, class Y>
[[nodiscard]] auto unique_dyn_cast(std::unique_ptr<Y> &&Val) {
    return unique_dyn_cast<X, Y>(Val);
}

// unique_dyn_cast_or_null<X> - Functionally identical to unique_dyn_cast,
// except that a null value is accepted.
template <class X, class Y>
[[nodiscard]] typename CastInfo<X, std::unique_ptr<Y>>::CastResultType unique_dyn_cast_or_null(std::unique_ptr<Y> &Val) {
    if (!Val)
        return nullptr;
    return unique_dyn_cast<X, Y>(Val);
}

template <class X, class Y>
[[nodiscard]] auto unique_dyn_cast_or_null(std::unique_ptr<Y> &&Val) {
    return unique_dyn_cast_or_null<X, Y>(Val);
}

//===----------------------------------------------------------------------===//
// Isa Predicates
//===----------------------------------------------------------------------===//

/// These are wrappers over isa* function that allow them to be used in generic
/// algorithms such as `einsums:all_of`, `einsums::none_of`, etc. This is accomplished
/// by exposing the isa* functions through function objects with a generic
/// function call operator.

namespace detail {
template <typename... Types>
struct IsaCheckPredicate {
    template <typename T>
    [[nodiscard]] bool operator()(T const &Val) const {
        return isa<Types...>(Val);
    }
};

template <typename... Types>
struct IsaAndPresentCheckPredicate {
    template <typename T>
    [[nodiscard]] bool operator()(T const &Val) const {
        return isa_and_present<Types...>(Val);
    }
};
} // namespace detail

/// Function object wrapper for the `einsums::isa` type check. The function call
/// operator returns true when the value can be cast to any type in `Types`.
/// Example:
/// ```
/// SmallVector<Type> myTypes = ...;
/// if (einsums::all_of(myTypes, einsums::IsaPred<VectorType>))
///   ...
/// ```
template <typename... Types>
inline constexpr detail::IsaCheckPredicate<Types...> IsaPred{};

/// Function object wrapper for the `einsums::isa_and_present` type check. The
/// function call operator returns true when the value can be cast to any type
/// in `Types`, or if the value is not present (e.g., nullptr). Example:
/// ```
/// SmallVector<Type> myTypes = ...;
/// if (einsums::all_of(myTypes, einsums::IsaAndPresentPred<VectorType>))
///   ...
/// ```
template <typename... Types>
inline constexpr detail::IsaAndPresentCheckPredicate<Types...> IsaAndPresentPred{};

} // namespace einsums