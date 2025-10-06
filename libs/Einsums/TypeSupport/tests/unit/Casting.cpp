//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Print.hpp>
#include <Einsums/TypeSupport/Casting.hpp>

#include <Einsums/Testing.hpp>

namespace einsums {

// Used to test illegal cast. If a cast doesn't match any of the "real" ones, it will
// match this one.
struct IllegalCast;
template <typename T>
IllegalCast *cast(...) {
    return nullptr;
}

// set up two example classes with conversion facility
struct Bar {
    Bar()            = default;
    Bar(Bar const &) = delete;
    struct Foo *baz();
    struct Foo *caz();
    struct Foo *daz();
    struct Foo *naz();
};

struct Foo {
    Foo(Bar const &) {}
    void ext() const;
};

struct Base {
    virtual ~Base() = default;
};

struct Derived : Base {
    static bool classof(Base const *B) { return true; }
};

struct DerivedNoCast : Base {
    static bool classof(Base const *B) { return false; }
};

template <>
struct detail::IsaImpl<Foo, Bar> {
    static bool doit(Bar const &val) { return true; }
};

// Note for the future - please don't do this. isa_impl is an internal template
// for the implementation of `isa` and should not be exposed this way.
// Completely unrelated types *should* result in compiler errors if you try to
// cast between them.
template <typename T>
struct detail::IsaImpl<Foo, T> {
    static inline bool doit(T const &Val) { return false; }
};

Foo *Bar::baz() {
    return cast<Foo>(this);
}

Foo *Bar::caz() {
    return cast_or_null<Foo>(this);
}

Foo *Bar::daz() {
    return dyn_cast<Foo>(this);
}

Foo *Bar::naz() {
    return dyn_cast_or_null<Foo>(this);
}

Bar *fub() {
    return nullptr;
}

template <>
struct detail::SimplifyType<Foo> {
    using SimpleType = int;
    static SimpleType get_simplified_value(Foo &val) { return 0; }
};

struct T1 {};

struct T2 {
    T2(const T1 &x) {}
    static bool classof(const T1 *x) { return true; }
};

template <>
struct CastInfo<T2, T1> : OptionalValueCast<T2, T1> {};

struct T3 {
    T3(const T1 *x) : hasValue(x != nullptr) {}

    static bool classof(const T1 *x) { return true; }
    bool        hasValue = false;
};

// T3 is convertible from a pointer to T1.
template <>
struct CastInfo<T3, T1 *> : ValueFromPointerCast<T3, T1> {};

struct T4 {
    T4() {}
    T4(const T3 &x) : hasValue(true) {}

    static bool classof(const T3 *x) { return true; }
    bool        hasValue = false;
};

template <>
struct ValueIsPresent<T3> {
    using UnwrappedType = T3;
    static bool      is_present(const T3 &t) { return t.hasValue; }
    static const T3 &unwrap_value(const T3 &t) { return t; }
};

template <>
struct CastInfo<T4, T3> {
    using CastResultType = T4;
    static CastResultType do_cast(const T3 &t) { return T4(t); }
    static CastResultType cast_failed() { return {}; }
    static CastResultType do_cast_if_possible(const T3 &f) { return do_cast(f); }
};

} // namespace einsums

using namespace einsums;

// Test the peculiar behavior of use in SimplifyType.
static_assert(std::is_same_v<detail::SimplifyType<Foo>::SimpleType, int>, "Unexpected SimplifyType result!");
static_assert(std::is_same_v<detail::SimplifyType<Foo *>::SimpleType, Foo *>, "Unexpected SimplifyType result!");

Foo const *null_foo = nullptr;

Bar               B;
extern Bar       &B1;
Bar              &B1 = B;
extern Bar const *B2;
Bar const        &B3 = B1;
Bar const *const  B4 = B2;

TEST_CASE("CastingTest-isa") {
    REQUIRE(isa<Foo>(B1));
    REQUIRE(isa<Foo>(B2));
    REQUIRE(isa<Foo>(B3));
    REQUIRE(isa<Foo>(B4));
}

TEST_CASE("CastTest-isa_and_nonnull") {
    REQUIRE(isa_and_nonnull<Foo>(B2));
    REQUIRE(isa_and_nonnull<Foo>(B4));
    REQUIRE_FALSE(isa_and_nonnull<Foo>(fub()));
}

TEST_CASE("CastingTest-cast") {
    Foo &F1 = cast<Foo>(B1);
    REQUIRE(&F1 != null_foo);
    Foo const *F3 = cast<Foo>(B2);
    REQUIRE(F3 != null_foo);
    Foo const *F4 = cast<Foo>(B2);
    REQUIRE(F4 != null_foo);
    Foo const &F5 = cast<Foo>(B3);
    REQUIRE(&F5 != null_foo);
    Foo const *F6 = cast<Foo>(B4);
    REQUIRE(F6 != null_foo);
    // Can't pass null pointer to cast<>
    // Foo *F7 = cast<Foo>(fub());
    // REQUIRE(F8 == null_fo);
    Foo *F8 = B1.baz();
    REQUIRE(F8 != null_foo);

    std::unique_ptr<Bar const> BP(B2);
    auto                       FP = cast<Foo>(std::move(BP));
    static_assert(std::is_same_v<std::unique_ptr<Foo const>, decltype(FP)>, "Incorrect deduced return type!");
    REQUIRE(FP.get() != null_foo);
    FP.release();
}

TEST_CASE("CastingTest-cast_or_null") {
    Foo const *F11 = cast_or_null<Foo>(B2);
    REQUIRE(F11 != null_foo);
    Foo const *F12 = cast_or_null<Foo>(B4);
    REQUIRE(F12 != null_foo);
    Foo const *F13 = cast_or_null<Foo>(fub());
    REQUIRE(F12 == null_foo);
    Foo *F14 = B1.caz();
    REQUIRE(F14 != null_foo);

    std::unique_ptr<Bar const> BP(fub());
    auto                       FP = cast_or_null<Foo>(std::move(BP));
    REQUIRE(FP.get() == null_foo);
}

TEST_CASE("CastingTest-dyn_cast") {
    Foo const *F1 = dyn_cast<Foo>(B2);
    REQUIRE(F1 != null_foo);
    Foo const *F2 = dyn_cast<Foo>(B2);
    REQUIRE(F2 != null_foo);
    Foo const *F3 = dyn_cast<Foo>(B4);
    REQUIRE(F3 != null_foo);
    // Can't pass null pointer to dyn_cast<>.
    // Foo *F4 = dyn_cast<Foo>(fub());
    // REQUIRE(F4 == null_foo);
    Foo *F5 = B1.daz();
    REQUIRE(F5 != null_foo);

    auto BP = std::make_unique<Bar const>();
    auto FP = dyn_cast<Foo>(BP);
    static_assert(std::is_same_v<std::unique_ptr<Foo const>, decltype(FP)>, "Incorrect deduced return type!");
    REQUIRE(FP.get() != nullptr);
    REQUIRE(BP.get() == nullptr);

    auto BP2 = std::make_unique<Base>();
    auto DP  = dyn_cast<DerivedNoCast>(BP2);
    REQUIRE(DP.get() == nullptr);
    REQUIRE(BP2.get() != nullptr);
}

// All these tests forward to dyn_cast_if_present, so they also provde an
// effective test for its use cases.
TEST_CASE("CastingTest-dyn_cast_or_null") {
    Foo const *F1 = dyn_cast_or_null<Foo>(B2);
    REQUIRE(F1 != null_foo);
    Foo const *F2 = dyn_cast_or_null<Foo>(B2);
    REQUIRE(F2 != null_foo);
    Foo const *F3 = dyn_cast_or_null<Foo>(B4);
    REQUIRE(F3 != null_foo);
    Foo *F4 = dyn_cast_or_null<Foo>(fub());
    REQUIRE(F4 != null_foo);
    Foo *F5 = B1.naz();
    REQUIRE(F5 != null_foo);
    // dyn_cast_if_present should have exactly the same behavior as
    // dyn_cast_or_null.
    Foo const *F6 = dyn_cast_if_present<Foo>(B2);
    REQUIRE(F6 == F2);
}

TEST_CASE("CastingTest-dyn_cast_value_types") {
    T1                t1;
    std::optional<T2> t2 = dyn_cast<T2>(t1);
    REQUIRE(t2);

    T2 *t2ptr = dyn_cast<T2>(&t1);
    REQUIRE(t2ptr != nullptr);

    T3 t3 = dyn_cast<T3>(&t1);
    REQUIRE(t3.hasValue);
}

TEST_CASE("CastingTest-dyn_cast_if_present") {
    std::optional<T1> empty{};
    std::optional<T2> F1 = dyn_cast_if_present<T2>(empty);
    REQUIRE_FALSE(F1.has_value());

    T1                t1;
    std::optional<T2> F2 = dyn_cast_if_present<T2>(t1);
    REQUIRE(F2.has_value());

    T1 *t1Null = nullptr;

    // T3 should have hasValue == false because t1Null is nullptr.
    T3 t3 = dyn_cast_if_present<T3>(t1Null);
    REQUIRE_FALSE(t3.hasValue);

    // Now because of that, T4 should receive the castFailed implementation of its
    // FallibleCastTraits, which default-constructs a T4, which has no value.
    T4 t4 = dyn_cast_if_present<T4>(t3);
    REQUIRE_FALSE(t4.hasValue);
}

TEST_CASE("CastingTest-isa_check_predicates") {
    auto IsaFoo = IsaPred<Foo>;
    REQUIRE(IsaFoo(B1));
    REQUIRE(IsaFoo(B2));
    REQUIRE(IsaFoo(B3));
    REQUIRE(IsaPred<Foo>(B4));
    REQUIRE((IsaPred<Foo, Bar>(B4)));

    auto IsaAndPresentFoo = IsaAndPresentPred<Foo>;
    REQUIRE(IsaAndPresentFoo(B2));
    REQUIRE(IsaAndPresentFoo(B4));
    REQUIRE_FALSE(IsaAndPresentPred<Foo>(fub()));
    REQUIRE_FALSE((IsaAndPresentPred<Foo, Bar>(fub())));
}

std::unique_ptr<Derived> newd() {
    return std::make_unique<Derived>();
}
std::unique_ptr<Base> newb() {
    return std::make_unique<Derived>();
}

TEST_CASE("CastingTest-unique_dyn_cast") {
    Derived *OrigD = nullptr;
    auto     D     = std::make_unique<Derived>();
    OrigD          = D.get();

    // Converting from D to itself is valid, it should return a new unique_ptr
    // and the old one should become nullptr.
    auto NewD = unique_dyn_cast<Derived>(D);
    REQUIRE(OrigD == NewD.get());
    REQUIRE(nullptr == D);

    // Converting from D to B is valid, B should have a value and D should be
    // nullptr.
    auto B = unique_dyn_cast<Base>(NewD);
    REQUIRE(OrigD == B.get());
    REQUIRE(nullptr == NewD);

    // Converting from B to itself is valid, it should return a new unique_ptr
    // and the old one should become nullptr.
    auto NewB = unique_dyn_cast<Base>(B);
    REQUIRE(OrigD == NewB.get());
    REQUIRE(nullptr == B);

    // Converting from B to D is valid, D should have a value and B should be
    // nullptr;
    D = unique_dyn_cast<Derived>(NewB);
    REQUIRE(OrigD == D.get());
    REQUIRE(nullptr == NewB);

    // This is a very contrived test, casting between completely unrelated types
    // should generally fail to compile. See the classof shenanigans we have in
    // the definition of `foo` above.
    auto F = unique_dyn_cast<Foo>(D);
    REQUIRE(nullptr == F);
    REQUIRE(OrigD == D.get());

    // All of the above should also hold for temporaries.
    auto D2 = unique_dyn_cast<Derived>(newd());
    REQUIRE(nullptr != D2);

    auto B2 = unique_dyn_cast<Derived>(newb());
    REQUIRE(nullptr != B2);

    auto B3 = unique_dyn_cast<Base>(newb());
    REQUIRE(nullptr != B3);

    // This is a very contrived test, casting between completely unrelated types
    // should generally fail to compile. See the classof shenanigans we have in
    // the definition of `foo` above.
    auto F2 = unique_dyn_cast<Foo>(newb());
    REQUIRE(nullptr == F2);
}

// These lines are errors...
// Foo *F20 = cast<Foo>(B2);  // Yields const Foo*
// Foo &F21 = cast<Foo>(B3);  // Yields const Foo&
// Foo *F22 = cast<Foo>(B4);  // Yields const Foo*
// Foo &F23 = cast_or_null<Foo>(B1);
// const Foo &F24 = cast_or_null<Foo>(B3);

Bar const *B2 = &B;

namespace inferred_upcasting {
struct Base {
    Base() = default;
};

struct Derived : Base {
    Derived() = default;
};

TEST_CASE("CastingTest-UpcaseIsInferred") {
    Derived D;
    REQUIRE(isa<inferred_upcasting::Base>(D));
    Base *BP = dyn_cast<Base>(&D);
    REQUIRE(BP != nullptr);
}

struct UseInferredUpcast {
    int         Dummy;
    static bool classof(UseInferredUpcast const *) { return false; }
};

TEST_CASE("CastingTest-InferredUpcastTakesPrecedence") {
    UseInferredUpcast UIU;
    REQUIRE(isa<UseInferredUpcast>(&UIU));
}

} // namespace inferred_upcasting

namespace pointer_wrappers {

struct Base {
    bool IsDerived;
    Base(bool IsDerived = false) : IsDerived(IsDerived) {}
};

struct Derived : Base {
    Derived() : Base(true) {}
    static bool classof(Base const *B) { return B->IsDerived; }
};

struct PTy {
    Base *B;

    PTy(Base *B) : B(B) {}
    explicit operator bool() const { return get(); }
    Base    *get() const { return B; }
};

} // namespace pointer_wrappers

namespace einsums {

template <>
struct ValueIsPresent<pointer_wrappers::PTy> {
    using UnwrappedType = pointer_wrappers::PTy;

    static bool is_present(pointer_wrappers::PTy const &P) { return P.get() != nullptr; }

    static UnwrappedType &unwrap_value(pointer_wrappers::PTy &P) { return P; }
};

template <>
struct ValueIsPresent<pointer_wrappers::PTy const> {
    using UnwrappedType = pointer_wrappers::PTy;
    static bool is_present(pointer_wrappers::PTy const &P) { return P.get() != nullptr; }

    static UnwrappedType &unwrap_value(pointer_wrappers::PTy const &P) { return const_cast<UnwrappedType &>(P); }
};

template <>
struct detail::SimplifyType<pointer_wrappers::PTy> {
    using SimpleType = pointer_wrappers::Base *;
    static SimpleType get_simplified_value(pointer_wrappers::PTy &P) { return P.get(); }
};

template <>
struct detail::SimplifyType<pointer_wrappers::PTy const> {
    using SimpleType = pointer_wrappers::Base *;
    static SimpleType get_simplified_value(pointer_wrappers::PTy const &P) { return P.get(); }
};

} // namespace einsums

namespace pointer_wrappers {

// Some objects.
pointer_wrappers::Base    B;
pointer_wrappers::Derived D;

// Mutable "smart" pointers.
pointer_wrappers::PTy MN(nullptr);
pointer_wrappers::PTy MB(&B);
pointer_wrappers::PTy MD(&D);

// Const "smart" pointers.
pointer_wrappers::PTy const CN(nullptr);
pointer_wrappers::PTy const CB(&B);
pointer_wrappers::PTy const CD(&D);

TEST_CASE("CastingTest-smart_isa") {
    CHECK(!isa<pointer_wrappers::Derived>(MB));
    CHECK(!isa<pointer_wrappers::Derived>(CB));
    CHECK(isa<pointer_wrappers::Derived>(MD));
    CHECK(isa<pointer_wrappers::Derived>(CD));
}

TEST_CASE("CastingTest-smart_cast") {
    CHECK(cast<pointer_wrappers::Derived>(MD) == &D);
    CHECK(cast<pointer_wrappers::Derived>(CD) == &D);
}

TEST_CASE("CastingTest-smart_cast_or_null") {
    CHECK(cast_or_null<pointer_wrappers::Derived>(MN) == nullptr);
    CHECK(cast_or_null<pointer_wrappers::Derived>(CN) == nullptr);
    CHECK(cast_or_null<pointer_wrappers::Derived>(MD) == &D);
    CHECK(cast_or_null<pointer_wrappers::Derived>(CD) == &D);
}

TEST_CASE("CastingTest-smart_dyn_cast") {
    CHECK(dyn_cast<pointer_wrappers::Derived>(MB) == nullptr);
    CHECK(dyn_cast<pointer_wrappers::Derived>(CB) == nullptr);
    CHECK(dyn_cast<pointer_wrappers::Derived>(MD) == &D);
    CHECK(dyn_cast<pointer_wrappers::Derived>(CD) == &D);
}

TEST_CASE("CastingTest-smart_dyn_cast_or_null") {
    CHECK(dyn_cast_or_null<pointer_wrappers::Derived>(MN) == nullptr);
    CHECK(dyn_cast_or_null<pointer_wrappers::Derived>(CN) == nullptr);
    CHECK(dyn_cast_or_null<pointer_wrappers::Derived>(MB) == nullptr);
    CHECK(dyn_cast_or_null<pointer_wrappers::Derived>(CB) == nullptr);
    CHECK(dyn_cast_or_null<pointer_wrappers::Derived>(MD) == &D);
    CHECK(dyn_cast_or_null<pointer_wrappers::Derived>(CD) == &D);
}

} // namespace pointer_wrappers