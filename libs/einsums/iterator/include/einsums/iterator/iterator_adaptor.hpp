//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/iterator/iterator_facade.hpp>
#include <einsums/iterator/traits/is_iterator.hpp>
#include <einsums/type_support/lazy_conditional.hpp>
#include <einsums/type_support/type_identity.hpp>

#include <algorithm>
#include <iterator>
#include <memory>
#include <type_traits>
#include <utility>

namespace einsums::util {
// Default template argument handling for iterator_adaptor
namespace detail {
///////////////////////////////////////////////////////////////////////
template <typename Iterator>
struct value_type_iterator_traits_helper {
    using type = typename std::iterator_traits<Iterator>::value_type;
};

template <typename Iterator>
struct reference_iterator_traits_helper {
    using type = typename std::iterator_traits<Iterator>::reference;
};

template <typename Iterator>
struct category_iterator_traits_helper {
    using type = typename std::iterator_traits<Iterator>::iterator_category;
};

template <typename Iterator>
struct difference_type_iterator_traits_helper {
    using type = typename std::iterator_traits<Iterator>::difference_type;
};

// A meta-function which computes an iterator_adaptor's base class,
// a specialization of iterator_facade.
template <typename Derived, typename Base, typename Value, typename Category, typename Reference, typename Difference, typename Pointer>
struct iterator_adaptor_base {
    // the following type calculations use lazy_conditional to avoid
    // premature instantiations
    using value_type =
        std::conditional_t<std::is_void<Value>::value,
                           ::einsums::detail::lazy_conditional_t<std::is_void<Reference>::value, value_type_iterator_traits_helper<Base>,
                                                                 std::remove_reference<Reference>>,
                           Value>;

    using reference_type =
        std::conditional_t<std::is_void<Reference>::value,
                           ::einsums::detail::lazy_conditional_t<std::is_void<Value>::value, reference_iterator_traits_helper<Base>,
                                                                 std::add_lvalue_reference<Value>>,
                           Reference>;

    using iterator_category = ::einsums::detail::lazy_conditional_t<std::is_void<Category>::value, category_iterator_traits_helper<Base>,
                                                                    ::einsums::detail::type_identity<Category>>;

    using difference_type =
        ::einsums::detail::lazy_conditional_t<std::is_void<Difference>::value, difference_type_iterator_traits_helper<Base>,
                                              ::einsums::detail::type_identity<Difference>>;

    using type = iterator_facade<Derived, value_type, iterator_category, reference_type, difference_type, Pointer>;
};
} // namespace detail

// Iterator adaptor
//
// The idea is that when the user needs
// to fiddle with the reference type it is highly likely that the
// iterator category has to be adjusted as well.  Any of the
// following four template arguments may be omitted or explicitly
// replaced by void.
//
//   Value - if supplied, the value_type of the resulting iterator, unless
//      const. If const, a conforming compiler strips const-ness for the
//      value_type. If not supplied, iterator_traits<Base>::value_type is used
//
//   Category - the traversal category of the resulting iterator. If not
//      supplied, iterator_traversal<Base>::type is used.
//
//   Reference - the reference type of the resulting iterator, and in
//      particular, the result type of operator*(). If not supplied but
//      Value is supplied, Value& is used. Otherwise
//      iterator_traits<Base>::reference is used.
//
//   Difference - the difference_type of the resulting iterator. If not
//      supplied, iterator_traits<Base>::difference_type is used.
//
template <typename Derived, typename Base, typename Value = void, typename Category = void, typename Reference = void,
          typename Difference = void, typename Pointer = void>
class iterator_adaptor
    : public einsums::util::detail::iterator_adaptor_base<Derived, Base, Value, Category, Reference, Difference, Pointer>::type {
  protected:
    using base_adaptor_type =
        typename einsums::util::detail::iterator_adaptor_base<Derived, Base, Value, Category, Reference, Difference, Pointer>::type;

    friend class einsums::util::iterator_core_access;

  public:
    iterator_adaptor() = default;

    EINSUMS_HOST_DEVICE explicit iterator_adaptor(Base const &iter) : iterator_(iter) {}

    using base_type = Base;

    EINSUMS_HOST_DEVICE EINSUMS_FORCEINLINE Base const &base() const { return iterator_; }

  protected:
    // for convenience in derived classes
    using iterator_adaptor_ = iterator_adaptor<Derived, Base, Value, Category, Reference, Difference, Pointer>;

    // lvalue access to the Base object for Derived
    EINSUMS_HOST_DEVICE EINSUMS_FORCEINLINE Base const &base_reference() const { return iterator_; }

    EINSUMS_HOST_DEVICE EINSUMS_FORCEINLINE Base &base_reference() { return iterator_; }

  private:
    // Core iterator interface for iterator_facade.  This is private
    // to prevent temptation for Derived classes to use it, which
    // will often result in an error.  Derived classes should use
    // base_reference(), above, to get direct access to m_iterator.
    EINSUMS_HOST_DEVICE EINSUMS_FORCEINLINE typename base_adaptor_type::reference dereference() const { return *iterator_; }

    template <typename OtherDerived, typename OtherIterator, typename V, typename C, typename R, typename D, typename P>
    EINSUMS_HOST_DEVICE EINSUMS_FORCEINLINE bool equal(iterator_adaptor<OtherDerived, OtherIterator, V, C, R, D, P> const &x) const {
        // Maybe re-add with same_distance
        //  static_assert(
        //      (detail::same_category_and_difference<Derived,OtherDerived>::value)
        //  );
        return iterator_ == x.base();
    }

    // prevent this function from being instantiated if not needed
    template <typename DifferenceType>
    EINSUMS_HOST_DEVICE EINSUMS_FORCEINLINE void advance(DifferenceType n) {
        std::advance(iterator_, n);
    }

    EINSUMS_HOST_DEVICE EINSUMS_FORCEINLINE void increment() { ++iterator_; }

    // prevent this function from being instantiated if not needed
    template <typename Iterator = Base, typename Enable = typename std::enable_if<traits::is_bidirectional_iterator<Iterator>::value>::type>
    EINSUMS_HOST_DEVICE EINSUMS_FORCEINLINE void decrement() {
        --iterator_;
    }

    template <typename OtherDerived, typename OtherIterator, typename V, typename C, typename R, typename D, typename P>
    EINSUMS_HOST_DEVICE EINSUMS_FORCEINLINE typename base_adaptor_type::difference_type
                        distance_to(iterator_adaptor<OtherDerived, OtherIterator, V, C, R, D, P> const &y) const {
        // Maybe re-add with same_distance
        //  static_assert(
        //      (detail::same_category_and_difference<Derived,OtherDerived>::value)
        //  );
        return std::distance(iterator_, y.base());
    }

  private: // data members
    Base iterator_;
};
} // namespace einsums::util
