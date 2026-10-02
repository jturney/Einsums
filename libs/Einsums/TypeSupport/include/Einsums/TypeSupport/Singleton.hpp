//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <memory>

/**
 * @def EINSUMS_SINGLETON_DEF
 *
 * Makes a class a singleton with a static <tt>Type &get_singleton()</tt>. Place at the start of
 * the class, define a private no-argument constructor, and pair with \c EINSUMS_SINGLETON_IMPL.
 *
 * @param Type The type of singleton to construct.
 *
 * @versionadded{1.0.0}
 */
#define EINSUMS_SINGLETON_DEF(Type)                                                                                                        \
  private:                                                                                                                                 \
    class PrivateConstructorStuff {};                                                                                                      \
                                                                                                                                           \
  public:                                                                                                                                  \
    Type(PrivateConstructorStuff /*ignore*/) : Type() {                                                                                    \
    }                                                                                                                                      \
    static auto get_singleton() -> Type &;                                                                                                 \
    Type(const Type &) = delete;                                                                                                           \
    Type(Type &&)      = delete;

/**
 * @def EINSUMS_SINGLETON_IMPL
 *
 * Creates the code for managing a singleton.
 *
 * @versionadded{1.0.0}
 */
#define EINSUMS_SINGLETON_IMPL(Type)                                                                                                       \
    auto Type::get_singleton() -> Type & {                                                                                                 \
        static std::unique_ptr<Type> singleton_instance = std::make_unique<Type>(PrivateConstructorStuff());                               \
        if (!singleton_instance) {                                                                                                         \
            singleton_instance = std::make_unique<Type>(PrivateConstructorStuff());                                                        \
        }                                                                                                                                  \
        return *singleton_instance;                                                                                                        \
    }
