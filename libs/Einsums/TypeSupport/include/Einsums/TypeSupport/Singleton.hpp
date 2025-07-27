//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

/**
 * @def EINSUMS_SINGLETON_DEF
 *
 * Turns a C++ class into a singleton. Place this at the beginning of the class. You will then need to define a private
 * constructor with no arguments to actually construct the singleton stuff. This macro is only for the definition of the
 * class and does not contain any code. Make sure to use a matching \c EINSUMS_SINGLETON_IMPL somewhere else to
 * get the code to compile.
 *
 * This will provide the two methods <tt>Type &get_singleton()</tt> and <tt>finalize_singleton()</tt>.
 *
 * @param Type The type of singleton to construct.
 */
#define EINSUMS_SINGLETON_DEF(Type)                                                                                                        \
  private:                                                                                                                                 \
    class PrivateConstructorStuff {};                                                                                                      \
                                                                                                                                           \
  public:                                                                                                                                  \
    Type(PrivateConstructorStuff) : Type() {                                                                                               \
    }                                                                                                                                      \
    static Type &get_singleton();                                                                                                          \
    Type(const Type &) = delete;                                                                                                           \
    Type(Type &&)      = delete;

/**
 * @def EINSUMS_SINGLETON_IMPL
 *
 * Creates the code for managing a singleton.
 */
#define EINSUMS_SINGLETON_IMPL(Type)                                                                                                       \
    Type &Type::get_singleton() {                                                                                                          \
        static std::unique_ptr<Type> singleton_instance = std::make_unique<Type>(PrivateConstructorStuff());                               \
        return *singleton_instance;                                                                                                        \
    }
