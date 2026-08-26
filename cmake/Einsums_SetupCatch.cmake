#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

include(FetchContent)

fetchcontent_declare(
  Catch2
  URL https://github.com/catchorg/Catch2/archive/v3.4.0.tar.gz
  URL_HASH SHA256=122928b814b75717316c71af69bd2b43387643ba076a6ec16e7882bfb2dfacbb
  FIND_PACKAGE_ARGS
  3
)
fetchcontent_makeavailable(Catch2)

# Catch2 decides PER TRANSLATION UNIT whether it has a StringMaker for
# std::string_view, from __cplusplus / _MSVC_LANG in
# internal/catch_compiler_capabilities.hpp. A prebuilt Catch2 whose own sources
# were compiled below C++17 therefore DECLARES the specialization to our C++20
# test TUs while never having compiled its definition into the library, and any
# test that lets Catch2 stringify a string_view fails to link with an undefined
# StringMaker<std::string_view>::convert. conda-forge's Windows Catch2 is such a
# build; its Linux and macOS builds are not.
#
# Ask the library whether the symbol is there rather than guessing from the
# platform, and when it is not, tell our TUs the same thing the library was
# built with. Catch2 then stringifies a string_view through its primary
# StringMaker, which streams it with operator<< - the same text, minus the
# quotes the specialization would add.
include(CheckCXXSourceCompiles)
include(CMakePushCheckState)

cmake_push_check_state()
set(CMAKE_REQUIRED_LIBRARIES Catch2::Catch2)
set(CMAKE_REQUIRED_QUIET TRUE)
check_cxx_source_compiles(
  "#include <catch2/catch_tostring.hpp>
   #include <string_view>
   int main() {
     return static_cast<int>(Catch::Detail::stringify(std::string_view{\"x\"}).size());
   }"
  EINSUMS_CATCH2_HAS_STRING_VIEW_STRINGMAKER
)
cmake_pop_check_state()

if(NOT EINSUMS_CATCH2_HAS_STRING_VIEW_STRINGMAKER)
  einsums_info(
    "Catch2 was built without its std::string_view StringMaker; defining CATCH_CONFIG_NO_CPP17_STRING_VIEW to match"
  )
  set_property(
    TARGET Catch2::Catch2 APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS CATCH_CONFIG_NO_CPP17_STRING_VIEW
  )
endif()
