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
