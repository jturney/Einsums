#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

include(FetchContent)

set(SNITCH_DEFINE_MAIN OFF)

fetchcontent_declare(
    snitch
    GIT_REPOSITORY https://github.com/snitch-org/snitch.git
    GIT_TAG v1.2.3
) # update version number as needed

fetchcontent_makeavailable(snitch)
