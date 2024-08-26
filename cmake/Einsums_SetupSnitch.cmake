#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

include(FetchContent)

FetchContent_Declare(snitch
        GIT_REPOSITORY https://github.com/snitch-org/snitch.git
        GIT_TAG v1.2.5) # update version number as needed

FetchContent_MakeAvailable(snitch)