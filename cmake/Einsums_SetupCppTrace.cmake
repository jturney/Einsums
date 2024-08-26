#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

if (NOT WIN32)
    include(FetchContent)
    fetchcontent_declare(
            cpptrace
            GIT_REPOSITORY https://github.com/jeremy-rifkin/cpptrace.git
            GIT_TAG v0.7.0 # <HASH or TAG>
    )
    fetchcontent_makeavailable(cpptrace)
endif ()
