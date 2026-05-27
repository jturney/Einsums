# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""
einsums
-------

This module allows for interaction with the C++ Einsums library.
"""

import sys
import os
import atexit

__modpath = os.path.dirname(__file__)

if __modpath not in sys.path:
    sys.path.append(__modpath)

try:
    from . import core
except (ModuleNotFoundError, ImportError):
    try:
        print("Importing core in a different way.")
        import core
    except (ModuleNotFoundError, ImportError) as e:
        raise RuntimeError(
            f"File is {__file__}, path is {sys.path} and version is {sys.version}"
        ) from e

from . import utils  # pylint: disable=wrong-import-position

def __initialize() :
    """
    Filter out Python arguments and pass on einsums arguments. Einsums arguments are prefixed with
    '--einsums'.
    """
    pass_args = [sys.argv[0]]

    if len(sys.argv) > 1 :
        einsums_arg = False
        for arg in sys.argv[1:] :
            if einsums_arg :
                pass_args.append(arg)
                einsums_arg = False
            elif arg == "--einsums" :
                einsums_arg = True

    core.initialize(pass_args)

__initialize()

atexit.register(core.finalize)
