# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""``python -m einsums.tune``: the same as ``einsums tune``, for an environment without the command on its path."""

import sys

if __name__ == "__main__":
    from einsums.cli import main

    sys.exit(main(["tune", *sys.argv[1:]], prog="python -m einsums"))
