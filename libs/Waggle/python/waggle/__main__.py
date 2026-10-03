# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""``python -m waggle``: see :mod:`waggle.cli`."""

import sys

from .cli import main

sys.exit(main(prog="python -m waggle"))
