#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------
"""Interop adapters that assemble einsums tensors from other packages' data.

Submodules are duck-typed against the foreign package's *data layout* (numpy
buffers plus a little metadata), never its extension types, so einsums and the
foreign package need not be compiled against each other.
"""

from . import psi4 as psi4

__all__ = ["psi4"]
