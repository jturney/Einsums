# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""Waggle's Python side: the terminal viewer, and ``report`` and ``diff`` for saved sessions.

The viewer connects to the server a program's profiler starts (``WAGGLE_SERVER=1``, or a
library's own option), opens saved session files, and replays recorded streams. Only the viewer
needs Textual, imported on use. Libraries add panels and actions through :mod:`waggle.plugin`.
"""

__version__ = "0.1.0"
