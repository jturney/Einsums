# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""``einsums.__version__`` is the release's version, tag included.

It used to drop the tag, so an alpha reported itself as the final release and
any version comparison (pip, conda, ``packaging``) treated it as one.
"""

from packaging.version import Version

import einsums
from einsums import _core


def test_version_includes_the_tag():
    v = _core._version
    numeric = f"{v.major_version()}.{v.minor_version()}.{v.patch_version()}"
    assert einsums.__version__ == numeric + v.tag()


def test_version_is_pep440_in_canonical_form():
    # The conda version and the git tag are spelled the same way; a separator
    # PEP 440 normalizes away would make them disagree.
    assert str(Version(einsums.__version__)) == einsums.__version__


def test_version_is_a_prerelease_exactly_when_tagged():
    assert Version(einsums.__version__).is_prerelease == bool(_core._version.tag())


def test_version_info_matches_version():
    major, minor, patch, tag = einsums.version_info
    assert einsums.__version__ == f"{major}.{minor}.{patch}{tag}"
