"""Tests for the top-level package metadata."""

import leaseguard


def test_version_is_defined() -> None:
    """The package exposes its current release version."""
    assert leaseguard.__version__ == "0.1.0"
