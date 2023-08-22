# -*- coding: utf-8 -*-

"""Top-level package for bioio_nd2."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("bioio-nd2")
except PackageNotFoundError:
    __version__ = "uninstalled"

__author__ = "bioio-devs"
__email__ = "brian.whitney@alleninstitute.org"

from .reader import Reader
from .reader_metadata import ReaderMetadata

__all__ = ["Reader", "ReaderMetadata"]
