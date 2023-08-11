#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Dict, Tuple

import nd2
import xarray as xr
from bioio_base import constants, exceptions
from bioio_base import io as io_utils
from bioio_base.reader import Reader as BaseReader
from bioio_base.types import PathLike, PhysicalPixelSizes
from fsspec.implementations.local import LocalFileSystem
from fsspec.spec import AbstractFileSystem

###############################################################################


class Reader(BaseReader):
    """Read NIS-Elements files using the Nikon nd2 SDK.

    This reader requires `nd2` to be installed in the environment.

    Parameters
    ----------
    image : Path or str
        path to file
    fs_kwargs: Dict[str, Any]
        Any specific keyword arguments to pass down to the fsspec created filesystem.
        Default: {}

    Raises
    ------
    exceptions.UnsupportedFileFormatError
        If the file is not supported by ND2.
    """

    @staticmethod
    def _is_supported_image(fs: AbstractFileSystem, path: str, **kwargs: Any) -> bool:
        return nd2.is_supported_file(path, fs.open)

    def __init__(self, image: PathLike, fs_kwargs: Dict[str, Any] = {}):
        self._fs, self._path = io_utils.pathlike_to_fs(
            image,
            enforce_exists=True,
            fs_kwargs=fs_kwargs,
        )
        # Catch non-local file system
        if not isinstance(self._fs, LocalFileSystem):
            raise ValueError(
                f"Cannot read ND2 from non-local file system. "
                f"Received URI: {self._path}, which points to {type(self._fs)}."
            )

        if not self._is_supported_image(self._fs, self._path):
            raise exceptions.UnsupportedFileFormatError(
                self.__class__.__name__, self._path
            )

    @property
    def scenes(self) -> Tuple[str, ...]:
        with nd2.ND2File(self._path) as rdr:
            return tuple(rdr._position_names())

    def _read_delayed(self) -> xr.DataArray:
        return self._xarr_reformat(delayed=True)

    def _read_immediate(self) -> xr.DataArray:
        return self._xarr_reformat(delayed=False)

    def _xarr_reformat(self, delayed: bool) -> xr.DataArray:
        with nd2.ND2File(self._path) as rdr:
            xarr = rdr.to_xarray(
                delayed=delayed, squeeze=False, position=self.current_scene_index
            )
            xarr.attrs[constants.METADATA_UNPROCESSED] = xarr.attrs.pop("metadata")
            if self.current_scene_index is not None:
                xarr.attrs[constants.METADATA_UNPROCESSED][
                    "frame"
                ] = rdr.frame_metadata(self.current_scene_index)
        return xarr.isel({nd2.AXIS.POSITION: 0}, missing_dims="ignore")

    @property
    def physical_pixel_sizes(self) -> PhysicalPixelSizes:
        """
        Returns
        -------
        sizes: PhysicalPixelSizes
            Using available metadata, the floats representing physical pixel sizes for
            dimensions Z, Y, and X.

        Notes
        -----
        We currently do not handle unit attachment to these values. Please see the file
        metadata for unit information.
        """
        with nd2.ND2File(self._path) as rdr:
            return PhysicalPixelSizes(*rdr.voxel_size()[::-1])
