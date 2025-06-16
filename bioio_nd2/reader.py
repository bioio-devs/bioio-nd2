#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Dict, Tuple

import nd2
import xarray as xr
from bioio_base import constants, exceptions, io, reader, types
from fsspec.implementations.cached import CachingFileSystem
from fsspec.implementations.local import LocalFileSystem
from fsspec.spec import AbstractFileSystem
from ome_types import OME

###############################################################################


class Reader(reader.Reader):
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
        if nd2.is_supported_file(path, fs.open):
            return True
        raise exceptions.UnsupportedFileFormatError(
            "bioio-nd2", path, "File is not supported by ND2."
        )

    def __init__(self, image: types.PathLike, fs_kwargs: Dict[str, Any] = {}):
        self._fs, self._path = io.pathlike_to_fs(
            image,
            enforce_exists=True,
            fs_kwargs=fs_kwargs,
        )
        # Catch non-local file system and non-caching file system
        if not isinstance(self._fs, LocalFileSystem) and not isinstance(
            self._fs, CachingFileSystem
        ):
            raise ValueError(
                f"Cannot read ND2 from non-local file system. "
                f"Received URI: {self._path}, which points to {type(self._fs)}."
            )

        self._is_supported_image(self._fs, self._path)

    @property
    def scenes(self) -> Tuple[str, ...]:
        with self._fs.open(self._path, "rb") as f:
            with nd2.ND2File(f) as rdr:
                return tuple(rdr._position_names())

    def _read_delayed(self) -> xr.DataArray:
        return self._xarr_reformat(delayed=True)

    def _read_immediate(self) -> xr.DataArray:
        return self._xarr_reformat(delayed=False)

    def _xarr_reformat(self, delayed: bool) -> xr.DataArray:
        with self._fs.open(self._path, "rb") as f:
            with nd2.ND2File(f) as rdr:
                xarr = rdr.to_xarray(
                    delayed=delayed, squeeze=False, position=self.current_scene_index
                )
                xarr.attrs[constants.METADATA_UNPROCESSED] = xarr.attrs.pop("metadata")
                if self.current_scene_index is not None:
                    xarr.attrs[constants.METADATA_UNPROCESSED][
                        "frame"
                    ] = rdr.frame_metadata(self.current_scene_index)

                # include OME metadata as attrs of returned xarray.DataArray if possible
                # (not possible with `nd2` version < 0.7.0; see PR #521)
                try:
                    xarr.attrs[constants.METADATA_PROCESSED] = self.ome_metadata
                except NotImplementedError:
                    pass

        return xarr.isel({nd2.AXIS.POSITION: 0}, missing_dims="ignore")

    @property
    def physical_pixel_sizes(self) -> types.PhysicalPixelSizes:
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
        with self._fs.open(self._path, "rb") as f:
            with nd2.ND2File(f) as rdr:
                return types.PhysicalPixelSizes(*rdr.voxel_size()[::-1])

    @property
    def ome_metadata(self) -> OME:
        """Return OME metadata.

        Returns
        -------
        metadata: OME
            The original metadata transformed into the OME specfication.
            This likely isn't a complete transformation but is guarenteed to
            be a valid transformation.

        Raises
        ------
        NotImplementedError
            No metadata transformer available.
        """
        if hasattr(nd2.ND2File, "ome_metadata"):
            with self._fs.open(self._path, "rb") as f:
                with nd2.ND2File(f) as rdr:
                    return rdr.ome_metadata()
        raise NotImplementedError()
