import logging
import re
from itertools import product
from numbers import Integral
from typing import Any, Dict, Optional, Tuple, cast

import nd2
import numpy as np
import xarray as xr
from bioio_base import constants, exceptions, io, reader, types
from bioio_base.dimensions import Dimensions
from bioio_base.standard_metadata import StandardMetadata
from fsspec.implementations.cached import CachingFileSystem
from fsspec.implementations.local import LocalFileSystem
from fsspec.spec import AbstractFileSystem
from ome_types import OME

from .plates import (
    Plate,
    WellPosition,
    extract_position_stage_xy_um,
    extract_scene_to_position_index,
    map_scenes_to_wells,
)

###############################################################################

log = logging.getLogger(__name__)

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
    plate : Plate | None
        Plate geometry used to assign scene positions to wells.
        If None, no well assignment is performed and row/column
        metadata will be omitted.
    Raises
    ------
    exceptions.UnsupportedFileFormatError
        If the file is not supported by ND2.
    """

    _scene_to_well_map: Dict[int, WellPosition | None] | None = None
    # must match the bioio-base Reader's signature for the `dims` property,
    # which is cached here for mypy type checking reasons
    _dims: Optional[Dimensions] = None
    _dtype: Optional[np.dtype] = None

    @staticmethod
    def _is_supported_image(fs: AbstractFileSystem, path: str, **kwargs: Any) -> bool:
        if nd2.is_supported_file(path, fs.open):
            return True
        raise exceptions.UnsupportedFileFormatError(
            "bioio-nd2", path, "File is not supported by ND2."
        )

    def __init__(
        self,
        image: types.PathLike,
        fs_kwargs: Dict[str, Any] = {},
        *,
        plate: Plate | None = None,
    ):
        self._plate = plate

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

    @property
    def shape(self) -> Tuple[int, ...]:
        """
        Returns
        -------
        shape: Tuple[int, ...]
            Tuple of the image array's dimensions.
        """
        return self.dims.shape

    @property
    def dtype(self) -> np.dtype:
        """
        Returns
        -------
        dtype: np.dtype
            Data-type of the image array's elements.
        """
        if self._dtype is None:
            with self._fs.open(self._path, "rb") as f:
                with nd2.ND2File(f) as rdr:
                    self._dtype = rdr.dtype

        return self._dtype

    @property
    def dims(self) -> Dimensions:
        """
        Returns
        -------
        dims: Dimensions
            Object with the paired dimension names and their sizes.
        """
        if self._dims is None:
            dims, shape = self._read_dims_and_shape()
            self._dims = Dimensions(dims=dims, shape=shape)

        return self._dims

    def _read_dims_and_shape(self) -> Tuple[Tuple[str, ...], Tuple[int, ...]]:
        """
        Read the dimension names and sizes for the current scene from the ND2
        metadata, without constructing the dask-backed xarray.

        Returns
        -------
        dims: Tuple[str, ...]
            The ordered dimension names for the current scene.
        shape: Tuple[int, ...]
            The size of each dimension, in the same order as ``dims``.
        """
        with self._fs.open(self._path, "rb") as f:
            with nd2.ND2File(f) as rdr:
                dims, shape = self._dims_and_shape_from_nd2(
                    rdr,
                    self.current_scene_index,
                )

        return tuple(dims), tuple(shape)

    @staticmethod
    def _dims_and_shape_from_nd2(
        rdr: nd2.ND2File,
        position: int | None = None,
    ) -> Tuple[list[str], list[int]]:
        """
        Derive the dimension names and sizes from an open ND2 file, dropping the
        position axis for the requested scene.

        Parameters
        ----------
        rdr: nd2.ND2File
            An open ND2 file to read dimension metadata from.
        position: int | None
            The scene (position) index to select. When the file has a position
            axis it is removed from the returned dims/shape. Defaults to None.

        Returns
        -------
        dims: list[str]
            The ordered dimension names, excluding the position axis.
        shape: list[int]
            The size of each dimension, in the same order as ``dims``.

        Raises
        ------
        IndexError
            If ``position`` is out of range for the available positions.
        """
        dims = list(rdr.sizes)
        shape = list(rdr.shape)
        coords = rdr._expand_coords(squeeze=False)

        for missing_dim in set(coords).difference(dims):
            dims.insert(0, missing_dim)
            shape.insert(0, len(coords[missing_dim]))

        try:
            position_index = dims.index(nd2.AXIS.POSITION)
        except ValueError:
            if position and position > 0:
                raise IndexError(
                    f"Position {position} is out of range. Only 1 position available"
                )
        else:
            if position is not None and position >= shape[position_index]:
                raise IndexError(
                    f"Position {position} is out of range. "
                    f"Only {shape[position_index]} positions available"
                )

            shape[position_index] = 1
            dims.pop(position_index)
            shape.pop(position_index)

        return dims, shape

    @staticmethod
    def _indices_from_dim_spec(dim_spec: Any, size: int) -> list[int]:
        """
        Resolve a single-dimension getitem spec into the concrete list of
        source indices it selects.

        Parameters
        ----------
        dim_spec: Any
            The getitem operation for one dimension (an integer, slice, range,
            tuple, or list of integers).
        size: int
            The size of the dimension being indexed.

        Returns
        -------
        indices: list[int]
            The selected indices into the dimension.

        Raises
        ------
        TypeError
            If ``dim_spec`` is not a supported indexer type.
        """
        dim_range = range(size)
        if isinstance(dim_spec, Integral):
            return [dim_range[int(dim_spec)]]

        if isinstance(dim_spec, slice):
            return list(dim_range[dim_spec])

        if isinstance(dim_spec, range):
            dim_spec = list(dim_spec)

        if isinstance(dim_spec, tuple):
            dim_spec = list(dim_spec)

        if isinstance(dim_spec, list):
            return [dim_range[int(index)] for index in dim_spec]

        raise TypeError(f"Unsupported dimension indexer: {type(dim_spec).__name__}")

    @staticmethod
    def _local_dim_spec(dim_spec: Any, size: int) -> Any:
        """
        Translate a source-dimension getitem spec into the equivalent indexer
        for the gathered subset array, which is already reduced to the selected
        indices.

        Parameters
        ----------
        dim_spec: Any
            The original getitem operation for one dimension.
        size: int
            The size of the dimension in the gathered subset array.

        Returns
        -------
        local_spec: Any
            The indexer to apply to the subset array: ``0`` for an integer
            spec (to drop the axis), ``slice(None)`` for a slice, or the full
            list of local indices otherwise.
        """
        if isinstance(dim_spec, Integral):
            return 0

        if isinstance(dim_spec, slice):
            return slice(None)

        return list(range(size))

    @staticmethod
    def _reshape_frame_to_dims(
        frame: np.ndarray,
        current_dims: list[str],
        target_dims: list[str],
        shape_by_dim: Dict[str, int],
    ) -> np.ndarray:
        """
        Reshape and transpose a raw ND2 frame so its axes match the requested
        target dimension order, inserting size-1 axes for any missing dims.

        Parameters
        ----------
        frame: np.ndarray
            The raw frame data as read from the ND2 file.
        current_dims: list[str]
            The dimension names currently present in ``frame``, in order. This
            list is mutated in place as axes are inserted.
        target_dims: list[str]
            The desired dimension names, in order, for the returned frame.
        shape_by_dim: Dict[str, int]
            Mapping of dimension name to its size, used to reshape ``frame``.

        Returns
        -------
        frame: np.ndarray
            The frame reshaped and transposed to match ``target_dims``.
        """
        if current_dims:
            frame = frame.reshape(tuple(shape_by_dim[dim] for dim in current_dims))
        else:
            frame = frame.reshape(())

        for dim in target_dims:
            if dim not in current_dims:
                frame = np.expand_dims(frame, axis=0)
                current_dims.insert(0, dim)

        if current_dims != target_dims:
            frame = frame.transpose([current_dims.index(dim) for dim in target_dims])

        return frame

    def _read_delayed(self) -> xr.DataArray:
        return self._xarr_reformat(delayed=True)

    def _read_immediate(self) -> xr.DataArray:
        return self._xarr_reformat(delayed=False)

    def _read_indexed(self, given_dims: str, dim_specs: list) -> np.ndarray:
        """
        Return the native-order array with ``dim_specs`` applied.

        This lets ``get_image_data`` read only the requested sub-region. It reads
        each requested frame one at a time and applies the selection, so only the
        requested planes are read off disk.

        Parameters
        ----------
        given_dims: str
            The native dimension ordering of the image (``self.dims.order``).
        dim_specs: list
            One getitem operation per dimension in ``given_dims``, as produced by
            ``transforms.compute_dim_specs``.

        Returns
        -------
        data: np.ndarray
            The indexed image data in native (reduced) dimension order.
        """
        position = self.current_scene_index
        with self._fs.open(self._path, "rb") as f:
            with nd2.ND2File(f) as rdr:
                dims, shape = self._dims_and_shape_from_nd2(
                    rdr,
                    position,
                )
                shape_by_dim = dict(zip(dims, shape))
                frame_coord_dims = nd2.AXIS.frame_coords()
                coord_dims = [dim for dim in rdr.sizes if dim not in frame_coord_dims]
                frame_dims = [dim for dim in given_dims if dim in frame_coord_dims]
                current_frame_dims = [
                    dim for dim in rdr.sizes if dim in frame_coord_dims
                ]

                selected_indices = {
                    dim: self._indices_from_dim_spec(
                        dim_specs[dim_index],
                        shape_by_dim[dim],
                    )
                    for dim_index, dim in enumerate(given_dims)
                }
                subset_shape = tuple(len(selected_indices[dim]) for dim in given_dims)
                subset = np.empty(subset_shape, dtype=rdr.dtype)
                local_indexer = tuple(
                    self._local_dim_spec(dim_specs[dim_index], subset_shape[dim_index])
                    for dim_index in range(len(given_dims))
                )

                if 0 in subset_shape:
                    return subset[local_indexer]

                coord_choices = []
                for dim in coord_dims:
                    if dim == nd2.AXIS.POSITION:
                        coord_choices.append([(0, position)])
                    else:
                        coord_choices.append(list(enumerate(selected_indices[dim])))

                for coord_selection in product(*coord_choices):
                    coord_indexes = tuple(index for _, index in coord_selection)
                    if not coord_dims:
                        frame_index = 0
                    else:
                        frame_index = cast(
                            int, rdr._seq_index_from_coords(coord_indexes)
                        )

                    frame = np.asarray(rdr.read_frame(frame_index))
                    frame = self._reshape_frame_to_dims(
                        frame,
                        current_frame_dims.copy(),
                        frame_dims,
                        shape_by_dim,
                    )

                    for frame_axis, dim in enumerate(frame_dims):
                        frame = np.take(
                            frame,
                            selected_indices[dim],
                            axis=frame_axis,
                        )

                    coord_ordinals = {
                        dim: ordinal
                        for dim, (ordinal, _) in zip(coord_dims, coord_selection)
                        if dim != nd2.AXIS.POSITION
                    }
                    subset_index = tuple(
                        coord_ordinals[dim] if dim in coord_ordinals else slice(None)
                        for dim in given_dims
                    )
                    subset[subset_index] = frame

        return subset[local_indexer]

    def _xarr_reformat(self, delayed: bool) -> xr.DataArray:
        with self._fs.open(self._path, "rb") as f:
            with nd2.ND2File(f) as rdr:
                xarr = rdr.to_xarray(
                    delayed=delayed, squeeze=False, position=self.current_scene_index
                )
                xarr.attrs[constants.METADATA_UNPROCESSED] = xarr.attrs.pop("metadata")
                if self.current_scene_index is not None:
                    xarr.attrs[constants.METADATA_UNPROCESSED]["frame"] = (
                        rdr.frame_metadata(self.current_scene_index)
                    )

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
    def binning(self) -> str | None:
        """
        Returns
        -------
        binning : str | None
            Binning value reported by the ND2File metadata, e.g., "1x1".
        """
        with self._fs.open(self._path, "rb") as f, nd2.ND2File(f) as rdr:
            desc = rdr.text_info.get("description", "")
            match = re.search(r"\bBinning:\s*(\d+x\d+)", desc)
            return match.group(1) if match else None

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

    def _get_scene_to_well_map(self) -> Dict[int, WellPosition | None]:
        """
        Compute and cache the mapping of absolute scene index to logical
        well position for this image.

        If no plate geometry is provided, no mapping is performed and all
        scenes map to None.
        """
        if self._scene_to_well_map is not None:
            return self._scene_to_well_map

        if self._plate is None:
            self._scene_to_well_map = {i: None for i in range(len(self.scenes))}
            return self._scene_to_well_map

        with self._fs.open(self._path, "rb") as f:
            with nd2.ND2File(f) as rdr:
                wells = self._plate.generate_wells()

                position_xy = extract_position_stage_xy_um(rdr)
                scene_to_position = extract_scene_to_position_index(
                    rdr, num_scenes=len(self.scenes)
                )

        self._scene_to_well_map = map_scenes_to_wells(
            scene_to_position,
            position_xy,
            wells,
            plate=self._plate,
        )

        return self._scene_to_well_map

    @property
    def row(self) -> str | None:
        """
        Extracts the well row index from XYPosLoop.

        Returns
        -------
        Optional[str]
            The row index as a string. Returns None if parsing fails.
        """
        try:
            pos = self._get_scene_to_well_map().get(self.current_scene_index)
            return pos.row if pos else None
        except Exception as exc:
            log.warning("Failed to extract row: %s", exc, exc_info=True)
            return None

    @property
    def column(self) -> str | None:
        """
        Extracts the well column index from XYPosLoop.

        Returns
        -------
        Optional[str]
            The column index as a string. Returns None if parsing fails.
        """
        try:
            pos = self._get_scene_to_well_map().get(self.current_scene_index)
            return pos.col if pos else None
        except Exception as exc:
            log.warning("Failed to extract column: %s", exc, exc_info=True)
            return None

    @property
    def standard_metadata(self) -> StandardMetadata:
        """
        Return the standard metadata for this reader, updating specific fields.

        This implementation calls the base reader’s standard_metadata property
        via super() and then assigns the new values.
        """
        metadata = super().standard_metadata

        metadata.column = self.column
        metadata.binning = self.binning
        metadata.row = self.row

        # ND2 does not currently support immersion parsing into ome object
        # This can be removed once they do.
        if not metadata.objective or metadata.objective.strip().endswith("Water"):
            return metadata

        try:
            with self._fs.open(self._path, "rb") as fh:
                with nd2.ND2File(fh) as f:
                    ri = f.metadata.channels[0].microscope.immersionRefractiveIndex

                    # 1.33 is the refractive index of water
                    if ri is not None and abs(float(ri) - 1.333) <= 1e-3:
                        metadata.objective = f"{metadata.objective}Water"

        except Exception as err:
            log.warning(f"Failed to patch ND2 objective immersion suffix: {err}")

        return metadata
