import logging
import re
from contextlib import contextmanager
from datetime import timedelta
from itertools import product
from numbers import Integral
from typing import Any, Dict, Iterator, Optional, Tuple, cast

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

    @contextmanager
    def _open_nd2(self) -> Iterator[nd2.ND2File]:
        """
        Open the backing ND2 file, memory-mapping it when possible.
        """
        if isinstance(self._fs, LocalFileSystem):
            with nd2.ND2File(self._path) as rdr:
                yield rdr
        else:
            with self._fs.open(self._path, "rb") as f:
                with nd2.ND2File(f) as rdr:
                    yield rdr

    @property
    def scenes(self) -> Tuple[str, ...]:
        with self._open_nd2() as rdr:
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
            with self._open_nd2() as rdr:
                self._dtype = rdr.dtype

        return self._dtype

    @property
    def dims(self) -> Dimensions:
        """
        Returns
        -------
        dims: Dimensions
            Paired dimension names and their sizes.
        """
        if self._dims is None:
            with self._open_nd2() as rdr:
                dims = list(rdr.sizes)
                shape = list(rdr.shape)
                coords = rdr._expand_coords(squeeze=False)

                for missing_dim in set(coords).difference(dims):
                    dims.insert(0, missing_dim)
                    shape.insert(0, len(coords[missing_dim]))

                position = self.current_scene_index
                try:
                    position_index = dims.index(nd2.AXIS.POSITION)
                except ValueError:
                    if position and position > 0:
                        raise IndexError(
                            f"Position {position} is out of range. "
                            f"Only 1 position available"
                        )
                else:
                    if position is not None and position >= shape[position_index]:
                        raise IndexError(
                            f"Position {position} is out of range. "
                            f"Only {shape[position_index]} positions available"
                        )
                    dims.pop(position_index)
                    shape.pop(position_index)

            self._dims = Dimensions(dims=tuple(dims), shape=tuple(shape))

        return self._dims

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
        shape_by_dim = dict(zip(self.dims.order, self.dims.shape))
        with self._open_nd2() as rdr:
            # nd2 splits dims into per-frame axes (C/Y/X/S, returned by
            # read_frame) and loop axes (P/T/Z, addressed by sequence index).
            frame_coord_dims = nd2.AXIS.frame_coords()
            coord_dims = [dim for dim in rdr.sizes if dim not in frame_coord_dims]
            frame_dims = [dim for dim in given_dims if dim in frame_coord_dims]

            # Source indices each spec selects
            selected_indices = {
                dim: np.atleast_1d(np.arange(shape_by_dim[dim])[spec]).tolist()
                for dim, spec in zip(given_dims, dim_specs)
            }
            subset_shape = tuple(len(selected_indices[dim]) for dim in given_dims)
            subset = np.empty(subset_shape, dtype=rdr.dtype)
            local_indexer = tuple(
                0 if isinstance(spec, Integral) else slice(None) for spec in dim_specs
            )

            if 0 in subset_shape:
                return subset[local_indexer]

            coord_choices = []
            for dim in coord_dims:
                if dim == nd2.AXIS.POSITION:
                    coord_choices.append([(0, position)])
                else:
                    coord_choices.append(list(enumerate(selected_indices[dim])))

            # Map each requested plane to its (sequence index, destination in
            # subset).
            planes = []
            for coord_selection in product(*coord_choices):
                coord_indexes = tuple(index for _, index in coord_selection)
                if not coord_dims:
                    frame_index = 0
                else:
                    frame_index = cast(int, rdr._seq_index_from_coords(coord_indexes))

                coord_ordinals = {
                    dim: ordinal
                    for dim, (ordinal, _) in zip(coord_dims, coord_selection)
                    if dim != nd2.AXIS.POSITION
                }
                subset_index = tuple(
                    coord_ordinals[dim] if dim in coord_ordinals else slice(None)
                    for dim in given_dims
                )
                planes.append((frame_index, subset_index))

            for frame_index, subset_index in sorted(planes, key=lambda p: p[0]):
                # reshape prepends size-1 axes for any singleton frame dims
                # missing from read_frame's output.
                frame = np.asarray(rdr.read_frame(frame_index)).reshape(
                    tuple(shape_by_dim[dim] for dim in frame_dims)
                )

                for frame_axis, dim in enumerate(frame_dims):
                    frame = np.take(
                        frame,
                        selected_indices[dim],
                        axis=frame_axis,
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
        with self._open_nd2() as rdr:
            return types.PhysicalPixelSizes(*rdr.voxel_size()[::-1])

    @staticmethod
    def _time_period_ms(experiment: list) -> Optional[float]:
        """
        Extract the inter-frame time interval, in milliseconds, from an ND2
        experiment's time loop.

        Parameters
        ----------
        experiment: list
            The ND2 experiment loops, as returned by `nd2.ND2File.experiment`.

        Returns
        -------
        period_ms: Optional[float]
            The interval in milliseconds, or None when there is no time loop with
            a single well-defined interval.
        """
        for loop in experiment:
            if isinstance(loop, nd2.structures.TimeLoop):
                return loop.parameters.periodMs
            if isinstance(loop, nd2.structures.NETimeLoop):
                periods = loop.parameters.periods
                if len(periods) == 1:
                    return periods[0].periodMs
        return None

    @property
    def time_interval(self) -> types.TimeInterval:
        """
        Returns
        -------
        interval: TimeInterval
            The time between frames for dimension T as a ``datetime.timedelta``,
            read from the ND2 experiment's time loop. ``None`` when the file has
            no time loop with a single well-defined interval.
        """
        with self._fs.open(self._path, "rb") as f:
            with nd2.ND2File(f) as rdr:
                period_ms = self._time_period_ms(rdr.experiment)

        if period_ms is None or period_ms <= 0:
            return None
        return timedelta(milliseconds=period_ms)

    @staticmethod
    def _ome_unit_to_pint(ome_unit: Any) -> Optional[types.Unit]:
        """
        Convert an OME unit enum into a `pint.Unit` from the shared BioIO
        registry.

        Parameters
        ----------
        ome_unit: Any
            An OME unit enum (e.g. `UnitsLength.MICROMETER`), whose `.value` is
            the unit symbol (`"µm"`, `"s"`) that `bioio_base.types.ureg` parses.

        Returns
        -------
        unit: Optional[types.Unit]
            The corresponding `pint.Unit`, or None if the input is absent or
            unrecognized.
        """
        if ome_unit is None:
            return None
        try:
            return types.ureg(ome_unit.value).units
        except Exception:
            return None

    @property
    def dimension_properties(self) -> types.DimensionProperties:
        """
        Per-dimension metadata describing semantic meaning and units.
        """
        s = self.scale
        if not hasattr(nd2.ND2File, "ome_metadata"):
            return super().dimension_properties

        try:
            with self._fs.open(self._path, "rb") as f:
                with nd2.ND2File(f) as rdr:
                    pixels = rdr.ome_metadata().images[0].pixels
        except Exception as err:
            log.warning(f"Failed to read ND2 dimension units from OME metadata: {err}")
            return super().dimension_properties

        time_unit = self._ome_unit_to_pint(pixels.time_increment_unit)
        z_unit = self._ome_unit_to_pint(pixels.physical_size_z_unit)
        y_unit = self._ome_unit_to_pint(pixels.physical_size_y_unit)
        x_unit = self._ome_unit_to_pint(pixels.physical_size_x_unit)

        return types.DimensionProperties(
            T=types.DimensionProperty(
                type="time" if s.T is not None else None,
                unit=time_unit if s.T is not None else None,
            ),
            C=types.DimensionProperty(
                type="channel" if s.C is not None else None,
                unit=None,
            ),
            Z=types.DimensionProperty(
                type="space" if s.Z is not None else None,
                unit=z_unit if s.Z is not None else None,
            ),
            Y=types.DimensionProperty(
                type="space" if s.Y is not None else None,
                unit=y_unit if s.Y is not None else None,
            ),
            X=types.DimensionProperty(
                type="space" if s.X is not None else None,
                unit=x_unit if s.X is not None else None,
            ),
        )

    @property
    def binning(self) -> str | None:
        """
        Returns
        -------
        binning : str | None
            Binning value reported by the ND2File metadata, e.g., "1x1".
        """
        with self._open_nd2() as rdr:
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
            with self._open_nd2() as rdr:
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

        with self._open_nd2() as rdr:
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
            with self._open_nd2() as f:
                ri = f.metadata.channels[0].microscope.immersionRefractiveIndex

                # 1.33 is the refractive index of water
                if ri is not None and abs(float(ri) - 1.333) <= 1e-3:
                    metadata.objective = f"{metadata.objective}Water"

        except Exception as err:
            log.warning(f"Failed to patch ND2 objective immersion suffix: {err}")

        return metadata
