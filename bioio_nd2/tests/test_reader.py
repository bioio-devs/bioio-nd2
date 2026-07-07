#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datetime import timedelta
from typing import Any, List, Tuple, Union

import numpy as np
import pytest
from bioio_base import exceptions, test_utilities
from ome_types import OME

from bioio_nd2 import Reader

from .conftest import LOCAL_RESOURCES_DIR

nd2 = pytest.importorskip("nd2")

# nd2 0.4.3 and above improves detection of position names
if tuple(int(x) for x in nd2.__version__.split(".")) >= (0, 4, 3):
    pos_names = ("point name 1", "point name 2", "point name 3", "point name 4")
else:
    pos_names = ("XYPos:0", "XYPos:1", "XYPos:2", "XYPos:3")


@pytest.mark.parametrize(
    "filename, "
    "set_scene, "
    "expected_scenes, "
    "expected_shape, "
    "expected_dtype, "
    "expected_dims_order, "
    "expected_channel_names, "
    "expected_physical_pixel_sizes, "
    "expected_metadata_type",
    [
        pytest.param(
            "example.txt",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            marks=pytest.mark.xfail(raises=exceptions.UnsupportedFileFormatError),
        ),
        pytest.param(
            "ND2_aryeh_but3_cont200-1.nd2",
            "XYPos:0",
            ("XYPos:0", "XYPos:1", "XYPos:2", "XYPos:3", "XYPos:4"),
            (1, 2, 1040, 1392),
            np.uint16,
            "TCYX",
            ["20phase", "20xDiO"],
            (1, 50, 50),
            dict,
        ),
        (
            "ND2_jonas_header_test2.nd2",
            "XYPos:0",
            ("XYPos:0",),
            (1, 4, 5, 520, 696),
            np.uint16,
            "CTZYX",
            ["Jonas_DIC"],
            (0.5, 0.12863494437945, 0.12863494437945),
            OME,
        ),
        (
            "ND2_maxime_BF007.nd2",
            "XYPos:0",
            ("XYPos:0",),
            (1, 156, 164),
            np.uint16,
            "CYX",
            ["405/488/561/633nm"],
            (1.0, 0.158389678930686, 0.158389678930686),
            OME,
        ),
        (
            "ND2_dims_p4z5t3c2y32x32.nd2",
            pos_names[0],
            pos_names,
            (3, 5, 2, 32, 32),
            np.uint16,
            "TZCYX",
            ["Widefield Green", "Widefield Red"],
            (1.0, 0.652452890023035, 0.652452890023035),
            OME,
        ),
        (
            "ND2_dims_c2y32x32.nd2",
            "XYPos:0",
            ("XYPos:0",),
            (2, 32, 32),
            np.uint16,
            "CYX",
            ["Widefield Green", "Widefield Red"],
            (1.0, 0.652452890023035, 0.652452890023035),
            OME,
        ),
        (
            "ND2_dims_p1z5t3c2y32x32.nd2",
            "XYPos:0",
            ("XYPos:0",),
            (3, 5, 2, 32, 32),
            np.uint16,
            "TZCYX",
            ["Widefield Green", "Widefield Red"],
            (1.0, 0.652452890023035, 0.652452890023035),
            OME,
        ),
        (
            "ND2_dims_p2z5t3-2c4y32x32.nd2",
            pos_names[1],
            pos_names[:2],
            (5, 5, 4, 32, 32),
            np.uint16,
            "TZCYX",
            ["Widefield Green", "Widefield Red", "Widefield Far-Red", "Brightfield"],
            (1.0, 0.652452890023035, 0.652452890023035),
            OME,
        ),
        (
            "ND2_dims_t3c2y32x32.nd2",
            "XYPos:0",
            ("XYPos:0",),
            (3, 2, 32, 32),
            np.uint16,
            "TCYX",
            ["Widefield Green", "Widefield Red"],
            (1.0, 0.652452890023035, 0.652452890023035),
            OME,
        ),
        (
            "ND2_dims_rgb_t3p2c2z3x64y64.nd2",
            "XYPos:1",
            ("XYPos:0", "XYPos:1"),
            (3, 3, 2, 32, 32, 3),
            np.uint8,
            "TZCYXS",
            ["Brightfield", "Brightfield"],
            (0.01, 0.34285714285714286, 0.34285714285714286),
            OME,
        ),
        (
            "ND2_dims_rgb.nd2",
            "XYPos:0",
            ("XYPos:0",),
            (1, 64, 64, 3),
            np.uint8,
            "CYXS",
            ["Brightfield"],
            (1.0, 0.34285714285714286, 0.34285714285714286),
            OME,
        ),
    ],
)
def test_nd2_reader(
    filename: str,
    set_scene: str,
    expected_scenes: Tuple[str, ...],
    expected_shape: Tuple[int, ...],
    expected_dtype: np.dtype,
    expected_dims_order: str,
    expected_channel_names: List[str],
    expected_physical_pixel_sizes: Tuple[float, float, float],
    expected_metadata_type: Union[type, Tuple[Union[type, Tuple[Any, ...]], ...]],
) -> None:
    # Construct full filepath
    uri = LOCAL_RESOURCES_DIR / filename

    # Run checks
    test_utilities.run_image_file_checks(
        ImageContainer=Reader,
        image=uri,
        set_scene=set_scene,
        expected_scenes=expected_scenes,
        expected_current_scene=set_scene,
        expected_shape=expected_shape,
        expected_dtype=expected_dtype,
        expected_dims_order=expected_dims_order,
        expected_channel_names=expected_channel_names,
        expected_physical_pixel_sizes=expected_physical_pixel_sizes,
        expected_metadata_type=expected_metadata_type,
    )


@pytest.mark.parametrize(
    "filename",
    [
        "ND2_jonas_header_test2.nd2",
        "ND2_maxime_BF007.nd2",
        "ND2_dims_p4z5t3c2y32x32.nd2",
    ],
)
def test_ome_metadata(filename: str) -> None:
    # Get full filepath
    uri = LOCAL_RESOURCES_DIR / filename

    # Init image
    img = Reader(uri)

    # Test the transform
    assert isinstance(img.ome_metadata, OME)


@pytest.mark.parametrize(
    "cache",
    [True, False],
)
def test_frame_metadata(cache: bool) -> None:
    uri = LOCAL_RESOURCES_DIR / "ND2_dims_rgb_t3p2c2z3x64y64.nd2"
    rdr = Reader("simplecache::" + str(uri) if cache else uri)
    rdr.set_scene(0)
    assert isinstance(
        rdr.xarray_data.attrs["unprocessed"]["frame"], nd2.structures.FrameMetadata
    )


@pytest.mark.parametrize(
    "filename, expected_interval",
    [
        # Time-lapse files carry a TimeLoop, so time_interval is populated.
        ("ND2_jonas_header_test2.nd2", timedelta(seconds=5)),
        ("ND2_dims_rgb_t3p2c2z3x64y64.nd2", timedelta(seconds=1)),
        ("ND2_dims_t3c2y32x32.nd2", timedelta(milliseconds=1)),
        # Legacy file (no OME metadata) still exposes its experiment loop.
        ("ND2_aryeh_but3_cont200-1.nd2", timedelta(minutes=15)),
        # Files without a time loop report no interval.
        ("ND2_dims_c2y32x32.nd2", None),
        ("ND2_maxime_BF007.nd2", None),
    ],
)
def test_time_interval(
    filename: str,
    expected_interval: object,
) -> None:
    rdr = Reader(LOCAL_RESOURCES_DIR / filename)

    assert rdr.time_interval == expected_interval


@pytest.mark.parametrize(
    "filename, expected_t, expected_space",
    [
        # Time-lapse with OME metadata: seconds on T, microns on ZYX.
        (
            "ND2_jonas_header_test2.nd2",
            ("time", "second"),
            "micrometer",
        ),
        # No time loop: T carries no type/unit; ZYX still microns.
        (
            "ND2_dims_c2y32x32.nd2",
            (None, None),
            "micrometer",
        ),
        # Legacy file: no OME metadata, so units fall back to None while the
        # semantic types are still derived from the (present) scale.
        (
            "ND2_aryeh_but3_cont200-1.nd2",
            ("time", None),
            None,
        ),
    ],
)
def test_dimension_properties(
    filename: str,
    expected_t: Tuple[object, object],
    expected_space: object,
) -> None:
    rdr = Reader(LOCAL_RESOURCES_DIR / filename)

    dp = rdr.dimension_properties

    expected_t_type, expected_t_unit = expected_t
    assert dp.T.type == expected_t_type
    assert (str(dp.T.unit) if dp.T.unit is not None else None) == expected_t_unit

    # Channels never carry a spatial/temporal unit.
    assert dp.C.type is None
    assert dp.C.unit is None

    for axis in (dp.Z, dp.Y, dp.X):
        assert axis.type == "space"
        assert (str(axis.unit) if axis.unit is not None else None) == expected_space


@pytest.mark.parametrize(
    "filename, set_scene, dim_specs",
    [
        # Single channel selection (drops the C axis).
        ("ND2_dims_c2y32x32.nd2", 0, [1, slice(None), slice(None)]),
        # Single-plane collapse across two dims (native order TZCYX).
        (
            "ND2_dims_p4z5t3c2y32x32.nd2",
            1,
            [1, 0, slice(None), slice(None), slice(None)],
        ),
        # Slices with a step + fancy indexing together, on a non-zero scene.
        (
            "ND2_dims_p4z5t3c2y32x32.nd2",
            2,
            [slice(0, 2), slice(0, 5, 2), [0, 1], slice(None), slice(None)],
        ),
        # RGB file (native order TZCYXS): the trailing sample axis must survive.
        (
            "ND2_dims_rgb_t3p2c2z3x64y64.nd2",
            1,
            [0, slice(None), slice(None), slice(None), slice(None), slice(None)],
        ),
    ],
)
def test_indexed_read_matches_full_read(
    filename: str,
    set_scene: int,
    dim_specs: list,
) -> None:
    """
    Checks that slice read matches full data slice
    """
    uri = LOCAL_RESOURCES_DIR / filename

    reader = Reader(uri)
    reader.set_scene(set_scene)

    expected = reader.data[tuple(dim_specs)]
    actual = reader._read_indexed(reader.dims.order, dim_specs)

    np.testing.assert_array_equal(actual, expected)
