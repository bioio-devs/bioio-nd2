#!/usr/bin/env python
# -*- coding: utf-8 -*-

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


def _count_frame_reads(
    monkeypatch: pytest.MonkeyPatch, work: Any
) -> Tuple[Any, int]:
    """Run ``work()`` while counting real ``nd2.ND2File.read_frame`` calls."""
    reads = 0
    original = nd2.ND2File.read_frame

    def counting_read_frame(self: Any, frame_index: int) -> Any:
        nonlocal reads
        reads += 1
        return original(self, frame_index)

    monkeypatch.setattr(nd2.ND2File, "read_frame", counting_read_frame)
    result = work()
    return result, reads


@pytest.mark.parametrize(
    "filename, set_scene, dimension_order_out, kwargs",
    [
        # Simplest case: single channel selection.
        ("ND2_dims_c2y32x32.nd2", 0, "YX", {"C": 1}),
        # Single-plane collapse across two dims.
        ("ND2_dims_p4z5t3c2y32x32.nd2", 1, "ZYX", {"T": 1, "C": 0}),
        # Slices with a step + fancy indexing together, on a non-zero scene.
        (
            "ND2_dims_p4z5t3c2y32x32.nd2",
            2,
            "TZCYX",
            {"T": slice(0, 2), "C": [0, 1], "Z": slice(0, 5, 2)},
        ),
        # Dimension not present in the file (T) is added with depth 1.
        ("ND2_dims_c2y32x32.nd2", 0, "TCYX", {}),
        # RGB file: the trailing sample (S) axis must survive the round trip.
        ("ND2_dims_rgb_t3p2c2z3x64y64.nd2", 1, "ZCYXS", {"T": 0}),
    ],
)
def test_indexed_read_matches_full_read(
    filename: str,
    set_scene: int,
    dimension_order_out: str,
    kwargs: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    The optimized indexed read must return exactly the same pixels as the
    naive "read the whole scene, then slice" path it replaces.
    """
    uri = LOCAL_RESOURCES_DIR / filename

    optimized = Reader(uri)
    optimized.set_scene(set_scene)
    actual = optimized.get_image_data(dimension_order_out, **kwargs)

    # Reference: fall back to the base-class behavior (materialize, then index).
    reference = Reader(uri)
    reference.set_scene(set_scene)
    monkeypatch.setattr(
        reference,
        "_read_indexed",
        lambda _given_dims, dim_specs: reference.data[tuple(dim_specs)],
    )
    expected = reference.get_image_data(dimension_order_out, **kwargs)

    np.testing.assert_array_equal(actual, expected)


def test_indexed_read_only_reads_requested_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An indexed read should pull only the requested planes off disk."""
    uri = LOCAL_RESOURCES_DIR / "ND2_dims_p4z5t3c2y32x32.nd2"

    rdr = Reader(uri)
    rdr.set_scene(1)

    # Selecting one T and one C leaves only the 5 Z planes to read, far fewer
    # than the 3 * 5 * 2 = 30 frames in the full scene.
    _, reads = _count_frame_reads(
        monkeypatch, lambda: rdr.get_image_data("ZYX", T=1, C=0)
    )
    assert reads == 5


def test_metadata_served_without_reading_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``dims``/``shape``/``dtype`` come from metadata, reading no pixel data."""
    uri = LOCAL_RESOURCES_DIR / "ND2_dims_p4z5t3c2y32x32.nd2"

    def read_metadata() -> None:
        rdr = Reader(uri)
        rdr.set_scene(1)
        _ = rdr.dims
        _ = rdr.shape
        _ = rdr.dtype

    _, reads = _count_frame_reads(monkeypatch, read_metadata)
    assert reads == 0


def test_position_out_of_range_raises() -> None:
    """Selecting a scene index beyond the available positions is an error."""
    uri = LOCAL_RESOURCES_DIR / "ND2_dims_p4z5t3c2y32x32.nd2"
    rdr = Reader(uri)
    rdr._current_scene_index = 99
    with pytest.raises(IndexError, match="out of range"):
        rdr.dims
