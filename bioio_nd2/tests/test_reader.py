#!/usr/bin/env python
# -*- coding: utf-8 -*-

from contextlib import nullcontext
from typing import Any, List, Tuple, Union

import numpy as np
import pytest
import xarray as xr
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


def test_read_indexed_reads_only_requested_nd2_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = np.arange(2 * 2 * 3 * 4 * 5).reshape(2, 2, 3, 4, 5)
    frame_reads = []

    class FakeND2File:
        sizes = {"P": 2, "T": 2, "C": 3, "Y": 4, "X": 5}
        shape = (2, 2, 3, 4, 5)
        dtype = data.dtype

        def __init__(self, file: object) -> None:
            pass

        def __enter__(self) -> "FakeND2File":
            return self

        def __exit__(self, *args: object) -> None:
            pass

        def _expand_coords(self, squeeze: bool = True) -> dict[str, object]:
            assert squeeze is False
            return {
                nd2.AXIS.POSITION: range(2),
                "T": range(2),
                "C": range(3),
                "Y": range(4),
                "X": range(5),
            }

        def _seq_index_from_coords(self, coords: Tuple[int, ...]) -> int:
            return int(np.ravel_multi_index(coords, (2, 2)))

        def read_frame(self, frame_index: int) -> np.ndarray:
            frame_reads.append(frame_index)
            return data.reshape(4, 3, 4, 5)[frame_index]

    class FakeFS:
        def open(self, path: str, mode: str) -> Any:
            return nullcontext(object())

    monkeypatch.setattr("bioio_nd2.reader.nd2.ND2File", FakeND2File)

    rdr = Reader.__new__(Reader)
    rdr._fs = FakeFS()
    rdr._path = "example.nd2"
    rdr._dims = None
    rdr._current_scene_index = 1

    def fail_delayed_read() -> xr.DataArray:
        raise AssertionError("_read_delayed should not be used for indexed reads")

    monkeypatch.setattr(rdr, "_read_delayed", fail_delayed_read)

    actual = rdr._read_indexed(
        "TCYX",
        [
            slice(None),
            slice(1, 3),
            slice(1, 4),
            slice(0, 5, 2),
        ],
    )

    np.testing.assert_array_equal(
        actual,
        data[1, :, 1:3, 1:4, 0:5:2],
    )
    assert frame_reads == [2, 3]

    frame_reads.clear()
    dim_specs = [
        slice(None),
        [0, 2],
        [1, 3],
        slice(0, 5, 2),
    ]
    actual = rdr._read_indexed("TCYX", dim_specs)

    np.testing.assert_array_equal(
        actual,
        data[1][tuple(dim_specs)],
    )
    assert frame_reads == [2, 3]

    frame_reads.clear()
    rdr._current_scene_index = 0
    actual = rdr._read_indexed(
        "TCYX",
        [
            slice(None),
            slice(1, 3),
            slice(1, 4),
            slice(0, 5, 2),
        ],
    )

    np.testing.assert_array_equal(
        actual,
        data[0, :, 1:3, 1:4, 0:5:2],
    )
    assert frame_reads == [0, 1]

    rdr._current_scene_index = 2
    with pytest.raises(IndexError, match="Position 2 is out of range"):
        rdr._read_indexed("TCYX", dim_specs)


def test_read_indexed_reads_single_middle_plane(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = np.arange(3 * 5 * 7 * 4 * 6 * 8).reshape(3, 5, 7, 4, 6, 8)
    frame_reads = []

    class FakeND2File:
        sizes = {
            nd2.AXIS.POSITION: 3,
            "T": 5,
            "Z": 7,
            "C": 4,
            "Y": 6,
            "X": 8,
        }
        shape = (3, 5, 7, 4, 6, 8)
        dtype = data.dtype

        def __init__(self, file: object) -> None:
            pass

        def __enter__(self) -> "FakeND2File":
            return self

        def __exit__(self, *args: object) -> None:
            pass

        def _expand_coords(self, squeeze: bool = True) -> dict[str, object]:
            assert squeeze is False
            return {
                nd2.AXIS.POSITION: range(3),
                "T": range(5),
                "Z": range(7),
                "C": range(4),
                "Y": range(6),
                "X": range(8),
            }

        def _seq_index_from_coords(self, coords: Tuple[int, ...]) -> int:
            return int(np.ravel_multi_index(coords, (3, 5, 7)))

        def read_frame(self, frame_index: int) -> np.ndarray:
            frame_reads.append(frame_index)
            return data.reshape(3 * 5 * 7, 4, 6, 8)[frame_index]

    class FakeFS:
        def open(self, path: str, mode: str) -> Any:
            return nullcontext(object())

    monkeypatch.setattr("bioio_nd2.reader.nd2.ND2File", FakeND2File)

    rdr = Reader.__new__(Reader)
    rdr._fs = FakeFS()
    rdr._path = "example.nd2"
    rdr._dims = None
    rdr._current_scene_index = 1

    actual = rdr._read_indexed(
        "TZCYX",
        [
            2,
            3,
            2,
            slice(None),
            slice(None),
        ],
    )

    np.testing.assert_array_equal(actual, data[1, 2, 3, 2])
    assert frame_reads == [int(np.ravel_multi_index((1, 2, 3), (3, 5, 7)))]


def test_dims_and_shape_use_nd2_metadata_without_xarray_dask_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeND2File:
        sizes = {nd2.AXIS.POSITION: 3, "Y": 4, "X": 5}
        shape = (3, 4, 5)

        def __init__(self, file: object) -> None:
            pass

        def __enter__(self) -> "FakeND2File":
            return self

        def __exit__(self, *args: object) -> None:
            pass

        def _expand_coords(self, squeeze: bool = True) -> dict[str, object]:
            assert squeeze is False
            return {
                nd2.AXIS.POSITION: range(3),
                "Y": range(4),
                "X": range(5),
                "C": ["channel"],
            }

    class FakeFS:
        def __init__(self) -> None:
            self.open_count = 0

        def open(self, path: str, mode: str) -> Any:
            self.open_count += 1
            return nullcontext(object())

    monkeypatch.setattr("bioio_nd2.reader.nd2.ND2File", FakeND2File)

    fs = FakeFS()
    rdr = Reader.__new__(Reader)
    rdr._fs = fs
    rdr._path = "example.nd2"
    rdr._dims = None
    rdr._current_scene_index = 2

    def fail_delayed_read() -> xr.DataArray:
        raise AssertionError("_read_delayed should not be used for dims or shape")

    monkeypatch.setattr(rdr, "_read_delayed", fail_delayed_read)

    assert rdr.dims.order == "CYX"
    assert rdr.shape == (1, 4, 5)
    assert fs.open_count == 1

    rdr._dims = None
    rdr._current_scene_index = 3
    with pytest.raises(IndexError, match="Position 3 is out of range"):
        rdr.dims
