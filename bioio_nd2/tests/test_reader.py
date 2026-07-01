#!/usr/bin/env python
# -*- coding: utf-8 -*-

from contextlib import nullcontext
from datetime import timedelta
from types import SimpleNamespace
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


def _make_timeloop(period_ms: float) -> Any:
    return nd2.structures.TimeLoop(
        count=1,
        nestingLevel=0,
        parameters=nd2.structures.TimeLoopParams(
            startMs=0.0,
            periodMs=period_ms,
            durationMs=0.0,
            periodDiff=nd2.structures.PeriodDiff(avg=0.0, max=0.0, min=0.0),
        ),
    )


def _fake_nd2_for_experiment(experiment: List[Any]) -> Any:
    class FakeND2File:
        def __init__(self, file: object) -> None:
            pass

        def __enter__(self) -> "FakeND2File":
            return self

        def __exit__(self, *args: object) -> None:
            pass

        @property
        def experiment(self) -> List[Any]:
            return experiment

    return FakeND2File


class _FakeFS:
    def open(self, path: str, mode: str) -> Any:
        return nullcontext(object())


def test_time_interval_from_timeloop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "bioio_nd2.reader.nd2.ND2File",
        _fake_nd2_for_experiment([_make_timeloop(360000.0)]),
    )

    rdr = Reader.__new__(Reader)
    rdr._fs = _FakeFS()
    rdr._path = "example.nd2"

    assert rdr.time_interval == timedelta(milliseconds=360000.0)


def test_time_interval_none_without_timeloop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "bioio_nd2.reader.nd2.ND2File",
        _fake_nd2_for_experiment([]),
    )

    rdr = Reader.__new__(Reader)
    rdr._fs = _FakeFS()
    rdr._path = "example.nd2"

    assert rdr.time_interval is None


def test_dimension_properties_attach_units_from_ome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # OME unit enums expose the symbol via ``.value``, which the shared
    # registry parses (µm -> micrometer, s -> second).
    pixels = SimpleNamespace(
        physical_size_x_unit=SimpleNamespace(value="µm"),
        physical_size_y_unit=SimpleNamespace(value="µm"),
        physical_size_z_unit=SimpleNamespace(value="µm"),
        time_increment_unit=SimpleNamespace(value="s"),
    )
    ome = SimpleNamespace(images=[SimpleNamespace(pixels=pixels)])

    class FakeND2File:
        def __init__(self, file: object) -> None:
            pass

        def __enter__(self) -> "FakeND2File":
            return self

        def __exit__(self, *args: object) -> None:
            pass

        @property
        def experiment(self) -> List[Any]:
            return [_make_timeloop(1000.0)]

        def voxel_size(self) -> Tuple[float, float, float]:
            return (0.5, 0.5, 2.0)

        def ome_metadata(self) -> Any:
            return ome

    monkeypatch.setattr("bioio_nd2.reader.nd2.ND2File", FakeND2File)

    rdr = Reader.__new__(Reader)
    rdr._fs = _FakeFS()
    rdr._path = "example.nd2"

    dp = rdr.dimension_properties

    assert dp.T.type == "time"
    assert str(dp.T.unit) == "second"
    assert dp.C.type is None
    assert dp.C.unit is None
    for axis in (dp.Z, dp.Y, dp.X):
        assert axis.type == "space"
        assert str(axis.unit) == "micrometer"
