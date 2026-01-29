import datetime
from typing import Any

import pytest

from bioio_nd2 import Reader

from .conftest import LOCAL_RESOURCES_DIR


@pytest.mark.parametrize(
    "filename, expected",
    [
        (
            "ND2_dims_p2z5t3-2c4y32x32.nd2",
            {
                "Binning": "1x1",
                "Column": "3",
                "Dimensions Present": "TZCYX",
                "Image Size C": 4,
                "Image Size T": 5,
                "Image Size X": 32,
                "Image Size Y": 32,
                "Image Size Z": 5,
                "Imaged By": None,
                "Imaging Datetime": datetime.datetime(
                    2021,
                    9,
                    28,
                    6,
                    55,
                    1,
                    935004,
                    tzinfo=datetime.timezone(datetime.timedelta(hours=-7), name="PDT"),
                ),
                "Objective": "10x/0.3",
                "Pixel Size X": 0.652452890023035,
                "Pixel Size Y": 0.652452890023035,
                "Pixel Size Z": 1.0,
                "Position Index": None,
                "Row": "D",
                "Timelapse": True,
                "Timelapse Interval": datetime.timedelta(
                    seconds=18, microseconds=495260
                ),
                "Total Time Duration": datetime.timedelta(
                    seconds=73, microseconds=981041
                ),
            },
        ),
        (
            "ND2_dims_p1z5t3c2y32x32.nd2",
            {
                "Binning": "1x1",
                "Column": "3",
                "Dimensions Present": "TZCYX",
                "Image Size C": 2,
                "Image Size T": 3,
                "Image Size X": 32,
                "Image Size Y": 32,
                "Image Size Z": 5,
                "Imaged By": None,
                "Imaging Datetime": datetime.datetime(
                    2021,
                    9,
                    28,
                    6,
                    38,
                    50,
                    381995,
                    tzinfo=datetime.timezone(datetime.timedelta(hours=-7), name="PDT"),
                ),
                "Objective": "10x/0.3",
                "Pixel Size X": 0.652452890023035,
                "Pixel Size Y": 0.652452890023035,
                "Pixel Size Z": 1.0,
                "Position Index": None,
                "Row": "D",
                "Timelapse": True,
                "Timelapse Interval": datetime.timedelta(
                    seconds=4, microseconds=293375
                ),
                "Total Time Duration": datetime.timedelta(
                    seconds=8, microseconds=586750
                ),
            },
        ),
    ],
)
def test_nd2_standard_metadata(filename: str, expected: dict[str, Any]) -> None:
    uri = LOCAL_RESOURCES_DIR / filename
    reader = Reader(uri)
    metadata = reader.standard_metadata.to_dict()

    for key, expected_value in expected.items():
        error_message = f"{key}: Expected: {expected_value}, Actual: {metadata[key]}"
        if isinstance(expected_value, float):
            assert metadata[key] == pytest.approx(expected_value), error_message
        else:
            assert metadata[key] == expected_value, error_message


@pytest.mark.parametrize(
    "filename, scene_index, expected_row, expected_col",
    [
        # Case 1: ND2 file without XYPosLoop fails to produce
        (
            "ND2_maxime_BF007.nd2",
            0,
            None,
            None,
        ),
        # Case 2: Plate scan file, scene maps to E3
        (
            "ND2_dims_rgb_t3p2c2z3x64y64.nd2",
            0,
            "E",
            "3",
        ),
        # Case 3: Same file, different scene maps to E2
        (
            "ND2_dims_rgb_t3p2c2z3x64y64.nd2",
            1,
            "E",
            "2",
        ),
    ],
)
def test_nd2_scene_row_and_column(
    filename: str,
    scene_index: int,
    expected_row: str,
    expected_col: str,
) -> None:
    """
    Verify row/column inference from ND2 scene index.
    """
    uri = LOCAL_RESOURCES_DIR / filename
    reader = Reader(uri)

    # Explicitly set scene
    reader.set_scene(scene_index)

    assert reader.row == expected_row
    assert reader.column == expected_col
