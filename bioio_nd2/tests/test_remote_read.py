from typing import Any, Iterator, List, Tuple, Union

import fsspec
import numpy as np
import pytest
from bioio_base import exceptions, test_utilities
from ome_types import OME

from bioio_nd2 import Reader

from .conftest import LOCAL_RESOURCES_DIR
from .test_reader import pos_names


@pytest.fixture(scope="module")
def mem_fs() -> Iterator[fsspec.AbstractFileSystem]:
    mem = fsspec.filesystem("memory")
    for filename in (
        "ND2_dims_t3c2y32x32.nd2",
        "ND2_dims_p4z5t3c2y32x32.nd2",
        "example.txt",
    ):
        mem.pipe(f"/{filename}", (LOCAL_RESOURCES_DIR / filename).read_bytes())
    yield mem
    mem.rm("/", recursive=True)


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
            "ND2_dims_p4z5t3c2y32x32.nd2",
            pos_names[2],
            pos_names,
            (3, 5, 2, 32, 32),
            np.uint16,
            "TZCYX",
            ["Widefield Green", "Widefield Red"],
            (1.0, 0.652452890023035, 0.652452890023035),
            OME,
        ),
    ],
)
def test_nd2_remote_reader(
    mem_fs: fsspec.AbstractFileSystem,
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
    test_utilities.run_image_file_checks(
        ImageContainer=Reader,
        image=f"memory:///{filename}",
        set_scene=set_scene,
        expected_scenes=expected_scenes,
        expected_current_scene=set_scene,
        expected_shape=expected_shape,
        expected_dtype=expected_dtype,
        expected_dims_order=expected_dims_order,
        expected_channel_names=expected_channel_names,
        expected_physical_pixel_sizes=expected_physical_pixel_sizes,
        expected_metadata_type=expected_metadata_type,
        reader_kwargs={},
    )


def test_remote_delayed_read_outlives_reader(
    mem_fs: fsspec.AbstractFileSystem,
) -> None:
    remote = Reader("memory:///ND2_dims_p4z5t3c2y32x32.nd2")
    remote.set_scene(2)
    delayed = remote.xarray_dask_data
    del remote

    local = Reader(LOCAL_RESOURCES_DIR / "ND2_dims_p4z5t3c2y32x32.nd2")
    local.set_scene(2)
    np.testing.assert_array_equal(delayed.data.compute(), local.data)


def test_remote_unsupported_file(mem_fs: fsspec.AbstractFileSystem) -> None:
    with pytest.raises(exceptions.UnsupportedFileFormatError):
        Reader("memory:///example.txt")
