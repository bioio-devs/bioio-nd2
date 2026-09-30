#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for reading ND2 files straight off a remote file system.

A moto server stands in for S3 so that the tests need no network access and no
credentials.
"""

import os
from typing import Iterator, Tuple

import numpy as np
import pytest
from bioio_base import exceptions

from bioio_nd2 import Reader

from .conftest import LOCAL_RESOURCES_DIR

pytest.importorskip("s3fs")
moto_server = pytest.importorskip("moto.server")
boto3 = pytest.importorskip("boto3")

FILENAME = "ND2_dims_p4z5t3c2y32x32.nd2"
BUCKET = "bioio-nd2-test"


@pytest.fixture(scope="module")
def s3_endpoint() -> Iterator[str]:
    # ThreadedMotoServer is a real HTTP server and does not patch boto3's
    # credential chain, so hand it dummy credentials.
    for key in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"):
        os.environ.setdefault(key, "testing")
    os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")

    server = moto_server.ThreadedMotoServer(port=0)
    server.start()
    try:
        yield "http://{}:{}".format(*server.get_host_and_port())
    finally:
        server.stop()


@pytest.fixture(scope="module")
def remote_nd2(s3_endpoint: str) -> Tuple[str, dict]:
    """Upload an ND2 to the mock bucket; return its URI and storage options."""
    s3 = boto3.resource("s3", endpoint_url=s3_endpoint, region_name="us-east-1")
    bucket = s3.create_bucket(Bucket=BUCKET)
    bucket.upload_file(str(LOCAL_RESOURCES_DIR / FILENAME), FILENAME)
    fs_kwargs = {
        "client_kwargs": {"endpoint_url": s3_endpoint},
        # moto's multipart uploads otherwise fail checksum validation
        "config_kwargs": {"response_checksum_validation": "when_required"},
    }
    return f"s3://{BUCKET}/{FILENAME}", fs_kwargs


@pytest.fixture()
def readers(remote_nd2: Tuple[str, dict]) -> Tuple[Reader, Reader]:
    uri, fs_kwargs = remote_nd2
    return Reader(uri, fs_kwargs=fs_kwargs), Reader(LOCAL_RESOURCES_DIR / FILENAME)


@pytest.mark.parametrize("scene", [0, 2])
@pytest.mark.parametrize(
    "attribute",
    [
        "scenes",
        "shape",
        "dtype",
        "channel_names",
        "physical_pixel_sizes",
        "time_interval",
        "binning",
        "dimension_properties",
        "ome_metadata",
        "standard_metadata",
        "acquisition_times",
    ],
)
def test_remote_metadata_matches_local(
    readers: Tuple[Reader, Reader], scene: int, attribute: str
) -> None:
    remote, local = readers
    remote.set_scene(scene)
    local.set_scene(scene)

    assert getattr(remote, attribute) == getattr(local, attribute)


@pytest.mark.parametrize("scene", [0, 2])
def test_remote_dims_match_local(readers: Tuple[Reader, Reader], scene: int) -> None:
    remote, local = readers
    remote.set_scene(scene)
    local.set_scene(scene)

    assert remote.dims.order == local.dims.order
    assert remote.dims.shape == local.dims.shape


@pytest.mark.parametrize("scene", [0, 2])
def test_remote_data_matches_local(readers: Tuple[Reader, Reader], scene: int) -> None:
    remote, local = readers
    remote.set_scene(scene)
    local.set_scene(scene)

    np.testing.assert_array_equal(remote.data, local.data)
    np.testing.assert_array_equal(remote.xarray_data.data, local.xarray_data.data)


@pytest.mark.parametrize("scene", [0, 2])
def test_remote_delayed_data_matches_local(
    readers: Tuple[Reader, Reader], scene: int
) -> None:
    remote, local = readers
    remote.set_scene(scene)
    local.set_scene(scene)

    delayed = remote.xarray_dask_data
    np.testing.assert_array_equal(delayed.data.compute(), local.data)
    # computing must not have consumed or closed the shared handle
    np.testing.assert_array_equal(delayed.data.compute(), local.data)


def test_remote_delayed_data_outlives_reader(remote_nd2: Tuple[str, dict]) -> None:
    """Deleting the Reader closes its handle; the array reopens the file itself."""
    uri, fs_kwargs = remote_nd2
    remote = Reader(uri, fs_kwargs=fs_kwargs)
    remote.set_scene(2)
    local = Reader(LOCAL_RESOURCES_DIR / FILENAME)
    local.set_scene(2)

    delayed = remote.xarray_dask_data
    handle = remote._nd2
    del remote
    assert handle is not None and handle.closed

    np.testing.assert_array_equal(delayed.data.compute(), local.data)


def test_remote_indexed_read_matches_local(readers: Tuple[Reader, Reader]) -> None:
    """Only the requested planes should be pulled over the wire."""
    remote, local = readers
    remote.set_scene(1)
    local.set_scene(1)

    np.testing.assert_array_equal(
        remote.get_image_data("ZYX", T=1, C=0), local.get_image_data("ZYX", T=1, C=0)
    )
    np.testing.assert_array_equal(
        remote.get_image_dask_data("YX", T=2, C=1, Z=3).compute(),
        local.get_image_data("YX", T=2, C=1, Z=3),
    )


def test_remote_support_check_reads_only_magic_bytes(
    remote_nd2: Tuple[str, dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Constructing a Reader must not pull a whole block off the network."""
    import s3fs

    uri, fs_kwargs = remote_nd2
    fetched = []

    # note: only the bulk read path is instrumented. Wrapping `cat_file` would
    # shadow s3fs's async dispatch and change what the check actually does.
    orig_fetch_range = s3fs.S3File._fetch_range

    def spy_fetch_range(self, start, end):  # type: ignore[no-untyped-def]
        fetched.append(end - start)
        return orig_fetch_range(self, start, end)

    monkeypatch.setattr(s3fs.S3File, "_fetch_range", spy_fetch_range)

    Reader(uri, fs_kwargs=fs_kwargs)

    # the magic number is read as a byte range, so no block of file data moves
    assert fetched == []


def test_remote_reader_reuses_one_handle(readers: Tuple[Reader, Reader]) -> None:
    """Metadata is parsed once, not re-fetched on every property access."""
    remote, _ = readers

    remote.scenes
    handle = remote._nd2
    assert handle is not None and not handle.closed

    remote.physical_pixel_sizes
    remote.ome_metadata
    assert remote._nd2 is handle


def test_remote_unsupported_file(
    remote_nd2: Tuple[str, dict], s3_endpoint: str
) -> None:
    _, fs_kwargs = remote_nd2
    s3 = boto3.resource("s3", endpoint_url=s3_endpoint, region_name="us-east-1")
    s3.Bucket(BUCKET).upload_file(
        str(LOCAL_RESOURCES_DIR / "example.txt"), "example.txt"
    )

    with pytest.raises(exceptions.UnsupportedFileFormatError):
        Reader(f"s3://{BUCKET}/example.txt", fs_kwargs=fs_kwargs)
