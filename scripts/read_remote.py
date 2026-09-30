#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Read an ND2 straight off a remote file system, without downloading it first.

Requires `nd2>=0.12.0`, which reads remote URLs via fsspec, and the fsspec
implementation for the protocol -- `aiohttp` for https://, `s3fs` for s3://.
"""

import sys

from bioio_nd2 import Reader

URL = (
    "https://s3.us-west-2.amazonaws.com/production.files.allencell.org/"
    "b47/47d/dae/9b5/4f2/43a/54c/af0/bbe/0c0/98/"
    "3500009020_nikon0_20260807_pretimelapse_C11_58.nd2"
)


def main(url: str = URL) -> None:
    reader = Reader(url)

    print(f"scenes     : {reader.scenes}")
    print(f"dims       : {reader.dims}")
    print(f"dtype      : {reader.dtype}")
    print(f"pixel sizes: {reader.physical_pixel_sizes}")

    # only the requested plane is pulled over the wire
    plane = reader.get_image_data("YX", T=0, C=0, Z=0)
    print(f"plane      : {plane.shape} {plane.dtype} min={plane.min()} max={plane.max()}")


if __name__ == "__main__":
    main(*sys.argv[1:])
