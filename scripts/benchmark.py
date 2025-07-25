"""
    Runs bioio_base's benchmark function against the test resources in this repository
"""
import pathlib

import bioio_base.benchmark

import bioio_nd2


# This file is under /scripts while the test resourcess are under /bioio_nd2/tests/resources
test_resources_dir = pathlib.Path(__file__).parent.parent / "bioio_nd2" / "tests" / "resources"
assert test_resources_dir.exists(), f"Test resources directory {test_resources_dir} does not exist"
test_files = [
    test_file
    for test_file in test_resources_dir.iterdir()
    if test_file.name.endswith(".nd2")
]
print(f"Test files: {[file.name for file in test_files]}")
bioio_base.benchmark.benchmark(bioio_nd2.reader.Reader, test_files)
