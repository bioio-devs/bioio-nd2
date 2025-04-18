# bioio-nd2

[![Build Status](https://github.com/bioio-devs/bioio-nd2/actions/workflows/ci.yml/badge.svg)](https://github.com/bioio-devs/bioio-nd2/actions)
[![PyPI version](https://badge.fury.io/py/bioio-nd2.svg)](https://badge.fury.io/py/bioio-nd2)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Python 3.10–3.13](https://img.shields.io/badge/python-3.10--3.13-blue.svg)](https://www.python.org/downloads/)

A BioIO reader plugin for reading nd2 (Nikon microscope) files using `nd2`

---


## Documentation

[See the full documentation on our GitHub pages site](https://bioio-devs.github.io/bioio/OVERVIEW.html) - the generic use and installation instructions there will work for this package.

Information about the base reader this package relies on can be found in the `bioio-base` repository [here](https://github.com/bioio-devs/bioio-base)

## Installation

**Stable Release:** `pip install bioio-nd2`<br>
**Development Head:** `pip install git+https://github.com/bioio-devs/bioio-nd2.git`

## Example Usage (see full documentation for more examples)

Install bioio-nd2 alongside bioio:

`pip install bioio bioio-nd2`


This example shows a simple use case for just accessing the pixel data of the image
by explicitly passing this `Reader` into the `BioImage`. Passing the `Reader` into
the `BioImage` instance is optional as `bioio` will automatically detect installed
plug-ins and auto-select the most recently installed plug-in that supports the file
passed in.
```python
from bioio import BioImage
import bioio_nd2

img = BioImage("my_file.nd2", reader=bioio_nd2.Reader)
img.data
```

## Issues
[_Click here to view all open issues in bioio-devs organization at once_](https://github.com/search?q=user%3Abioio-devs+is%3Aissue+is%3Aopen&type=issues&ref=advsearch) or check this repository's issue tab.


## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for information related to developing the code.
