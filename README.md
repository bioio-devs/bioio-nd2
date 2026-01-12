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

## Multi-Well Plate & Well Assignment Support (ND2)

`bioio-nd2` includes support for mapping ND2 XY stage positions to
**multi-well plate positions** (e.g. A1, B3, H12). This is primarily
intended for experiments acquired using Nikon’s **XYPosLoop** functionality
on multi-well plates.

> ⚠️ If the ND2 file does **not** contain XY position metadata (`XYPosLoop`),
> no well mapping is performed and `row` / `column` will be `None`.

### Default behavior (96-well plate)

The ND2 reader assumes a **standard 96-well plate geometry** (See [Plates](https://github.com/bioio-devs/bioio-nd2/blob/main/bioio_nd2/plates.py) for specification) and assigns each scene to the nearest well center. Users may define their own plate geometry and pass it to the reader to override these defaults.


```python
from bioio import BioImage
import bioio_nd2

img = BioImage("my_file.nd2", reader=bioio_nd2.Reader)

img.set_scene(0)

img.reader.row     # e.g. "B"
img.reader.column  # e.g. "3"
```

### Custom Plate Geometries

Users may define their own plate geometry and pass it to the reader.

```python
from bioio import BioImage
import bioio_nd2
from bioio_nd2.plates import Plate, WellAssignmentMode

custom_plate = Plate(
    name="custom_96",
    rows=list("ABCDEFGH")[::-1],
    cols=[str(i) for i in range(1, 13)],
    plate_width_mm=126.6,
    plate_height_mm=85.7,
    a1_offset_mm=(14.3, 11.36),
    well_spacing_um=9000.0,
    well_radius_um=6210.0 / 2,
    assignment_mode=WellAssignmentMode.CLOSEST,
)

img = BioImage(
    "my_file.nd2",
    reader=bioio_nd2.Reader,
    plate=custom_plate,
)
```

---

### Controlling well assignment strictness

The plate definition includes a **well assignment mode** that controls how
strictly stage positions must correspond to a physical well.

Available modes:

| Mode           | Behavior                                             |
| -------------- | ---------------------------------------------------- |
| `CLOSEST`      | Always assign the nearest well (most permissive)     |
| `WITHIN_WELL`  | Assign only if position falls within the well radius |
| `HALF_SPACING` | Assign only if within half the inter-well spacing    |

In strict modes, scenes that fall outside the allowed region will have
`row` / `column` set to `None`.

---

## Issues
[_Click here to view all open issues in bioio-devs organization at once_](https://github.com/search?q=user%3Abioio-devs+is%3Aissue+is%3Aopen&type=issues&ref=advsearch) or check this repository's issue tab.


## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for information related to developing the code.
