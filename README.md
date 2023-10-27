# bioio-nd2

[![Build Status](https://github.com/bioio-devs/bioio-nd2/actions/workflows/ci.yml/badge.svg)](https://github.com/bioio-devs/bioio-nd2/actions)
[![Documentation](https://github.com/bioio-devs/bioio-nd2/actions/workflows/docs.yml/badge.svg)](https://bioio-devs.github.io/bioio-nd2)

A BioIO reader plugin for reading ND2 images.

This plugin is intended to be used in conjunction with [bioio](https://github.com/bioio-devs/bioio)
---

## Installation

**Stable Release:** `pip install bioio-nd2`<br>
**Development Head:** `pip install git+https://github.com/bioio-devs/bioio-nd2.git`

## Quickstart

```python
from bioio_nd2 import Reader 

r = Reader("my-image.ext")
r.dims
```

## Documentation

For full package documentation please visit [bioio-devs.github.io/bioio-nd2](https://bioio-devs.github.io/bioio-nd2).

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for information related to developing the code.

**MIT License**
