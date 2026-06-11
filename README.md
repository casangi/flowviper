# FlowvVIPER

Workflow examples for radio astronomy data processing with [astroviper](https://github.com/casangi/astroviper).

Requires Python >= 3.11, < 3.14.

## Installation

### Base install

```bash
pip install flowviper
```

This is sufficient for the bundled Prefect demo (synthetic MSv4 data) and workflows that start from MSv4 or zarr data already converted. It does **not** include reading legacy CASA MeasurementSets (`.ms` files).

### macOS — MSv2 support

On macOS, `python-casacore` is not available via pip. Install a CASA table backend before using workflows that read or convert MSv2 data.

**Path A (pip-only, recommended)**

```bash
pip install --extra-index-url https://casa-pip.nrao.edu/repository/pypi-group/simple \
  casaconfig casatools
pip install flowviper
```

Works with a normal Python virtual environment. Do **not** install `python-casacore` alongside `casatools` (namespace conflict).

**Path B (conda)**

For users already using Miniforge, mamba, or conda:

```bash
conda create -n flowviper python=3.12 pip
conda activate flowviper
conda install -c conda-forge python-casacore
pip install flowviper
```

`casatools` is published on the [NRAO pip index](https://casa-pip.nrao.edu/repository/pypi-group/simple), not on PyPI, so it must be installed manually as shown above.

## Prefect workflows

Example workflows live in [`prefect_workflow/`](prefect_workflow/). See [`prefect_workflow/README.md`](prefect_workflow/README.md) for usage notes.

Quick start:

```bash
pip install flowviper
python prefect_workflow/cube_imaging_example.py
```
