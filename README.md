# FlowvVIPER

Workflow examples for radio astronomy data processing with [astroviper](https://github.com/casangi/astroviper).

Requires Python >= 3.11, < 3.15 (tested on 3.12, 3.13 and 3.14).

## Installation

### Base install

```bash
pip install flowviper
```

This is sufficient for the bundled Prefect demo (downloads a small real MSv4
processing set via toolviper), for workflows that start from MSv4 or zarr data
already converted, and for workflows that read or convert legacy CASA
MeasurementSets (MSv2, `.ms` directories): XRADIO reads them with
[arcae](https://github.com/ska-sa/arcae), which `astroviper[all]` installs and
which has wheels for Linux and macOS.

## Prefect workflows

Example workflows live in [`prefect_workflow/`](prefect_workflow/). See [`prefect_workflow/README.md`](prefect_workflow/README.md) for usage notes.

Quick start:

```bash
pip install flowviper
python prefect_workflow/single_field_cube_imaging_example.py
```
