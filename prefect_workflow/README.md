# Prefect workflows

Example Prefect flows for radio astronomy data processing with astroviper.

## Prerequisites

Install flowviper first. See the [main README](../README.md#installation) for platform-specific instructions.

The bundled cube imaging demo uses synthetic MSv4 data and works with the base install:

```bash
pip install flowviper prefect
```

Workflows that read legacy CASA MeasurementSets (`.ms` files) require the macOS MSv2 backend described in the [main README](../README.md#macos--msv2-support).

## Example: cube imaging

```bash
python cube_imaging_example.py
```
