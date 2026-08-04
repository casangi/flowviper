# Prefect workflows

Example Prefect flows for radio astronomy data processing with astroviper.

## Prerequisites

Install flowviper first. See the [main README](../README.md#installation) for platform-specific instructions.

The bundled cube imaging demo downloads a small real TW Hya MSv4 processing set
via toolviper and runs AstroVIPER's science-layer
`image_cube_single_field` CLEAN loop. It works with the base install:

```bash
pip install flowviper
```

Workflows that read legacy CASA MeasurementSets (`.ms` files) require the macOS MSv2 backend described in the [main README](../README.md#macos--msv2-support).

## Example: cube imaging

```bash
python cube_imaging_example.py
```

Pass `interactive=True` (or call `imaging_flow(interactive=True)`) to pause for
Prefect UI overrides of CLEAN iteration controls. The default CLI path is
headless.
