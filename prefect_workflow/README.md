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
Run with python in a non-interactive way, headless by default. 
```bash
python cube_imaging_example.py
```

To pause for Prefect UI overrides of CLEAN iteration controls:

```bash
# In terminal 1, start server and open the UI using the given URL in a browser
prefect server start

# In terminal 2, point the client at the server, then run:
prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api
python prefect_workflow/cube_imaging_example.py --interactive
```

