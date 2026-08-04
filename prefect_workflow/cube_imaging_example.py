# An example imaging workflow based on astroviper's imaging-loop tutorial.
# Uses the science-layer CLEAN entry point
# ``astroviper.processing_functions.imaging.image_cube_single_field``
# (the old ``run_imaging_loop`` / ``generate_ms4_with_point_sources`` API
# has been removed).
from __future__ import annotations

import base64
import pickle
from io import BytesIO
from typing import Any

import numpy as np
import xarray as xr
from prefect import flow, task
from prefect.artifacts import create_image_artifact, create_markdown_artifact
from prefect.flow_runs import pause_flow_run
from prefect.input import RunInput
from toolviper.utils.data import download, update
from xradio.image import make_empty_sky_image, write_image
from xradio.measurement_set import load_processing_set, open_processing_set

from astroviper.processing_functions.imaging.image_cube_single_field import (
    image_cube_single_field,
)
from astroviper.processing_functions.imaging.utils import format_deconvolve_dict

# Small real TW Hya ALMA cube (5 LSRK channels, XX/YY) — same dataset as the
# processing-functions imaging-loop tutorial. Downloaded via toolviper.
PS_STORE = "twhya_selfcal_5chans_lsrk_compare_weights.ps.zarr"
DEFAULT_IMAGE_NAME = "twhya_clean_cube"


# User-specifiable CLEAN controls (Prefect UI pause input).
class ImagingParamsInput(RunInput):
    gain: float
    niter: int
    threshold: float
    nmajor: int
    cyclefactor: float
    minpsffraction: float
    maxpsffraction: float


@task(log_prints=True)
def data_prep(ps_store: str = PS_STORE) -> tuple[Any, np.ndarray, np.ndarray]:
    """Download and load the demo processing set.

    Returns
    -------
    ps_xdt
        Eager in-memory processing set (science-layer input).
    phase_center
        Field phase-center direction (radians).
    frequency_coords
        Frequency axis values (Hz).
    """
    update()
    download(file=ps_store)

    ps_open = open_processing_set(ps_store)
    ps_xdt = load_processing_set(
        ps_store, data_group_name="base", load_sub_datasets=False
    )

    combined = ps_open.xr_ps.get_combined_field_and_source_xds()
    phase_center = combined.FIELD_PHASE_CENTER_DIRECTION.sel(
        field_name=combined.attrs["center_field_name"]
    ).values
    frequency_coords = ps_open.xr_ps.get_freq_axis().values

    ms_name = list(ps_xdt.keys())[0]
    dims = ps_xdt[ms_name].sizes
    print(f"Loaded processing set: {ps_store}")
    print(f"Measurement set: {ms_name}")
    print(
        f"Dimensions: time={dims['time']}, baseline_id={dims['baseline_id']}, "
        f"frequency={dims['frequency']}, polarization={dims['polarization']}"
    )
    print(f"Polarizations: {list(ps_xdt[ms_name].polarization.values)}")
    return ps_xdt, phase_center, frequency_coords


@task(log_prints=True)
def configure_imaging(
    phase_center: np.ndarray,
    frequency_coords: np.ndarray,
    image_name: str = DEFAULT_IMAGE_NAME,
) -> dict[str, Any]:
    """Build image / weights / iteration-control parameter dicts."""
    image_size = [200, 200]
    cell_size = np.array([-0.1, 0.1]) * np.pi / (180 * 3600)  # 0.1 arcsec

    image_params = {
        "image_size": image_size,
        "cell_size": cell_size,
        "phase_direction": phase_center,
        "frequency_coords": frequency_coords,
        "polarization_coords": ["I", "Q"],  # 2-pol linear -> Stokes I, Q
        "time_coords": [0],
        "fft_padding": 1.2,
    }
    imaging_weights_params = {
        "weighting": "briggs",
        "robust": 0.5,
        "casa_weighting_implementation": True,
    }
    # Capped at 3 residual-update cycles for a fast demo.
    iteration_control_params = {
        "niter": 300,
        "nmajor": 3,
        "threshold": 0.001,  # Jy
        "primary_beam_limit": 0.2,
        "gain": 0.1,
        "cyclefactor": 1.5,
        "cycleniter": -1,
        "minpsffraction": 0.05,
        "maxpsffraction": 0.8,
    }

    params = {
        "image_name": image_name,
        "image_params": image_params,
        "imaging_weights_params": imaging_weights_params,
        "iteration_control_params": iteration_control_params,
        "instrument_polarization_basis": "linear",
        "processing_set_data_group_name": "base",
    }
    print(f"Configured imaging parameters: {params}")
    return params


@flow(log_prints=True)
def modify_imaging_params(params: dict[str, Any]) -> dict[str, Any]:
    """Pause for Prefect UI input to override CLEAN iteration controls.

    Only intended for the initial setting before running the imaging loop —
    not an interactive clean.
    """
    ic = params["iteration_control_params"]
    user_input: ImagingParamsInput = pause_flow_run(
        wait_for_input=ImagingParamsInput.with_initial_data(
            gain=ic["gain"],
            niter=ic["niter"],
            threshold=ic["threshold"],
            nmajor=ic["nmajor"],
            cyclefactor=ic["cyclefactor"],
            minpsffraction=ic["minpsffraction"],
            maxpsffraction=ic["maxpsffraction"],
        )
    )
    print("Applying user overrides to iteration_control_params")
    ic["gain"] = user_input.gain
    ic["niter"] = user_input.niter
    ic["threshold"] = user_input.threshold
    ic["nmajor"] = user_input.nmajor
    ic["cyclefactor"] = user_input.cyclefactor
    ic["minpsffraction"] = user_input.minpsffraction
    ic["maxpsffraction"] = user_input.maxpsffraction
    print(f"Modified iteration_control_params: {ic}")
    return params


@task(log_prints=True)
def run_imaging_loop_task(
    ps_xdt, params: dict[str, Any]
) -> tuple[xr.Dataset, object, object]:
    """Run the science-layer cube CLEAN loop."""
    image_params = params["image_params"]

    # Empty image in the instrument (correlation) basis; the science function
    # transforms to the Stokes output basis internally.
    img_xds = make_empty_sky_image(
        phase_center=image_params["phase_direction"],
        image_size=image_params["image_size"],
        cell_size=image_params["cell_size"],
        frequency_coords=image_params["frequency_coords"],
        pol_coords=["XX", "YY"],
        time_coords=image_params["time_coords"],
        do_sky_coords=True,
    )

    img_xds, timing_df, deconvolve_dict = image_cube_single_field(
        ps_xdt,
        img_xds,
        image_params,
        params["imaging_weights_params"],
        params["iteration_control_params"],
        processing_set_data_group_name=params["processing_set_data_group_name"],
        deconvolver="hogbom",
        instrument_polarization_basis=params["instrument_polarization_basis"],
        single_precision_image=False,
        processing_function_threads=1,
        fft_backend="scipy",
        image_data_variables_keep=[
            "sky_residual",
            "sky_model",
            "point_spread_function",
            "primary_beam",
        ],
        restore=True,
    )

    n_major = int(timing_df["n_major_cycles"].iloc[0])
    print(f"CLEAN loop complete. Residual-update cycles: {n_major}")
    print(f"Planes cleaned: {len(deconvolve_dict.data)}")
    return img_xds, timing_df, deconvolve_dict


@task
def create_imaging_report(timing_df, deconvolve_dict):
    """Publish CLEAN summary stats as a Prefect markdown artifact."""
    n_major = int(timing_df["n_major_cycles"].iloc[0])
    first_key = next(iter(deconvolve_dict.data), None)
    stop_desc = (
        deconvolve_dict.data[first_key].get("stop_description", "n/a")
        if first_key is not None
        else "n/a"
    )
    report = f"""
### Imaging loop report

- Residual-update cycles: {n_major}
- Planes cleaned: {len(deconvolve_dict.data)}
- Example plane stop reason ({first_key}): {stop_desc}

```
{format_deconvolve_dict(deconvolve_dict)}
```
"""
    create_markdown_artifact(
        key="imaging-loop-report",
        markdown=report,
        description="Summary of the cube imaging CLEAN loop",
    )


@task(log_prints=True)
def make_summary_image(img_xds: xr.Dataset, chan: int = 2):
    """Create a model / residual / restored summary plot as a Prefect artifact."""
    import matplotlib.pyplot as plt

    n_freq = img_xds.sizes["frequency"]
    chan = min(chan, n_freq - 1)

    model_I = img_xds["SKY_MODEL"].isel(time=0, frequency=chan, polarization=0).values
    residual_I = (
        img_xds["SKY_RESIDUAL"].isel(time=0, frequency=chan, polarization=0).values
    )
    restored_I = (
        img_xds["SKY_RESTORED"].isel(time=0, frequency=chan, polarization=0).values
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    im0 = axes[0].imshow(model_I, origin="lower")
    axes[0].set_title(f"SKY_MODEL (Stokes I)\nTotal flux: {np.sum(model_I):.3f} Jy")
    plt.colorbar(im0, ax=axes[0], label="Jy/pixel")

    im1 = axes[1].imshow(residual_I, origin="lower")
    axes[1].set_title(
        f"SKY_RESIDUAL (Stokes I)\nPeak: {np.max(np.abs(residual_I)):.4f} Jy"
    )
    plt.colorbar(im1, ax=axes[1], label="Jy/pixel")

    im2 = axes[2].imshow(restored_I, origin="lower")
    axes[2].set_title("SKY_RESTORED (model*beam + residual)")
    plt.colorbar(im2, ax=axes[2], label="Jy/pixel")

    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    b64_encoded_image = base64.b64encode(buf.read()).decode()

    create_image_artifact(
        key="imaging-summary",
        image_url=f"data:image/png;base64,{b64_encoded_image}",
        description="Summary of imaging results (Model, Residual, Restored)",
    )


@task(log_prints=True)
def save_results(img_xds: xr.Dataset, deconvolve_dict, image_name: str):
    """Write the image dataset (zarr) and pickle the deconvolution ReturnDict."""
    write_image(img_xds, image_name + ".img.zarr", out_format="zarr", overwrite=True)
    with open(image_name + "_imaging_results.pkl", "wb") as f:
        pickle.dump(deconvolve_dict.data, f)
    print(f"Wrote {image_name}.img.zarr and {image_name}_imaging_results.pkl")


@flow(log_prints=True)
def imaging_flow(interactive: bool = False, image_name: str = DEFAULT_IMAGE_NAME):
    """Cube imaging Prefect workflow.

    Parameters
    ----------
    interactive : bool, optional
        If ``True``, pause for Prefect UI input to override CLEAN controls.
        Defaults to ``False`` so ``python cube_imaging_example.py`` runs
        headlessly.
    image_name : str, optional
        Basename for on-disk outputs (``.img.zarr`` and ``_imaging_results.pkl``).
    """
    ps_xdt, phase_center, frequency_coords = data_prep()
    params = configure_imaging(phase_center, frequency_coords, image_name=image_name)
    if interactive:
        params = modify_imaging_params(params)

    img_xds, timing_df, deconvolve_dict = run_imaging_loop_task(ps_xdt, params)
    create_imaging_report(timing_df, deconvolve_dict)
    make_summary_image(img_xds)
    save_results(img_xds, deconvolve_dict, params["image_name"])
    return img_xds, timing_df, deconvolve_dict


if __name__ == "__main__":
    imaging_flow(interactive=False)
