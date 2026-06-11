# Prefect workflow based on astroviper_tutorial_mosaics notebook
# It has been updated to work with the PR branch of astroviper 183-add-dask-parallelism-for-cube-single-field-imaging
from __future__ import annotations

import os
import shutil
from io import BytesIO
import base64

import numpy as np
from prefect import flow, task
from prefect.artifacts import create_image_artifact, create_markdown_artifact


DEFAULT_PS_STORE = "Antennae_North.cal.lsrk.ps.zarr"
DEFAULT_IMAGE_NAME = "Antennae_North_Cube.img.zarr"
DEFAULT_SCAN_INTENTS = ["OBSERVE_TARGET#ON_SOURCE"]
DEFAULT_IMAGE_DATA_VARIABLES_KEEP = [
    "sky_residual",
    "point_spread_function",
    "primary_beam",
]


@task(log_prints=True)
def download_data(ps_store: str = DEFAULT_PS_STORE) -> str:
    """Download tutorial processing-set data if not already present locally."""
    from toolviper.utils.data import download

    download(file=ps_store)
    print(f"Downloaded (or verified) processing set: {ps_store}")
    return ps_store


@task(log_prints=True)
def inspect_processing_set(
    ps_store: str,
    scan_intents: list[str],
    ms_key: str | None = None,
) -> dict:
    """Open the processing set, print a summary, and return metadata for imaging."""
    from xradio.measurement_set import open_processing_set
    import pandas as pd

    pd.options.display.max_colwidth = 100

    ps = open_processing_set(ps_store, scan_intents=scan_intents)
    ps.xr_ps.summary()

    if ms_key is None:
        ms_key = next(iter(ps.keys()))

    print(ps[ms_key])

    combined_field_and_source_xds = ps.xr_ps.get_combined_field_and_source_xds()
    center_field_name = combined_field_and_source_xds.attrs["center_field_name"]
    phase_direction = combined_field_and_source_xds.FIELD_PHASE_CENTER_DIRECTION.sel(
        field_name=center_field_name
    )
    frequency_coord = ps[ms_key].frequency

    metadata = {
        "ps_store": ps_store,
        "scan_intents": scan_intents,
        "ms_key": ms_key,
        "center_field_name": center_field_name,
        "phase_direction": phase_direction.values,
        "frequency_coords": frequency_coord.values,
    }
    print(f"Derived imaging metadata from MS key: {ms_key}")
    return metadata


@task(log_prints=True)
def configure_image_params(
    metadata: dict,
    image_size: tuple[int, int] = (500, 500),
    cell_arcsec: float = 0.13,
    polarization_coords: list[str] | None = None,
) -> dict:
    """Build image_params and related imaging configuration from processing-set metadata."""
    if polarization_coords is None:
        polarization_coords = ["I", "Q"]

    cell_size = np.array([-cell_arcsec, cell_arcsec]) * np.pi / (180 * 3600)

    image_params = {
        "image_size": list(image_size),
        "cell_size": cell_size,
        "phase_direction": metadata["phase_direction"],
        "frequency_coords": metadata["frequency_coords"],
        "polarization_coords": polarization_coords,
        "time_coords": [0],
        "fft_padding": 1.0,
    }

    config = {
        "image_params": image_params,
        "imaging_weights_params": {"weighting": "natural"},
        "iteration_control_params": {
            "niter": 0,
            "nmajor": 0,
            "threshold": 0.0,
            "gain": 0.1,
            "cyclefactor": 1.5,
            "cycleniter": 1,
            "minpsffraction": 0.05,
            "maxpsffraction": 0.8,
        },
        "image_data_variables_keep": list(DEFAULT_IMAGE_DATA_VARIABLES_KEEP),
        "processing_set_data_group_name": "base",
        "n_chunks": None,
        "overwrite": True,
    }

    print(f"Configured image parameters: {image_params}")
    return config


@task(log_prints=True)
def prepare_image_store(image_name: str) -> str:
    """Remove an existing image store so imaging can overwrite it cleanly."""
    if os.path.exists(image_name):
        shutil.rmtree(image_name)
        print(f"Removed existing image store: {image_name}")
    return image_name


@task(log_prints=True)
def run_cube_imaging(
    ps_store: str,
    image_name: str,
    scan_intents: list[str],
    imaging_config: dict,
    dask_cores: int = 4,
    dask_memory_limit: str = "4GB",
) -> str:
    """Run distributed-graph cube imaging for a single field."""
    from toolviper.dask.client import local_client
    import astroviper.distributed_graphs as distributed_graphs

    local_client(cores=dask_cores, memory_limit=dask_memory_limit)

    distributed_graphs.imaging.image_cube_single_field(
        ps_store=ps_store,
        image_store=image_name,
        image_params=imaging_config["image_params"],
        imaging_weights_params=imaging_config["imaging_weights_params"],
        iteration_control_params=imaging_config["iteration_control_params"],
        scan_intents=scan_intents,
        image_data_variables_keep=imaging_config["image_data_variables_keep"],
        processing_set_data_group_name=imaging_config[
            "processing_set_data_group_name"
        ],
        n_chunks=imaging_config["n_chunks"],
        overwrite=imaging_config["overwrite"],
    )

    print(f"Cube imaging completed: {image_name}")
    return image_name


@task(log_prints=True)
def load_image_cube(image_name: str):
    """Open the image zarr store and log its contents."""
    import xarray as xr

    img_xds = xr.open_zarr(image_name)
    print(img_xds)
    return image_name


@task
def create_imaging_summary_artifact(image_name: str) -> None:
    """Publish a short markdown summary of the produced image store to Prefect."""
    import xarray as xr

    img_xds = xr.open_zarr(image_name)
    summary = f"""
    Image store: `{image_name}`
    Dimensions: {dict(img_xds.sizes)}
    Data variables: {list(img_xds.data_vars)}
    """
    create_markdown_artifact(
        key="mosaics-cube-imaging-report",
        markdown=summary,
        description="Summary of mosaics cube imaging output",
    )
    print(f"Created imaging summary artifact: {summary}")

@task(log_prints=True)
def save_results(image_name: str, imaging_config: dict, metadata: dict) -> str:
    """Save imaging configuration and processing-set metadata alongside the image store."""
    import pickle

    results_path = image_name + "_imaging_results.pkl"
    results = {
        "image_name": image_name,
        "imaging_config": imaging_config,
        "metadata": metadata,
    }
    with open(results_path, "wb") as f:
        pickle.dump(results, f)

    print(f"Saved imaging results to: {results_path}")
    return results_path


@task(log_prints=True)
def plot_image_products(
    image_name: str,
    frequency_index: int = 82,
    polarization_index: int = 0,
) -> None:
    """
    Plot PSF, primary beam, and sky residual for one channel.

    Creates a Prefect image artifact (base64-encoded PNG) for the UI.
    """
    import matplotlib.pyplot as plt
    import xarray as xr

    img_xds = xr.open_zarr(image_name)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    psf = img_xds.POINT_SPREAD_FUNCTION.isel(
        polarization=polarization_index, frequency=frequency_index
    )
    psf.plot(ax=axes[0], cmap="viridis", vmin=0.0)
    axes[0].set_title(f"Point Spread Function (chan {frequency_index})")

    pb = img_xds.PRIMARY_BEAM.isel(
        polarization=polarization_index, frequency=frequency_index
    )
    pb.plot(ax=axes[1])
    axes[1].set_title(f"Primary Beam (chan {frequency_index})")

    residual = img_xds.SKY_RESIDUAL.isel(
        polarization=polarization_index, frequency=frequency_index
    )
    residual.plot(ax=axes[2], cmap="viridis", vmin=0.0)
    axes[2].set_title(f"Sky Residual (chan {frequency_index})")

    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    b64_encoded_image = base64.b64encode(buf.read()).decode()

    create_image_artifact(
        key="mosaics-cube-imaging-summary",
        image_url=f"data:image/png;base64,{b64_encoded_image}",
        description=(
            "PSF, primary beam, and sky residual "
            f"(pol={polarization_index}, freq={frequency_index})"
        ),
    )


@flow(log_prints=True)
def mosaics_cube_imaging_flow(
    ps_store: str = DEFAULT_PS_STORE,
    image_name: str = DEFAULT_IMAGE_NAME,
    scan_intents: list[str] | None = None,
    ms_key: str | None = None,
    image_size: tuple[int, int] = (500, 500),
    cell_arcsec: float = 0.13,
    polarization_coords: list[str] | None = None,
    create_plots: bool = False,
    plot_frequency_index: int = 82,
    plot_polarization_index: int = 0,
    dask_cores: int = 4,
    dask_memory_limit: str = "4GB",
):
    """Mosaics tutorial cube imaging workflow with optional diagnostic plots."""
    if scan_intents is None:
        scan_intents = list(DEFAULT_SCAN_INTENTS)

    download_data(ps_store)
    metadata = inspect_processing_set(ps_store, scan_intents, ms_key=ms_key)
    imaging_config = configure_image_params(
        metadata,
        image_size=image_size,
        cell_arcsec=cell_arcsec,
        polarization_coords=polarization_coords,
    )
    prepare_image_store(image_name)
    run_cube_imaging(
        ps_store,
        image_name,
        scan_intents,
        imaging_config,
        dask_cores=dask_cores,
        dask_memory_limit=dask_memory_limit,
    )
    load_image_cube(image_name)
    save_results(image_name, imaging_config, metadata)
    create_imaging_summary_artifact(image_name)

    if create_plots:
        plot_image_products(
            image_name,
            frequency_index=plot_frequency_index,
            polarization_index=plot_polarization_index,
        )


if __name__ == "__main__":
    mosaics_cube_imaging_flow()
