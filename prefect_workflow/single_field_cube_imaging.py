# An example Prefect workflow for single field cube imaging based on AstroVIPER's
# single field cube tutorial. The workflow make use of distributed applications layer
# of AstroVIPER to run the imaging in a distributed manner using Dask as local client.
import os
import shutil
from io import BytesIO
import base64
from typing import Literal
from typing import Any
import numpy as np
import matplotlib.pyplot as plt

from prefect import flow, task
from prefect.artifacts import create_image_artifact, create_markdown_artifact
from prefect.flow_runs import pause_flow_run
from prefect.input import RunInput

DEFAULT_PS_STORE = "twhya_selfcal_5chans_lsrk_compare_weights.ps.zarr"
DEFAULT_IMAGE_NAME = "twhya_selfcal_5chans_lsrk_compare_weights.img.zarr"
DEFAULT_SCAN_INTENTS = ["OBSERVE_TARGET#ON_SOURCE"]
DEFAULT_IMAGE_DATA_VARIABLES_KEEP = [
    "sky_residual",
    "point_spread_function",
    "primary_beam",
]


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
def download_data(
    ps_store: str = DEFAULT_PS_STORE,
    ps_store_id: str = None,
    location: Literal["Cloudflare", "GoogleDrive"] = "Cloudflare",
) -> str:
    """Download tutorial processing-set data if not already present locally.

    Parameters
    ----------
    ps_store: str
        The name of the processing set zarr store to download.
    ps_store_id: str, optional
        The identifier for the processing set in the data repository. Defaults to None.
    location: Literal['Cloudflare', 'GoogleDrive'], optional
        The data location from which to download the processing set. Defaults to 'Cloudflare'.
    """

    if os.path.exists(ps_store):
        return  # use existing store if already present

    if location == "GoogleDrive":
        if ps_store_id is None:
            raise ValueError(
                "ps_store_id must be provided when using GoogleDrive location."
            )
        import gdown

        zip_path = ps_store + ".zip"
        gdown.download(id=ps_store_id, output=zip_path, quiet=False)
        shutil.unpack_archive(zip_path, extract_dir=os.path.dirname(ps_store))
        os.remove(zip_path)  # Clean up the zip file after extraction
    else:
        from toolviper.utils.data import download

        download(file=ps_store)
    print(f"Downloaded (or verified) processing set: {ps_store}")
    return ps_store


@task(log_prints=True)
def inspect_processing_set(
    ps_store: str,
    scan_intents: list[str] | None = None,
) -> dict:
    """Open the processing set, print a summary, and return metadata for imaging.

    Parameters
    ----------
    ps_store: str
        The name of the processing set zarr store to load.
    scan_intents: list[str], optional
        A list of scan intents to filter the processing set. Defaults to None."""
    from xradio.measurement_set import open_processing_set
    import pandas as pd

    pd.options.display.max_colwidth = 100

    ps_xdt = open_processing_set(ps_store, scan_intents=scan_intents)
    ps_xdt.xr_ps.summary()

    ms_name, ms_xdt = list(ps_xdt.items())[0]

    combined_field_and_source_xds = ps_xdt.xr_ps.get_combined_field_and_source_xds()
    center_field_name = combined_field_and_source_xds.attrs["center_field_name"]
    phase_direction = combined_field_and_source_xds.FIELD_PHASE_CENTER_DIRECTION.sel(
        field_name=center_field_name
    ).values
    frequency_coords = ps_xdt.xr_ps.get_freq_axis().values

    return ps_xdt, scan_intents, phase_direction, frequency_coords


@task(log_prints=True)
def configure_imaging_params(
    # image parameters
    phase_direction: np.ndarray,
    frequency_coords: np.ndarray,
    image_size: tuple[int, int] = (250, 250),
    cell_arcsec: float = 0.13,
    polarization_coords: list[str] | None = None,
    # weighting params
    weighting: str = "briggs",
    robust: float = 0.5,
    # Gridding params
    support: int = 7,
    oversampling: int = 100,
    # Deconvolution params
    algorithm: str = "hogbom",
    gain: float = 0.1,
    niter: int = 100,
    threshold: float = 0.0,
    # major cycle control
    nmajor: int = -1,
    cyclefactor: float = 1.5,
    cycleniter: int = -1,
    minpsffraction: float = 0.05,
    maxpsffraction: float = 0.8,
) -> dict:
    """Build image_params and related imaging configuration from processing-set metadata."""
    # Convert cell size to radians
    cell_size = np.array([-cell_arcsec, cell_arcsec]) * np.pi / (180 * 3600)

    image_params = {
        "image_size": list(image_size),
        "cell_size": cell_size,
        "phase_direction": phase_direction,
        "frequency_coords": frequency_coords,
        "polarization_coords": polarization_coords,
        "time_coords": [0],
        "fft_padding": 1.0,
    }

    imaging_weights_params = {
        "weighting": weighting,
        "robust": robust,
    }

    iteration_control_params = {
        "niter": niter,
        "nmajor": nmajor,
        "threshold": threshold,
        "primary_beam_limit": 0.2,
        "gain": gain,
        "cyclefactor": cyclefactor,
        "cycleniter": cycleniter,
        "minpsffraction": minpsffraction,
        "maxpsffraction": maxpsffraction,
    }

    # Configure imaging parameters
    params = {
        # Image geometry - use the npix from generated data
        "image_params": image_params,
        "imaging_weights_params": imaging_weights_params,
        # Deconvolution
        "algorithm": "hogbom",
        "iteration_control_params": iteration_control_params,
        "image_data_variables_keep": list(DEFAULT_IMAGE_DATA_VARIABLES_KEEP),
        "processing_set_data_group_name": "base",
        "overwrite": True,
    }
    print(f"Configured imaging loop parameters:{params}")
    return params


@task(log_prints=True)
def modify_imaging_params(params: dict[str, Any]) -> dict[str, Any]:
    """Pause the *calling* flow for Prefect UI CLEAN-control overrides.

    Safe to run as a ``@task``: since Prefect 3.6.3 (see
    https://github.com/PrefectHQ/prefect/pull/19457), ``pause_flow_run``
    always targets the enclosing flow run's context, even when called from
    within a task — unlike a nested ``@flow``, which would only pause the
    subflow and leave the parent ``imaging_flow`` run ``Running`` (no UI
    Resume form). Decorating this as a task also makes it appear as its own
    node in the Prefect UI graph.

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
    from astroviper.distributed_applications.imaging import image_cube_single_field

    viper_client = local_client(cores=dask_cores, memory_limit=dask_memory_limit)

    clean_dict = image_cube_single_field(
        ps_store=ps_store,
        image_store=image_name,
        image_params=imaging_config["image_params"],
        imaging_weights_params=imaging_config["imaging_weights_params"],
        iteration_control_params=imaging_config["iteration_control_params"],
        scan_intents=scan_intents,
        image_data_variables_keep=imaging_config["image_data_variables_keep"],
        processing_set_data_group_name=imaging_config["processing_set_data_group_name"],
        # n_chunks=imaging_config["n_chunks"],
        overwrite=imaging_config["overwrite"],
    )
    viper_client.close()
    print(f"Cube imaging completed: {image_name}")
    return clean_dict


@task
def create_imaging_summary_artifact(imaging_ret_dict: dict) -> None:
    """Publish a short markdown summary of the imaging result to Prefect."""
    from astroviper.processing_functions.imaging.utils import format_deconvolve_dict

    deconvolve_summary = f"""
```
{format_deconvolve_dict(imaging_ret_dict["deconvolution"], float_format="{:.6g}")}
```
"""

    create_markdown_artifact(
        key="single-field-cube-imaging-report",
        markdown=deconvolve_summary,
        description="Summary of single field cube imaging output",
    )
    print(f"Created imaging summary artifact: {deconvolve_summary}")


@task(log_prints=True)
def create_imaging_timing_artifact(imaging_ret_dict: dict) -> None:
    from astroviper.distributed_applications.imaging.image_cube_single_field import (
        DISTRIBUTED_APPLICATION_TIMING_PHASES,
        DISTRIBUTED_APPLICATION_TIMING_TOTAL_KEY,
    )
    from astroviper.utils.timing import format_timing_summary

    timing_summary = f"""
```
{format_timing_summary(
    imaging_ret_dict["timing_distributed_application"],
    DISTRIBUTED_APPLICATION_TIMING_PHASES,
    total_key=DISTRIBUTED_APPLICATION_TIMING_TOTAL_KEY,
    title="AstroVIPER distributed-application timing (driver, seconds)",
    total_label="TOTAL (driver wall time)",
)}
```
"""

    create_markdown_artifact(
        key="single-field-cube-imaging-distributed-applications-timing",
        markdown=timing_summary,
        description="Summary of the distributed applications timing",
    )


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
    frequency_index: int = 2,
    polarization_index: int = 0,
) -> None:
    """
    Plot PSF, primary beam, and sky residual for one channel.

    Creates a Prefect image artifact (base64-encoded PNG) for the UI.
    """
    # import matplotlib.pyplot as plt
    import xarray as xr

    img_xds = xr.open_zarr(image_name)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, var in zip(
        axes, ["POINT_SPREAD_FUNCTION", "PRIMARY_BEAM", "SKY_RESIDUAL"], strict=False
    ):
        plane = img_xds[var].isel(
            time=0, frequency=frequency_index, polarization=polarization_index
        )
        im = ax.imshow(plane.values, origin="lower", cmap="viridis")
        ax.set_title(f"{var}\nchannel {frequency_index}, Stokes I")
        fig.colorbar(im, ax=ax)
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    b64_encoded_image = base64.b64encode(buf.read()).decode()

    create_image_artifact(
        key="single-field-cube-imaging-summary",
        image_url=f"data:image/png;base64,{b64_encoded_image}",
        description=(
            "PSF, primary beam, and sky residual "
            f"(pol={polarization_index}, freq={frequency_index})"
        ),
    )


def plot_image_statistics(imaging_ret_dict: dict) -> None:
    """Plot per-plane image statistics for sky_residual, sky_model, and sky_restored."""
    image_statistics = imaging_ret_dict["image_statistics"]
    stat_panels = [
        ("peak", "Signed peak"),
        ("rms", "RMS"),
        ("mad_sigma", "MAD sigma (robust noise)"),
        ("mean", "Mean"),
        ("median", "Median"),
        ("sum", "Sum over pixels"),
    ]
    variables = [
        v
        for v in ("sky_residual", "sky_model", "sky_restored")
        if v in image_statistics
    ]
    fig, axes = plt.subplots(
        len(stat_panels),
        len(variables),
        figsize=(5 * len(variables), 2.6 * len(stat_panels)),
        squeeze=False,
        constrained_layout=True,
    )
    for col, variable in enumerate(variables):
        stats = image_statistics[variable].isel(time=0)
        channel = np.arange(
            stats.sizes["frequency"]
        )  # frequency values: stats["frequency"]
        for row, (stat, title) in enumerate(stat_panels):
            ax = axes[row, col]
            for i, pol in enumerate(stats["polarization"].values):
                ax.plot(
                    channel,
                    stats[stat].sel(polarization=pol),
                    "o-",
                    color=f"C{i}",
                    label=str(pol),
                )
                ax.plot(
                    channel,
                    stats[stat + "_masked"].sel(polarization=pol),
                    "s--",
                    color=f"C{i}",
                    label=f"{pol} (masked)",
                )
            ax.set_title(f"{variable}: {title}", fontsize=10)
            ax.set_ylabel("Jy/beam")
            ax.grid(True, color="lightgray")
        axes[-1, col].set_xlabel("Channel")
    axes[0, 0].legend(fontsize=8)
    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    b64_encoded_image = base64.b64encode(buf.read()).decode("utf-8")

    create_image_artifact(
        key="single-field-cube-imaging-image-statistics",
        image_url=f"data:image/png;base64,{b64_encoded_image}",
        description="Per-plane image statistics for sky_residual, sky_model, and sky_restored.",
    )


@flow(log_prints=True)
def single_field_cube_imaging_flow(
    interactive: bool = False,
    ps_store: str = DEFAULT_PS_STORE,
    image_name: str = DEFAULT_IMAGE_NAME,
    scan_intents: list[str] | None = None,
    image_size: tuple[int, int] = (500, 500),
    cell_arcsec: float = 0.13,
    polarization_coords: list[str] | None = None,
    create_plots: bool = False,
    plot_frequency_index: int = 2,
    plot_polarization_index: int = 0,
    dask_cores: int = 4,
    dask_memory_limit: str = "4GB",
):
    if scan_intents is None:
        scan_intents = list(DEFAULT_SCAN_INTENTS)

    download_data(ps_store)
    ps_xdt, scan_intents, phase_direction, frequency_coords = inspect_processing_set(
        ps_store, scan_intents
    )
    imaging_config = configure_imaging_params(
        phase_direction=phase_direction,
        frequency_coords=frequency_coords,
        image_size=image_size,
        cell_arcsec=cell_arcsec,
        polarization_coords=polarization_coords,
    )

    if interactive:
        print(
            "interactive=True: pausing for Prefect UI input "
            "(open this flow run and click Resume)"
        )
        imaging_config = modify_imaging_params(imaging_config)

    returned_clean_dict = run_cube_imaging(
        ps_store,
        image_name,
        scan_intents,
        imaging_config,
        dask_cores=dask_cores,
        dask_memory_limit=dask_memory_limit,
    )
    metadata = {
        "ps_store": ps_store,
        "scan_intents": scan_intents,
        "image_size": image_size,
        "cell_arcsec": cell_arcsec,
        "polarization_coords": polarization_coords,
    }
    save_results(image_name, imaging_config, metadata)
    create_imaging_summary_artifact(returned_clean_dict)
    create_imaging_timing_artifact(returned_clean_dict)

    if create_plots:
        plot_image_products(
            image_name,
            frequency_index=plot_frequency_index,
            polarization_index=plot_polarization_index,
        )
        plot_image_statistics(returned_clean_dict)


# Run single_field_cube_imaging_flow with the default data defined in
# DEFAULT_PS_STORE
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Single Field Cube imaging Prefect demo"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Pause for Prefect UI overrides of CLEAN iteration controls",
    )
    args = parser.parse_args()
    single_field_cube_imaging_flow(
        interactive=args.interactive,
        polarization_coords=["I", "Q"],
        create_plots=True,
    )
