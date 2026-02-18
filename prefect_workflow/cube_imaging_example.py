# An example imaging workflow based on astroviper's imaging loop
# demo notebook
from prefect import flow, task
from prefect.artifacts import (
    create_markdown_artifact,
    create_image_artifact,
)
from prefect.flow_runs import pause_flow_run
from prefect.input import RunInput
from astroviper.core.imaging.imager import run_imaging_loop
from astroviper.core.imaging.imaging_utils.standard_gridding_example import (
    generate_ms4_with_point_sources,
)
from xradio.image import write_image

import random
import numpy as np
import xarray as xr


# define input class for user specifiable imaging parameters
class ImagingParamsInput(RunInput):
    gain: float
    niter: int
    threshold: float  # Stop at 10 mJy
    nmajor: int
    cyclefactor: float
    minpsffraction: float
    maxpsffraction: float


@task(log_prints=True)
def data_prep():
    """
    Data preparation:
      generate synthetic MSv4 data with point sources
    """
    nsources = 4
    source_fluxes = np.ones(nsources)
    sources, npix, cell, ms4 = generate_ms4_with_point_sources(nsources, source_fluxes)
    print(f"Generated MS4 with {nsources} point sources")
    print(f"Image size: {npix} x {npix} pixels")
    print(f"Cell size: {cell}")
    ms4["WEIGHT"] = xr.ones_like(ms4["WEIGHT"])
    ms4["FLAG"] = xr.zeros_like(ms4["FLAG"])
    return sources, npix, cell, ms4


@task(log_prints=True)
def configure_imaging_loop(npix, cell) -> dict:
    """Configure imaging loop parameters based on the generated data geometry and other settings"""
    print("Enter configure_imaging_loop")
    # Convert cell size to radians
    cell_rad = cell.to("rad").value

    # Configure imaging parameters
    params = {
        # Image geometry - use the npix from generated data
        "image_name:": "test_cube",
        "image_size": (npix, npix),
        "cell_size": (-cell_rad, cell_rad),  # RA typically negative
        # Gridding
        "support": 7,
        "oversampling": 100,
        # Deconvolution
        "algorithm": "hogbom",
        "gain": 0.1,
        "niter": 10000,  # Max total iterations
        "threshold": 0.01,  # Stop at 10 mJy
        # Major cycle control - CAPPED AT 3 FOR THIS DEMO
        "nmajor": 3,
        "cyclefactor": 1.5,
        "minpsffraction": 0.05,
        "maxpsffraction": 0.8,
        # Spectral/polarization mode
        "chan_mode": "cube",
        "corr_type": "linear",  # XX, YY -> Stokes I, Q
    }

    print(f"Configured imaging loop parameters:{params}")
    return params


@flow(log_prints=True)
def modify_imaging_params(params: dict) -> dict:
    """Modify imaging parameters based on user input from Prefect UI
    Not intended to be interactive clean so it is only for the initial setting before running the imaging loop
    """
    # pause for user input to modify imaging parameters
    user_input: ImagingParamsInput = pause_flow_run(
        wait_for_input=ImagingParamsInput.with_initial_data(
            image_name="test_cube",
            gain=0.1,
            niter=10000,
            threshold=0.01,
            nmajor=3,
            cyclefactor=1.5,
            minpsffraction=0.05,
            maxpsffraction=0.8,
        )
    )
    print("Enter modify_imaging_params")
    # Example modification: change the gain and niter for testing
    params["gain"] = user_input.gain
    params["niter"] = user_input.niter
    params["threshold"] = user_input.threshold
    params["nmajor"] = user_input.nmajor
    params["cyclefactor"] = user_input.cyclefactor
    params["minpsffraction"] = user_input.minpsffraction
    params["maxpsffraction"] = user_input.maxpsffraction

    print(f"Modified imaging loop parameters:{params}")
    return params


@task(log_prints=True)
def run_imaging_loop_task(
    ms4, imaging_params
) -> tuple[xr.Dataset, xr.Dataset, dict, object]:
    """Run cube imaging loop"""
    # Run the imaging loop
    model, residual, return_dict, controller = run_imaging_loop(
        ms4=ms4,
        params=imaging_params,
        initial_model=None,
        output_dir=".",
    )

    return model, residual, return_dict, controller


@task
def create_imaging_report(controller):
    """Report imaging states to Prefect Artifact to send to UI"""

    controller_state = f"""
    Major cycles completed: {controller.major_done}
    Total iterations: {controller.total_iter_done}
    Stop code: {controller.stopcode}
    Stop reason: {controller.stopdescription}
    """

    create_markdown_artifact(
        key="imaging-loop-report",
        markdown=controller_state,
        description="Current state of the imaging ontroller",
    )


@task(log_prints=True)
def make_summary_image(model: np.ndarray, residual: np.ndarray):
    """
    Create a summary image plot
    The plot is avaiable as a Prefect Artifact. The image link
    that the Prefect UI can display has to be a public URL. For demo purposes,
    it is encoded as a Base64 string and embed it in the UI.
    """
    # Code adapted from the imaging-loop notebook
    import matplotlib.pyplot as pl

    #
    from io import BytesIO
    import base64

    fig, axes = pl.subplots(1, 3, figsize=(15, 5))

    # Extract Stokes I (index 0) from shape (chan, stokes, y, x)
    model_I = model[0, 0, :, :].real
    residual_I = residual[0, 0, :, :].real

    # Model (Stokes I)
    im0 = axes[0].imshow(model_I, origin="lower")
    axes[0].set_title(f"Model (Stokes I)\nTotal flux: {np.sum(model_I):.3f} Jy")
    pl.colorbar(im0, ax=axes[0], label="Jy/pixel")

    # Residual (Stokes I)
    im1 = axes[1].imshow(residual_I, origin="lower")
    axes[1].set_title(f"Residual (Stokes I)\nPeak: {np.max(np.abs(residual_I)):.4f} Jy")
    pl.colorbar(im1, ax=axes[1], label="Jy/pixel")

    # Restored (Model + Residual, approximate)
    restored = model_I + residual_I
    im2 = axes[2].imshow(restored, origin="lower")
    axes[2].set_title("Restored (Model + Residual)")
    pl.colorbar(im2, ax=axes[2], label="Jy/pixel")

    pl.tight_layout()
    # pl.savefig("image_summary.png")
    # not recommended by Prefect by for demo purposes
    # encode image as binary stream to embed the image in the UI
    buf = BytesIO()
    pl.savefig(buf, format="png")
    buf.seek(0)
    # image bytes to Base64
    b64_encoded_image = base64.b64encode(buf.read()).decode()

    create_image_artifact(
        key="imaging-summary",
        # image_url="./imaging_summary.png",
        image_url=f"data:image/png;base64,{b64_encoded_image}",
        description="Summary of imaging results (Model, Residual, Restored)",
    )


@task(log_prints=True)
def save_results(
    model: np.ndarray, residual: np.ndarray, return_dict: dict, image_name: str
):
    """Save imiging results to disk, currently only saves the imaging loop return dictionary"""
    # write_image(model, image_name + ".model")
    # write_image(residual, image_name + ".residual")

    # save return_dict as json
    import pickle

    with open(image_name + "_imaging_results.pkl", "wb") as f:
        pickle.dump(return_dict.data, f)


@flow
def imaging_flow():
    """Cube imaging stage workflow"""
    sources, npix, cell, ms4 = data_prep()
    imaging_params = configure_imaging_loop(npix, cell)
    updated_params = modify_imaging_params(imaging_params)
    model, residual, return_dict, controller = run_imaging_loop_task(
        ms4, updated_params
    )
    create_imaging_report(controller)
    make_summary_image(model, residual)
    save_results(model, residual, return_dict, updated_params["image_name:"])


if __name__ == "__main__":
    imaging_flow()
