"""Smoke tests for flowviper.

They need no test data and no network access beyond a local Prefect test
server, so they run quickly in CI on every supported Python version.
"""

import importlib
import pkgutil

import numpy as np
import pytest

import flowviper.prefect_workflow as prefect_workflow

MODULES = sorted(
    m.name
    for m in pkgutil.walk_packages(
        prefect_workflow.__path__, prefix=prefect_workflow.__name__ + "."
    )
)


@pytest.mark.parametrize("name", MODULES)
def test_import(name):
    importlib.import_module(name)


@pytest.fixture(scope="module")
def prefect_harness():
    from prefect.testing.utilities import prefect_test_harness

    with prefect_test_harness(server_startup_timeout=120):
        yield


def test_math_flow(prefect_harness):
    from flowviper.prefect_workflow.example_template import compute_data, math_flow

    assert compute_data.fn(3, 4) == 7
    assert math_flow(3, 4) == 7


def test_configure_imaging_defaults():
    from flowviper.prefect_workflow import single_field_cube_imaging_example

    params = single_field_cube_imaging_example.configure_imaging_params.fn(
        np.zeros(2), np.array([1.0e11, 1.1e11])
    )
    assert params["cycle_control_params"]["max_iter"] == 100
    assert params["cycle_control_params"]["primary_beam_limit"] == 0.2
    assert "iteration_control_params" not in params
    # Gridding defaults are astroviper's; its default kernel needs padding >= 1.05.
    assert params["image_params"]["fft_padding"] == 1.2
    assert params["gridder"] == "anti_alias"
    assert params["gridding_kernel_params"] is None
    # Defaults of the image_cube_single_field run settings are astroviper's.
    assert params["imaging_weights_params"]["casa_weighting_implementation"] is False
    assert params["image_data_variables_keep"] == list(
        single_field_cube_imaging_example.DEFAULT_IMAGE_DATA_VARIABLES_KEEP
    )
    assert params["algorithm"] == "hogbom"
    assert params["single_precision_image"] is True
    assert params["processing_function_threads"] == 1
    assert params["n_mapping_parallelism"] is None
    assert params["restore"] is False
    assert params["fft_backend"] == "pyfftw"
    assert params["write_visibility_model_to_ps"] is False

    run_input = single_field_cube_imaging_example.ImagingParamsInput(
        gain=0.1,
        max_iter=1,
        threshold=0.0,
        max_cycles=1,
        psf_sidelobe_factor=1.5,
        min_psf_fraction=0.1,
        max_psf_fraction=0.8,
    )
    assert run_input.max_iter == 1


def test_configure_imaging_run_settings():
    from flowviper.prefect_workflow import single_field_cube_imaging_example

    keep = ["sky_residual", "point_spread_function", "primary_beam"]
    kernel = {"type": "prolate_spheroidal_casa"}
    params = single_field_cube_imaging_example.configure_imaging_params.fn(
        np.zeros(2),
        np.array([1.0e11, 1.1e11]),
        algorithm="hogbom_many_threads",
        primary_beam_limit=0.0,
        gridder="anti_alias",
        gridding_kernel_params=kernel,
        fft_padding=1.0,
        casa_weighting_implementation=True,
        image_data_variables_keep=keep,
        single_precision_image=False,
        processing_function_threads=12,
        n_mapping_parallelism=5,
        restore=True,
        fft_backend="scipy",
        write_visibility_model_to_ps=True,
    )
    assert params["algorithm"] == "hogbom_many_threads"
    assert params["image_params"]["fft_padding"] == 1.0
    assert params["cycle_control_params"]["primary_beam_limit"] == 0.0
    assert params["gridding_kernel_params"] == kernel
    assert params["gridding_kernel_params"] is not kernel
    assert params["imaging_weights_params"]["casa_weighting_implementation"] is True
    assert params["image_data_variables_keep"] == keep
    assert params["image_data_variables_keep"] is not keep
    assert params["single_precision_image"] is False
    assert params["processing_function_threads"] == 12
    assert params["n_mapping_parallelism"] == {"frequency": 5}
    assert params["restore"] is True
    assert params["fft_backend"] == "scipy"
    assert params["write_visibility_model_to_ps"] is True
