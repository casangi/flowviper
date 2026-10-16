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
    assert params["iteration_control_params"]["max_iter"] == 100

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
