"""Unit tests for the mosaics cube imaging workflow tasks.

Every task is called through ``.fn`` so it runs as a plain function, without
data downloads, astroviper imaging or a Dask cluster. Only the tasks that
publish Prefect artifacts need the local Prefect test server. The end to end
runs of the flow live in testviper's integration tests.
"""

import base64
import pickle

import matplotlib
import numpy as np
import pytest
import xarray as xr

# Import Prefect tasks
from flowviper.prefect_workflow.mosaics_cube_imaging import (
    DEFAULT_IMAGE_DATA_VARIABLES_KEEP,
    configure_image_params,
    create_imaging_summary_artifact,
    load_image_cube,
    plot_image_products,
    prepare_image_store,
    save_results,
)

# Select non-GUI backend for plot task
matplotlib.use("Agg")

N_FREQUENCY = 3
POLARIZATION_COORDS = ["I", "Q"]
N_PIXELS = 8


@pytest.fixture(scope="module")
def prefect_harness():
    from prefect.testing.utilities import prefect_test_harness

    with prefect_test_harness(server_startup_timeout=120):
        yield


@pytest.fixture
def imaging_metadata():
    """Metadata dict with the structure inspect_processing_set returns."""
    return {
        "ps_store": "dummy.ps.zarr",
        "scan_intents": ["OBSERVE_TARGET#ON_SOURCE"],
        "ms_key": "dummy_ms",
        "center_field_name": "field_0",
        "phase_direction": np.array([0.1, -0.2]),
        "frequency_coords": np.array([1.4e9, 1.401e9, 1.402e9, 1.403e9]),
    }


@pytest.fixture
def image_store(tmp_path):
    """Small image store with the data variables the flow writes and plots."""
    shape = (N_FREQUENCY, len(POLARIZATION_COORDS), N_PIXELS, N_PIXELS)
    dims = ("frequency", "polarization", "l", "m")
    rng = np.random.default_rng(seed=1)
    img_xds = xr.Dataset(
        {
            "POINT_SPREAD_FUNCTION": (dims, rng.random(shape)),
            "PRIMARY_BEAM": (dims, rng.random(shape)),
            "SKY_RESIDUAL": (dims, rng.random(shape)),
        },
        coords={
            "frequency": np.linspace(1.4e9, 1.402e9, N_FREQUENCY),
            "polarization": POLARIZATION_COORDS,
            "l": np.arange(N_PIXELS, dtype=float),
            "m": np.arange(N_PIXELS, dtype=float),
        },
    )
    image_name = str(tmp_path / "cube.img.zarr")
    img_xds.to_zarr(image_name)
    return image_name


def _expected_cell_size(cell_arcsec: float) -> np.ndarray:
    return np.array([-cell_arcsec, cell_arcsec]) * np.pi / (180 * 3600)


def test_configure_image_params_defaults(imaging_metadata):
    config = configure_image_params.fn(imaging_metadata)

    image_params = config["image_params"]
    assert image_params["image_size"] == [500, 500]
    np.testing.assert_allclose(image_params["cell_size"], _expected_cell_size(0.13))
    np.testing.assert_array_equal(
        image_params["phase_direction"], imaging_metadata["phase_direction"]
    )
    np.testing.assert_array_equal(
        image_params["frequency_coords"], imaging_metadata["frequency_coords"]
    )
    assert image_params["polarization_coords"] == ["I", "Q"]
    assert image_params["time_coords"] == [0]
    assert image_params["fft_padding"] == 1.0

    assert config["imaging_weights_params"] == {"weighting": "natural"}
    assert config["iteration_control_params"] == {
        "max_iter": 0,
        "max_cycles": 0,
        "threshold": 0.0,
        "gain": 0.1,
        "psf_sidelobe_factor": 1.5,
        "max_iter_per_cycle": 1,
        "min_psf_fraction": 0.05,
        "max_psf_fraction": 0.8,
    }
    assert config["image_data_variables_keep"] == list(
        DEFAULT_IMAGE_DATA_VARIABLES_KEEP
    )
    assert config["processing_set_data_group_name"] == "base"
    assert config["n_mapping_parallelism"] is None
    assert config["overwrite"] is True


@pytest.mark.parametrize(
    "image_size", [(256, 256), (500, 500), (1024, 512)], ids=["256", "500", "wide"]
)
@pytest.mark.parametrize(
    "cell_arcsec", [0.065, 0.13, 0.26], ids=["fine", "default", "coarse"]
)
def test_configure_image_params_geometry(image_size, cell_arcsec, imaging_metadata):
    config = configure_image_params.fn(
        imaging_metadata, image_size=image_size, cell_arcsec=cell_arcsec
    )

    image_params = config["image_params"]
    assert image_params["image_size"] == list(image_size)
    assert isinstance(image_params["image_size"], list)
    np.testing.assert_allclose(
        image_params["cell_size"], _expected_cell_size(cell_arcsec)
    )


@pytest.mark.parametrize(
    "polarization_coords,expected",
    [
        (None, ["I", "Q"]),
        (["I", "Q"], ["I", "Q"]),
        (["I", "Q", "U", "V"], ["I", "Q", "U", "V"]),
    ],
    ids=["default", "stokes-IQ", "stokes-IQUV"],
)
def test_configure_image_params_polarization(
    polarization_coords, expected, imaging_metadata
):
    config = configure_image_params.fn(
        imaging_metadata, polarization_coords=polarization_coords
    )
    assert config["image_params"]["polarization_coords"] == expected


def test_configure_image_params_returns_independent_keep_list(imaging_metadata):
    config = configure_image_params.fn(imaging_metadata)
    config["image_data_variables_keep"].append("sky_model")

    assert "sky_model" not in DEFAULT_IMAGE_DATA_VARIABLES_KEEP
    assert configure_image_params.fn(imaging_metadata)[
        "image_data_variables_keep"
    ] == list(DEFAULT_IMAGE_DATA_VARIABLES_KEEP)


def test_prepare_image_store_removes_existing_directory(tmp_path):
    image_name = tmp_path / "existing.img.zarr"
    (image_name / "SKY_RESIDUAL").mkdir(parents=True)
    (image_name / "SKY_RESIDUAL" / "marker").write_text("old")

    result = prepare_image_store.fn(str(image_name))

    assert result == str(image_name)
    assert not image_name.exists()
    assert tmp_path.exists()


def test_prepare_image_store_returns_path_for_missing_directory(tmp_path):
    image_name = tmp_path / "new.img.zarr"

    result = prepare_image_store.fn(str(image_name))

    assert result == str(image_name)
    assert not image_name.exists()


def test_save_results_writes_pickle(tmp_path, imaging_metadata):
    image_name = str(tmp_path / "cube.img.zarr")
    imaging_config = configure_image_params.fn(imaging_metadata)

    results_path = save_results.fn(image_name, imaging_config, imaging_metadata)

    assert results_path == image_name + "_imaging_results.pkl"
    with open(results_path, "rb") as handle:
        results = pickle.load(handle)

    assert set(results) == {"image_name", "imaging_config", "metadata"}
    assert results["image_name"] == image_name

    saved_metadata = results["metadata"]
    assert set(saved_metadata) == set(imaging_metadata)
    for key in ("ps_store", "scan_intents", "ms_key", "center_field_name"):
        assert saved_metadata[key] == imaging_metadata[key]
    for key in ("phase_direction", "frequency_coords"):
        np.testing.assert_array_equal(saved_metadata[key], imaging_metadata[key])

    saved_config = results["imaging_config"]
    assert saved_config["iteration_control_params"] == (
        imaging_config["iteration_control_params"]
    )
    assert saved_config["n_mapping_parallelism"] is None
    saved_params = saved_config["image_params"]
    expected_params = imaging_config["image_params"]
    assert saved_params["image_size"] == expected_params["image_size"]
    assert saved_params["polarization_coords"] == expected_params["polarization_coords"]
    for key in ("cell_size", "phase_direction", "frequency_coords"):
        np.testing.assert_allclose(saved_params[key], expected_params[key])


def test_save_results_overwrites_previous_results(tmp_path, imaging_metadata):
    image_name = str(tmp_path / "cube.img.zarr")
    save_results.fn(
        image_name,
        configure_image_params.fn(imaging_metadata, image_size=(256, 256)),
        imaging_metadata,
    )

    results_path = save_results.fn(
        image_name,
        configure_image_params.fn(imaging_metadata, image_size=(1024, 512)),
        imaging_metadata,
    )

    with open(results_path, "rb") as handle:
        results = pickle.load(handle)
    assert results["imaging_config"]["image_params"]["image_size"] == [1024, 512]


def test_load_image_cube_returns_image_name(image_store, capsys):
    assert load_image_cube.fn(image_store) == image_store
    assert "POINT_SPREAD_FUNCTION" in capsys.readouterr().out


def test_load_image_cube_missing_store_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_image_cube.fn(str(tmp_path / "missing.img.zarr"))


def test_create_imaging_summary_artifact(prefect_harness, image_store):
    from prefect.artifacts import Artifact

    create_imaging_summary_artifact.fn(image_store)

    artifact = Artifact.get("mosaics-cube-imaging-report")
    assert artifact is not None
    assert artifact.type == "markdown"
    assert image_store in artifact.data
    for name in ("POINT_SPREAD_FUNCTION", "PRIMARY_BEAM", "SKY_RESIDUAL"):
        assert name in artifact.data
    assert f"'l': {N_PIXELS}" in artifact.data


@pytest.mark.parametrize(
    "frequency_index,polarization_index",
    [(0, 0), (N_FREQUENCY - 1, len(POLARIZATION_COORDS) - 1)],
    ids=["first-plane", "last-plane"],
)
def test_plot_image_products_creates_png_artifact(
    prefect_harness, image_store, frequency_index, polarization_index
):
    from prefect.artifacts import Artifact

    plot_image_products.fn(
        image_store,
        frequency_index=frequency_index,
        polarization_index=polarization_index,
    )

    artifact = Artifact.get("mosaics-cube-imaging-summary")
    assert artifact is not None
    assert artifact.type == "image"
    assert f"pol={polarization_index}" in artifact.description
    assert f"freq={frequency_index}" in artifact.description

    prefix = "data:image/png;base64,"
    assert artifact.data.startswith(prefix)
    png = base64.b64decode(artifact.data[len(prefix) :])
    assert png.startswith(b"\x89PNG\r\n\x1a\n")


def test_plot_image_products_frequency_index_out_of_range(image_store):
    with pytest.raises(IndexError):
        plot_image_products.fn(image_store, frequency_index=N_FREQUENCY)
