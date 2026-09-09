"""Scientific checks for depth-supported Sofular analytical-error propagation."""

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MIS6 import sofular_chronology as chronology


EVENT_AGES = np.array([176.779, 179.831, 180.309, 190.972, 194.238])


@pytest.fixture(scope="module")
def sources():
    return chronology.load_sources()


def test_original_units_full_controls_and_hiatus(sources):
    so4, so57 = sources["So-4"], sources["So-57"]
    assert len(so4.series) == 2034 and len(so57.series) == 854
    assert len(so4.controls) == 24 and len(so57.controls) == 7
    np.testing.assert_array_equal(so4.hiatus, [781.5])
    assert so57.hiatus.size == 0
    assert so57.series["age_ka_bp"].duplicated().sum() == 22
    control = so57.controls.set_index("control_id").loc["So57-1"]
    assert control.age_ka_bp == pytest.approx(172.800282392294)
    assert control.age_error_2sigma_ka == pytest.approx(0.730762278731393)
    assert control.source_age_unit == "yr"
    assert so4.controls.source_age_unit.eq("ka").all()
    diagnostics = chronology.control_diagnostics(sources)
    offset = diagnostics.set_index("control_id").loc["So4-M21", "model_minus_uth_age_ka"]
    assert offset == pytest.approx(1.359175784076)
    assert not diagnostics.set_index("control_id").loc["So4-M19", "model_age_available"]
    assert pd.isna(diagnostics.set_index("control_id").loc["So4-M19", "nominal_model_age_ka_bp"])


def test_full_so57_dates_support_69_but_proxy_does_not_cover_613(sources):
    context = chronology.build_context(EVENT_AGES[:4], "So-57", sources=sources)
    assert context.bracket_left_ids[0] == "So57-2"
    assert context.bracket_right_ids[0] == "So57-3"
    with pytest.raises(ValueError, match="outside proxy support"):
        chronology.build_context(EVENT_AGES, "So-57", sources=sources)


def test_so4_projection_and_local_error_covariance(sources):
    context = chronology.build_context(EVENT_AGES, sources=sources)
    assert len(context.control_ids) == 10
    assert context.control_ids[0] == "So4-M20"
    assert context.control_ids[-1] == "So4-M27"
    np.testing.assert_allclose(context.interpolation_weights.sum(axis=1), 1.0)
    assert np.all(np.sum(context.interpolation_weights > 0.0, axis=1) <= 2)
    covariance = (context.interpolation_weights * context.control_sigmas_ka**2) @ context.interpolation_weights.T
    np.testing.assert_allclose(context.nominal_covariance_ka2, covariance)
    np.testing.assert_allclose(context.nominal_sigma_ka**2, np.diag(covariance))
    assert np.linalg.matrix_rank(covariance) > 1
    assert covariance[0, -1] == 0.0
    assert covariance[0, 1] > 0.0
    diagnostics = chronology.projection_diagnostics(context)
    assert diagnostics.projection_method.eq("assumed_age_coordinate_projection").all()
    np.testing.assert_allclose(diagnostics.weight_left + diagnostics.weight_right, 1)


@pytest.fixture
def simple_source():
    # Nonlinear published growth curve: the midpoint in depth is not the
    # midpoint in age, so this distinguishes age warp from depth interpolation.
    series = pd.DataFrame({
        "depth_mm": [0.0, 1.0, 2.0, 3.0], "age_ka_bp": [10.0, 11.0, 13.0, 16.0],
        "d13C": [-5.0] * 4, "d18O": [-10.0] * 4, "segment_id": [0] * 4,
    })
    controls = pd.DataFrame({
        "control_id": ["a", "b", "c"], "depth_mm": [0.0, 1.5, 3.0],
        "age_ka_bp": [10.2, 12.1, 15.8], "age_error_2sigma_ka": [0.2, 0.4, 0.6],
        "chronology_sigma_ka": [0.1, 0.2, 0.3], "segment_id": [0] * 3,
    })
    return chronology.SofularSource("So-4", series, controls, np.array([]), Path("synthetic.txt"))


def test_analytical_weights_variance_and_simulated_covariance(simple_source):
    context = chronology.build_context(np.array([11.0, 14.0]), sources={"So-4": simple_source})
    np.testing.assert_allclose(context.nominal_model_control_ages_ka, [10, 12, 16])
    np.testing.assert_allclose(context.interpolation_weights, [[0.5, 0.5, 0], [0, 0.5, 0.5]])
    expected = np.array([[0.0125, 0.0100], [0.0100, 0.0325]])
    np.testing.assert_allclose(context.nominal_covariance_ka2, expected)
    rng = np.random.default_rng(17)
    draws = rng.normal(0.0, context.control_sigmas_ka, size=(100000, 3)) @ context.interpolation_weights.T
    np.testing.assert_allclose(np.cov(draws, rowvar=False), expected, rtol=0.02)


class FixedErrors:
    def __init__(self, errors):
        self.errors = np.asarray(errors)
        self.calls = 0

    def normal(self, loc, scale):
        self.calls += 1
        return self.errors.copy()


def test_rejection_is_one_whole_proposal_without_sorting(simple_source):
    context = chronology.build_context(np.array([11.0, 14.0]), sources={"So-4": simple_source})
    rng = FixedErrors([0.0, -3.0, 0.0])
    offsets, errors = chronology.propose_offsets(context, rng)
    assert offsets is None and rng.calls == 1
    np.testing.assert_array_equal(errors, [0, -3, 0])


def test_positive_knots_preserve_entire_nonlinear_curve(simple_source):
    ages = np.linspace(10.0, 16.0, 601)
    context = chronology.build_context(ages, sources={"So-4": simple_source})
    offsets, errors = chronology.propose_offsets(context, FixedErrors([0.3, -0.4, 0.2]))
    assert offsets is not None
    assert np.all(np.diff(ages + offsets) > 0.0)
    np.testing.assert_allclose(offsets[[0, 200, -1]], errors)


def test_zero_errors_preserve_points_and_lag_only_moves_projection(sources):
    context = chronology.build_context(EVENT_AGES, sources=sources)
    offsets, _ = chronology.propose_offsets(context, FixedErrors(np.zeros(10)))
    np.testing.assert_array_equal(offsets, np.zeros(5))
    shifted = chronology.build_context(EVENT_AGES, mapping_lag_ka=0.5, sources=sources)
    np.testing.assert_array_equal(shifted.event_ages_ka, EVENT_AGES)
    np.testing.assert_allclose(shifted.projected_ages_ka, EVENT_AGES + 0.5)
    assert np.all(shifted.event_depth_mm > context.event_depth_mm)


@pytest.mark.parametrize("ages,component,message", [
    ([100.0], "So-4", "hiatus"),
    ([160.5], "So-4", "outside dated support"),
    ([171.0], "So-57", "outside dated support"),
    ([172.154], "So-57", "non-unique depth"),
    ([205.0], "So-4", "outside dated support"),
    ([180.0, 179.0], "So-4", "strictly increase"),
    ([np.nan], "So-4", "finite"),
    ([], "So-4", "nonempty"),
])
def test_invalid_age_projection_cannot_extrapolate(sources, ages, component, message):
    with pytest.raises(ValueError, match=message):
        chronology.build_context(np.asarray(ages), component, sources=sources)


def test_source_schema_and_coordinate_corruption_rejected(simple_source):
    missing = replace(simple_source, series=simple_source.series.drop(columns="d13C"))
    with pytest.raises(ValueError, match="schema"):
        chronology.build_context(np.array([11.0]), sources={"So-4": missing})
    series = simple_source.series.copy()
    series.loc[1, "depth_mm"] = 0.0
    with pytest.raises(ValueError, match="depths must strictly increase"):
        chronology.build_context(np.array([11.0]), sources={"So-4": replace(simple_source, series=series)})
    controls = simple_source.controls.copy()
    controls.loc[0, "age_error_2sigma_ka"] = -1.0
    with pytest.raises(ValueError, match="uncertainties must be positive"):
        chronology.build_context(np.array([11.0]), sources={"So-4": replace(simple_source, controls=controls)})


def test_original_file_schema_error_fails_before_numeric_guessing(tmp_path):
    for component, filename in chronology.SOURCE_FILES.items():
        content = (chronology.DEFAULT_RAW_DIR / filename).read_text()
        if component == "So-57":
            content = content.replace("age_corr_BP1950", "age_corr_kaBP1950")
        (tmp_path / filename).write_text(content)
    with pytest.raises(ValueError, match="schema or age units"):
        chronology.load_sources(tmp_path)
