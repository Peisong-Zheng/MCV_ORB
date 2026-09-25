"""Scientific checks for the separate NGRIP cooling and warming experiments."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from NGRIP import ngrip_transition_phase_sensitivity as analysis
from toolbox import combined_likelihood as likelihood


NGRIP_DIR = Path(__file__).resolve().parents[1]
EXPECTED = {
    "cooling": dict(count=35, anchor="GS-26", age=119.09, valid=9263,
                    gain=0.135703244352517, p=0.04083895305343282,
                    phase=340.0846153057898),
    "warming": dict(count=34, anchor="GI-25", age=115.32, valid=9982,
                    gain=0.19356196333138304, p=0.01194420970146628,
                    phase=327.1265794732237),
}


@pytest.fixture(scope="module")
def contexts():
    return {kind: analysis.load_catalogue(kind) for kind in EXPECTED}


@pytest.fixture(scope="module")
def source_draws():
    return pd.read_csv(
        NGRIP_DIR / "data/processed/ngrip_event_age_uncertainty/ngrip_event_age_realizations.csv",
        float_precision="round_trip",
    )


@pytest.mark.parametrize("kind", EXPECTED)
def test_catalogue_anchor_exposure_and_model_match_main_method(contexts, kind):
    context, expected = contexts[kind], EXPECTED[kind]
    events = context.events
    assert len(events) == expected["count"]
    assert events.event_id.is_unique
    assert set(events.segment_id) == {"NGRIP"}
    assert events.event_age_kyr_bp.is_monotonic_increasing
    assert events.event_label.str.startswith("GS-" if kind == "cooling" else "GI-").all()
    assert events.iloc[-1].event_label == expected["anchor"]
    segment = context.segments["NGRIP"]
    assert segment.observation_start_kyr_bp == 12.0
    assert segment.observation_end_kyr_bp == 123.0
    assert segment.anchor_age_kyr_bp == pytest.approx(expected["age"])
    assert context.response_exposure_kyr == pytest.approx(expected["age"] - 12)
    assert context.history_tau_ka == 1.5 and context.initial_history == 0
    assert context.reduced_terms == (likelihood.HISTORY_TERM, "lr04_scaled", "co2_scaled")
    assert context.full_terms == context.reduced_terms + ("pre_phase_sin", "pre_phase_cos")

    fit = likelihood.fit_catalogue(events, context)
    assert fit.summary["n_response_events"] == expected["count"] - 1
    assert fit.summary["gain_bits_per_event"] == pytest.approx(expected["gain"], abs=1e-8)
    assert fit.summary["nominal_LR_p"] == pytest.approx(expected["p"], abs=1e-8)
    assert fit.summary["pre_phase_preferred_deg"] == pytest.approx(expected["phase"], abs=1e-5)
    assert fit.summary["beta_history"] <= 0


@pytest.mark.parametrize("kind", EXPECTED)
def test_saved_realizations_keep_label_alignment_order_and_outside_support(contexts, source_draws, kind):
    context = contexts[kind]
    draws, columns = analysis.load_age_realizations(context, n_realizations=10000)
    # GS/GI labels identify the age columns; source labels may have extra suffixes.
    expected_columns = ["age_ka_bp__" + label for label in context.events.event_label]
    assert columns == expected_columns
    assert len(draws) == 10000 and draws.realization_id.is_unique
    pd.testing.assert_series_equal(draws.realization_id, source_draws.realization_id)
    np.testing.assert_array_equal(draws[columns], source_draws[expected_columns])
    ages = draws[columns].to_numpy(float)
    assert np.isfinite(ages).all() and np.all(np.diff(ages, axis=1) > 0)
    valid = (ages.min(axis=1) >= 12) & (ages.max(axis=1) <= 123)
    assert valid.sum() == EXPECTED[kind]["valid"]
    assert (ages.min(axis=1) < 12).sum() == 0
    # Unsupported rows remain present for downstream flagging, not redrawing.
    assert (~valid).sum() == 10000 - EXPECTED[kind]["valid"]


@pytest.fixture(scope="module")
def small_run():
    fitted = likelihood.fit_catalogue
    calls = []

    def inspect_refit(events, context, **kwargs):
        fit = fitted(events, context, **kwargs)
        # Check the actual refits, including simulated and age-perturbed events.
        assert fit.context.scaling == context.scaling
        ages = events.event_age_kyr_bp.to_numpy(float)
        frame = fit.design.event_frame
        expected_history = [np.exp(-(ages[ages > age] - age) / 1.5).sum()
                            for age in frame.age_kyr_bp]
        np.testing.assert_allclose(frame[likelihood.HISTORY_TERM], expected_history, atol=1e-13)
        assert fit.design.weights.sum() == pytest.approx(ages.max() - 12)
        calls.append(dict(catalogue_id=context.catalogue_id,
                          fixed_support=kwargs.get("fixed_support", False),
                          scaling={name: scale.copy() for name, scale in context.scaling.items()},
                          nominal_anchor=context.segments["NGRIP"].anchor_age_kyr_bp,
                          fitted_anchor=fit.context.segments["NGRIP"].anchor_age_kyr_bp))
        return fit

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(likelihood, "fit_catalogue", inspect_refit)
        result = analysis.run_analysis(n_bootstrap=2, n_realizations=3,
                                       seed=20260921, n_workers=1, show_progress=False)
    return result, calls


def test_small_analysis_uses_plus_one_p_valid_age_denominators_and_fixed_scaling(small_run):
    result, calls = small_run
    assert result["summary"].event_type.tolist() == ["cooling", "warming"]
    assert result["age_summary"].event_type.tolist() == ["cooling", "warming"]
    for kind in EXPECTED:
        summary = result["summary"].set_index("event_type").loc[kind]
        bootstrap = result["bootstrap_replicates"].query("event_type == @kind")
        age = result["age_realizations"].query("event_type == @kind")
        age_summary = result["age_summary"].set_index("event_type").loc[kind]
        assert len(bootstrap) == 2 and bootstrap.fit_valid.all()
        assert len(age) == 3 and age.fit_valid.all()
        exceedances = (bootstrap.LR_statistic >= summary.LR_statistic).sum()
        assert summary.empirical_p_plus_one == pytest.approx((1 + exceedances) / 3)
        assert summary.n_failed_replicates == 0
        assert age_summary.n_valid == 3
        assert age_summary.fraction_nominal_p_below_0p05 == pytest.approx(
            age.loc[age.fit_valid, "nominal_LR_p"].lt(.05).mean())

        context = result["contexts"][kind]
        refits = [call for call in calls if call["catalogue_id"] == context.catalogue_id]
        assert all(call["scaling"] == context.scaling for call in refits)
        simulations = [call for call in refits if call["fixed_support"]]
        assert len(simulations) == 2
        assert all(call["fitted_anchor"] == call["nominal_anchor"] for call in simulations)
        # Age perturbations move the anchor and exposure, while scaling is fixed.
        assert sum(call["fitted_anchor"] != call["nominal_anchor"] for call in refits) == 3
        assert result["fits"][kind].context.scaling == context.scaling


def test_small_analysis_reproduces_across_worker_counts(small_run):
    serial, _ = small_run
    parallel = analysis.run_analysis(n_bootstrap=2, n_realizations=3,
                                    seed=20260921, n_workers=2, show_progress=False)
    for key in ("bootstrap_replicates", "age_realizations", "summary", "age_summary"):
        pd.testing.assert_frame_equal(serial[key], parallel[key])
