"""Scientific checks for the two-term LR04 modulation experiment."""

import numpy as np
import pandas as pd
import pytest

from toolbox import climate_phase_interaction as interaction


def synthetic_response():
    rng = np.random.default_rng(917)
    phase = rng.uniform(0, 2 * np.pi, 1000)
    climate = rng.uniform(-0.5, 0.5, len(phase))
    frame = pd.DataFrame(dict(lr04=climate + 4.0, lr04_scaled=climate,
        pre_phase_sin=np.sin(phase), pre_phase_cos=np.cos(phase), dt_ka=0.2))
    eta = 1.0 + 0.3 * climate + 0.5 * np.cos(phase) + 2.5 * climate * np.sin(phase)
    frame["event_count"] = rng.poisson(np.exp(eta) * frame.dt_ka)
    return frame


def test_interaction_is_nested_and_uses_the_same_exposure():
    frame = synthetic_response()
    original = frame.copy(deep=True)
    terms = ("lr04_scaled", "pre_phase_sin", "pre_phase_cos")
    result = interaction.fit_models(frame, terms)
    comparison = result["comparison"]
    pd.testing.assert_frame_equal(frame, original)
    assert comparison["fit_valid"] and comparison["likelihood_nesting_ok"]
    assert comparison["df"] == 2
    assert comparison["n_events"] == frame.event_count.sum()
    assert comparison["exposure_kyr"] == pytest.approx(frame.dt_ka.sum())
    gain = comparison["loglik_interaction"] - comparison["loglik_full"]
    assert comparison["LR_statistic"] == pytest.approx(2 * gain)
    assert comparison["info_bits_per_event"] == pytest.approx(gain / (frame.event_count.sum() * np.log(2)))
    assert comparison["delta_AIC"] == pytest.approx(4 - comparison["LR_statistic"])
    assert comparison["nominal_p"] < 0.01


def test_background_uses_duration_weights_without_rescaling():
    frame = pd.DataFrame(dict(lr04=[3., 4., 5.], lr04_scaled=[-.5, 0., .5], dt_ka=[1., 1., 8.]))
    result = interaction.background_quantiles(frame)
    assert result.lr04_permil.iloc[1] > 4.5  # Most exposed time is at the high background.
    np.testing.assert_allclose(result.lr04_scaled, (result.lr04_permil - 4) / 2)
    assert np.all(np.diff(result.lr04_permil) >= 0)


def test_conditional_phase_retains_peak_and_amplitude_changes():
    coefficients = pd.DataFrame(dict(model_id="interaction",
        term=["pre_phase_sin", "pre_phase_cos", "lr04_phase_sin", "lr04_phase_cos"],
        beta=[0., 1., 1., 0.]))
    backgrounds = pd.DataFrame(dict(background_id=["low", "mid", "high"], lr04_scaled=[-1., 0., 1.]))
    estimates, curves = interaction.conditional_phase(coefficients, backgrounds)
    np.testing.assert_allclose(estimates.preferred_phase_deg, [315., 0., 45.])
    np.testing.assert_allclose(estimates.phase_amplitude, [np.sqrt(2), 1., np.sqrt(2)])
    for row in estimates.itertuples(index=False):
        curve = curves.loc[curves.background_id.eq(row.background_id)]
        assert curve.rate_multiplier.max() / curve.rate_multiplier.min() == pytest.approx(row.rate_ratio_max_vs_min)
        assert curve.loc[curve.rate_multiplier.idxmax(), "phase_deg"] == pytest.approx(row.preferred_phase_deg)


def test_unsupported_realization_stays_in_denominator():
    response = synthetic_response()
    terms = ("lr04_scaled", "pre_phase_sin", "pre_phase_cos")
    selected = pd.DataFrame(dict(realization_id=["valid1", "outside", "valid2"], age=[1., -1., 2.]))

    def frame_for_ages(ages):
        if ages[0] < 0:
            raise ValueError("outside observation support")
        return response

    result = interaction.analyze_realizations(response, terms, selected, ["age"], frame_for_ages,
                                             show_progress=False)
    summary = result["comparison_summary"].iloc[0]
    assert (summary.n_mc_total, summary.n_mc_valid, summary.n_mc_invalid) == (3, 2, 1)
    assert result["mc_comparisons"].realization_id.tolist() == selected.realization_id.tolist()
    outside = result["mc_comparisons"].set_index("realization_id").loc["outside"]
    assert not outside.fit_valid and not outside.within_observation_support
    assert result["mc_phase_estimates"].realization_id.nunique() == 2
    # Identical accepted input frames give an exact point/MC curve agreement.
    np.testing.assert_allclose(result["phase_curves"].rate_multiplier_point,
                               result["phase_curves"].rate_multiplier_median)
