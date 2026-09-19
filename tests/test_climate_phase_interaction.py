"""Continuous LR04 modulation, phase convention and chronology validity."""

import numpy as np
import pandas as pd
import pytest

from toolbox import combined_likelihood
from toolbox import climate_phase_interaction as interaction
from toolbox import orbital_driver_reporting as reporting


@pytest.fixture(scope="module")
def nominal_design():
    context = interaction.add_interactions(combined_likelihood.build_context())
    return combined_likelihood.prepare_catalogue(context.events, context)


def test_interaction_uses_both_event_and_integral_products(nominal_design):
    design = nominal_design
    assert not {"lr04_phase_sin", "lr04_phase_cos"}.intersection(design.context.events.columns)
    for frame in (design.event_frame, design.integration_frame):
        np.testing.assert_allclose(frame.lr04_phase_sin, frame.lr04_scaled * frame.pre_phase_sin)
        np.testing.assert_allclose(frame.lr04_phase_cos, frame.lr04_scaled * frame.pre_phase_cos)
    result = interaction.fit_models(design, design.context.full_terms)
    comparison = result["comparison"]
    assert comparison["fit_valid"] and comparison["likelihood_nesting_ok"]
    assert comparison["df"] == 2 and comparison["n_events"] == 53
    assert comparison["exposure_kyr"] == pytest.approx(165.058)
    gain = comparison["loglik_interaction"] - comparison["loglik_full"]
    assert comparison["LR_statistic"] == pytest.approx(2 * gain)
    assert comparison["gain_bits_per_event"] == pytest.approx(gain / (53 * np.log(2)))
    assert comparison["delta_AIC"] == pytest.approx(4 - comparison["LR_statistic"])
    assert "n_bins" not in comparison and "delta_AICc" not in comparison
    history = result["coefficients"].query("term == 'same_type_exponential_history'")
    assert history.beta.le(0).all()


def test_background_uses_time_weights_and_retains_nominal_scale():
    frame = pd.DataFrame(dict(lr04=[3., 4., 5.], lr04_scaled=[-.5, 0., .5], weight=[1., 1., 8.]))
    result = interaction.background_quantiles(frame)
    assert result.lr04_permil.iloc[1] > 4.5
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



def test_unsupported_chronology_is_not_replaced(nominal_design):
    context = nominal_design.context
    columns = [f"age_kyr_bp__{event_id}" for event_id in context.events.event_id]
    selected = reporting.read_selected_realizations(
        "data/processed/NGRIP_MIS6_orbital_driver_sensitivity/selected_realizations.csv", columns, 3)
    selected.loc[1, "age_kyr_bp__NGRIP:GI-25"] = 124.0
    result = interaction.analyze_realizations(context, context.full_terms, selected, columns,
                                             show_progress=False)
    summary = result["comparison_summary"].iloc[0]
    assert (summary.n_mc_total, summary.n_mc_valid, summary.n_mc_invalid) == (3, 2, 1)
    assert result["mc_comparisons"].realization_id.tolist() == selected.realization_id.tolist()
    outside = result["mc_comparisons"].iloc[1]
    assert not outside.fit_valid and not outside.within_observation_support
    assert outside.comparison_id == "lr04_phase_interaction" and outside.df == 2
    assert result["mc_phase_estimates"].realization_id.nunique() == 2
    assert result["mc_comparisons"].loc[result["mc_comparisons"].fit_valid, "n_events"].eq(53).all()
    assert result["mc_comparisons"].exposure_kyr.dropna().nunique() == 2
