"""Source preservation and continuous-time Barker definition comparison."""
from dataclasses import replace
import hashlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from Barker2011 import Barker2011_event_phase_analysis as barker
from toolbox import combined_likelihood


@pytest.fixture(scope="module")
def result():
    return barker.run_analysis()


@pytest.fixture(scope="module")
def fixed():
    return barker.run_analysis("fixed_threshold")


def test_source_picks_and_descriptive_phases_are_unchanged(result):
    assert hashlib.sha256(barker.BARKER_XLS.read_bytes()).hexdigest() == (
        "5b028eb1d9edc5250abbe403efeb0a8fd80171ed0eab42a8bd46c338619258cd")
    events = result["events"]
    assert len(events) == 70 and events.event_id.is_unique
    assert events.event_id.tolist() == [f"Barker_S3_{row:03d}" for row in events.source_excel_row]
    assert events.age_column.unique().tolist() == [barker.AGE_COLUMN]
    np.testing.assert_allclose(events.event_age_ka.iloc[[0, -1]],
                               [11.402407613184174, 396.4642638777152], atol=1e-12)
    ray = result["rayleigh"]
    np.testing.assert_allclose([ray["mean_phase_deg"], ray["mean_resultant_length"], ray["rayleigh_p"]],
                               [0.8300241493611098, 0.19001179622527256, 0.07948875469940513], atol=1e-10)


def test_exact_support_history_and_age_sampler_interface(result):
    fit = result["fit"]
    assert result["context"].history_tau_ka == 1.5
    assert "mis6_segment" not in fit.full.terms
    assert len(fit.design.event_frame) == 69
    assert fit.design.weights.sum() == pytest.approx(396.4642638777152)
    assert result["events"].event_role.eq("conditioning").sum() == 1
    ages = result["events"].event_age_ka.to_numpy()
    frame = fit.design.event_frame
    expected = [np.sum(np.exp(-(ages[ages > a] - a) / 1.5)) for a in frame.age_kyr_bp]
    np.testing.assert_allclose(frame.same_type_exponential_history, expected, atol=1e-12)
    # The upstream age sampler still inserts IDs itself, without duplication.
    from Barker2011 import Barker2011_event_age_uncertainty as chronology
    prepared = chronology.prepare_events(chronology.prepare_controls())
    assert prepared.event_id.tolist() == result["events"].event_id.tolist()
    np.testing.assert_array_equal(prepared.event_age_ka, result["events"].event_age_ka)


def test_models_and_integration_refinement(result, fixed):
    for current in (result, fixed):
        fit = current["fit"]
        for model in (fit.reduced, fit.full):
            assert model.converged and model.gradient_max <= 1e-6
            assert dict(zip(model.terms, model.beta))[combined_likelihood.HISTORY_TERM] <= 0
        s = current["summary"].iloc[0]
        assert s.gain_bits_per_event == pytest.approx(s.LR_statistic / (2 * s.n_response_events * np.log(2)))
        assert s.delta_AIC_full_minus_reduced == pytest.approx(4 - s.LR_statistic)
        refined = combined_likelihood.fit_catalogue(current["events"], replace(current["context"], quadrature_order=8))
        assert abs(refined.full.log_likelihood - fit.full.log_likelihood) < 1e-6
        assert abs(refined.reduced.log_likelihood - fit.reduced.log_likelihood) < 1e-6
    _, multiplier = barker.phase_response_curve(0.3, -0.4)
    assert multiplier.max() / multiplier.min() == pytest.approx(np.exp(1), rel=1e-4)


def test_fixed_picks_use_their_own_exact_anchor(fixed):
    source = pd.read_excel(barker.BARKER_XLS, sheet_name="Sheet1", header=8)
    ages = pd.to_numeric(source[barker.AGE_COLUMN], errors="coerce")
    selected = pd.to_numeric(source["DO pick"], errors="coerce").eq(1) & ages.between(0, 400)
    assert set(fixed["events"].source_row) == set(source.index[selected])
    assert len(fixed["events"]) == 59
    assert fixed["events"].included_in_response.sum() == 58
    assert fixed["context"].response_exposure_kyr == pytest.approx(392.2456958970233)
    assert fixed["likelihood_tests"].dataset_id.iloc[0] == "barker_fixed_threshold_speleo_0_400"


def test_compact_outputs_and_overlay(result, fixed, tmp_path):
    barker.write_outputs(result, tmp_path)
    assert (tmp_path / "fitted_rates.csv").exists()
    assert (tmp_path / "support.csv").exists()
    assert not any("binned" in p.name for p in tmp_path.iterdir())
    assert not barker.build_parameters(result).parameter.str.contains("bin_width|origin").any()
    with plt.rc_context():
        fig = barker.plot_results(result, fixed)
        assert len(fig.axes) == 3 and fig.axes[1].name == "polar"
        assert fig.axes[0].get_xlim() == (400, 0)
        assert [axis.texts[-1].get_text() for axis in fig.axes] == ["a", "b", "c"]
        plt.close(fig)
