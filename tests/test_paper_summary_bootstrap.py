"""Paper annotations must use the bootstrap belonging to the plotted fit."""
from pathlib import Path

import pandas as pd
import pytest

import paper_summary_figures as figures


@pytest.fixture
def saved_bootstrap(tmp_path, monkeypatch):
    monkeypatch.setattr(figures, "RESULT_ROOT", tmp_path)
    folder = Path("data/processed/test_bootstrap")
    (tmp_path / folder).mkdir(parents=True)
    fitted = pd.Series(dict(model_version="continuous", event_definition="fixed_threshold",
                            n_source_events=59, n_response_events=58,
                            LR_statistic=8.5767, response_exposure_kyr=392.2457,
                            nominal_LR_p=0.0137))
    summary = fitted.to_dict() | dict(n_bootstrap=99, n_valid_replicates=99,
                                     n_failed_replicates=0,
                                     n_bootstrap_exceeding_or_equal_observed=2,
                                     empirical_p_plus_one=0.03)
    path = tmp_path / folder / "summary.csv"
    pd.DataFrame([summary]).to_csv(path, index=False)
    return folder, fitted, path


def test_paper_uses_bootstrap_instead_of_nominal_p(saved_bootstrap):
    folder, fitted, _ = saved_bootstrap
    assert figures.bootstrap_p_for_fit(folder, fitted) == pytest.approx(0.03)


@pytest.mark.parametrize("field,value", [
    ("event_definition", "variable_threshold"),
    ("n_response_events", 69),
    ("response_exposure_kyr", 396.4643),
    ("LR_statistic", 10.5825),
    ("n_failed_replicates", 1),
    ("empirical_p_plus_one", 0.0137),
])
def test_paper_rejects_wrong_or_incomplete_calibration(saved_bootstrap, field, value):
    folder, fitted, path = saved_bootstrap
    summary = pd.read_csv(path)
    summary.loc[0, field] = value
    summary.to_csv(path, index=False)
    with pytest.raises(ValueError):
        figures.bootstrap_p_for_fit(folder, fitted)
