"""The So-57 sensitivity must recover ages by the saved source realization IDs."""
import numpy as np
import pandas as pd
import pytest
import sofular_component_likelihood_sensitivity as analysis


def test_saved_ids_recover_same_ages_after_source_row_and_column_reordering():
    events = pd.read_csv(analysis.EVENT_CATALOGUE_CSV)
    pairs = pd.read_csv(analysis.PAIRINGS_CSV, usecols=analysis.ID_COLUMNS).iloc[[2, 0, 1]].reset_index(drop=True)
    ngrip = pd.read_csv(analysis.NGRIP_MC_INPUT, dtype={"realization_id": str}, float_precision="round_trip")
    mis6 = pd.read_csv(analysis.MIS6_MC_INPUT, dtype={"realization_id": str}, float_precision="round_trip")
    draws = analysis.restore_pairings(events, pairs, ngrip, mis6, 3)
    reordered = analysis.restore_pairings(events, pairs, ngrip.iloc[::-1, ::-1], mis6.iloc[::-1, ::-1], 3)
    pd.testing.assert_frame_equal(draws, reordered, check_exact=True)
    pd.testing.assert_frame_equal(draws[analysis.ID_COLUMNS], pairs)
    for index, pair in pairs.iterrows():
        expected = mis6.set_index("realization_id").loc[str(pair.mis6_realization_id), "MIS6_DO_21_age_ka_bp"]
        assert draws.loc[index, "age_kyr_bp__MIS6:MIS6_DO_21"] == expected
    assert np.all(np.diff(draws.iloc[:, 3:].to_numpy(), axis=1) > 0)
    duplicated = ngrip.copy()
    duplicated.loc[1, "realization_id"] = duplicated.loc[0, "realization_id"]
    with pytest.raises(ValueError, match="source IDs"):
        analysis.restore_pairings(events, pairs, duplicated, mis6, 3)
