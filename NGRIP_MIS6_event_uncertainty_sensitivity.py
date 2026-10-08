#!/usr/bin/env python3
"""Refit the saved joint age ensemble with the continuous event likelihood."""
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from toolbox import event_model, model_stats, age_sensitivity
from toolbox.point_process import fit_point_process
from toolbox.project_config import (
    PROJECT_ROOT, EVENT_CATALOGUE_CSV, OBSERVATION_SEGMENTS_CSV, MODEL_VERSION,
    LR04_CSV, CO2_CSV, ORBITAL_CSV, PRECESSION_PHASE_CSV,
)
from toolbox.plotting import plot_sensitivity as draw_sensitivity
from paper_figure_export import copy_pdf_to_paper
from toolbox.project_config import generated_notes_dir

# Adjust run parameters and paths here before running the script.
RUN_NAME='NGRIP_MIS6_event_uncertainty_sensitivity'
OUT_DATA_DIR=PROJECT_ROOT/'data/processed'/RUN_NAME
OUT_FIG_DIR=PROJECT_ROOT/'figures'/RUN_NAME
NGRIP_MC_INPUT=PROJECT_ROOT/'NGRIP/data/processed/ngrip_event_age_uncertainty/ngrip_event_age_realizations.csv'
MIS6_MC_INPUT=PROJECT_ROOT/'MIS6/data/processed/MIS6_event_age_uncertainty/mis6_event_age_realizations.csv'
COMBINED_AGE_INPUT = PROJECT_ROOT / 'data/processed' / RUN_NAME / 'combined_event_age_realizations.csv'
REDRAW_INPUT_DIR = PROJECT_ROOT / 'data/processed' / RUN_NAME
NOTE_DIR = generated_notes_dir(PROJECT_ROOT)
N_WORKERS = 1
REDRAW = False
EXPORT_PAPER = True
N_REALIZATIONS=10000
PAIRING_SEED=20260906
HISTORY_TAU_KYR=1.5
P_THRESHOLD=.05
HISTORY_TERM = "same_type_exponential_history"
REDUCED_TERMS = ["intercept", HISTORY_TERM, "lr04_scaled", "co2_scaled", "mis6_segment"]
FULL_TERMS = REDUCED_TERMS + ["pre_phase_sin", "pre_phase_cos"]
CATALOGUE_ID = "ngrip_warming_plus_mis6"


def combined_age_columns(events: pd.DataFrame) -> list[str]:
    """Name one wide-table age column per stable curated event ID."""

    return [f"age_kyr_bp__{event_id}" for event_id in events["event_id"]]

def source_age_columns(events: pd.DataFrame) -> dict[str, list[str]]:
    """Map curated event IDs to their columns in the two source ensembles."""

    ngrip = events.loc[events["segment_id"].eq("NGRIP")]
    mis6 = events.loc[events["segment_id"].eq("MIS6")]

    # NGRIP uncertainty columns use the simple GI label, not Table 2 suffixes
    # such as GI-1e that are retained only in source_event_label.
    ngrip_columns = [f"age_ka_bp__{label}" for label in ngrip["event_label"]]
    mis6_columns = [
        f"{event_id.removeprefix('MIS6:')}_age_ka_bp"
        for event_id in mis6["event_id"]
    ]
    return {"NGRIP": ngrip_columns, "MIS6": mis6_columns}

def _validate_source_ensemble(
    table: pd.DataFrame,
    columns: list[str],
    source: str,
) -> np.ndarray:
    if (
        table["realization_id"].isna().any()
        or table["realization_id"].duplicated().any()
    ):
        raise ValueError(f"{source} realization IDs must be complete and unique")

    ages = table.loc[:, columns].to_numpy(float)
    if not np.isfinite(ages).all():
        raise ValueError(f"{source} age ensemble contains non-finite values")
    if not np.all(np.diff(ages, axis=1) > 0.0):
        raise ValueError(
            f"{source} realizations must preserve event rank; ages are never sorted"
        )
    return ages

def pair_source_ensembles(
    events: pd.DataFrame,
    ngrip_table: pd.DataFrame,
    mis6_table: pd.DataFrame,
    *,
    n_realizations: int,
    seed: int,
) -> pd.DataFrame:
    """Pair independent source rows and return 55 ordered ages per realization."""

    if n_realizations < 1:
        raise ValueError("n_realizations must be positive")
    if n_realizations > min(len(ngrip_table), len(mis6_table)):
        raise ValueError("Requested more pairs than available source realizations")

    columns = source_age_columns(events)
    ngrip_ages = _validate_source_ensemble(ngrip_table, columns["NGRIP"], "NGRIP")
    mis6_ages = _validate_source_ensemble(mis6_table, columns["MIS6"], "MIS 6")

    # Independent permutations avoid imposing row-wise correspondence between
    # separately generated chronology ensembles.  With 10,000 source rows,
    # every row enters the joint experiment exactly once.
    rng = np.random.default_rng(seed)
    ngrip_rows = rng.permutation(len(ngrip_table))[:n_realizations]
    mis6_rows = rng.permutation(len(mis6_table))[:n_realizations]
    paired_ngrip = ngrip_ages[ngrip_rows]
    paired_mis6 = mis6_ages[mis6_rows]

    if not np.all(paired_ngrip[:, -1] < paired_mis6[:, 0]):
        raise ValueError("At least one paired realization interleaves the two segments")

    paired_ages = np.column_stack((paired_ngrip, paired_mis6))

    draws = pd.DataFrame(paired_ages, columns=combined_age_columns(events))
    draws.insert(
        0,
        "mis6_realization_id",
        mis6_table["realization_id"].to_numpy()[mis6_rows],
    )
    draws.insert(
        0,
        "ngrip_realization_id",
        ngrip_table["realization_id"].to_numpy()[ngrip_rows],
    )
    draws.insert(
        0,
        "realization_id",
        [f"joint_{index:05d}" for index in range(1, n_realizations + 1)],
    )
    return draws

def main():
    events = pd.read_csv(EVENT_CATALOGUE_CSV)
    observations = pd.read_csv(OBSERVATION_SEGMENTS_CSV)
    if (events.groupby("segment_id", sort=False).size().to_dict() != {"NGRIP": 34, "MIS6": 21}
            or events.event_id.isna().any() or not events.event_id.is_unique):
        raise ValueError("Check the curated 34 NGRIP and 21 MIS6 event identities")
    lr04 = pd.read_csv(LR04_CSV, float_precision="round_trip")
    co2 = pd.read_csv(CO2_CSV, float_precision="round_trip")
    orbital = pd.read_csv(ORBITAL_CSV, float_precision="round_trip")
    anchors = pd.read_csv(PRECESSION_PHASE_CSV, float_precision="round_trip")
    forcings = {"lr04": (lr04.age_kyr_bp.to_numpy(), lr04.lr04.to_numpy()),
                "co2": (co2.age_kyr_bp.to_numpy(), co2.co2_ppm.to_numpy()),
                "precession_index": (orbital.age_kyr_bp.to_numpy(), orbital.precession_index.to_numpy())}
    phase_anchors = (anchors.age_kyr_bp.to_numpy(), anchors.phase_unwrapped_rad.to_numpy())
    windows = event_model.response_windows(events, observations)
    scaling = event_model.nominal_scaling({name: forcings[name] for name in ("lr04", "co2")}, windows)
    event_x, integral_x = event_model.build_design(
        events, windows, forcings, phase_anchors, scaling, tau=HISTORY_TAU_KYR,
    )
    for frame in (event_x, integral_x):
        frame["mis6_segment"] = frame.segment_id.eq("MIS6").astype(float)
    reduced = fit_point_process(event_x[REDUCED_TERMS], integral_x[REDUCED_TERMS],
                                integral_x.weight, REDUCED_TERMS, nonpositive_terms=(HISTORY_TERM,))
    full = fit_point_process(event_x[FULL_TERMS], integral_x[FULL_TERMS],
                             integral_x.weight, FULL_TERMS, nonpositive_terms=(HISTORY_TERM,),
                             start_beta=np.r_[reduced.beta, 0., 0.])
    point = model_stats.fit_summary(reduced, full, event_x, windows, n_source_events=len(events),
                                    catalogue_id=CATALOGUE_ID, tau=HISTORY_TAU_KYR)
    data = OUT_DATA_DIR
    figures = OUT_FIG_DIR
    notes = NOTE_DIR
    for directory in (data,figures,notes): directory.mkdir(parents=True,exist_ok=True)
    if REDRAW:
        results=pd.read_csv(REDRAW_INPUT_DIR / 'gain_realizations.csv')
    else:
        # The frozen pairings are analysis inputs; do not resample or rewrite them.
        draws=pd.read_csv(COMBINED_AGE_INPUT,float_precision='round_trip').iloc[:N_REALIZATIONS].copy()
        results,diagnostics=age_sensitivity.fit_realizations(
            events, observations, forcings, phase_anchors, scaling, REDUCED_TERMS, FULL_TERMS,
            draws, [f"age_kyr_bp__{event_id}" for event_id in events.event_id],
            catalogue_id=CATALOGUE_ID, tau=HISTORY_TAU_KYR,
            show_progress=True, n_workers=N_WORKERS)
        age_sensitivity.compact_results(results).to_csv(data/'gain_realizations.csv',index=False)
        pd.DataFrame([diagnostics]).to_csv(data/'fitting_diagnostics.csv',index=False)
        if data.resolve() != COMBINED_AGE_INPUT.parent.resolve():
            shutil.copy2(COMBINED_AGE_INPUT, data / COMBINED_AGE_INPUT.name)
        if diagnostics['n_numerical_failures']: raise RuntimeError('Unresolved numerical age fits; inspect diagnostics')
    summary=age_sensitivity.summarize(results,point)
    summary.to_csv(data/'summary.csv',index=False)
    pd.DataFrame([dict(parameter=k,value=v) for k,v in dict(model_version=MODEL_VERSION,
                    history_tau_kyr=HISTORY_TAU_KYR,history_coefficient_domain='nonpositive',
                    age_input='saved combined_event_age_realizations.csv',n_realizations=len(results)).items()]).to_csv(data/'parameters_and_provenance.csv',index=False)
    fig=draw_sensitivity(results,point)
    for ext in ('pdf','png'): fig.savefig(figures/f'{RUN_NAME}.{ext}',dpi=450)
    plt.close(fig)
    if EXPORT_PAPER: copy_pdf_to_paper(figures/f'{RUN_NAME}.pdf')
    row=summary.iloc[0]
    text=(f"Continuous conditional likelihood was refitted to {len(results):,} saved joint chronological realizations. "
          f"Each segment conditions on its exact oldest event; history decays with tau = 1.5 kyr and its coefficient is nonpositive. "
          f"Nominal climate scaling and event identities were retained. {row.n_valid:g} realizations were within observation support. "
          f"G had a 2.5–97.5% range of {row.gain_bits_per_event_q025:.4f}–{row.gain_bits_per_event_q975:.4f} bits/event. "
          f"Nominal LR p < 0.05 occurred in {row.n_nominal_p_below_0p05:g}/{row.n_valid:g} valid realizations. "
          "This proportion measures chronology sensitivity; it is not an empirical significance probability. "
          "Event membership is fixed here; additional missed-event sensitivity is a separate experiment.\n")
    (notes/f'{RUN_NAME}_Methods_and_results.txt').write_text(text)
    (notes/f'{RUN_NAME}_Caption.txt').write_text('Chronological sensitivity of the continuous event model. (a) G; (b) nominal LR p; (c) preferred phase; (d) phase maximum/minimum rate ratio; (e) AIC(full) − AIC(reduced). Black lines mark point-age fits, dashed lines MC medians, and dotted lines decision references. Phase ranges are circularly unwrapped about the point estimate.\n')
    print(summary.to_string(index=False))

if __name__=='__main__': main()
