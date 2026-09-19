#!/usr/bin/env python3
"""Refit the saved joint age ensemble with the continuous event likelihood."""
from pathlib import Path
import argparse
import json
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from toolbox import combined_likelihood, age_sensitivity
from toolbox.age_sensitivity_plotting import plot_sensitivity as draw_sensitivity
from toolbox.project_config import PROJECT_ROOT
from paper_figure_export import copy_pdf_to_paper

RUN_NAME='NGRIP_MIS6_event_uncertainty_sensitivity'
OUT_DATA_DIR=PROJECT_ROOT/'data/processed'/RUN_NAME
OUT_FIG_DIR=PROJECT_ROOT/'figures'/RUN_NAME
NGRIP_MC_INPUT=PROJECT_ROOT/'NGRIP/data/processed/ngrip_event_age_uncertainty/ngrip_event_age_realizations.csv'
MIS6_MC_INPUT=PROJECT_ROOT/'MIS6/data/processed/MIS6_event_age_uncertainty/mis6_event_age_realizations.csv'
COMBINED_AGE_OUTPUT=OUT_DATA_DIR/'combined_event_age_realizations.csv'
N_REALIZATIONS=10000
PAIRING_SEED=20260906
HISTORY_TAU_KYR=1.5
P_THRESHOLD=.05

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
    if len(ngrip_columns) != 34 or len(mis6_columns) != 21:
        raise ValueError("Expected 34 NGRIP and 21 MIS 6 uncertainty columns")
    return {"NGRIP": ngrip_columns, "MIS6": mis6_columns}

def _validate_source_ensemble(
    table: pd.DataFrame,
    columns: list[str],
    source: str,
) -> np.ndarray:
    required = {"realization_id", *columns}
    if missing := required.difference(table.columns):
        raise ValueError(f"{source} age ensemble is missing columns: {sorted(missing)}")
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
    if not np.all(np.diff(paired_ages, axis=1) > 0.0):
        raise RuntimeError("Joint event rank changed during source pairing")

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

def load_joint_realizations(
    events: pd.DataFrame,
    *,
    n_realizations: int = N_REALIZATIONS,
    seed: int = PAIRING_SEED,
    ngrip_path: Path = NGRIP_MC_INPUT,
    mis6_path: Path = MIS6_MC_INPUT,
) -> pd.DataFrame:
    """Load the existing combined-error ensembles and form fixed random pairs."""

    if not ngrip_path.exists():
        raise FileNotFoundError(f"Missing NGRIP age ensemble: {ngrip_path}")
    if not mis6_path.exists():
        raise FileNotFoundError(f"Missing MIS 6 age ensemble: {mis6_path}")
    return pair_source_ensembles(
        events,
        pd.read_csv(ngrip_path),
        pd.read_csv(mis6_path),
        n_realizations=n_realizations,
        seed=seed,
    )


def fit_realizations(events,draws,context,show_progress=False,n_workers=1):
    return age_sensitivity.fit_realizations(context,draws,combined_age_columns(events),n_workers,show_progress)


def compact_gain_results(results):
    return age_sensitivity.compact_results(results)


def build_summary(results,point_fit):
    return age_sensitivity.summarize(results,point_fit.summary)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',type=Path,default=PROJECT_ROOT)
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--n-realizations',type=int,default=N_REALIZATIONS)
    parser.add_argument('--no-paper-export',action='store_true')
    parser.add_argument('--redraw',action='store_true')
    args=parser.parse_args()
    context=combined_likelihood.build_context()
    point=combined_likelihood.fit_catalogue(context.events,context)
    data=args.output_root/'data/processed'/RUN_NAME
    figures=args.output_root/'figures'/RUN_NAME
    notes=args.output_root/'experiment_note'
    for directory in (data,figures,notes): directory.mkdir(parents=True,exist_ok=True)
    if args.redraw:
        results=pd.read_csv(data/'gain_realizations.csv')
    else:
        # The frozen pairings are analysis inputs; do not resample or rewrite them.
        draws=pd.read_csv(COMBINED_AGE_OUTPUT,float_precision='round_trip').iloc[:args.n_realizations].copy()
        results,diagnostics=fit_realizations(context.events,draws,context,True,args.workers)
        compact_gain_results(results).to_csv(data/'gain_realizations.csv',index=False)
        pd.DataFrame([diagnostics]).to_csv(data/'fitting_diagnostics.csv',index=False)
        if data.resolve()!=OUT_DATA_DIR.resolve(): shutil.copy2(COMBINED_AGE_OUTPUT,data/COMBINED_AGE_OUTPUT.name)
        if diagnostics['n_numerical_failures']: raise RuntimeError('Unresolved numerical age fits; inspect diagnostics')
    summary=build_summary(results,point)
    summary.to_csv(data/'summary.csv',index=False)
    pd.DataFrame([dict(parameter=k,value=v) for k,v in dict(model_version=combined_likelihood.MODEL_VERSION,
                    history_tau_kyr=HISTORY_TAU_KYR,history_coefficient_domain='nonpositive',
                    age_input='saved combined_event_age_realizations.csv',n_realizations=len(results)).items()]).to_csv(data/'parameters_and_provenance.csv',index=False)
    fig=draw_sensitivity(results,point)
    for ext in ('pdf','png'): fig.savefig(figures/f'{RUN_NAME}.{ext}',dpi=450)
    plt.close(fig)
    if not args.no_paper_export: copy_pdf_to_paper(figures/f'{RUN_NAME}.pdf')
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
