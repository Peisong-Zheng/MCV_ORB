#!/usr/bin/env python3
"""Effect precision for Barker's varying-threshold SpeleoAge events.

Chronology reuses all saved age fits. Sampling simulates 5,000 catalogues at nominal ages.
Combined selects 200 age realizations and simulates 50 catalogues per fitted model.
The simulation, interval construction and figure template match NGRIP–MIS6.
"""

import argparse
import hashlib
import os
from pathlib import Path
import sys
import time

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

import NGRIP_MIS6_effect_uncertainty as shared
from toolbox import combined_likelihood as likelihood
from toolbox import effect_uncertainty as effect
from toolbox.project_config import PROJECT_ROOT, CO2_XLSX, LR04_XLSX, PRE_TXT


RUN_NAME = "Barker2011_effect_uncertainty"
BARKER_ROOT = PROJECT_ROOT / "Barker2011"
AGE_DRAWS = BARKER_ROOT / "data/processed/Barker2011_event_age_uncertainty/event_age_realizations.csv"
AGE_RESULTS = BARKER_ROOT / "data/processed/Barker2011_event_uncertainty_sensitivity/gain_realizations.csv"
DEFAULT_SEED = 20260925


def load_age_inputs(events):
    """Match the unchanged age ensemble to its saved continuous-time fits."""
    draws = pd.read_csv(AGE_DRAWS, float_precision="round_trip")
    results = pd.read_csv(AGE_RESULTS)
    if draws.realization_id.duplicated().any() or results.realization_id.duplicated().any():
        raise ValueError("Age realization IDs must be unique")
    pd.testing.assert_series_equal(draws.realization_id, results.realization_id)
    if not results.fit_valid.isin([True, False]).all():
        raise ValueError("Each age realization needs a fit status")
    columns = [f"age_ka_bp__{event_id}" for event_id in events.event_id]
    ages = draws[columns].to_numpy(float)
    if not np.isfinite(ages).all() or not np.all(np.diff(ages, axis=1) > 0):
        raise ValueError("Saved BP ages must be finite and ordered")
    values = results.loc[results.fit_valid, ["pre_phase_preferred_deg",
                                           "pre_phase_rate_ratio_max_vs_min"]]
    if not np.isfinite(values.to_numpy(float)).all():
        raise ValueError("Valid age fits need finite phase-effect estimates")
    return draws, results, columns


def input_hashes():
    paths = [AGE_DRAWS, AGE_RESULTS, CO2_XLSX, LR04_XLSX, PRE_TXT,
             BARKER_ROOT / "data/raw/Barker et al-2011-SOM.xls",
             PROJECT_ROOT / "toolbox/point_process.py", Path(likelihood.__file__),
             Path(effect.__file__), Path(shared.__file__), Path(__file__)]
    return pd.DataFrame([dict(path=str(path.relative_to(PROJECT_ROOT)),
                              sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                         for path in paths])


def write_notes(summary, regions, selected, parameters, notes_dir):
    identified = not regions["B_sampling"]["origin_in_region"]
    note = (
        "Effect precision for Barker varying-threshold events\n\n"
        "Chronology reuses all valid SpeleoAge fits, including their realized conditioning events. "
        "Sampling generates new continuous event sequences from the nominal full model. "
        "Combined samples distinct valid age realizations without replacement, fits a full model to "
        "each, and generates equally many new sequences per model. All full and BG coefficients "
        "are refitted. The oldest conditioning event and the 0-kyr BP endpoint remain fixed "
        "within each generator; response-event totals vary. Exponential history is updated "
        "after every generated event (tau=1.5 kyr, beta_H<=0). Climate scaling remains nominal; "
        "climate and phase are interpolated at simulated ages. Simulated events receive no "
        "additional age perturbation.\n\n"
        "All three sources use the primary catalogue's coefficient-ellipse construction: nominal "
        "center, sample covariance, and size set by the 95th percentile of squared standardized "
        "distances from the nominal fit. Each complete region is projected to phase, rate ratio "
        "and curve bounds. Sampling gives approximate conditional confidence bounds. Chronology "
        "and combined regions summarize sensitivity to the working age-error model; they are "
        "not calibrated confidence intervals.\n\n"
        f"Sampling confidence region excludes zero coefficients: {identified}.\n"
        f"Combined exposure range: {selected.response_exposure_kyr.min():.6f}–"
        f"{selected.response_exposure_kyr.max():.6f} kyr.\n\n"
        + parameters.to_string(index=False) + "\n\n" + summary.to_string(index=False) + "\n"
    )
    (notes_dir / f"{RUN_NAME}_Methods_and_results.txt").write_text(note)
    caption = shared.effect_caption("Barker varying-threshold")
    (notes_dir / f"{RUN_NAME}_Caption.txt").write_text(caption)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-point", type=int, default=5000)
    parser.add_argument("--n-outer", type=int, default=200)
    parser.add_argument("--n-inner", type=int, default=50)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--redraw", action="store_true")
    parser.add_argument("--no-paper-export", action="store_true")
    args = parser.parse_args()
    if args.n_point < 20 or min(args.n_outer, args.n_inner, args.workers) < 1 or args.seed < 0:
        parser.error("Require n-point >= 20, positive counts/workers and a nonnegative seed")
    root = args.output_root / "Barker2011"
    data_dir = root / "data/processed" / RUN_NAME
    figure_dir = root / "figures" / RUN_NAME
    notes_dir = root / "experiment_note"
    for directory in (data_dir, figure_dir, notes_dir):
        directory.mkdir(parents=True, exist_ok=True)

    context = likelihood.build_barker_context("variable_threshold")
    point_fit = likelihood.fit_catalogue(context.events, context)
    effect.validate_effect_fit(point_fit)
    draws, ages, age_columns = load_age_inputs(context.events)
    hashes = input_hashes()
    if args.redraw:
        # Plot-only edits may change the entry scripts; inputs and fitting code must match.
        plotting_scripts = {str(Path(__file__).relative_to(PROJECT_ROOT)),
                            str(Path(shared.__file__).relative_to(PROJECT_ROOT))}
        saved_hashes = pd.read_csv(data_dir / "input_sha256.csv")
        pd.testing.assert_frame_equal(
            saved_hashes.loc[~saved_hashes.path.isin(plotting_scripts)].reset_index(drop=True),
            hashes.loc[~hashes.path.isin(plotting_scripts)].reset_index(drop=True))
        saved = pd.read_csv(data_dir / "point_generator.csv")
        np.testing.assert_allclose(saved.iloc[0].to_numpy(float), point_fit.full.beta,
                                   rtol=1e-9, atol=1e-10)
        replicates = pd.read_csv(data_dir / "effect_replicates.csv")
        selected = pd.read_csv(data_dir / "selected_age_generators.csv")
        parameters = pd.read_csv(data_dir / "parameters_and_provenance.csv")
    else:
        started = time.perf_counter()
        generators, selected = shared.build_generators(
            context.events, context, point_fit, draws, ages, args.n_outer, args.seed,
            age_columns=age_columns, source_id_columns=())
        selected.to_csv(data_dir / "selected_age_generators.csv", index=False)
        pd.DataFrame([dict(zip(point_fit.full.terms, point_fit.full.beta))]).to_csv(
            data_dir / "point_generator.csv", index=False)
        replicates = shared.run_simulations(context, generators, args.n_point, args.n_inner,
                                             args.seed, args.workers)
        # Save every replicate before checking failures; never replace difficult draws.
        replicates.to_csv(data_dir / "effect_replicates.csv", index=False, float_format="%.12g")
        settings = dict(model_version=likelihood.MODEL_VERSION, event_definition="variable_threshold",
                        n_point=args.n_point, n_outer=args.n_outer, n_inner=args.n_inner,
                        seed=args.seed, workers=args.workers, n_age_total=len(ages),
                        n_age_valid=int(ages.fit_valid.sum()), history_tau_kyr=context.history_tau_ka,
                        history_coefficient_domain="beta_H <= 0", age_reference="BP1950",
                        response_exposure_kyr=context.response_exposure_kyr,
                        elapsed_seconds=time.perf_counter() - started,
                        n_failed=int((~replicates.fit_valid).sum()))
        parameters = pd.DataFrame(dict(parameter=list(settings), value=list(settings.values())))
        parameters.to_csv(data_dir / "parameters_and_provenance.csv", index=False)
        hashes.to_csv(data_dir / "input_sha256.csv", index=False)

    summary, regions = shared.summarize_effects(point_fit, ages, replicates)
    curves = shared.build_curve_table(point_fit, regions)
    shared.save_effect_outputs(summary, curves, regions, data_dir)
    parameters = pd.read_csv(data_dir / "parameters_and_provenance.csv")
    export = not args.no_paper_export and args.output_root.resolve() == PROJECT_ROOT.resolve()
    shared.plot_results(summary, curves, regions, figure_dir, export, run_name=RUN_NAME,
                        compact_ratio_ticks=True)
    write_notes(summary, regions, selected, parameters, notes_dir)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
