#!/usr/bin/env python3
"""Compare chronology, sampling and combined phase-effect uncertainty.

Chronology uses all saved age fits. Sampling simulates 5,000 full-model
catalogues at nominal ages. Combined uses 200 chronologies with 50 refits each.
All displayed ranges project coefficient ellipses constructed in the same way;
only nominal-age sampling gives an approximate confidence region.
Both ensembles condition on observed initial events. The reduced-model null
bootstrap remains a separate experiment and is not rerun by this script.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import tempfile
import time

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mcv_orb_matplotlib"))
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy

from toolbox.project_config import PROJECT_ROOT, CO2_CSV, LR04_CSV, ORBITAL_CSV, PRECESSION_PHASE_CSV
from toolbox.project_config import generated_notes_dir


from toolbox import event_model, sampling
from toolbox.point_process import fit_point_process
from toolbox.plotting import plot_effect_uncertainty
from toolbox.model_stats import SCENARIOS, INTERVAL_TYPES, summarize_effects, build_curve_table
from toolbox.project_config import MODEL_VERSION, EVENT_CATALOGUE_CSV, OBSERVATION_SEGMENTS_CSV

RUN_NAME = "NGRIP_MIS6_effect_uncertainty"
AGE_DIR = PROJECT_ROOT / "data/processed/NGRIP_MIS6_event_uncertainty_sensitivity"
AGE_DRAWS = AGE_DIR / "combined_event_age_realizations.csv"
AGE_RESULTS = AGE_DIR / "gain_realizations.csv"
RANDOM_SEED = 20260908
N_POINT_DRAWS = 5_000
N_OUTER_DRAWS = 200
N_INNER_DRAWS = 50
N_WORKERS = 3
OUTPUT_ROOT = PROJECT_ROOT
REDRAW = False
EXPORT_PAPER = True
HISTORY_TERM = "same_type_exponential_history"


def load_age_inputs(events, age_results_path=AGE_RESULTS):
    """Read the paired ensemble, retaining the original support-failure audit."""
    draws = pd.read_csv(AGE_DRAWS, float_precision="round_trip")
    results = pd.read_csv(age_results_path)
    ids = ["realization_id", "ngrip_realization_id", "mis6_realization_id"]
    if draws.realization_id.duplicated().any() or results.realization_id.duplicated().any():
        raise ValueError("Age input IDs must be unique")
    if not np.array_equal(draws[ids].to_numpy(), results[ids].to_numpy()):
        raise ValueError("Age realizations and fits must have the same paired IDs")
    if not results.fit_valid.isin([True, False]).all():
        raise ValueError("Every saved age realization needs an explicit support status")
    columns = [f"age_kyr_bp__{event_id}" for event_id in events.event_id]
    ages = draws[columns].to_numpy(float)
    if not np.isfinite(ages).all() or not np.all(np.diff(ages, axis=1) > 0):
        raise ValueError("Saved age sequences must be finite and ordered")
    valid = results.fit_valid.eq(True)
    if results.loc[~valid, "invalid_reason"].fillna("").eq("").any():
        raise ValueError("Outside-support age rows must retain their reason")
    metrics = ["pre_phase_preferred_deg", "pre_phase_rate_ratio_max_vs_min"]
    if not np.isfinite(results.loc[valid, metrics].to_numpy(float)).all():
        raise ValueError("Valid age fits need finite effect estimates")
    return draws, results


def save_figure(summary, curves, regions, figure_dir, export=False, *, compact_ratio_ticks=False):
    """Write the common effect figure in this study's output folder."""
    from paper_figure_export import copy_pdf_to_paper
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig = plot_effect_uncertainty(summary, curves, regions, compact_ratio_ticks=compact_ratio_ticks)
    for suffix in ("png", "pdf"):
        fig.savefig(figure_dir / f"{RUN_NAME}.{suffix}", dpi=450)
    plt.close(fig)
    if export:
        copy_pdf_to_paper(figure_dir / f"{RUN_NAME}.pdf")


def save_provenance(output_dir, age_results, windows, selected, elapsed):
    parameters = dict(n_point=N_POINT_DRAWS, n_outer=N_OUTER_DRAWS,
        n_inner=N_INNER_DRAWS, seed=RANDOM_SEED, workers=N_WORKERS,
        output_root=OUTPUT_ROOT, age_results=AGE_RESULTS,
        no_paper_export=not EXPORT_PAPER, redraw=REDRAW)
    parameters.update(model_version=MODEL_VERSION,
        age_reference="BP1950", history_tau_ka=1.5,
        quadrature_order=4, history_coefficient_domain="beta_H <= 0",
        response_exposure_kyr=float((windows.response_end_kyr_bp-windows.response_start_kyr_bp).sum()),
        C_response_exposure_min_kyr=selected.response_exposure_kyr.min(),
        C_response_exposure_max_kyr=selected.response_exposure_kyr.max(),
        age_rows=len(age_results), age_valid=int(age_results.fit_valid.sum()),
        elapsed_seconds=elapsed,
        generator="fitted continuous full model; event-driven exponential history",
        initialization="exact oldest event fixed; earlier history zero; independent segments",
        C_support="selected chronology determines its own anchor; younger endpoints remain fixed",
        B_interval="bootstrap-error-calibrated 2D ellipse; approximate joint confidence projection",
        A_interval="95% coefficient ellipse projection; chronology working region",
        C_interval="95% coefficient ellipse projection; combined working region, not calibrated confidence",
        C_outer="uniform valid chronologies without replacement; equal inner weights",
        failures="retain each original replicate; withhold intervals if failures remain",
        python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__)
    pd.DataFrame([dict(parameter=k, value=v) for k,v in parameters.items()]).to_csv(
        output_dir / "parameters_and_provenance.csv", index=False)


def save_effect_outputs(summary, curves, regions, output_dir):
    """Update interval products while leaving simulated sequences and their provenance intact."""
    summary.to_csv(output_dir / "effect_summary.csv", index=False, float_format="%.12g")
    curves.to_csv(output_dir / "phase_response_bands.csv", index=False, float_format="%.12g")
    encoded = {}
    for scenario, region in regions.items():
        encoded[SCENARIOS[scenario]] = {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in region.items() if key != "bootstrap_error_quadratic"}
    (output_dir / "coefficient_regions.json").write_text(json.dumps(encoded, indent=2) + "\n")
    # Retain the existing sampling-only audit product for earlier diagnostic consumers.
    (output_dir / "sampling_joint_region.json").write_text(json.dumps(encoded["sampling"], indent=2) + "\n")
    parameters_path = output_dir / "parameters_and_provenance.csv"
    parameters = pd.read_csv(parameters_path).set_index("parameter")
    for scenario, interval in INTERVAL_TYPES.items():
        parameters.loc[f"{scenario[0]}_interval", "value"] = interval
    parameters.to_csv(parameters_path)


def effect_caption(catalogue):
    return (
        f"Uncertainty in the {catalogue} phase effect. "
        "Chronology (gray dotted), sampling (blue solid), and sampling + chronology (orange dashed) "
        "use the same coefficient-ellipse construction. (a,b) Preferred-phase and maximum/minimum "
        "rate-ratio bounds projected from the 95% regions in (c); dots and dashed vertical lines "
        "mark nominal estimates, and phase angles continue across 360 degrees. (c) Coefficient "
        "regions, nominal estimate (black dot), and no phase effect (cross). (d) Curve envelopes "
        "from the complete regions and the nominal phase multiplier (black); blue shading marks "
        "the sampling envelope. Minimum/Maximum refer to the precession index; unity means no "
        "phase contribution. Sampling gives an approximate conditional confidence region; "
        "chronology and combined regions summarize sensitivity to the assumed age errors.\n"
    )


def main():
    output_dir = OUTPUT_ROOT / "data/processed" / RUN_NAME
    figure_dir = OUTPUT_ROOT / "figures" / RUN_NAME
    if REDRAW:
        # Redraw the saved experiment; current upstream files need not recreate it.
        summary = pd.read_csv(output_dir / "effect_summary.csv")
        curves = pd.read_csv(output_dir / "phase_response_bands.csv")
        saved_regions = json.loads((output_dir / "coefficient_regions.json").read_text())
        regions = {scenario: saved_regions[name] for scenario, name in SCENARIOS.items()}
        save_figure(summary, curves, regions, figure_dir, EXPORT_PAPER)
        print(summary.to_string(index=False), flush=True)
        return
    notes_dir = generated_notes_dir(OUTPUT_ROOT)
    for directory in (output_dir, figure_dir, notes_dir):
        directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    quadrature_order = 4
    events = pd.read_csv(EVENT_CATALOGUE_CSV)
    observations = pd.read_csv(OBSERVATION_SEGMENTS_CSV)
    catalogue_id = "ngrip_warming_plus_mis6"
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
    event_x, integral_x = event_model.build_design(events, windows, forcings, phase_anchors, scaling,
                                                  quadrature_order=quadrature_order)
    for frame in (event_x, integral_x):
        frame["mis6_segment"] = frame.segment_id.eq("MIS6").astype(float)
    reduced_terms = ("intercept", HISTORY_TERM, "lr04_scaled", "co2_scaled", "mis6_segment")
    full_terms = reduced_terms + ("pre_phase_sin", "pre_phase_cos")
    reduced = fit_point_process(event_x[list(reduced_terms)], integral_x[list(reduced_terms)],
        integral_x.weight, reduced_terms, nonpositive_terms=(HISTORY_TERM,))
    full = fit_point_process(event_x[list(full_terms)], integral_x[list(full_terms)],
        integral_x.weight, full_terms, nonpositive_terms=(HISTORY_TERM,), start_beta=np.r_[reduced.beta, 0., 0.])
    sampling.validate_effect_fit(reduced, full)
    draws, age_results = load_age_inputs(events, AGE_RESULTS)
    age_columns = [f"age_kyr_bp__{event_id}" for event_id in events.event_id]
    generators, selected = sampling.effect_generators(events, observations, windows, forcings,
        phase_anchors, scaling, full, draws, age_results, age_columns,
        reduced_terms=reduced_terms, full_terms=full_terms, n_outer=N_OUTER_DRAWS, seed=RANDOM_SEED,
        source_id_columns=("ngrip_realization_id", "mis6_realization_id"))
    selected.to_csv(output_dir / "selected_age_generators.csv", index=False)
    pd.DataFrame([{f"beta__{term}": beta for term, beta in zip(full.terms, full.beta)}]).to_csv(
        output_dir / "point_generator.csv", index=False)
    age_results[["realization_id", "fit_valid", "invalid_reason"]].to_csv(
        output_dir / "age_support_status.csv", index=False)
    replicates = sampling.sample_effect(generators, forcings, phase_anchors, scaling,
        reduced_terms=reduced_terms, full_terms=full_terms, n_point=N_POINT_DRAWS,
        n_inner=N_INNER_DRAWS, seed=RANDOM_SEED, workers=N_WORKERS)
    replicates.to_csv(output_dir / "effect_replicates.csv", index=False, float_format="%.12g")
    # Reuse the nominal full-model refits for model diagnostics.
    gof_columns = ["replicate_id", "scenario", "fit_valid", "invalid_reason",
                   "n_response_events", "ks_uniform", "adjacent_dependence", "residual_status"]
    replicates.loc[replicates.scenario.eq("B_sampling"), gof_columns].to_csv(
        output_dir / "gof_replicates.csv", index=False, float_format="%.12g")
    save_provenance(output_dir, age_results, windows, selected, time.perf_counter()-started)
    summary, regions = summarize_effects(full, age_results, replicates)
    curves = build_curve_table(full, regions)
    save_effect_outputs(summary, curves, regions, output_dir)
    calibration = replicates.loc[replicates.scenario.eq("B_sampling"), ["inner_id"]].copy()
    calibration["error_quadratic"] = regions["B_sampling"]["bootstrap_error_quadratic"]
    calibration.to_csv(output_dir / "sampling_region_calibration.csv", index=False)
    save_figure(summary, curves, regions, figure_dir, EXPORT_PAPER)
    text = ("Effect uncertainty\n\n"
        "Chronology uses every valid fit in the unchanged age ensemble. Sampling simulates the fitted continuous full model "
        "at nominal ages, holding each oldest event and younger endpoint fixed. Each catalogue is "
        "refitted with the same exact-event likelihood and exponential history (tau=1.5 kyr; beta_H<=0). "
        "Sampling + chronology repeats full-model simulation under selected age realizations, with equal weight per chronology.\n\n"
        "All three sources use coefficient ellipses centered on the nominal fit. Each sample covariance "
        "sets the ellipse shape, and the 95th percentile of squared standardized distances from the "
        "nominal fit sets its size. Projecting each whole ellipse gives phase/rate-ratio bounds "
        "and a curve envelope. Sampling retains its approximate conditional confidence interpretation; "
        "chronology and combined regions are working uncertainty regions, not calibrated confidence intervals. "
        "If a region contains the origin, preferred phase is unidentified.\n\n" + summary.to_string(index=False) + "\n")
    (notes_dir / f"{RUN_NAME}_Methods_and_results.txt").write_text(text)
    (notes_dir / f"{RUN_NAME}_Caption.txt").write_text(effect_caption("NGRIP–MIS6"))
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
