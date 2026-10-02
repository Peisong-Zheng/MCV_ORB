#!/usr/bin/env python3
"""Continuous-time refits of Barker's saved SpeleoAge realizations."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from toolbox import combined_likelihood as c, age_sensitivity
from toolbox.plotting import plot_sensitivity
from toolbox.project_config import PROJECT_ROOT, BARKER_EVENT_CSVS
from paper_figure_export import copy_pdf_to_paper

ROOT = PROJECT_ROOT / "Barker2011"
RUN_NAME = "Barker2011_event_uncertainty_sensitivity"
OUT_DATA_DIR = ROOT / "data/processed" / RUN_NAME
OUT_FIG_DIR = ROOT / "figures" / RUN_NAME
AGE_INPUT = ROOT / "data/processed/Barker2011_event_age_uncertainty/event_age_realizations.csv"
GAIN_COLUMNS = [
    "realization_id", "fit_valid", "invalid_reason", "n_response_events", "response_exposure_kyr",
    "gain_bits_per_event", "LR_statistic", "nominal_LR_p", "delta_AIC_full_minus_reduced",
    "pre_phase_preferred_deg", "pre_phase_rate_ratio_max_vs_min", "beta_pre_phase_sin", "beta_pre_phase_cos",
]
SUMMARY_COLUMNS = [
    "n_realizations", "n_valid", "n_invalid", "n_numerical_failures", "n_outside_observation_support",
    "n_nominal_p_below_0p05", "fraction_nominal_p_below_0p05", "model_version", "history_tau_kyr",
] + [
    name for metric in ("gain_bits_per_event", "nominal_LR_p", "pre_phase_preferred_deg",
                        "pre_phase_rate_ratio_max_vs_min")
    for name in (f"point_{metric}", f"{metric}_q025", f"{metric}_median", f"{metric}_q975")
]


def fit_realizations(realizations, context, show_progress=False, n_workers=1):
    columns = [f"age_ka_bp__{event_id}" for event_id in context.events.event_id]
    return age_sensitivity.fit_realizations(context, realizations, columns, n_workers, show_progress)[0]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--n-realizations", type=int, default=10000)
    parser.add_argument("--no-paper-export", action="store_true")
    parser.add_argument("--redraw", action="store_true")
    args = parser.parse_args(argv)
    if args.n_realizations < 1:
        raise ValueError("n_realizations must be positive")
    events = pd.read_csv(BARKER_EVENT_CSVS["variable_threshold"], float_precision="round_trip")
    context = c.build_barker_context(events)
    point = c.fit_catalogue(context.events, context)
    root = args.output_root / "Barker2011"
    data = root / "data/processed" / RUN_NAME
    figures = root / "figures" / RUN_NAME
    for directory in (data, figures):
        directory.mkdir(parents=True, exist_ok=True)
    if args.redraw:
        results = pd.read_csv(data / "gain_realizations.csv", float_precision="round_trip")
    else:
        draws = pd.read_csv(AGE_INPUT, float_precision="round_trip").iloc[:args.n_realizations]
        columns = [f"age_ka_bp__{event_id}" for event_id in context.events.event_id]
        results, diagnostics = age_sensitivity.fit_realizations(context, draws, columns, args.workers, True)
        results.reindex(columns=GAIN_COLUMNS).to_csv(data / "gain_realizations.csv", index=False)
        print(pd.Series(diagnostics).to_string())

    # Phase quantiles remain unwrapped about the nominal phase in the shared calculation.
    summary = age_sensitivity.summarize(results, point.summary)
    reasons = results.invalid_reason.fillna("")
    summary["n_numerical_failures"] = int(reasons.str.startswith("numerical_fit").sum())
    summary["n_outside_observation_support"] = int(reasons.str.startswith("outside_").sum())
    summary["history_tau_kyr"] = context.history_tau_ka
    summary = summary[SUMMARY_COLUMNS]
    summary.to_csv(data / "summary.csv", index=False)
    if summary.n_numerical_failures.item():
        raise RuntimeError("Unresolved numerical age fits saved; figure publication withheld")
    fig = plot_sensitivity(results, point)
    for ext in ("pdf", "png"):
        fig.savefig(figures / f"{RUN_NAME}.{ext}", dpi=450)
    plt.close(fig)
    if not args.no_paper_export and args.output_root.resolve() == PROJECT_ROOT.resolve():
        copy_pdf_to_paper(figures / f"{RUN_NAME}.pdf")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
