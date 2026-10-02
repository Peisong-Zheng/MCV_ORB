#!/usr/bin/env python3
"""History utility and full-model fit checks; reuse Barker S4 sampling refits."""

import argparse
import os
from pathlib import Path
import sys

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from toolbox import combined_likelihood as likelihood
from toolbox import point_process_diagnostics as diagnostics
from toolbox.project_config import PROJECT_ROOT, BARKER_EVENT_CSVS


RUN_NAME = "Barker2011_model_diagnostics"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--n-history", type=int, default=4999)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--quadrature-order", type=int, default=4)
    parser.add_argument("--gof-replicates", type=Path)
    parser.add_argument("--gof-only", action="store_true")
    args = parser.parse_args()
    events = pd.read_csv(BARKER_EVENT_CSVS["variable_threshold"], float_precision="round_trip")
    context = likelihood.build_barker_context(events, quadrature_order=args.quadrature_order)
    gof_source = args.gof_replicates or (
        args.output_root / "Barker2011/data/processed/Barker2011_effect_uncertainty/effect_replicates.csv")
    design = likelihood.prepare_catalogue(context.events, context, fixed_support=True)
    full = likelihood.fit_terms(design, context.full_terms)
    sampling, gof_parameters = diagnostics.load_sampling_gof_replicates(
        gof_source, context, full_model=full)
    result = {} if args.gof_only else diagnostics.run_history_test(
        context, n_bootstrap=args.n_history, seed=args.seed, workers=args.workers)
    result.update(diagnostics.run_gof(context, replicates=sampling, full_model=full))
    data_dir = args.output_root / "Barker2011/data/processed" / RUN_NAME
    data_dir.mkdir(parents=True, exist_ok=True)
    settings = dict(model_version=likelihood.MODEL_VERSION, history_tau_kyr=context.history_tau_ka,
                    initial_history=context.initial_history, quadrature_order=context.quadrature_order,
                    event_input_csv=str(BARKER_EVENT_CSVS["variable_threshold"].relative_to(PROJECT_ROOT)))
    if "history_test" in result:
        scales = {f"{forcing}_{field}": context.scaling[forcing][field]
                  for forcing in ("lr04", "co2") for field in ("mean", "range")}
        result["history_test"] = result["history_test"].drop(
            columns=["null_model", "parameter_domain"]).assign(
                history_seed=args.seed, **settings, **scales)
        for name in ("history_test", "history_replicates", "history_coefficients"):
            result[name].to_csv(data_dir / f"{name}.csv", index=False, float_format="%.17g")
        print(result["history_test"].to_string(index=False))
    result["gof_summary"] = result["gof_summary"].drop(columns="calibration").assign(
        **settings, gof_source=gof_parameters["gof_source"],
        gof_seed=gof_parameters["gof_seed"], gof_scenario="B_sampling")
    result["gof_summary"].to_csv(data_dir / "gof_summary.csv", index=False, float_format="%.17g")
    print(result["gof_summary"].to_string(index=False))
    print("Observed residual segments:")
    print(result["residual_segments"].to_string(index=False))
    print(f"Saved {data_dir}")
    if any(not frame.fit_valid.all() for name, frame in result.items() if name.endswith("replicates")):
        raise RuntimeError("Model-check failures are saved; resolve or report them before publication")


if __name__ == "__main__":
    main()
