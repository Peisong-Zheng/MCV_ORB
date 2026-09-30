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
    sampling, gof_parameters = diagnostics.load_sampling_gof_replicates(gof_source, context)
    result = {} if args.gof_only else diagnostics.run_history_test(
        context, n_bootstrap=args.n_history, seed=args.seed, workers=args.workers)
    result.update(diagnostics.run_gof(context, replicates=sampling))
    data_dir = diagnostics.save_results(result, context, args.output_root / "Barker2011", RUN_NAME,
        dict(history_bootstrap_replicates=args.n_history, history_seed=args.seed,
             event_input_csv=str(BARKER_EVENT_CSVS["variable_threshold"].relative_to(PROJECT_ROOT)),
             **gof_parameters),
        diagnostics_root=args.output_root / "tests/diagnostics")
    print(f"Saved {data_dir}")
    if any(not frame.fit_valid.all() for name, frame in result.items() if name.endswith("replicates")):
        raise RuntimeError("Model-check failures are saved; resolve or report them before publication")


if __name__ == "__main__":
    main()
