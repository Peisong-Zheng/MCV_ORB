#!/usr/bin/env python3
"""History utility and full-model fit checks; reuse primary B_sampling for GOF."""

import argparse
import os
from pathlib import Path

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import pandas as pd
from toolbox import combined_likelihood as likelihood
from toolbox import point_process_diagnostics as diagnostics
from toolbox.project_config import PROJECT_ROOT


RUN_NAME = "NGRIP_MIS6_model_diagnostics"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--n-history", type=int, default=4999)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--quadrature-order", type=int, default=4)
    parser.add_argument("--gof-replicates", type=Path)
    parser.add_argument("--gof-only", action="store_true")
    args = parser.parse_args()
    context = likelihood.build_context(quadrature_order=args.quadrature_order)
    result = {} if args.gof_only else diagnostics.run_history_test(
        context, n_bootstrap=args.n_history, seed=args.seed, workers=args.workers)
    gof_source = args.gof_replicates or (args.output_root / "data/processed/NGRIP_MIS6_effect_uncertainty/gof_replicates.csv")
    if gof_source.is_file():
        result.update(diagnostics.run_gof(context, replicates=pd.read_csv(gof_source)))
    elif args.gof_only:
        raise FileNotFoundError(f"Nominal full-model residual replicates are unavailable: {gof_source}")
    else:
        print("Primary GOF remains pending B_sampling; no duplicate full ensemble is simulated.")
    data_dir = diagnostics.save_results(result, context, args.output_root, RUN_NAME,
        dict(history_bootstrap_replicates=args.n_history, history_seed=args.seed,
             gof_source=str(gof_source), gof_source_role="nominal B_sampling full-model refits only"))
    print(f"Saved {data_dir}")
    if any(not frame.fit_valid.all() for name, frame in result.items() if name.endswith("replicates")):
        raise RuntimeError("Model-check failures are saved; resolve or report them before publication")


if __name__ == "__main__":
    main()
