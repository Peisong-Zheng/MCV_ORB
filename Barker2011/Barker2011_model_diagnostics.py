#!/usr/bin/env python3
"""History utility and full-model fit checks for Barker varying-threshold events."""

import argparse
import os
from pathlib import Path
import sys

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from toolbox import combined_likelihood as likelihood
from toolbox import point_process_diagnostics as diagnostics
from toolbox.project_config import PROJECT_ROOT


RUN_NAME = "Barker2011_model_diagnostics"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--n-history", type=int, default=4999)
    parser.add_argument("--n-gof", type=int, default=1999)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--quadrature-order", type=int, default=4)
    parser.add_argument("--gof-only", action="store_true")
    args = parser.parse_args()
    context = likelihood.build_barker_context(quadrature_order=args.quadrature_order)
    result = {} if args.gof_only else diagnostics.run_history_test(
        context, n_bootstrap=args.n_history, seed=args.seed, workers=args.workers)
    result.update(diagnostics.run_gof(context, n_bootstrap=args.n_gof,
                                      seed=args.seed + 1, workers=args.workers))
    data_dir = diagnostics.save_results(result, context, args.output_root / "Barker2011", RUN_NAME,
        dict(history_bootstrap_replicates=args.n_history, history_seed=args.seed,
             gof_bootstrap_replicates=args.n_gof, gof_seed=args.seed + 1,
             gof_source_role="new nominal varying-threshold full-model refits"),
        diagnostics_root=args.output_root / "tests/diagnostics")
    print(f"Saved {data_dir}")
    if any(not frame.fit_valid.all() for name, frame in result.items() if name.endswith("replicates")):
        raise RuntimeError("Model-check failures are saved; resolve or report them before publication")


if __name__ == "__main__":
    main()
