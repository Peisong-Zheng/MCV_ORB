#!/usr/bin/env python3
"""Extra random event-deletion sensitivity for Barker varying-threshold events."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from toolbox import combined_likelihood as likelihood
from toolbox import event_detection_sensitivity as detection
from toolbox.project_config import PROJECT_ROOT


RUN_NAME = "Barker2011_event_detection_sensitivity"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--n-replicates", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--quadrature-order", type=int, default=4)
    args = parser.parse_args()
    context = likelihood.build_barker_context(quadrature_order=args.quadrature_order)
    result = detection.run_catalogue_analysis(context, scopes={"all": None},
        n_replicates=args.n_replicates, seed=args.seed)
    data_dir = detection.save_results(result, context, args.output_root / "Barker2011", RUN_NAME,
                                      diagnostics_root=args.output_root / "tests/diagnostics")
    print(f"Saved {data_dir}; valid fits {result['replicates'].fit_valid.sum()}/{len(result['replicates'])}.")
    if not result["replicates"].fit_valid.all():
        raise RuntimeError("Deletion failures are saved; resolve or report them before publication")


if __name__ == "__main__":
    main()
