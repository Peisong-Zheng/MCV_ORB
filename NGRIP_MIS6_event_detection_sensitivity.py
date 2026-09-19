#!/usr/bin/env python3
"""Extra random event-deletion sensitivity for NGRIP--MIS6; tables and notes."""

import argparse
from pathlib import Path

from toolbox import combined_likelihood as likelihood
from toolbox import event_detection_sensitivity as detection
from toolbox.project_config import PROJECT_ROOT


RUN_NAME = "NGRIP_MIS6_event_detection_sensitivity"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--n-replicates", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--quadrature-order", type=int, default=4)
    args = parser.parse_args()
    context = likelihood.build_context(quadrature_order=args.quadrature_order)
    result = detection.run_catalogue_analysis(context,
        scopes={"both": None, "MIS6_only": ("MIS6",)},
        n_replicates=args.n_replicates, seed=args.seed)
    data_dir = detection.save_results(result, context, args.output_root, RUN_NAME)
    print(f"Saved {data_dir}; valid fits {result['replicates'].fit_valid.sum()}/{len(result['replicates'])}.")
    if not result["replicates"].fit_valid.all():
        raise RuntimeError("Deletion failures are saved; resolve or report them before publication")


if __name__ == "__main__":
    main()
