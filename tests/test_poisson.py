"""Numerical contracts for the shared binned Poisson likelihood."""

import numpy as np

from toolbox import poisson


def test_negative_loglik_gradient_matches_central_difference():
    beta = np.array([-0.8, 0.35, -0.22])
    predictors = np.array(
        [
            [-1.0, 0.2],
            [-0.4, -0.8],
            [0.0, 0.5],
            [0.7, -0.3],
            [1.2, 1.0],
            [0.3, -1.1],
        ]
    )
    event_counts = np.array([0.0, 1.0, 2.0, 0.0, 1.0, 3.0])
    bin_widths = np.array([0.10, 0.20, 0.15, 0.40, 0.25, 0.30])

    diagnostics = poisson.eta_clip_diagnostics(beta, predictors)
    assert diagnostics["n_eta_clipped_low"] == 0
    assert diagnostics["n_eta_clipped_high"] == 0

    step = 1e-6
    finite_difference = np.empty_like(beta)
    for index in range(len(beta)):
        perturbation = np.zeros_like(beta)
        perturbation[index] = step
        upper = poisson.negative_binned_poisson_loglik(
            beta + perturbation, predictors, event_counts, bin_widths
        )
        lower = poisson.negative_binned_poisson_loglik(
            beta - perturbation, predictors, event_counts, bin_widths
        )
        finite_difference[index] = (upper - lower) / (2.0 * step)

    analytic = poisson.negative_binned_poisson_gradient(
        beta, predictors, event_counts, bin_widths
    )
    np.testing.assert_allclose(analytic, finite_difference, rtol=1e-7, atol=5e-9)
