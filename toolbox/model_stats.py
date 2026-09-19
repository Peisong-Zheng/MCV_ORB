"""Likelihood gains for nested continuous-time event models."""
import numpy as np
from scipy.stats import chi2


def information_criteria(log_likelihood, n_params):
    # Quadrature resolution is not a statistical sample size.
    return {"AIC": float(2*n_params-2*log_likelihood)}


def likelihood_gain(loglik_full, loglik_reduced):
    return float(loglik_full-loglik_reduced)


def likelihood_ratio_p_value(ll_gain, df):
    lr=2*float(ll_gain)
    return lr, float(chi2.sf(max(lr,0.),df))


def bits_from_loglik_gain(ll_gain, denominator):
    return float(ll_gain/np.log(2)/denominator) if denominator>0 else np.nan


def nested_likelihood_metrics(*,loglik_full,loglik_reduced,df,n_events,aic_full,aic_reduced):
    gain=likelihood_gain(loglik_full,loglik_reduced)
    lr,p=likelihood_ratio_p_value(gain,df)
    if gain < -1e-7:
        raise RuntimeError("Full-model likelihood is below the nested reference")
    return dict(df=int(df),n_events=int(n_events),loglik_reduced=float(loglik_reduced),
                loglik_full=float(loglik_full),ll_gain_nats=gain,
                gain_bits_per_event=bits_from_loglik_gain(gain,n_events),
                LR_statistic=lr,LR_p_value=p,delta_AIC_full_minus_reduced=float(aic_full-aic_reduced))
