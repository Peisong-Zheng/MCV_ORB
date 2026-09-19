#!/usr/bin/env python3
"""Continuous-time refits of Barker's saved SpeleoAge realizations."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from toolbox import combined_likelihood as c, age_sensitivity
from toolbox.age_sensitivity_plotting import plot_sensitivity
from toolbox.project_config import PROJECT_ROOT
from paper_figure_export import copy_pdf_to_paper
ROOT=PROJECT_ROOT/'Barker2011'
RUN_NAME='Barker2011_event_uncertainty_sensitivity'
OUT_DATA_DIR=ROOT/'data/processed'/RUN_NAME
OUT_FIG_DIR=ROOT/'figures'/RUN_NAME
AGE_INPUT=ROOT/'data/processed/Barker2011_event_age_uncertainty/event_age_realizations.csv'

def prepare_context():
    return c.build_barker_context()

def fit_realizations(realizations,context,show_progress=False,n_workers=1):
    columns=[f'age_ka_bp__{event_id}' for event_id in context.events.event_id]
    return age_sensitivity.fit_realizations(context,realizations,columns,n_workers,show_progress)[0]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',type=Path,default=PROJECT_ROOT)
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--n-realizations',type=int,default=10000)
    parser.add_argument('--no-paper-export',action='store_true')
    parser.add_argument('--redraw',action='store_true')
    args=parser.parse_args()
    context=c.build_barker_context();point=c.fit_catalogue(context.events,context)
    root=args.output_root/'Barker2011'
    data=root/'data/processed'/RUN_NAME;figures=root/'figures'/RUN_NAME;notes=root/'experiment_note'
    for directory in (data,figures,notes): directory.mkdir(parents=True,exist_ok=True)
    if args.redraw: results=pd.read_csv(data/'gain_realizations.csv')
    else:
        draws=pd.read_csv(AGE_INPUT,float_precision='round_trip').iloc[:args.n_realizations]
        columns=[f'age_ka_bp__{event_id}' for event_id in context.events.event_id]
        results,diagnostics=age_sensitivity.fit_realizations(context,draws,columns,args.workers,True)
        age_sensitivity.compact_results(results).to_csv(data/'gain_realizations.csv',index=False)
        pd.DataFrame([diagnostics]).to_csv(data/'fitting_diagnostics.csv',index=False)
        if diagnostics['n_numerical_failures']: raise RuntimeError('Unresolved numerical age fits')
    summary=age_sensitivity.summarize(results,point.summary);summary.to_csv(data/'summary.csv',index=False)
    phase=np.linspace(0,360,721);r=np.deg2rad(phase);v=results.loc[results.fit_valid]
    curves=np.exp(v.beta_pre_phase_sin.to_numpy()[:,None]*np.sin(r)+v.beta_pre_phase_cos.to_numpy()[:,None]*np.cos(r))
    quantiles=np.quantile(curves,[.025,.5,.975],axis=0)
    pd.DataFrame(dict(phase_deg=phase,mc_q025=quantiles[0],mc_median=quantiles[1],mc_q975=quantiles[2])).to_csv(data/'phase_response_summary.csv',index=False)
    fig=plot_sensitivity(results,point)
    for ext in ('pdf','png'): fig.savefig(figures/f'{RUN_NAME}.{ext}',dpi=450)
    plt.close(fig)
    if not args.no_paper_export: copy_pdf_to_paper(figures/f'{RUN_NAME}.pdf')
    s=summary.iloc[0]
    text=(f"The continuous conditional models were refitted to {len(results):,} unchanged SpeleoAge realizations for the varying-threshold catalogue. "
          "The exact oldest event initializes each realization; the response ends at 0 kyr BP. "
          "History uses exponential decay (tau = 1.5 kyr) with a nonpositive coefficient. "
          f"Among {s.n_valid:g} valid realizations, G ranged from {s.gain_bits_per_event_q025:.4f} to {s.gain_bits_per_event_q975:.4f} bits/event (2.5–97.5%). "
          f"Nominal LR p < 0.05 in {s.n_nominal_p_below_0p05:g}/{s.n_valid:g} realizations. "
          "These ranges describe chronology sensitivity, not sampling confidence or missed-event uncertainty. "
          "Age-control uncertainties and the upstream ordering rule were unchanged.\n")
    (notes/f'{RUN_NAME}_Methods_and_results.txt').write_text(text)
    (notes/f'{RUN_NAME}_Caption.txt').write_text('Sensitivity to the saved Barker SpeleoAge realizations. (a) G; (b) nominal LR p; (c) preferred phase; (d) phase maximum/minimum rate ratio; (e) AIC(full) − AIC(reduced). Lines mark the point-age estimate, MC median and decision references. The fixed-threshold catalogue has no separate chronology ensemble.\n')
    pd.DataFrame([dict(parameter=k,value=v) for k,v in dict(model_version=c.MODEL_VERSION,history_tau_kyr=1.5,
        history_coefficient_domain='nonpositive',source=str(AGE_INPUT.relative_to(PROJECT_ROOT)),n_realizations=len(results)).items()]).to_csv(data/'parameters_and_provenance.csv',index=False)
    print(summary.to_string(index=False))

if __name__=='__main__':main()
