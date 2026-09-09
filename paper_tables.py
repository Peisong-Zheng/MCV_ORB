"""Export manuscript tables from saved results; run from the project root.

No models are fitted here. CSVs keep source precision and source paths; TeX
rounding is presentation only. Table groups correspond to SI Tables S1--S5.
"""
from pathlib import Path
import hashlib

import pandas as pd

PROJECT = Path(__file__).resolve().parent
OUT = PROJECT / 'monsoon_paper/tables'
PRIMARY = Path('data/processed')
BARKER = Path('Barker2011/data/processed')
INPUTS = {}


def read(path):
    path = Path(path)
    source = PROJECT / path
    INPUTS[str(path)] = hashlib.sha256(source.read_bytes()).hexdigest()
    return pd.read_csv(source)


def save(frame, name):
    frame.to_csv(OUT / f'{name}.csv', index=False)


def tex_table(frame, title, columns=None, formats=None, widths=None):
    """A readable longtable; full numeric precision remains in its CSV."""
    shown = frame.copy() if columns is None else frame[list(columns)].copy()
    if formats:
        for column, precision in formats.items():
            shown[column] = shown[column].map(lambda x: '--' if pd.isna(x) else f'{x:.{precision}f}')
    if columns:
        shown = shown.rename(columns=columns)
    shown = shown.fillna('--')
    # Render display-only Unicode notation in portable LaTeX text.
    shown = shown.map(lambda x: x.replace('₂', '2').replace('²', ' squared').replace('×', ' x ') if isinstance(x, str) else x)
    latex = shown.to_latex(index=False, longtable=True, escape=True,
                          column_format=widths.replace('p{', 'P{') if widths else None,
                          float_format=lambda x: f'{x:.4g}', na_rep='--')
    latex = latex.replace('CO2', r'CO$_2$').replace('65 N', r'65$^\circ$N')
    latex = latex.replace('a b2k / 1000 - 0.05', r'$t_{\rm b2k}({\rm yr})/1000-0.05$')
    return r'\Needspace{8\baselineskip}' + '\n\\subsection*{' + title + '}\n{\\small\n' + latex + '}\n'


def age_range(row, prefix):
    return f'{row[prefix + "_median"]:.3f} [{row[prefix + "_q025"]:.3f}, {row[prefix + "_q975"]:.3f}]'


def inventory_tables():
    primary = read('data/curated/ngrip_mis6_warming_events.csv')
    variable = read(BARKER / 'Barker2011_event_phase_analysis/event_catalogue_used.csv')
    fixed = read(BARKER / 'Barker2011_event_phase_analysis/fixed_threshold/event_catalogue_used.csv')
    variable['fixed_threshold'] = variable.source_excel_row.isin(fixed.source_excel_row)
    variable['event_id'] = variable.source_excel_row.map(lambda x: f'Barker S3 row {x}')
    save(primary, 'S1_primary_inventory')
    save(variable, 'S1_barker_inventory')
    support = read('data/curated/observation_segments.csv')
    support['response_start_kyr_bp'] = support.observation_start_kyr_bp
    support['response_end_kyr_bp'] = support.observation_end_kyr_bp - 1.5
    support = pd.concat([support, pd.DataFrame([dict(segment_id='Barker',
        observation_start_kyr_bp=0, observation_end_kyr_bp=400,
        response_start_kyr_bp=0, response_end_kyr_bp=398.5)])], ignore_index=True)
    support['response_exposure_kyr'] = support.response_end_kyr_bp - support.response_start_kyr_bp
    save(support, 'S1_support')
    s = tex_table(support, '(a) Nominal observation and response support (kyr BP)',
        {'segment_id':'Segment','observation_start_kyr_bp':'Obs. start','observation_end_kyr_bp':'Obs. end',
         'response_start_kyr_bp':'Resp. start','response_end_kyr_bp':'Resp. end','response_exposure_kyr':'Exposure'},
        {c:1 for c in ['observation_start_kyr_bp','observation_end_kyr_bp','response_start_kyr_bp','response_end_kyr_bp','response_exposure_kyr']})
    s += 'Exposure is in kyr; ages increase into the past. The combined primary exposure is 180 kyr.\n'
    s += tex_table(primary, '(b) Primary published-event inventory',
        {'event_label':'Event','source_record':'Record','source_event_label':'Source label','event_age_kyr_bp':'Age (kyr BP)'},
        {'event_age_kyr_bp':3}, widths='lllr')
    s += 'NGRIP ages are published GI starts; MF and Sofular ages are maximum-gradient assignments near published labels (Text S1).\n'
    # Use reader-facing membership labels in print; retain booleans in the CSV.
    variable_display = variable.copy()
    variable_display['fixed_threshold'] = variable_display.fixed_threshold.map({True: 'Yes', False: 'No'})
    s += tex_table(variable_display, '(c) Barker variable-threshold inventory and fixed-threshold membership',
        {'event_id':'Source identifier','event_age_ka':'Age (kyr BP)','fixed_threshold':'Fixed threshold'},
        {'event_age_ka':3}, widths='lrl')
    return s


def epoch_tables():
    audit = read('data/curated/age_epoch_audit.csv')
    orbital = read('data/curated/orbital_driver_input_audit.csv')
    save(audit, 'S2_source_epoch_audit')
    save(orbital, 'S2_orbital_input_audit')
    # Concise display descriptions retain the source audit's verification status.
    epochs = pd.DataFrame([
        ['NGRIP starts and GICC05','2000 CE','a b2k / 1000 - 0.05','Explicit source definition'],
        ['MF stack','1950 CE','kyr ages unchanged','Archive calendar-BP unit'],
        ['Sofular stack / So-4','1950 CE','kyr ages unchanged','Explicit article / raw headers'],
        ['So-57 U-Th controls','1950 CE','yr ages / 1000','Explicit raw header'],
        ['LR04','1950 CE','kyr ages unchanged','Archive and values checked'],
        ['CO2 composite','1950 CE','yr ages / 1000','Archive and values checked'],
        ['La2004 precession, eccentricity, obliquity','J2000','-signed kyr - 0.05','Official epoch and values checked'],
        ['65 N solstice insolation','J2000 inferred','positive kyr - 0.05','Numerical reconstruction; metadata not explicit'],
        ['Barker SpeleoAge','1950 CE assumed','kyr ages unchanged','Exact source epoch unverified'],
    ], columns=['source','source_epoch','conversion_to_kyr_bp','evidence'])
    save(epochs, 'S2_epochs_display')
    uncertainty = pd.DataFrame([
        ['NGRIP definition','Published location codes; working Gaussian SDs','Independent event errors: a 20, b 50, c 200, d 100, f 30 yr; explicit 4 yr retained','Reject complete crossed warming/cooling sequence'],
        ['NGRIP counted','MCE / 2 as working SD','Cumulative independent variance increments at age knots; interpolate offsets','Reject nonmonotone knot map'],
        ['NGRIP extension','4.5% of b2k age as approximate 2 SD envelope','Same cumulative process; preserve counted endpoint','Envelope is not a hard bound or published posterior'],
        ['MF chronology','Outer approximate 95% envelope / 1.95996','One Gaussian factor shared by all MF events','Repair envelope to include point age; no interpolation across source gap'],
        ['Sofular chronology','Reported U-Th 2 SD errors / 2','Independent dated-depth errors transferred through reference growth curve','Monotone controls; no hiatus crossing or extrapolation'],
        ['Speleothem timing','Nine equally weighted parameter settings','One smoothing/window choice per source record','Fixed event identity and final order'],
        ['Barker chronology','Published combined errors as uniform half-widths','Independent control proposals; linear offset interpolation on SpeleoAge','Reject crossed controls; gap interpolation and auxiliary endpoint'],
    ],columns=['source','scale','dependence','boundary'])
    save(uncertainty, 'S2_working_uncertainty')
    s=tex_table(epochs,'(a) Epoch harmonization', {'source':'Source','source_epoch':'Epoch','conversion_to_kyr_bp':'Conversion','evidence':'Evidence'}, widths='p{3.0cm}p{2.0cm}p{3.0cm}p{6.0cm}')
    s+=tex_table(uncertainty,'(b) Working uncertainty assumptions',{'source':'Source','scale':'Error scale','dependence':'Dependence','boundary':'Boundary rule'},widths='p{2.1cm}p{3.5cm}p{4.5cm}p{3.9cm}')
    s+='Error widths are durations and receive unit conversion, never an epoch shift. Full source evidence and file paths are supplied in the accompanying CSVs.\n'
    return s


def estimate_tables():
    points=[]; ages=[]; boot=[]
    for label,base,prefix in [('Primary',PRIMARY,'NGRIP_MIS6'),('Barker',BARKER,'Barker2011')]:
        p=read(base/f'{prefix}_event_phase_analysis/analysis_summary.csv').iloc[0]
        points.append(dict(catalogue=label,**p.to_dict()))
        ages.append(dict(catalogue=label,**read(base/f'{prefix}_event_uncertainty_sensitivity/summary.csv').iloc[0].to_dict()))
        boot.append(dict(catalogue=label,**read(base/f'{prefix}_PI_bootstrap/summary.csv').iloc[0].to_dict()))
    points,ages,boot=map(pd.DataFrame,[points,ages,boot])
    effect=read(PRIMARY/'NGRIP_MIS6_effect_uncertainty/effect_summary.csv')
    for name,frame in [('S3_point_estimates',points),('S3_age_sensitivity',ages),('S3_null_bootstrap',boot),('S3_effect_precision',effect)]:save(frame,name)
    metrics={'n_predictive_events':'Response events','response_exposure_kyr':'Exposure (kyr)',
             'info_bits_per_event':'PI (bits/event)','LR_statistic':'Likelihood ratio','nominal_LR_p':'Nominal p',
             'pre_phase_preferred_deg':'Preferred phase (degrees)','pre_phase_rate_ratio_max_vs_min':'Maximum/minimum rate ratio',
             'delta_AICc_full_minus_reduced':'Full-minus-reduced AICc','rayleigh_p':'Descriptive Rayleigh p'}
    display=pd.DataFrame({'Quantity':list(metrics.values())})
    for _,row in points.iterrows():display[row.catalogue]=[f'{row[key]:.5g}' for key in metrics]
    s=tex_table(display,'(a) Point-age estimates',widths='lrr')
    s+=tex_table(boot,'(b) Reduced-model null calibration',{'catalogue':'Catalogue','n_bootstrap':'Simulations',
        'n_bootstrap_exceeding_or_equal_observed':'Exceedances','empirical_p_plus_one':'Bootstrap p',
        'empirical_p_ci95_low':'95% lower','empirical_p_ci95_high':'95% upper'},
        {c:5 for c in ['empirical_p_plus_one','empirical_p_ci95_low','empirical_p_ci95_high']})
    s+='The interval is an exact binomial interval for the unadjusted null exceedance probability; the reported p uses the plus-one correction.\n'
    display=[]
    for _,row in ages.iterrows():
        for prefix,label in [('info_bits_per_event','PI (bits/event)'),('pre_phase_preferred_deg','Phase (degrees)'),('pre_phase_rate_ratio_max_vs_min','Rate ratio')]:
            display.append([row.catalogue,label,age_range(row,prefix),f'{int(row.n_valid)}/{int(row.n_realizations)}'])
    s+=tex_table(pd.DataFrame(display,columns=['Catalogue','Quantity','Median [2.5%, 97.5%]','Valid/total']), '(c) Chronology sensitivity',widths='llrl')
    effect_display = effect.copy()
    effect_display['scenario'] = effect_display.scenario.map({'A_chronology':'A: chronology','B_sampling':'B: sampling','C_joint':'C: joint'})
    effect_display['quantity'] = effect_display.quantity.map({'max_min_rate_ratio':'Rate ratio','preferred_phase_deg':'Phase (deg)'})
    s+=tex_table(effect_display,'(d) Primary effect precision',{'scenario':'Scenario','quantity':'Quantity','center':'Center','low':'Lower','high':'Upper','n':'Fits'},
        {c:2 for c in ['center','low','high']},widths='llrrrr')
    s+='A/C are central 95\\% working ranges centered on medians. B projects an approximate joint 95\\% coefficient confidence region and is centered on the point estimate. Angles are unwrapped around 330 degrees. C is not a calibrated combined interval; no Barker sampling interval is available (Text S5).\n'
    return s


def sensitivity_tables():
    design=read(PRIMARY/'NGRIP_MIS6_PI_design_sensitivity/design_sensitivity.csv')
    models=read(PRIMARY/'NGRIP_MIS6_PI_model_sensitivity/model_sensitivity.csv')
    support=read(PRIMARY/'NGRIP_MIS6_PI_design_sensitivity/support_sensitivity.csv')
    pool=read(PRIMARY/'NGRIP_MIS6_PI_design_sensitivity/pooling_diagnostic.csv')
    knot=read(PRIMARY/'NGRIP_MIS6_event_uncertainty_sensitivity/ngrip_knot_spacing/summary.csv')
    component=read(PRIMARY/'NGRIP_MIS6_event_uncertainty_sensitivity/sofular_so57_overlap/summary.csv')
    fixed=read(BARKER/'Barker2011_event_phase_analysis/event_definition_sensitivity.csv')
    projection=read('MIS6/data/processed/MIS6_event_age_uncertainty/sofular_component_mapping_sensitivity.csv')
    for name,frame in [('design',design),('models',models),('support',support),('pooling',pool),('knot_spacing',knot),('so57_overlap',component),('barker_definition',fixed),('projection',projection)]:save(frame,'S4_'+name)
    stats={'info_bits_per_event':'PI','nominal_LR_p':'Nominal p','pre_phase_preferred_deg':'Phase','pre_phase_rate_ratio_max_vs_min':'Rate ratio','delta_AICc_full_minus_reduced':'dAICc'}
    fmts={'info_bits_per_event':4,'nominal_LR_p':5,'pre_phase_preferred_deg':1,'pre_phase_rate_ratio_max_vs_min':2,'delta_AICc_full_minus_reduced':2}
    s=tex_table(design,'(a) All 30 designs on 173-kyr common support; 55 events',
        {'history_window_kyr':'History','bin_width_kyr':'Bin','origin_fraction':'Origin',**stats},fmts,widths='rrrrrrrr')
    s+='History and bin width are in kyr; origin is a fraction of bin width. Phase is in degrees and PI in bits/event throughout Tables S4--S5.\n'
    # Explicit shorter descriptions avoid transporting implementation identifiers into the article.
    models['display_model']=models.variant_label
    replacements={'Linear LR04 + CO2':'Linear','LR04 quadratic':'LR04 squared','CO2 quadratic':'CO2 squared',
        'Both quadratic':'Both squares','Both quadratic + interaction':'Squares + product',
        'Prior 1.5-kyr count':'Prior count','Time since last event':'Elapsed time','log(1 + time / 1 kyr)':'Log elapsed time'}
    models['display_model']=models.display_model.replace(replacements)
    s+=tex_table(models,'(b) Background shape and memory alternatives',{'display_model':'Specification','response_exposure_kyr':'kyr',**stats},fmts,widths='p{3.3cm}rrrrrr')
    s+='The five climate specifications have 55 events and 180 kyr; the three memory specifications have 53 events and 164.8 kyr. Each dAICc compares full with its own reduced model on identical support.\n'
    s+=tex_table(support,'(c) Response-support comparison at primary settings',{'response_exposure_kyr':'Exposure (kyr)',**stats},fmts)
    s+=tex_table(pool,'(d) Segment-specific phase coefficients versus a common response',{'n_events':'Events','LR_statistic':'LR','nominal_LR_p':'Nominal p','info_bits_per_event_for_interaction':'PI','delta_AICc_segment_specific_minus_common':'dAICc'},
        {'LR_statistic':4,'nominal_LR_p':5,'info_bits_per_event_for_interaction':5,'delta_AICc_segment_specific_minus_common':3})
    s+='Both phase interactions are added together (2 degrees of freedom); response exposure is 180 kyr. This test does not establish equivalence between segments.\n'
    chronology=[]
    for _,row in knot.iterrows():chronology.append(dict(case=f'NGRIP knots {row.knot_spacing_ka:g} kyr',**row.to_dict()))
    chronology.append(dict(case='Sofular So-57 overlap',**component.iloc[0].to_dict()))
    chronology=pd.DataFrame(chronology)
    chronology['PI median [range]']=[age_range(r,'info_bits_per_event') for _,r in chronology.iterrows()]
    chronology['valid / total']=[f'{int(r.n_valid)}/{int(r.n_realizations)}' for _,r in chronology.iterrows()]
    chronology['nominal_below_percent']=100*chronology.fraction_nominal_p_below_0p05
    s+=tex_table(chronology,'(e) Alternative chronology models',{'case':'Case','valid / total':'Valid/total','PI median [range]':'PI median [95% range]','nominal_below_percent':'p < 0.05 (%)'}, {'nominal_below_percent':2},widths='lrrr')
    s+='Percentages are conditional robustness fractions, not bootstrap p values. The So-57 case changes only four supported Sofular projections; the oldest event retains So-4.\n'
    fixed_display = fixed.copy()
    fixed_display['event_definition'] = fixed_display.event_definition.str.replace('_', ' ')
    s+=tex_table(fixed_display,'(f) Barker event definitions',{'event_definition':'Definition','n_predictive_events':'Events',**stats},fmts,widths='lrrrrrr')
    summary=projection.groupby(['component','mapping_lag_ka']).chronology_sigma_before_conditioning_ka.agg(['min','max']).reset_index()
    save(summary,'S4_projection_ranges')
    s+=tex_table(summary,'(g) Sofular projection-position diagnostic',{'component':'Component','mapping_lag_ka':'Coordinate shift (kyr)','min':'Minimum SD (kyr)','max':'Maximum SD (kyr)'},{'min':3,'max':3},widths='lrrr')
    s+='Ranges span supported events and all nine detector settings before order conditioning. Coordinate shifts move error-field evaluation positions without changing nominal event ages; they do not represent climate lags. The complete 405-row diagnostic accompanies this table.\n'
    return s


def orbital_tables():
    pieces=[]; points=[]; summaries=[]; interactions=[]
    modelnames={'B':'BG','BP':'BG + Pre','B_ecc':'BG + Ecc','BP_ecc':'BG + Pre + Ecc',
        'B_obl':'BG + Obl','BP_obl':'BG + Pre + Obl','B_insol65n':'BG + Insol','BP_insol65n':'BG + Pre + Insol'}
    for label,base,prefix in [('Primary',PRIMARY,'NGRIP_MIS6'),('Barker',BARKER,'Barker2011')]:
        point=read(base/f'{prefix}_orbital_driver_sensitivity/point_comparisons.csv')
        summary=read(base/f'{prefix}_orbital_driver_sensitivity/comparison_summary.csv')
        interaction=read(base/f'{prefix}_climate_phase_interaction/comparison_summary.csv')
        point.insert(0,'catalogue',label); summary.insert(0,'catalogue',label); interaction.insert(0,'catalogue',label)
        points.append(point);summaries.append(summary);interactions.append(interaction)
        point['comparison']=[modelnames[f]+' vs '+modelnames[r] for f,r in zip(point.full_model_id,point.reduced_model_id)]
        pieces.append(tex_table(point,f'({"a" if label=="Primary" else "c"}) {label}: all point-age comparisons',
            {'comparison':'Comparison','df':'df','info_bits_per_event':'PI','nominal_p':'Nominal p','holm_nominal_p':'Holm p','delta_AIC':'dAIC','delta_AICc':'dAICc'},
            {'info_bits_per_event':4,'nominal_p':5,'holm_nominal_p':5,'delta_AIC':2,'delta_AICc':2},widths='p{5cm}rrrrrr'))
        summary['comparison']=point.comparison
        summary['range']=[age_range(r,'info_bits_per_event') for _,r in summary.iterrows()]
        pieces.append(tex_table(summary,f'({"b" if label=="Primary" else "d"}) {label}: chronology sensitivity',
            {'comparison':'Comparison','range':'PI median [2.5%, 97.5%]','n_mc_valid':'Valid','n_mc_total':'Total'},widths='p{6cm}rrr'))
    save(pd.concat(points,ignore_index=True),'S5_orbital_point')
    save(pd.concat(summaries,ignore_index=True),'S5_orbital_chronology')
    interactions=pd.concat(interactions,ignore_index=True);save(interactions,'S5_lr04_interaction')
    pieces.append(tex_table(interactions,'(e) Additional LR04-by-phase interactions (2 degrees of freedom)',
        {'catalogue':'Catalogue','info_bits_per_event_point':'PI','LR_statistic_point':'LR','nominal_p_point':'Nominal p','delta_AIC_point':'dAIC','delta_AICc_point':'dAICc','n_mc_valid':'Valid MC'},
        {'info_bits_per_event_point':4,'LR_statistic_point':4,'nominal_p_point':5,'delta_AIC_point':2,'delta_AICc_point':2}))
    pieces.append('BG denotes intercept, prior-event count, LR04 and CO2, plus a segment term in the primary catalogue; Pre is the sine/cosine phase block; Ecc is eccentricity; Obl is obliquity; Insol is 65-degree-N summer-solstice daily mean insolation. Holm adjustment includes nine additional orbital comparisons per catalogue; the phase reference is outside that family. LR04 interactions are separate exploratory tests. All p values in this table are nominal; the point-age main null bootstrap is not reused for different hypotheses. Negative dAIC/dAICc favors the augmented model.\n')
    return ''.join(pieces)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    for number,builder in enumerate([inventory_tables,epoch_tables,estimate_tables,sensitivity_tables,orbital_tables],1):
        (OUT/f'TableS{number}.tex').write_text('% Generated by paper_tables.py; edit the exporter, not this file.\n'+builder())
    pd.DataFrame([dict(source_file=p,sha256=s) for p,s in sorted(INPUTS.items())]).to_csv(OUT/'source_manifest.csv',index=False)
    print(f'Exported five table groups from {len(INPUTS)} saved CSV inputs to {OUT}.')

if __name__=='__main__':
    main()
