#!/usr/bin/env python3
"""Compare precession associations of NGRIP stadial and interstadial starts.

Run from the project root: python NGRIP/ngrip_transition_phase_sensitivity.py
Reuses the main continuous-time likelihood, BG bootstrap and saved combined
chronology ensemble. GI and GS are fitted separately, not as independent data.
Source: Rasmussen et al. (2014), doi:10.1016/j.quascirev.2014.09.007.
"""
import hashlib
import os
from pathlib import Path
import sys
import tempfile

# Each worker uses one numerical thread; no figures are required for this test.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mcv_orb_matplotlib"))
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from toolbox import age_sensitivity, event_model, model_stats, sampling
from toolbox.point_process import fit_point_process
from toolbox.project_config import (
    PROJECT_ROOT, LR04_CSV, CO2_CSV, ORBITAL_CSV, PRECESSION_PHASE_CSV,
    OBSERVATION_SEGMENTS_CSV, MODEL_VERSION,
)
from toolbox.project_config import generated_notes_dir

ROOT = PROJECT_ROOT / "NGRIP"
RUN_NAME = "ngrip_transition_phase_sensitivity"
EVENT_INPUT = ROOT / "data/processed/ngrip_warming_cooling_starts.csv"
AGE_INPUT = ROOT / "data/processed/ngrip_event_age_uncertainty/ngrip_event_age_realizations.csv"
OUT_DIR = ROOT / "data/processed" / RUN_NAME
NOTE_DIR = generated_notes_dir(ROOT)
EVENT_TYPES = ("cooling", "warming")
N_BOOTSTRAP = 9_999
N_REALIZATIONS = 10_000
SEED = 20260921
N_WORKERS = 3
HISTORY_TAU_KYR = 1.5
HISTORY_TERM = "same_type_exponential_history"
REDUCED_TERMS = ["intercept", HISTORY_TERM, "lr04_scaled", "co2_scaled"]
FULL_TERMS = REDUCED_TERMS + ["pre_phase_sin", "pre_phase_cos"]


def select_catalogue(raw, event_type):
    """Keep one transition type and its original BP1950 ages."""
    if event_type not in EVENT_TYPES:
        raise ValueError("event_type must be cooling or warming")
    events = raw.loc[raw.event_type.eq(event_type)].sort_values("age_ka_bp").reset_index(drop=True)
    if len(events) != {"cooling": 35, "warming": 34}[event_type] or events.event_label.duplicated().any():
        raise ValueError("Unexpected NGRIP event count or duplicate labels")
    # These ages are already BP1950. The common core explicitly uses u=anchor-age.
    events["event_age_kyr_bp"] = events.age_ka_bp
    events["event_id"] = "NGRIP:" + events.event_label
    events["segment_id"] = "NGRIP"
    return events


def select_age_realizations(source, events, n_realizations=N_REALIZATIONS):
    """Keep original IDs and label correspondence; never sort perturbed ages."""
    if not 1 <= n_realizations <= len(source):
        raise ValueError("Requested age realizations exceed the available ensemble")
    columns = [f"age_ka_bp__{label}" for label in events.event_label]
    draws = source.loc[:, ["realization_id", *columns]].iloc[:n_realizations].copy()
    if draws.realization_id.isna().any() or draws.realization_id.duplicated().any():
        raise ValueError("Age realizations must have unique, nonmissing IDs")
    return draws, columns


def run_analysis(*, n_bootstrap=N_BOOTSTRAP, n_realizations=N_REALIZATIONS,
                 seed=SEED, n_workers=3, show_progress=True):
    raw = pd.read_csv(EVENT_INPUT)
    source_ages = pd.read_csv(AGE_INPUT)
    observations = pd.read_csv(OBSERVATION_SEGMENTS_CSV).query("segment_id == 'NGRIP'").copy()
    lr04 = pd.read_csv(LR04_CSV, float_precision="round_trip")
    co2 = pd.read_csv(CO2_CSV, float_precision="round_trip")
    orbital = pd.read_csv(ORBITAL_CSV, float_precision="round_trip")
    anchors = pd.read_csv(PRECESSION_PHASE_CSV, float_precision="round_trip")
    forcings = {"lr04": (lr04.age_kyr_bp.to_numpy(), lr04.lr04.to_numpy()),
                "co2": (co2.age_kyr_bp.to_numpy(), co2.co2_ppm.to_numpy()),
                "precession_index": (orbital.age_kyr_bp.to_numpy(), orbital.precession_index.to_numpy())}
    phase_anchors = (anchors.age_kyr_bp.to_numpy(), anchors.phase_unwrapped_rad.to_numpy())
    summaries, age_summaries, boot_tables, age_tables = [], [], [], []
    nominal_events, nominal_coefficients, nominal_windows, nominal_scaling = [], [], [], []
    for index, event_type in enumerate(EVENT_TYPES):
        events = select_catalogue(raw, event_type)
        windows = event_model.response_windows(events, observations)
        scaling = event_model.nominal_scaling({name: forcings[name] for name in ("lr04", "co2")}, windows)
        event_x, integral_x = event_model.build_design(
            events, windows, forcings, phase_anchors, scaling, tau=HISTORY_TAU_KYR,
        )
        reduced = fit_point_process(event_x[REDUCED_TERMS], integral_x[REDUCED_TERMS],
                                    integral_x.weight, REDUCED_TERMS, nonpositive_terms=(HISTORY_TERM,))
        full = fit_point_process(event_x[FULL_TERMS], integral_x[FULL_TERMS],
                                 integral_x.weight, FULL_TERMS, nonpositive_terms=(HISTORY_TERM,),
                                 start_beta=np.r_[reduced.beta, 0., 0.])
        point = model_stats.fit_summary(reduced, full, event_x, windows, n_source_events=len(events),
                                        catalogue_id=f"ngrip_{event_type}", tau=HISTORY_TAU_KYR)
        if not point["all_models_converged"] or not point["likelihood_nesting_ok"]:
            raise RuntimeError(f"Invalid nominal {event_type} fit")
        if show_progress:
            print(f"\nNGRIP {event_type}: nominal p={point['nominal_LR_p']:.6g}, "
                  f"phase={point['pre_phase_preferred_deg']:.2f} deg", flush=True)

        # BG simulations retain nominal anchors, endpoints and scaling.
        replicates, failures = sampling.phase_bootstrap(
            events, windows, forcings, phase_anchors, scaling, reduced, full,
            reduced_terms=REDUCED_TERMS, full_terms=FULL_TERMS, tau=HISTORY_TAU_KYR,
            n_bootstrap=n_bootstrap, seed=seed + index, workers=n_workers, show_progress=show_progress)
        summary = model_stats.phase_bootstrap_summary(point, replicates, failures)
        summary.insert(0, "event_type", event_type)
        summary["seed"] = seed + index
        summaries.append(summary)
        boot_tables.append(replicates.assign(event_type=event_type))

        # Each age row moves its own anchor; unsupported rows are retained.
        draws, columns = select_age_realizations(source_ages, events, n_realizations)
        age_results, diagnostics = age_sensitivity.fit_realizations(
            events, observations, forcings, phase_anchors, scaling, REDUCED_TERMS, FULL_TERMS,
            draws, columns, catalogue_id=f"ngrip_{event_type}", tau=HISTORY_TAU_KYR,
            n_workers=n_workers, show_progress=show_progress)
        age_summary = age_sensitivity.summarize(age_results, point)
        age_summary.insert(0, "event_type", event_type)
        age_summary["n_outside_support"] = age_results.invalid_reason.str.startswith("outside_").sum()
        age_summary["n_numerical_failures"] = diagnostics["n_numerical_failures"]
        age_summaries.append(age_summary)
        age_tables.append(age_sensitivity.compact_results(age_results).assign(event_type=event_type))
        events["event_role"] = np.where(events.event_age_kyr_bp.eq(windows.anchor_age_kyr_bp.item()),
                                        "conditioning", "response")
        nominal_events.append(events[["event_label", "source_event_label", "event_age_kyr_bp",
                                       "event_role", "event_type"]])
        nominal_coefficients.append(pd.DataFrame([
            dict(model_id=name, term=term, beta=beta, rate_ratio_per_unit=np.exp(beta), event_type=event_type)
            for name, model in (("reduced", reduced), ("full", full))
            for term, beta in zip(model.terms, model.beta)
        ]))
        nominal_windows.append(windows.assign(event_type=event_type))
        nominal_scaling.append(scaling.reset_index().assign(event_type=event_type))
    return dict(summary=pd.concat(summaries, ignore_index=True),
                age_summary=pd.concat(age_summaries, ignore_index=True),
                bootstrap_replicates=pd.concat(boot_tables, ignore_index=True),
                age_realizations=pd.concat(age_tables, ignore_index=True),
                nominal_events=pd.concat(nominal_events, ignore_index=True),
                nominal_coefficients=pd.concat(nominal_coefficients, ignore_index=True),
                nominal_windows=pd.concat(nominal_windows, ignore_index=True),
                nominal_scaling=pd.concat(nominal_scaling, ignore_index=True), n_workers=n_workers)


def save_results(result, output_dir=OUT_DIR):
    """Save readable research outputs, including unsuccessful realizations."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "summary.csv": result["summary"][["event_type", "n_source_events", "n_response_events",
            "response_exposure_kyr", "gain_bits_per_event", "LR_statistic", "nominal_LR_p",
            "pre_phase_preferred_deg", "pre_phase_rate_ratio_max_vs_min", "beta_history",
            "empirical_p_plus_one", "empirical_p_ci95_low", "empirical_p_ci95_high",
            "n_bootstrap", "n_bootstrap_exceeding_or_equal_observed", "n_failed_replicates", "seed"]],
        "age_summary.csv": result["age_summary"][["event_type", "n_realizations", "n_valid",
            "n_outside_support", "n_numerical_failures", "n_nominal_p_below_0p05",
            "fraction_nominal_p_below_0p05", "gain_bits_per_event_q025", "gain_bits_per_event_median",
            "gain_bits_per_event_q975", "pre_phase_preferred_deg_q025",
            "pre_phase_preferred_deg_median", "pre_phase_preferred_deg_q975"]],
        "bootstrap_replicates.csv": result["bootstrap_replicates"][["event_type", "bootstrap_id",
            "n_events_response", "LR_statistic", "fit_valid", "status", "solver_attempts", "failure_reason"]],
        "age_realizations.csv": result["age_realizations"][["event_type", "realization_id", "fit_valid",
            "invalid_reason", "n_response_events", "response_exposure_kyr", "gain_bits_per_event",
            "LR_statistic", "nominal_LR_p", "pre_phase_preferred_deg",
            "pre_phase_rate_ratio_max_vs_min", "beta_history"]],
        "nominal_coefficients.csv": result["nominal_coefficients"],
        "nominal_events.csv": result["nominal_events"],
    }
    parameters = []
    for kind in EVENT_TYPES:
        row = result["summary"].set_index("event_type").loc[kind]
        values = dict(model_version=MODEL_VERSION, age_epoch="BP1950",
            history_tau_kyr=HISTORY_TAU_KYR, history_domain="beta_H <= 0",
            history_events="same transition type only", initial_unobserved_history=0.,
            quadrature_order=4, seed=int(row.seed), n_bootstrap=int(row.n_bootstrap),
            n_workers=result["n_workers"],
            n_age_realizations=len(result["age_realizations"].query("event_type == @kind")),
            reduced_terms=" + ".join(REDUCED_TERMS[1:]), full_terms=" + ".join(FULL_TERMS[1:]),
            age_rejections="outside observation support; no truncation or replacement",
            p_rule="bootstrap: (1+exceedances)/(B+1); age fraction: nominal LR p<0.05 among valid fits")
        values.update(result["nominal_windows"].set_index("event_type").loc[kind].to_dict())
        scales = result["nominal_scaling"].query("event_type == @kind").set_index("forcing_id")
        for forcing, scale in scales.drop(columns="event_type").iterrows():
            values.update({f"{forcing}_{key}": value for key, value in scale.items()})
        parameters.extend(dict(event_type=kind, parameter=key, value=value) for key, value in values.items())
    paths = [EVENT_INPUT, AGE_INPUT, OBSERVATION_SEGMENTS_CSV, Path(__file__),
             Path(sampling.__file__), Path(age_sensitivity.__file__),
             PROJECT_ROOT / "toolbox/point_process.py", PROJECT_ROOT / "toolbox/model_stats.py",
             PROJECT_ROOT / "toolbox/event_model.py",
             PROJECT_ROOT / "toolbox/project_config.py",
             LR04_CSV, CO2_CSV, ORBITAL_CSV, PRECESSION_PHASE_CSV]
    parameters.extend(dict(event_type="both", parameter="sha256:" + str(path.relative_to(PROJECT_ROOT)),
                           value=hashlib.sha256(path.read_bytes()).hexdigest()) for path in paths)
    tables["parameters_and_provenance.csv"] = pd.DataFrame(parameters)
    for name, table in tables.items():
        table.to_csv(output_dir / name, index=False, float_format="%.12g")


def write_notes(result, notes_dir=NOTE_DIR):
    notes_dir = Path(notes_dir)
    notes_dir.mkdir(parents=True, exist_ok=True)
    lines = ["NGRIP WARMING AND COOLING: CONTINUOUS-TIME PHASE ASSOCIATIONS", "",
        "Rasmussen et al. (2014) GI and GS starts are fitted separately on the main NGRIP observation window (12-123 kyr BP1950). Each direction conditions on its exact oldest event and retains all exposure down to 12 kyr BP. This gives different response lengths for GI and GS. BG includes LR04, CO2 and same-type exponential history (tau=1.5 kyr; beta_H<=0); full adds phase sine and cosine. There is no segment term. Climate scaling uses each nominal catalogue's response-time mean and range.", "",
        "Bootstrap retains the nominal anchor and younger endpoint, simulates the fitted BG model in continuous time, allows event counts to vary, and refits both models. Climate values are interpolated at simulated event times and history updates at each event. Bootstrap p=(1+number of simulated LR >= observed LR)/(B+1). Failed fits are never replaced, and p is withheld if any bootstrap replicate remains unresolved.", "",
        "Age sensitivity reuses all saved combined NGRIP draws, including chronology and definition errors. GI and GS retain the same source realization IDs. Each in-support draw updates its exact conditioning age, exposure, forcing values and history while retaining nominal scaling. Draws outside [12,123] kyr BP are flagged rather than replaced. The fraction with nominal LR p<0.05 is calculated among valid draws; it is not a bootstrap p or a probability that the effect exists. Phase quantiles are unwrapped about the nominal maximum.", ""]
    for kind in EVENT_TYPES:
        s = result["summary"].set_index("event_type").loc[kind]
        a = result["age_summary"].set_index("event_type").loc[kind]
        anchor = result["nominal_windows"].set_index("event_type").loc[kind, "anchor_age_kyr_bp"]
        lines = lines + [kind.upper(),
            f"Inventory={int(s.n_source_events)}; fitted={int(s.n_response_events)}; anchor={anchor:.3f} kyr BP; response duration={s.response_exposure_kyr:.3f} kyr.",
            f"G={s.gain_bits_per_event:.6f} bits/event; LR={s.LR_statistic:.6f}; nominal p={s.nominal_LR_p:.6g}; preferred phase={s.pre_phase_preferred_deg:.2f} deg; phase rate ratio={s.pre_phase_rate_ratio_max_vs_min:.3f}.",
            f"Bootstrap B={int(s.n_bootstrap)}; exceedances={s.n_bootstrap_exceeding_or_equal_observed:g}; p={s.empirical_p_plus_one:.6g}; 95% Monte Carlo interval=[{s.empirical_p_ci95_low:.6g}, {s.empirical_p_ci95_high:.6g}]; failed={int(s.n_failed_replicates)}; seed={int(s.seed)}.",
            f"Age draws={int(a.n_realizations)}; valid={int(a.n_valid)}; outside support={int(a.n_outside_support)}; numerical failures={int(a.n_numerical_failures)}. Nominal p<0.05 in {int(a.n_nominal_p_below_0p05)}/{int(a.n_valid)} ({100*a.fraction_nominal_p_below_0p05:.2f}%).",
            f"Age median G={a.gain_bits_per_event_median:.6f}, central 95%=[{a.gain_bits_per_event_q025:.6f}, {a.gain_bits_per_event_q975:.6f}]. Preferred-phase median={a.pre_phase_preferred_deg_median:.2f} deg, central 95%=[{a.pre_phase_preferred_deg_q025:.2f}, {a.pre_phase_preferred_deg_q975:.2f}] deg.", ""]
    lines.append("Interpretation: These exploratory directions share chronology and alternate in the same record. Similar phases are not independent evidence or a test of equality of phase responses. The rate is measured over total response time, not only time spent in the state from which the transition starts. Connection to an orbitally favorable oscillatory regime remains a physical interpretation, not identification of the trigger of individual warmings or coolings.")
    (notes_dir / f"{RUN_NAME}_Methods_and_results.txt").write_text("\n".join(lines) + "\n")


def main():
    result = run_analysis(n_bootstrap=N_BOOTSTRAP, n_realizations=N_REALIZATIONS,
                          seed=SEED, n_workers=N_WORKERS)
    save_results(result, OUT_DIR)
    if result["summary"].n_failed_replicates.any() or result["age_summary"].n_numerical_failures.any():
        raise RuntimeError("Unresolved numerical fits saved for inspection; final notes withheld")
    write_notes(result, NOTE_DIR)
    print(result["summary"][["event_type", "gain_bits_per_event", "pre_phase_preferred_deg",
                               "nominal_LR_p", "empirical_p_plus_one"]].to_string(index=False))
    print(result["age_summary"][["event_type", "n_valid", "fraction_nominal_p_below_0p05"]].to_string(index=False))


if __name__ == "__main__":
    main()
