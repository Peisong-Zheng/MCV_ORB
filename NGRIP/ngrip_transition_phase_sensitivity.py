#!/usr/bin/env python3
"""Compare precession associations of NGRIP stadial and interstadial starts.

Run from the project root: python NGRIP/ngrip_transition_phase_sensitivity.py
Reuses the main continuous-time likelihood, BG bootstrap and saved combined
chronology ensemble. GI and GS are fitted separately, not as independent data.
Source: Rasmussen et al. (2014), doi:10.1016/j.quascirev.2014.09.007.
"""
import argparse
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
import pandas as pd
import NGRIP_MIS6_likelihood_bootstrap as bootstrap
from toolbox import age_sensitivity, combined_likelihood as likelihood
from toolbox.project_config import PROJECT_ROOT

ROOT = PROJECT_ROOT / "NGRIP"
RUN_NAME = "ngrip_transition_phase_sensitivity"
EVENT_INPUT = ROOT / "data/processed/ngrip_warming_cooling_starts.csv"
AGE_INPUT = ROOT / "data/processed/ngrip_event_age_uncertainty/ngrip_event_age_realizations.csv"
OUT_DIR = ROOT / "data/processed" / RUN_NAME
NOTE_DIR = ROOT / "experiment_note"
EVENT_TYPES = ("cooling", "warming")
N_BOOTSTRAP = 9_999
N_REALIZATIONS = 10_000
SEED = 20260921


def load_catalogue(event_type):
    """Use the main NGRIP support, conditioning on this type's oldest event."""
    if event_type not in EVENT_TYPES:
        raise ValueError("event_type must be cooling or warming")
    raw = pd.read_csv(EVENT_INPUT)
    events = raw.loc[raw.event_type.eq(event_type)].sort_values("age_ka_bp").reset_index(drop=True)
    if len(events) != {"cooling": 35, "warming": 34}[event_type] or events.event_label.duplicated().any():
        raise ValueError("Unexpected NGRIP event count or duplicate labels")
    # These ages are already BP1950. The common core explicitly uses u=anchor-age.
    events[likelihood.EVENT_AGE_COLUMN] = events.age_ka_bp
    events["event_id"] = "NGRIP:" + events.event_label
    events["segment_id"] = "NGRIP"
    support = likelihood.load_observation_segments().query("segment_id == 'NGRIP'").copy()
    return likelihood.build_context(support, events=events, catalogue_id=f"ngrip_{event_type}")


def load_age_realizations(context, n_realizations=N_REALIZATIONS):
    """Keep original IDs and label correspondence; never sort perturbed ages."""
    source = pd.read_csv(AGE_INPUT)
    if not 1 <= n_realizations <= len(source):
        raise ValueError("Requested age realizations exceed the available ensemble")
    columns = [f"age_ka_bp__{label}" for label in context.events.event_label]
    draws = source.loc[:, ["realization_id", *columns]].iloc[:n_realizations].copy()
    if draws.realization_id.isna().any() or draws.realization_id.duplicated().any():
        raise ValueError("Age realizations must have unique, nonmissing IDs")
    return draws, columns


def run_analysis(*, n_bootstrap=N_BOOTSTRAP, n_realizations=N_REALIZATIONS,
                 seed=SEED, n_workers=3, show_progress=True):
    summaries, age_summaries, boot_tables, age_tables = [], [], [], []
    fits, contexts = {}, {}
    for index, event_type in enumerate(EVENT_TYPES):
        context = load_catalogue(event_type)
        point = likelihood.fit_catalogue(context.events, context)
        if not point.summary["all_models_converged"] or not point.summary["likelihood_nesting_ok"]:
            raise RuntimeError(f"Invalid nominal {event_type} fit")
        if show_progress:
            print(f"\nNGRIP {event_type}: nominal p={point.summary['nominal_LR_p']:.6g}, "
                  f"phase={point.summary['pre_phase_preferred_deg']:.2f} deg", flush=True)

        # Simulate the fitted BG model at nominal ages, exactly as in the paper.
        replicates, failures = bootstrap.run_bootstrap(
            point, context, n_bootstrap=n_bootstrap, seed=seed + index,
            n_workers=n_workers, show_progress=show_progress)
        summary = bootstrap.build_summary(point, replicates, failures)
        summary.insert(0, "event_type", event_type)
        summary["seed"] = seed + index
        summaries.append(summary)
        boot_tables.append(replicates.assign(event_type=event_type))

        # Each direction uses the same source rows, including all invalid draws.
        # Shared scaling remains nominal; each age draw rebuilds anchor/history.
        draws, columns = load_age_realizations(context, n_realizations)
        age_results, diagnostics = age_sensitivity.fit_realizations(
            context, draws, columns, n_workers=n_workers, show_progress=show_progress)
        age_summary = age_sensitivity.summarize(age_results, point.summary)
        age_summary.insert(0, "event_type", event_type)
        age_summary["n_outside_support"] = age_results.invalid_reason.str.startswith("outside_").sum()
        age_summary["n_numerical_failures"] = diagnostics["n_numerical_failures"]
        age_summaries.append(age_summary)
        age_tables.append(age_sensitivity.compact_results(age_results).assign(event_type=event_type))
        fits[event_type], contexts[event_type] = point, context
    return dict(summary=pd.concat(summaries, ignore_index=True),
                age_summary=pd.concat(age_summaries, ignore_index=True),
                bootstrap_replicates=pd.concat(boot_tables, ignore_index=True),
                age_realizations=pd.concat(age_tables, ignore_index=True),
                fits=fits, contexts=contexts, n_workers=n_workers)


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
        "nominal_coefficients.csv": pd.concat([
            likelihood.coefficient_table(fit).assign(event_type=kind)
            for kind, fit in result["fits"].items()], ignore_index=True),
        "nominal_events.csv": pd.concat([
            fit.design.all_events[["event_label", "source_event_label", likelihood.EVENT_AGE_COLUMN,
                                   "event_role", "event_type"]]
            for fit in result["fits"].values()], ignore_index=True),
    }
    parameters = []
    for kind, context in result["contexts"].items():
        row = result["summary"].set_index("event_type").loc[kind]
        values = dict(model_version=likelihood.MODEL_VERSION, age_epoch="BP1950",
            history_tau_kyr=context.history_tau_ka, history_domain="beta_H <= 0",
            history_events="same transition type only", initial_unobserved_history=context.initial_history,
            quadrature_order=context.quadrature_order, seed=int(row.seed), n_bootstrap=int(row.n_bootstrap),
            n_workers=result["n_workers"],
            n_age_realizations=len(result["age_realizations"].query("event_type == @kind")),
            reduced_terms=" + ".join(context.reduced_terms), full_terms=" + ".join(context.full_terms),
            age_rejections="outside observation support; no truncation or replacement",
            p_rule="bootstrap: (1+exceedances)/(B+1); age fraction: nominal LR p<0.05 among valid fits")
        values.update(vars(context.segments["NGRIP"]))
        for forcing, scaling in context.scaling.items():
            values.update({f"{forcing}_{key}": value for key, value in scaling.items()})
        parameters.extend(dict(event_type=kind, parameter=key, value=value) for key, value in values.items())
    paths = [EVENT_INPUT, AGE_INPUT, likelihood.OBSERVATION_SEGMENTS_CSV, Path(__file__),
             Path(bootstrap.__file__), Path(likelihood.__file__), Path(age_sensitivity.__file__),
             PROJECT_ROOT / "toolbox/point_process.py", PROJECT_ROOT / "toolbox/model_stats.py",
             PROJECT_ROOT / "toolbox/orbital_phase.py", PROJECT_ROOT / "toolbox/event_inputs.py",
             PROJECT_ROOT / "toolbox/project_config.py",
             PROJECT_ROOT / "data/raw/lr04.xlsx", PROJECT_ROOT / "data/raw/composite_co2.xlsx",
             PROJECT_ROOT / "data/raw/pre_1000_60_inter100.txt"]
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
        anchor = result["contexts"][kind].segments["NGRIP"].anchor_age_kyr_bp
        lines += [kind.upper(),
            f"Inventory={int(s.n_source_events)}; fitted={int(s.n_response_events)}; anchor={anchor:.3f} kyr BP; response duration={s.response_exposure_kyr:.3f} kyr.",
            f"G={s.gain_bits_per_event:.6f} bits/event; LR={s.LR_statistic:.6f}; nominal p={s.nominal_LR_p:.6g}; preferred phase={s.pre_phase_preferred_deg:.2f} deg; phase rate ratio={s.pre_phase_rate_ratio_max_vs_min:.3f}.",
            f"Bootstrap B={int(s.n_bootstrap)}; exceedances={s.n_bootstrap_exceeding_or_equal_observed:g}; p={s.empirical_p_plus_one:.6g}; 95% Monte Carlo interval=[{s.empirical_p_ci95_low:.6g}, {s.empirical_p_ci95_high:.6g}]; failed={int(s.n_failed_replicates)}; seed={int(s.seed)}.",
            f"Age draws={int(a.n_realizations)}; valid={int(a.n_valid)}; outside support={int(a.n_outside_support)}; numerical failures={int(a.n_numerical_failures)}. Nominal p<0.05 in {int(a.n_nominal_p_below_0p05)}/{int(a.n_valid)} ({100*a.fraction_nominal_p_below_0p05:.2f}%).",
            f"Age median G={a.gain_bits_per_event_median:.6f}, central 95%=[{a.gain_bits_per_event_q025:.6f}, {a.gain_bits_per_event_q975:.6f}]. Preferred-phase median={a.pre_phase_preferred_deg_median:.2f} deg, central 95%=[{a.pre_phase_preferred_deg_q025:.2f}, {a.pre_phase_preferred_deg_q975:.2f}] deg.", ""]
    lines.append("Interpretation: These exploratory directions share chronology and alternate in the same record. Similar phases are not independent evidence or a test of equality of phase responses. The rate is measured over total response time, not only time spent in the state from which the transition starts. Connection to an orbitally favorable oscillatory regime remains a physical interpretation, not identification of the trigger of individual warmings or coolings.")
    (notes_dir / f"{RUN_NAME}_Methods_and_results.txt").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--n-realizations", type=int, default=N_REALIZATIONS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--notes-dir", type=Path, default=NOTE_DIR)
    args = parser.parse_args()
    result = run_analysis(n_bootstrap=args.n_bootstrap, n_realizations=args.n_realizations,
                          seed=args.seed, n_workers=args.workers)
    save_results(result, args.output_dir)
    if result["summary"].n_failed_replicates.any() or result["age_summary"].n_numerical_failures.any():
        raise RuntimeError("Unresolved numerical fits saved for inspection; final notes withheld")
    write_notes(result, args.notes_dir)
    print(result["summary"][["event_type", "gain_bits_per_event", "pre_phase_preferred_deg",
                               "nominal_LR_p", "empirical_p_plus_one"]].to_string(index=False))
    print(result["age_summary"][["event_type", "n_valid", "fraction_nominal_p_below_0p05"]].to_string(index=False))


if __name__ == "__main__":
    main()
