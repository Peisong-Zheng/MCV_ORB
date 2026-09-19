# Orbital-event manuscript workspace

This folder contains the English manuscript and Supporting Information for the
continuous-time NGRIP–speleothem warming-event study and the separate Barker
SpeleoAge comparison. The scientific draft uses a GRL-style `article` layout
with the AGU bibliography style. Author declarations and permanent public
archive identifiers remain to be completed.

| File | Purpose |
|---|---|
| `main.tex`, `main.pdf` | Main manuscript and four figures |
| `SI.tex`, `SI.pdf` | Detailed methods, sensitivity results, one event-age table and supplementary figures |
| `reference.bib` | Shared bibliography and stable citation keys |
| `reference_audit.md`, `literature_notes.md` | Source verification and claim checks |
| `paper_plan.md` | Scientific narrative and allocation of evidence |
| `draft_review.md` | Current review status and author revision items |
| `figure_manifest.csv` | Canonical source PDF to manuscript figure mapping |
| `figures/`, `tables/` | Synchronized figures and CSV snapshots for manual table checks |

## Scientific configuration

The event-rate model uses actual event times, inhibitory exponential history
with fixed tau = 1.5 kyr, background climate, and an optional two-term
precession-phase contribution. Each segment conditions on its exact oldest
event. The primary fit has 53 response events across 165.058 kyr; Barker
varying/fixed fits have 69/58 response events across 396.464264/392.245696 kyr.
All inventory events remain in descriptive plots. Source chronology ensembles
are unchanged; the model-dependent fits and simulations have been recomputed.

G is the log-likelihood gain per response event, in bits/event. The phase test
uses a reduced-model bootstrap. History utility is tested conditional on
phase; full-model checks use time rescaling with simulation/refitting
calibration. Chronology ranges, sampling confidence-region projections,
combined working ranges and deletion stress tests have separate meanings.
The new history, fit and deletion diagnostics are reported concisely in text
and saved tables; they do not add default manuscript figures.

Thirumalai (2020) motivates orbital modulation of millennial variability in
the introduction, while its amplitude question is distinguished from event
occurrence. Barker's shared chronology dependencies remain disclosed; its
BP1950 convention is supported by the source chronology audit in
`reference_audit.md`. The current study neither includes
the discarded Rousseau catalogue nor claims held-out predictive performance.

## Build from saved results

From the project root:

```bash
make -C orbital_event_paper pdf
make -C orbital_event_paper check
```

The build draws main Figures 1, 2 and 4 with `paper_summary_figures.py`,
composes the
paired bootstrap and LR04-interaction PDFs, copies the manifest-listed figures,
and compiles both TeX files. It does not refit models, generate new event-age
ensembles or rewrite prose. After a result changes, numerical statements and
captions must be checked against the saved CSVs before building.

To compile prose changes without redrawing figures:

```bash
cd orbital_event_paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error SI.tex
```

The build requires the project Python environment and a LaTeX installation
with `latexmk`, `natbib`, `agu.bst` and standard table packages. `make clean`
removes auxiliary TeX files and retains the PDFs. Figure synchronization can
also be checked with `python paper_figure_export.py --check` from the root.

## Figure maintenance

`figure_manifest.csv` is authoritative for manuscript numbering. Source names
remain descriptive. Main Figure 1 displays orbital drivers, events on
precession, LR04 and CO2. Figure 2 groups fitted full-minus-reduced rates,
descriptive phase counts and conditional phase multipliers for each catalogue.
Its rates are evaluated from saved continuous-model coefficients and observed
histories. Their difference is not local G or an isolated phase contribution.
Figure 3 distinguishes the three effect-uncertainty constructions; Figure 4
compares precession with other orbital predictors.

All current absolute-age axes put younger ages on the right; LR04's vertical
axis is reversed in Figure 1. The catalogue palette is shared through
`toolbox/catalogue_colors.py`: primary blue, Barker varying rose and Barker
fixed green. Phase multipliers mark orbital-index minima/maxima separately
from fitted event-rate peaks. Captions must name what bands and reference
lines represent.

Selected source exporters automatically follow this chain:

```text
analysis script -> source PDF -> Figure_<description>.py -> paper_figure_export.py
```

Both source PDFs must exist before composition. Staged review runs defer paper
export. `make sync` composes and copies existing source PDFs without fitting.
Main summary compositions require `paper_summary_figures.py` or `make pdf`
after their saved inputs change. Table S1 combines the 21 MIS6 ages assigned
in this study with their minimum and maximum timing offsets across the nine
definition settings, including both MF and Sofular. The table is inline in
`SI.tex` and maintained manually. Automated exports update figures only and
never rewrite manuscript TeX or tables. The CSV in `tables/` is a snapshot for
manual checking; original analysis outputs remain the source of numerical
results. The retired exporter is in `archive/table_export_2026-09-19/` at the
project root. Published inventories and exhaustive fit tables remain in the
research data folders.

## Validation and remaining work

The [migration audit](../docs/reviews/continuous-time-migration-2026-09-12.md)
records current numerical, figure and manuscript validation. Earlier completed
builds do not certify this revision. The complete previous project is frozen
in `../archive/pre_continuous_time_2026-09-12/`.

Before submission, supply author names, affiliations, contributions, funding,
declarations and real public code/data identifiers. Scientific prose follows
Peisong's scientific writing style: physical question, concise evidence,
explicit uncertainty definitions and restrained interpretation.
