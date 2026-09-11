# Orbital-event manuscript workspace

This folder contains the first complete English manuscript for the current
NGRIP–speleothem warming-event study and its Barker SpeleoAge comparison.
It follows the earlier project's flat LaTeX layout. The scientific text is a
GRL-style draft; author details and permanent archive identifiers remain to be
completed before submission.

The folder name `orbital_event_paper` reflects the study's orbital-event focus.
Paths recording the previous project's bibliography source retain that source's
original name.

| File | Purpose |
|---|---|
| `main.tex` / `main.pdf` | Main manuscript, four figures, key points, abstract and plain-language summary |
| `SI.tex` / `SI.pdf` | Seven methods/results sections, one MIS6 event-age table and eleven figures |
| `reference.bib` | Shared bibliography: 57 entries with stable citation keys |
| `reference_audit.md` | Bibliographic verification and source limitations |
| `literature_notes.md` | Claim checks, including the requested Thirumalai (2020) paper |
| `draft_review.md` | Completed scientific/build checks and remaining author revision items |
| `paper_plan.md` | Scientific narrative, figure decisions and original writing plan |
| `figure_manifest.csv` | Canonical research PDF → numbered manuscript figure mapping |
| `figures/` | All fifteen numbered PDF figures |
| `tables/` | MIS6 event-age table, companion CSV and input provenance |

Thirumalai (2020) is cited in the introduction and discussion. Its analysis of
millennial variability amplitude is distinguished from the present conditional
warming-onset rate. The current result is also placed alongside prior work on
state residence times and interstadial occurrence, without claiming that event
occurrence was previously unstudied.

## Build from saved research results

From the project root:

```bash
make -C orbital_event_paper pdf
make -C orbital_event_paper check
```

The PDF target runs these short export steps before compiling:

1. `paper_summary_figures.py` draws main Figs. 1, 2 and 4 from saved CSVs.
2. `paper_tables.py` exports the 21 MIS6 event-age assignments in Table S1.
3. `Figure_PI_bootstrap.py` joins the two bootstrap PDFs side by side;
   `Figure_climate_phase_interaction.py` stacks the two LR04-interaction PDFs.
4. `paper_figure_export.py` copies all fifteen mapped PDFs into `figures/`.

These commands do not rerun model fits, event detection or Monte Carlo
simulations. Fig. 3 and the other SI figures use their existing canonical research
PDFs. The compositions preserve the underlying event ages, coefficients,
bootstrap results and chronology summaries. Source hashes are recorded beside
the generated outputs.

To compile only after editing prose:

```bash
cd orbital_event_paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error SI.tex
```

Dependencies are the project's Python environment and a LaTeX installation
with `latexmk`, `natbib`, `agu.bst`, and standard table packages. The drafts
use `article` with the AGU bibliography style, following the previous workspace;
the final submission template and review formatting can be applied after author
revision. `make clean` removes auxiliary LaTeX files and retains the PDFs.

## Figure and table editing

Edit `../paper_summary_figures.py` for the three new main compositions; source
captions and transformation notes are in
`../experiment_note/paper_summary_figures_Caption.txt` and the companion
`Methods_and_results` file. Main-manuscript captions are self-contained edited
versions and must be checked when a figure changes.

Absolute-age axes in all current research figures run from older ages at the
left to younger ages at the right. Phase, age-offset, duration and statistical
axes retain their own conventions. Fig. 1 shows three orbital inputs above
precession and all three event definitions, then LR04 (reversed vertical axis)
and CO2. Its caption explains the independent orbital scales and overlapping
Barker picks. Historical archive figures retain their original presentation.

Fig. 2 has two three-panel groups. Panels a/d show full-minus-reduced fitted
rates using observed histories, b/e show 20-degree event-phase counts, and c/f
show conditional phase multipliers with compact statistical labels in the upper
margin. Positive differences mean a higher full-model rate; they are not local
PI or isolated phase contributions. Both Barker definitions appear throughout.
Event ticks sit within the same axes, between the curves and x-axis; vertical
tick offsets have no rate meaning. The shared polar count-2 ring has no numeral. Primary
rates come from the matched orbital-driver export and are checked against the
main coefficients, event total, exposure and LR. These are in-sample fits;
bootstrap calibration remains in the text and Fig. S8.

Catalogue colors are shared through `../toolbox/catalogue_colors.py`: NGRIP–MIS6
blue (`#3E6C8E`), Barker varying threshold rose (`#CC6677`), and Barker fixed
threshold green (`#228833`). Figs. 1–2 and the standalone main analyses use this
palette; dashed green curves also distinguish the fixed-threshold fit.

Fitted phase-response curves use `../toolbox/phase_response_plotting.py` to
distinguish precession-index minima/maxima from fitted event-rate peaks. They
show point-age preferred phases and maximum/minimum rate ratios; LR04-specific
values appear in the interaction legends. The curves and uncertainty bands
retain their original numerical definitions.

Selected existing research exporters also copy their PDF automatically after
saving. The two bootstrap exporters and the shared LR04-interaction exporter
now call their PDF compositor after saving either source plot. The compositor
then calls the paper exporter, completing this chain:

```text
analysis script -> source PDF -> Figure_<description>.py -> paper_figure_export.py
```

The two source PDFs must be available before their combined figure can be
built. Their descriptive filenames stay fixed; manuscript numbers are assigned
only in `figure_manifest.csv`. To compose and synchronize without refitting, run
`make -C orbital_event_paper sync`. To copy already composed PDFs or check them, run:

```bash
python paper_figure_export.py
python paper_figure_export.py --status
python paper_figure_export.py --check
```

Run those commands from the project root. Matching uses complete canonical
source paths, so archive or temporary figures cannot replace the active source.
The exporter never deletes unlisted manuscript files. The new automatic chain
applies to the two combined SI figures. Main summary compositions still require
their composition script or `make pdf` after a component analysis changes.

Edit `../paper_tables.py` for table content/formatting; generated
`tables/TableS1.tex` will be overwritten. The printed table contains only the
21 speleothem event times assigned by this study. Published NGRIP/Barker ages,
epoch audits, bootstrap replicates and full sensitivity results remain in the
original research `data/` folders; they are not duplicated as manuscript tables.

## Author revision

The first draft keeps the existing scientific scope and results. It does not
add lag searches, fixed-threshold bootstraps, or new tests. Chronology working
ranges are distinguished from sampling confidence regions and empirical null
probabilities. The Barker comparison is not described as wholly independent.

Before submission, supply author names/affiliations, contributions, funding and
other declarations, and establish real public code/data archive identifiers.
The exact epoch of Barker's SpeleoAge column remains a documented working
assumption. Those administrative and provenance limits are not filled with
invented information. The previous Rousseau catalogue and results are excluded
from the manuscript.
