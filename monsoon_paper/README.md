# Manuscript workspace

This folder contains the first complete English manuscript for the current
NGRIP–speleothem warming-event study and its Barker SpeleoAge comparison.
It follows the earlier project's flat LaTeX layout. The scientific text is a
GRL-style draft; author details and permanent archive identifiers remain to be
completed before submission.

| File | Purpose |
|---|---|
| `main.tex` / `main.pdf` | Main manuscript, four figures, key points, abstract and plain-language summary |
| `SI.tex` / `SI.pdf` | Seven methods/results sections, five table groups and fourteen figures |
| `reference.bib` | Shared bibliography: 57 entries with stable citation keys |
| `reference_audit.md` | Bibliographic verification and source limitations |
| `literature_notes.md` | Claim checks, including the requested Thirumalai (2020) paper |
| `draft_review.md` | Completed scientific/build checks and remaining author revision items |
| `paper_plan.md` | Scientific narrative, figure decisions and original writing plan |
| `figure_manifest.csv` | Canonical research PDF → numbered manuscript figure mapping |
| `figures/` | All eighteen numbered PDF figures |
| `tables/` | Generated table fragments, CSVs and input provenance |

Thirumalai (2020) is cited in the introduction and discussion. Its analysis of
millennial variability amplitude is distinguished from the present conditional
warming-onset rate. The current result is also placed alongside prior work on
state residence times and interstadial occurrence, without claiming that event
occurrence was previously unstudied.

## Build from saved research results

From the project root:

```bash
make -C monsoon_paper pdf
make -C monsoon_paper check
```

The PDF target runs three short export steps before compiling:

1. `paper_summary_figures.py` draws main Figs. 1, 2 and 4 from saved CSVs.
2. `paper_tables.py` exports the five SI table groups and their underlying CSVs.
3. `paper_figure_export.py` copies all eighteen mapped PDFs into `figures/`.

These commands do not rerun model fits, event detection or Monte Carlo
simulations. Fig. 3 and all SI figures retain their existing canonical research
PDFs. The new compositions preserve the underlying event ages, coefficients,
bootstrap results and chronology summaries. Source hashes are recorded beside
the generated outputs.

To compile only after editing prose:

```bash
cd monsoon_paper
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

Selected existing research exporters also copy their PDF automatically after
saving. To copy without plotting, run:

```bash
python paper_figure_export.py
python paper_figure_export.py --status
python paper_figure_export.py --check
```

Run those commands from the project root. Matching uses complete canonical
source paths, so archive or temporary figures cannot replace the active source.
The exporter never deletes unlisted manuscript files. Changing a component
analysis does not update a composed main figure until its composition script or
`make pdf` is run.

Edit `../paper_tables.py` for table content/formatting; generated
`tables/TableS*.tex` files will be overwritten. The CSVs retain more source
fields than the compact printed tables where appropriate.

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
