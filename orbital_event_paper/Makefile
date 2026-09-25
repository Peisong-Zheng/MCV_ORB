PYTHON ?= python
LATEXMK ?= latexmk

.PHONY: all pdf assets compose sync check clean
all: pdf

# Update figures only. Manuscript tables are maintained manually in SI.tex.
assets:
	$(PYTHON) ../paper_summary_figures.py
	$(MAKE) compose
	$(PYTHON) ../paper_figure_export.py

compose:
	$(PYTHON) ../Figure_orbital_mechanism.py

sync: compose
	$(PYTHON) ../paper_figure_export.py

check:
	$(PYTHON) ../paper_figure_export.py --check

pdf: assets
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error main.tex
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error SI.tex

clean:
	$(LATEXMK) -c main.tex SI.tex
