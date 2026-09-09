PYTHON ?= python
LATEXMK ?= latexmk

.PHONY: all pdf assets sync check clean
all: pdf

# These quick exports read saved results; they never refit scientific models.
assets:
	$(PYTHON) ../paper_summary_figures.py
	$(PYTHON) ../paper_tables.py
	$(PYTHON) ../paper_figure_export.py

sync:
	$(PYTHON) ../paper_figure_export.py

check:
	$(PYTHON) ../paper_figure_export.py --check

pdf: assets
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error main.tex
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error SI.tex

clean:
	$(LATEXMK) -c main.tex SI.tex
