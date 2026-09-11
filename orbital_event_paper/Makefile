PYTHON ?= python
LATEXMK ?= latexmk

.PHONY: all pdf assets compose sync check clean
all: pdf

# These quick exports read saved results; they never refit scientific models.
assets:
	$(PYTHON) ../paper_summary_figures.py
	$(PYTHON) ../paper_tables.py
	$(MAKE) compose
	$(PYTHON) ../paper_figure_export.py

compose:
	$(PYTHON) ../Figure_PI_bootstrap.py
	$(PYTHON) ../Figure_climate_phase_interaction.py

sync: compose
	$(PYTHON) ../paper_figure_export.py

check:
	$(PYTHON) ../paper_figure_export.py --check

pdf: assets
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error main.tex
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error SI.tex

clean:
	$(LATEXMK) -c main.tex SI.tex
