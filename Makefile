PYTHON ?= python3
PYTHONPATH := .

.PHONY: all install test reproduce figures paper clean help

help:
	@echo "Targets:"
	@echo "  install     -- pip install -e . in editable mode"
	@echo "  test        -- run the unit-test suite"
	@echo "  reproduce   -- run every paper experiment (writes results/*.json)"
	@echo "  figures     -- regenerate paper/figures/*.{png,pdf} from results"
	@echo "  paper       -- build paper/thgtm.pdf"
	@echo "  all         -- reproduce + figures + paper"
	@echo "  clean       -- delete build artefacts (keeps results/ and figures/)"

install:
	pip install -e .

test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest -q tests/

reproduce: results/noisy_xor.json results/temporal_xor.json results/depth_n_parity.json results/trajectory_verification.json

results/noisy_xor.json: experiments/noisy_xor.py thgtm/*.py
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) experiments/noisy_xor.py

results/temporal_xor.json: experiments/temporal_xor.py thgtm/*.py
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) experiments/temporal_xor.py

results/depth_n_parity.json: experiments/depth_n_parity.py thgtm/*.py
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) experiments/depth_n_parity.py

results/trajectory_verification.json: experiments/trajectory_verification.py thgtm/*.py
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) experiments/trajectory_verification.py

figures: reproduce
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/make_figures.py

paper: figures
	cd paper && pdflatex -interaction=nonstopmode thgtm.tex \
	  && bibtex thgtm \
	  && pdflatex -interaction=nonstopmode thgtm.tex \
	  && pdflatex -interaction=nonstopmode thgtm.tex

all: paper

clean:
	rm -f paper/*.aux paper/*.log paper/*.bbl paper/*.blg paper/*.out paper/*.toc
	find . -type d -name '__pycache__' -prune -exec rm -rf {} +
	find . -type d -name '.pytest_cache' -prune -exec rm -rf {} +
