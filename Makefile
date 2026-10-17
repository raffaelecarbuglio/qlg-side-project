.PHONY: test estimate smoke

PYTHON ?= python3

test:
	$(PYTHON) -m unittest discover -s tests -v

estimate:
	$(PYTHON) -m qlg estimate configs/benchmark.json

smoke:
	$(PYTHON) -m qlg prepare configs/smoke.json runs/smoke
	$(PYTHON) -m qlg run runs/smoke --threads 2
	$(PYTHON) -m qlg analyze runs/smoke --reference
