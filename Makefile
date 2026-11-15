PYTHON ?= python
MEMCAP ?=
DATA_HOME ?= data

# Pin BLAS/OpenMP threads: deterministic timings and no oversubscription on small machines.
export OMP_NUM_THREADS ?= 2
export OPENBLAS_NUM_THREADS ?= 2
export MKL_NUM_THREADS ?= 2
export VECLIB_MAXIMUM_THREADS ?= 2

.PHONY: install data run test lint clean

install:
	$(PYTHON) -m pip install -e ".[dev]"

# Download (and cache) the UCI Bike Sharing hourly data into $(DATA_HOME)/.
data:
	$(PYTHON) -m demandcast.cli download --data-home $(DATA_HOME)

# Full reproduction: 12-origin backtest, settings A and B -> reports/metrics.json + figures.
run: data
	$(MEMCAP) $(PYTHON) -m demandcast.cli run --data-home $(DATA_HOME) --out reports

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check .

clean:
	rm -rf .pytest_cache .ruff_cache build dist src/*.egg-info
