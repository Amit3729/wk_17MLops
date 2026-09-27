# One command per step. Every target goes through `uv run`, so it uses whichever
# project environment uv manages in .venv.
PROMPT ?= v1

.PHONY: setup test docs ingest experiment golden regression export promote nightly nightly-dry mlflow-ui

setup:            ## install the locked environment
	uv sync --frozen

test:             ## unit tests (no network, no API key)
	uv run pytest -q

docs:             ## download the CPython 3.14 HTML docs (the corpus) into data/
	uv run python -m mlops.fetch_docs

ingest:           ## build / verify the Chroma index (skipped when the corpus is unchanged)
	uv run python -c "from app.config import get_settings; from app.retrieval import Retriever; print(Retriever(get_settings()).ingest())"

experiment:       ## one tracked run: make experiment PROMPT=v2
	uv run python -m mlops.run_experiment --prompt $(PROMPT)

golden:           ## golden-reference candidates from a baseline run: make golden RUN=results/runs/<name>
	uv run python -m mlops.build_golden --run $(RUN)

regression:       ## re-run the Evidently suite on a stored run: make regression RUN=results/runs/<name>
	uv run python -m mlops.regression --run $(RUN) --log-mlflow

export:           ## MLflow runs -> results/mlflow_runs_comparison.{csv,md}
	uv run python -m mlops.export_runs

promote:          ## gate-checked promotion: make promote RUN_NAME=<name>
	uv run python -m mlops.promote --run-name $(RUN_NAME)

nightly:          ## one nightly cycle without Airflow
	uv run python -m mlops.nightly

nightly-dry:      ## nightly decision logic only, no API calls
	uv run python -m mlops.nightly --dry-run

mlflow-ui:        ## browse runs, params, metrics, artifacts and traces
	uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
