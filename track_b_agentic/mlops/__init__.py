"""W17 MLOps layer for the W15/W16 agentic assistant.

Modules:

* ``common``         - paths, the versioned prompt/config registry, pricing.
* ``runner``         - runs the query set and records a structured per-step trace.
* ``run_experiment`` - one prompt version end to end, tracked in MLflow.
* ``regression``     - Evidently LLM-as-a-judge regression suite with a promotion gate.
* ``build_golden``   - turns a baseline run into golden-reference candidates.
* ``export_runs``    - MLflow runs -> results/mlflow_runs_comparison.{csv,md}.
* ``promote``        - marks a gated run as the production baseline.
* ``nightly``        - the plain-Python logic behind the Airflow DAG.
"""
