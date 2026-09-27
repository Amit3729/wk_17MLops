# wk_17MLops: MLOps for a churn model and an agentic assistant

Week 17 applies three MLOps disciplines, **uv** (reproducible environments), **MLflow**
(experiment tracking) and **Evidently AI** (monitoring and regression testing), plus an
optional **Airflow** DAG, to two systems. Each track lives on its own branch, in its own
folder, with its own `pyproject.toml`, `uv.lock` and README.

| Track | Branch | Folder | What it is |
| --- | --- | --- | --- |
| **A: Data Science MLOps** | [`track-a-data-science`](../../tree/track-a-data-science/track_a_churn) | `track_a_churn/` | Telco Customer Churn classifier: 6 tracked model runs, registry Staging → Production, FastAPI serving, Evidently data/target drift with custom metrics, drift-triggered retraining DAG |
| **B: Agentic AI MLOps** | [`track-b-agentic-ai`](../../tree/track-b-agentic-ai/track_b_agentic) | `track_b_agentic/` | The W15 RAG assistant + W16 agentic feature: versioned prompts driven by step-level traces, MLflow comparison, Evidently LLM-as-a-judge regression suite with a promotion gate, nightly regression DAG |

Each folder's README is the full write-up, with setup, one command per step, and the required
sections: (a) environment and reproducibility, (b) experiment tracking strategy, (c) monitoring
and drift/regression strategy, and (d) orchestration.

## Quick start

The only prerequisite is [uv](https://docs.astral.sh/uv/getting-started/installation/); it
downloads Python 3.12 itself.

```bash
# Track A
git clone -b track-a-data-science https://github.com/Amit3729/wk_17MLops.git wk17-a
cd wk17-a/track_a_churn && uv sync --frozen && uv run churn-pipeline

# Track B (needs a free Groq API key in .env)
git clone -b track-b-agentic-ai https://github.com/Amit3729/wk_17MLops.git wk17-b
cd wk17-b/track_b_agentic && uv sync --frozen && cp .env.example .env   # set GROQ_API_KEY
make docs && make ingest && make test
```

## Headline results

**Track A.** `xgb_shallow_weighted` was registered: best F1 (0.621) and ROC-AUC (0.841) of 6 runs,
while the highest-accuracy run (0.804) caught only 57% of churners. Evidently flagged both
perturbed features and the target drift, and raised no alerts on a clean control set. The
retrained challenger beat the champion on recent data (F1 0.646 vs 0.622) and was promoted.

**Track B.** Four prompt versions, each fixing a failure found in the previous version's traces:
v1's answers never arrived (100% HTTP 400 from a pseudo-tool call), v2 stopped searching,
v3 fixed search, and v4 cut cost. v4 was promoted: 88% task completion, the same as v3, for 37%
fewer tokens, and 96% of regression checks passed (gate 80%).
