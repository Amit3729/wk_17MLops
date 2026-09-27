"""Evidently regression suite: current responses vs approved golden references.

*Reference* is ``regression/golden.jsonl``: one approved answer per query,
generated from the best earlier version and then hand-reviewed and corrected
against the retrieved documentation. *Current* is whatever the prompt version
under test just produced for the same queries.

Three row-level checks, each an Evidently descriptor test:

1. **No contradiction** - LLM judge (``BinaryClassificationPromptTemplate``,
   reference-based, Groq ``openai/gpt-oss-120b``): does the answer contradict the
   reference? ``consistent`` / ``contradicts`` + reasoning.
2. **No information loss** - second LLM judge, same template type: does the
   answer lose a key fact the reference uses to answer the question?
   ``complete`` / ``incomplete`` + reasoning. Asking the brief's two questions
   separately means a failing case says *which* way it regressed.
3. **Cites sources / declines** - deterministic: an answerable question must
   cite at least one expected document in its ``citations``; an unanswerable one
   must be declined; arithmetic needs nothing.

``correctness_rate`` is the share of cases passing both judge checks.

``pct_tests_passed`` = passed row checks / all row checks. The **promotion gate**
is ``pct_tests_passed >= GATE`` (0.80); below it the run is a regression and is
not promoted. Evidently's own dataset-level test (mean per-row success rate >=
the gate) is part of the HTML report.

    uv run python -m mlops.regression --run results/runs/<run_name>
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from .common import GOLDEN_PATH, JUDGE_MODEL, JUDGE_PROVIDER, REPORTS_DIR, ROOT, read_jsonl

GATE = 0.80
# Two LLM-judge checks (the brief's "contradict" and "lose information" questions, asked
# separately so a failure says which one happened) plus one deterministic check.
CHECKS = ("No contradiction", "No information loss", "Cites sources")
JUDGE_CHECKS = CHECKS[:2]

_GRADING_CONTEXT = """You are grading an engineering assistant's ANSWER to a QUESTION about the CPython 3.14 documentation or this project's own engineering notes. Compare it with the approved REFERENCE answer, which was checked against the documentation.
Different wording, different examples, extra detail and different formatting are all fine. Ignore the "Citations:" line when judging facts."""

CONTRADICTION_CRITERIA = _GRADING_CONTEXT + """

Question to decide: does the ANSWER CONTRADICT the REFERENCE?
The ANSWER is "contradicts" when it states something the REFERENCE shows is wrong: a wrong function, exception, number, default or behaviour, or a claim that directly conflicts with the REFERENCE. It is also "contradicts" when the REFERENCE says the documents do not contain the answer and the ANSWER nevertheless gives a confident substantive answer.
The ANSWER is "consistent" when nothing in it conflicts with the REFERENCE. An answer that is incomplete but not wrong is "consistent" (completeness is graded separately). An empty or error answer states nothing wrong, so it is "consistent".

QUESTION: {question}

REFERENCE: {reference}"""

COMPLETENESS_CRITERIA = _GRADING_CONTEXT + """

Question to decide: does the ANSWER LOSE INFORMATION that the REFERENCE uses to answer the QUESTION?
The ANSWER is "incomplete" when it omits a key fact the REFERENCE relies on, leaves part of a multi-part QUESTION unanswered, is empty, is an error message, or never answers what the QUESTION asks.
The ANSWER is "complete" when every key fact the REFERENCE needs to answer the QUESTION is present, even if worded differently. Minor supporting detail from the REFERENCE may be missing. When the REFERENCE says the documents do not contain the answer, the ANSWER is "complete" if it also declines or says the information is not available.

QUESTION: {question}

REFERENCE: {reference}"""


def _judge_options():
    """Evidently LLM options for the judge provider, with a rate limit that fits Groq's free tier."""
    from evidently.llm.utils import wrapper
    from evidently.llm.utils.wrapper import RateLimits

    if JUDGE_PROVIDER == "groq":
        options_type = getattr(wrapper, "GroqOptions")  # generated per litellm provider at import
        # Evidently's litellm wrapper passes no extra arguments by default. gpt-oss
        # is a reasoning model: keep its reasoning short, and let litellm retry a
        # 429 with backoff (Evidently's own retry loop does not wait between tries).
        options_type.get_additional_kwargs = lambda self: {"reasoning_effort": "low", "num_retries": 6}
        # Throttle by requests, not tokens. Evidently's limiter estimates a request as
        # len(prompt) characters plus a 100,000-token output guess until it has history,
        # so any `tpm` cap below that blocks each call behind a full one-minute window
        # (16 judge calls ran for 30+ minutes). One judge call is ~1K real tokens, so
        # JUDGE_RPM=6 stays under Groq's 8K tokens/minute; litellm retries any 429.
        return options_type(
            api_key=os.getenv("GROQ_API_KEY") or None,
            limits=RateLimits(rpm=int(os.getenv("JUDGE_RPM", "6"))),
        )
    return None


def citation_ok(trace: dict[str, Any]) -> bool:
    case = trace["case"]
    harness = trace["harness"]
    if case["expect_decline"]:
        return bool(harness.get("declined"))
    if not case["expect_documents"]:
        return True  # arithmetic: nothing to cite
    citations = trace["final"]["citations"]
    return any(expected in str(citation) for expected in case["expect_documents"] for citation in citations)


def build_frame(traces: list[dict[str, Any]], golden: list[dict[str, Any]]) -> pd.DataFrame:
    by_id = {row["case_id"]: row for row in golden}
    rows = []
    for trace in traces:
        case_id = trace["case"]["id"]
        if case_id not in by_id:
            continue
        final = trace["final"]
        citations = ", ".join(final["citations"]) or "(none)"
        rows.append({
            "case_id": case_id,
            "question": trace["case"]["question"],
            "reference": by_id[case_id]["reference"],
            "response": f"{final['answer']}\n\nCitations: {citations}",
            "cites_ok": citation_ok(trace),
        })
    missing = sorted(set(by_id) - {row["case_id"] for row in rows})
    if missing:
        raise SystemExit(f"current run is missing golden cases: {missing}")
    return pd.DataFrame(rows)


def evaluate(traces: list[dict[str, Any]], html_path: Path, golden_path: Path = GOLDEN_PATH) -> dict[str, Any]:
    """Run the suite, save the Evidently HTML, and return a JSON-able summary."""
    from dotenv import load_dotenv
    from evidently import DataDefinition, Dataset, Report
    from evidently.descriptors import ColumnTest, LLMEval, TestSummary
    from evidently.llm.templates import BinaryClassificationPromptTemplate
    from evidently.metrics import MeanValue
    from evidently.presets import TextEvals
    from evidently.tests import eq, gte

    load_dotenv(ROOT / ".env")
    golden = read_jsonl(golden_path)
    frame = build_frame(traces, golden)

    contradiction = BinaryClassificationPromptTemplate(
        criteria=CONTRADICTION_CRITERIA,
        target_category="contradicts",
        non_target_category="consistent",
        uncertainty="unknown",
        include_reasoning=True,
    )
    completeness = BinaryClassificationPromptTemplate(
        criteria=COMPLETENESS_CRITERIA,
        target_category="incomplete",
        non_target_category="complete",
        uncertainty="unknown",
        include_reasoning=True,
    )
    columns = {"question": "question", "reference": "reference"}
    descriptors = [
        LLMEval("response", provider=JUDGE_PROVIDER, model=JUDGE_MODEL, template=contradiction,
                additional_columns=columns, alias="Contradiction",
                tests=[eq("consistent", column="Contradiction", alias=CHECKS[0])]),
        LLMEval("response", provider=JUDGE_PROVIDER, model=JUDGE_MODEL, template=completeness,
                additional_columns=columns, alias="Completeness",
                tests=[eq("complete", column="Completeness", alias=CHECKS[1])]),
        ColumnTest("cites_ok", eq(True), alias=CHECKS[2]),
        TestSummary(success_all=True, success_rate=True, alias="Tests"),
    ]
    dataset = Dataset.from_pandas(
        frame,
        data_definition=DataDefinition(
            id_column="case_id",
            text_columns=["question", "reference", "response"],
            categorical_columns=["cites_ok"],
        ),
        descriptors=descriptors,
        options=_judge_options(),
    )
    report = Report(
        [TextEvals(), MeanValue(column="Tests_success_rate", tests=[gte(GATE)])],
        include_tests=True,
    )
    snapshot = report.run(dataset, None)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(html_path))

    scored = dataset.as_dataframe()
    per_case = []
    for _, row in scored.iterrows():
        contradiction_ok = row.get("Contradiction") == "consistent"
        completeness_ok = row.get("Completeness") == "complete"
        per_case.append({
            "case_id": row["case_id"],
            "contradiction": row.get("Contradiction"),
            "contradiction_reasoning": row.get("Contradiction reasoning"),
            "completeness": row.get("Completeness"),
            "completeness_reasoning": row.get("Completeness reasoning"),
            # Both judge checks pass: the answer neither contradicts nor loses reference information.
            "correctness": "correct" if contradiction_ok and completeness_ok else "incorrect",
            "cites_ok": bool(row["cites_ok"]),
            "checks_passed": int(sum(bool(row[check]) for check in CHECKS)),
            "all_passed": bool(row["Tests_success_all"]),
            "response": row["response"],
            "reference": row["reference"],
        })
    total_checks = len(per_case) * len(CHECKS)
    passed_checks = sum(case["checks_passed"] for case in per_case)
    pct = passed_checks / total_checks if total_checks else 0.0
    evidently_tests = [
        {"name": test["name"], "status": str(getattr(test["status"], "value", test["status"])),
         "description": test.get("description", "")}
        for test in snapshot.dict()["tests"]
    ]
    return {
        # Relative to track_b_agentic, so the committed JSON is machine-independent.
        "html": _relative(html_path),
        "golden_source_runs": sorted({row.get("source_run", "") for row in golden}),
        "n_cases": len(per_case),
        "pct_tests_passed": pct,
        "pct_cases_passed": sum(case["all_passed"] for case in per_case) / len(per_case) if per_case else 0.0,
        "correctness_rate": sum(case["correctness"] == "correct" for case in per_case) / len(per_case),
        "no_contradiction_rate": sum(case["contradiction"] == "consistent" for case in per_case) / len(per_case),
        "completeness_rate": sum(case["completeness"] == "complete" for case in per_case) / len(per_case),
        "citation_rate": sum(case["cites_ok"] for case in per_case) / len(per_case),
        "gate_threshold": GATE,
        "gate_passed": pct >= GATE,
        "evidently_tests": evidently_tests,
        "evidently_tests_failed": sum(test["status"] == "FAIL" for test in evidently_tests),
        "per_case": per_case,
    }


def _relative(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def load_traces(run_dir: Path) -> list[dict[str, Any]]:
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted((run_dir / "traces").glob("*.json"))]


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-run the Evidently regression suite on a stored run.")
    parser.add_argument("--run", required=True, help="results/runs/<run_name>")
    parser.add_argument("--log-mlflow", action="store_true", help="log the result into the run's MLflow run")
    args = parser.parse_args()
    run_dir = Path(args.run)
    summary = evaluate(load_traces(run_dir), REPORTS_DIR / f"evidently_{run_dir.name}.html")
    (run_dir / "regression.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in {"per_case", "evidently_tests"}}, indent=2))
    if args.log_mlflow:
        from .run_experiment import log_regression_to_existing_run

        log_regression_to_existing_run(run_dir, summary)


if __name__ == "__main__":
    main()
