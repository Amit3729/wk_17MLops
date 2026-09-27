"""Turn a baseline run into golden-reference *candidates* for hand review.

    uv run python -m mlops.build_golden --run results/runs/<baseline_run>

Writes ``regression/golden_candidates.jsonl``. Nothing becomes a reference
until a person has read each candidate against the documentation, corrected it
where it is wrong or incomplete, and saved the approved set as
``regression/golden.jsonl`` with a ``review`` note per row (see
regression/README.md). The candidate file keeps the raw model output so the
edits stay auditable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import REGRESSION_DIR, write_jsonl


def candidates(run_dir: Path) -> list[dict]:
    rows = []
    for path in sorted((run_dir / "traces").glob("*.json")):
        trace = json.loads(path.read_text(encoding="utf-8"))
        rows.append({
            "case_id": trace["case"]["id"],
            "question": trace["case"]["question"],
            "reference": trace["final"]["answer"],
            "reference_citations": trace["final"]["citations"],
            "expected_documents": trace["case"]["expect_documents"],
            "expect_decline": trace["case"]["expect_decline"],
            "source_run": trace["run_name"],
            "harness_completed": trace["harness"]["completed"],
            "harness_reason": trace["harness"]["reason"],
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--out", default=str(REGRESSION_DIR / "golden_candidates.jsonl"))
    args = parser.parse_args()
    rows = candidates(Path(args.run))
    write_jsonl(Path(args.out), rows)
    print(f"wrote {len(rows)} candidates to {args.out}; review them, then save regression/golden.jsonl")


if __name__ == "__main__":
    main()
