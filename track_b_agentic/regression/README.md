# Regression set

`golden.jsonl` is the **reference**: one approved answer for each query in the
fixed 8-query set (`mlops/common.py: CASE_IDS`). After every prompt or config
change the same 8 queries are re-run (the **current** data) and compared with the
references by the Evidently suite in `mlops/regression.py`.

How the references were made:

1. The best version at the time (v3) was turned into candidates with
   `uv run python -m mlops.build_golden --run results/runs/v3`, which writes
   `golden_candidates.jsonl` (the raw model output, kept so the edits can be audited).
2. Each candidate was checked against the indexed documentation
   (`data/python-3.14-docs-html/...`, `data/serving.md`) and either approved or
   corrected. The `review` field on every row says which and why. Approved
   (facts unchanged): json-indent, dataclass-json, arith-units (checked by hand),
   out-of-corpus-pricing (reworded, same meaning). Corrected or extended:
   int-error, read-lines, tz-convert, project-serving.
3. A reference states the facts a correct answer must contain, not a style to
   copy, so the judge compares substance rather than wording.

**Provenance caveat.** Every row has `source_run = "v3"`. v3's own score against
this set is circular: it is partly graded against its own answers. v1 and v2 are
graded against references written from a later version. The non-circular numbers
are the ones for runs made after the set was frozen: `v3_rerun` (same config, fresh
responses) and `v4`.

Row fields: `case_id`, `question`, `reference`, `expected_documents`,
`expect_decline`, `source_run`, `review`.
