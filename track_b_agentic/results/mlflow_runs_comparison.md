# MLflow run comparison

Exported from the `track-b-agentic-prompts` experiment (sqlite:///mlflow.db) by `mlops.export_runs`. One row per finished run, in start order. Percentages are over the fixed 8-query set; `pct_tests_passed` is the Evidently regression suite (gate >= 80%). Latencies `*_net_s` exclude time slept waiting on Groq's rate limits; `p50_wall_s` includes it. Cost uses Groq's published gpt-oss-120b price ($0.15 / $0.60 per million input / output tokens) and covers the assistant's calls only, not the judge. `golden` says how the regression references relate to the run: all were reviewed from v3's answers, so v3's own score is circular.

| run | prompt | mode | temp | top_k | max_iter | recover | completion | tool_ok | citation_hit | decline_acc | pct_tests_passed | no_contradict | complete | judge_both | cites_check | gate | mean_iter | pct_max_iter | pct_error | tok_in/q | tok_out/q | p50_net_s | p95_net_s | p50_wall_s | cost_usd | golden |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v1 | v1 | single | 0.2 | 4 | 4 | - | 0% | 0% | 0% | 0% | 6% | - | - | 0.0 | 12% | failed | 0.4 | 0% | 100% | 591 | 21 | 1.3 | 3.4 | 2.0 | 0.0008 | graded against references reviewed from v3 (a later version) |
| v2 | v2 | single | 0.2 | 4 | 4 | - | 50% | 38% | 33% | 100% | 50% | - | - | 0.5 | 50% | failed | 1.2 | 0% | 50% | 3,159 | 120 | 1.9 | 4.7 | 2.0 | 0.0044 | graded against references reviewed from v3 (a later version) |
| v3 | v3 | single | 0.2 | 4 | 4 | True | 88% | 100% | 83% | 100% | 100% | - | - | 1.0 | 100% | passed | 2.1 | 0% | 0% | 3,335 | 138 | 2.1 | 3.8 | 45.0 | 0.0047 | circular: golden references were reviewed from this run's own answers |
| v4 | v4 | single | 0.2 | 3 | 3 | True | 88% | 100% | 83% | 100% | 96% | 1.0 | 0.875 | 0.875 | 100% | passed | 2.0 | 0% | 0% | 2,094 | 89 | 1.8 | 2.7 | 44.8 | 0.0029 | graded against references reviewed from v3 |
