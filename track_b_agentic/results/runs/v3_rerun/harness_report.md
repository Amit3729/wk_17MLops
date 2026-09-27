# Evaluation report

Generated 2026-09-26 21:39 UTC · provider `groq` · model `openai/gpt-oss-120b` · top_k 4 · researcher top_k 5 · researcher turns 3

## Summary by configuration

| configuration | n | completion | tool-calls | mean turns | mean tokens | total tokens | hard | soft | cascading |
|---|---|---|---|---|---|---|---|---|---|
| `v3_rerun` | 8 | 88% | 100% | 2.2 | 4,015 | 32,120 | 0 | 1 | 0 |

## Completion by query kind

| kind | `v3_rerun` |
|---|---|
| arithmetic | 1/1 |
| multi_hop | 1/2 |
| single_hop | 4/4 |
| unanswerable | 1/1 |

## Per-query results

| query | config | ran as | completed | tools ok | turns | tokens | confidence | failure |
|---|---|---|---|---|---|---|---|---|
| `arith-units` | `v3_rerun` | single | yes | yes | 2 | 1,894 | 1.00 | - |
| `dataclass-json` | `v3_rerun` | single | yes | yes | 2 | 2,106 | 0.97 | - |
| `int-error` | `v3_rerun` | single | yes | yes | 4 | 15,758 | 0.90 | - |
| `json-indent` | `v3_rerun` | single | yes | yes | 2 | 5,461 | 0.97 | - |
| `out-of-corpus-pricing` | `v3_rerun` | single | yes | yes | 2 | 1,690 | 0.90 | - |
| `project-serving` | `v3_rerun` | single | yes | yes | 2 | 1,589 | 0.93 | - |
| `read-lines` | `v3_rerun` | single | yes | yes | 2 | 1,777 | 0.97 | - |
| `tz-convert` | `v3_rerun` | single | no | yes | 2 | 1,845 | 0.97 | soft |

## Failure log

### hard (0)

*No usable answer was produced: an exception, or an empty response.*

None.

### soft (1)

*An answer was produced and looks fine, but is wrong on its own terms: the wrong tool, invalid arguments, a missing requirement, or an ungrounded citation.*

- **`tz-convert`** (`v3_rerun`, multi_hop) — cited ['library/datetime.html'], needed 2 of ['library/datetime.html', 'library/zoneinfo.html']
  - confidence 0.97, 2 turns, answer: "Use a tzinfo object (e.g. an instance of ``datetime.timezone`` or a custom ``tzinfo`` subclass) when you create the ``datetime``. That makes the object *aware*.…"

### cascading soft (0)

*An upstream step failed or returned nothing usable, and the agent still produced a confident answer on top of it.*

None.
