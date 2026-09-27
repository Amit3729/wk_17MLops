# Evaluation report

Generated 2026-09-26 16:15 UTC · provider `groq` · model `openai/gpt-oss-120b` · top_k 4 · researcher top_k 5 · researcher turns 3

## Summary by configuration

| configuration | n | completion | tool-calls | mean turns | mean tokens | total tokens | hard | soft | cascading |
|---|---|---|---|---|---|---|---|---|---|
| `v3` | 8 | 88% | 100% | 2.1 | 3,473 | 27,784 | 0 | 1 | 0 |

## Completion by query kind

| kind | `v3` |
|---|---|
| arithmetic | 1/1 |
| multi_hop | 1/2 |
| single_hop | 4/4 |
| unanswerable | 1/1 |

## Per-query results

| query | config | ran as | completed | tools ok | turns | tokens | confidence | failure |
|---|---|---|---|---|---|---|---|---|
| `arith-units` | `v3` | single | yes | yes | 2 | 1,895 | 1.00 | - |
| `dataclass-json` | `v3` | single | yes | yes | 2 | 2,106 | 0.97 | - |
| `int-error` | `v3` | single | yes | yes | 3 | 4,871 | 0.93 | - |
| `json-indent` | `v3` | single | yes | yes | 2 | 5,460 | 0.97 | - |
| `out-of-corpus-pricing` | `v3` | single | yes | yes | 2 | 1,690 | 0.90 | - |
| `project-serving` | `v3` | single | yes | yes | 2 | 4,516 | 0.94 | - |
| `read-lines` | `v3` | single | yes | yes | 2 | 5,403 | 0.97 | - |
| `tz-convert` | `v3` | single | no | yes | 2 | 1,843 | 0.97 | soft |

## Failure log

### hard (0)

*No usable answer was produced: an exception, or an empty response.*

None.

### soft (1)

*An answer was produced and looks fine, but is wrong on its own terms: the wrong tool, invalid arguments, a missing requirement, or an ungrounded citation.*

- **`tz-convert`** (`v3`, multi_hop) — cited ['library/datetime.html'], needed 2 of ['library/datetime.html', 'library/zoneinfo.html']
  - confidence 0.97, 2 turns, answer: "To make a datetime object timezone‑aware you must attach a tzinfo object (either a fixed‑offset subclass such as `datetime.timezone` or a zoneinfo object). The …"

### cascading soft (0)

*An upstream step failed or returned nothing usable, and the agent still produced a confident answer on top of it.*

None.
