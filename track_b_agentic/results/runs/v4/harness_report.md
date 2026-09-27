# Evaluation report

Generated 2026-09-26 18:42 UTC · provider `groq` · model `openai/gpt-oss-120b` · top_k 3 · researcher top_k 5 · researcher turns 3

## Summary by configuration

| configuration | n | completion | tool-calls | mean turns | mean tokens | total tokens | hard | soft | cascading |
|---|---|---|---|---|---|---|---|---|---|
| `v4` | 8 | 88% | 100% | 2.0 | 2,184 | 17,468 | 0 | 1 | 0 |

## Completion by query kind

| kind | `v4` |
|---|---|
| arithmetic | 1/1 |
| multi_hop | 1/2 |
| single_hop | 4/4 |
| unanswerable | 1/1 |

## Per-query results

| query | config | ran as | completed | tools ok | turns | tokens | confidence | failure |
|---|---|---|---|---|---|---|---|---|
| `arith-units` | `v4` | single | yes | yes | 2 | 3,440 | 1.00 | - |
| `dataclass-json` | `v4` | single | yes | yes | 2 | 1,883 | 0.97 | - |
| `int-error` | `v4` | single | yes | yes | 2 | 1,490 | 0.93 | - |
| `json-indent` | `v4` | single | yes | yes | 2 | 4,427 | 0.97 | - |
| `out-of-corpus-pricing` | `v4` | single | yes | yes | 2 | 1,451 | 0.90 | - |
| `project-serving` | `v4` | single | yes | yes | 2 | 1,401 | 0.93 | - |
| `read-lines` | `v4` | single | yes | yes | 2 | 1,606 | 0.97 | - |
| `tz-convert` | `v4` | single | no | yes | 2 | 1,770 | 0.97 | soft |

## Failure log

### hard (0)

*No usable answer was produced: an exception, or an empty response.*

None.

### soft (1)

*An answer was produced and looks fine, but is wrong on its own terms: the wrong tool, invalid arguments, a missing requirement, or an ungrounded citation.*

- **`tz-convert`** (`v4`, multi_hop) — cited ['library/datetime.html'], needed 2 of ['library/datetime.html', 'library/zoneinfo.html']
  - confidence 0.97, 2 turns, answer: "Create an aware datetime by passing a tzinfo (e.g. datetime.timezone.utc or a custom tzinfo subclass) to the constructor or by using datetime.now(tz). Then call…"

### cascading soft (0)

*An upstream step failed or returned nothing usable, and the agent still produced a confident answer on top of it.*

None.
