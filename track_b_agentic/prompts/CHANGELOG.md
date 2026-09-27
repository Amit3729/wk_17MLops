# Prompt / config changelog

Every version below was written *after* reading the previous version's traces
(`results/runs/<version>/traces/*.json`, representative ones copied to
`traces/<version>/`). Each entry quotes the steps that motivated it. The model
is fixed for all of them: `openai/gpt-oss-120b` on Groq, `reasoning_effort=low`,
single-agent W15 tool-calling loop (`agent_mode=single`), the same 8 queries
(`mlops/common.py: CASE_IDS`).

| version | prompt file | temperature | top_k | max_iterations | recover_tool_answers | what changed |
|---|---|---|---|---|---|---|
| v1 | `prompt_v1.txt` | 0.2 | 4 | 4 | false | baseline: the W15 `SYSTEM_PROMPT`, verbatim |
| v2 | `prompt_v2.txt` | 0.2 | 4 | 4 | false | + "how to deliver the answer" rule (no pseudo-tool) |
| v3 | `prompt_v3.txt` | 0.2 | 4 | 4 | **true** | + search-before-answer, targeted queries, max two searches; recover pseudo-tool answers |
| v4 | `prompt_v4.txt` | 0.2 | **3** | **3** | true | cost cut after the golden set was frozen: `top_k` 3 per search, short answers, one fewer turn |

---

## v1 → v2: the answer is being "sent" to a tool that does not exist

**What v1 did.** 8 of 8 queries terminated with `termination = "error"`,
task completion 0%, 4,896 tokens in total. Every one of the eight errors is the
same HTTP 400 from Groq:

```
Error code: 400 - {'error': {'message': "Tool call validation failed: tool call validation
failed: attempted to call tool 'json' which was not in request.tools",
'type': 'invalid_request_error', 'code': 'tool_use_failed', ...
```

The model's generation is attached to the rejection (`failed_generation`), and
it is a *finished, correct-looking answer*, wrapped as a call to a tool named
`json`:

- `v1/json-indent`, step 1 (no search at all; it answered from the
  pre-retrieved context):
  `{"name": "json", "arguments": {"answer": "The *indent* argument controls pretty-printing of the JSON output. If it is a non-negative integer, each level of nesting is indented ...`
- `v1/arith-units`, step 1 `call_tools` → `calculator({"expression": "16535 * 4096 / 1048576"})`,
  reasoning `"We need compute expression. Use calculator tool."`; step 2
  `error`, `failed_generation`:
  `{"name": "json", "arguments": {"answer":"64.58984375","confidence":1.0,"citations":[]}}`

**Diagnosis.** Not a knowledge or retrieval failure: the tool results were read
correctly, and the answer is right. The W15 prompt says *"Reply with a single
JSON object and nothing else"* while tools are offered. Groq rejects
`response_format=json_object` together with tools (HTTP 400 "json mode cannot be
combined with tool/function calling", found during the smoke test, so tool
turns run without JSON mode), and gpt-oss then routes "the JSON object" to a tool
call named `json`. The prompt was written for OpenAI/qwen, where JSON mode and
tools co-exist.

**Change (prompt only; config identical, so the prompt is the only variable).**
`prompt_v2.txt` adds:

> The only tools that exist are `search_knowledge_base`, `calculator` and
> `current_time`. There is no tool called `json`, `JSON`, `answer` or
> `final_answer`: never call a tool to hand over your answer. When you have what
> you need, stop calling tools and write the JSON object below directly as the
> text of your message.

---

## v2 → v3: the rule halves the failure but does not remove it; and half the answers never search

**What v2 did.** Completion 0% → 50%, errors 8/8 → 4/8, 26,228 tokens.

1. *The pseudo-tool call survives in 4 of 8 runs* (`json-indent`, `read-lines`,
   `tz-convert`, `project-serving`), with the same 400 and the same shape:
   - `v2/tz-convert`, step 1 `search_knowledge_base({"query": "build a timezone-aware datetime and convert it to another timezone", "source": "python-3.14-docs-html", "top_k": 5})`,
     reasoning `"We need answer. Use docs. Provide steps. Cite docs."`; step 2
     `error`, `failed_generation` =
     `{'name': 'json', 'arguments': {'answer': "Create an aware datetime by supplying a tzinfo object (e.g. an instance of ``datetime.timezone`` or any ``tzinfo`` sub...`
   - `v2/project-serving`, step 1 searched `source: "root"` and got `serving.md` ×3;
     step 2 `failed_generation` = `{'name': 'json', 'arguments': {'answer': 'The project uses **vLLM** as its serving engine. ...`

   Four explicit sentences did not stop it, so this is a model/provider habit a
   prompt can reduce but not remove. The rejected generation is the finished
   answer, so the fix belongs in the loop: **`recover_tool_answers=true`**
   (`app/llm.py: _recover_answer_tool_call`) accepts a `tool_use_failed`
   generation *only* when the called name is not an offered tool and its
   arguments carry an `answer`. The trace keeps `recovered_from:
   "tool_use_failed: pseudo-tool 'json'"` on that step. It is a config flag, off
   for v1/v2, so their numbers stay as measured.

2. *4 of 8 answered on turn 1 from the pre-retrieved context without searching*
   (`json-indent`, `read-lines`, `dataclass-json`, `out-of-corpus-pricing`).
   - `v2/dataclass-json`, step 1 `final_answer`, reasoning
     `"Need answer: use dataclasses.asdict then json.dumps, or custom JSONEncoder. Provide answer with citation."`
     → harness: soft failure, *"never called ['search_knowledge_base']"*.

   This is the W16 finding (eval/ANALYSIS.md: the single loop "decides it
   already knows enough and answers on a single turn without calling
   search_knowledge_base at all") reproduced on a different model.

3. *When it did search, it searched vaguely and too often.* `v2/int-error`
   spent 4 turns and **15,819 tokens (60% of the whole run)** on three
   generic queries that kept retrieving `whatsnew/3.11.html`,
   `howto/argparse.html`, `howto/descriptor.html`:
   `"int() raises ValueError non-numeric string"` → `"int(string) raises ValueError non-numeric"` → `"int('abc') ValueError"`,
   step 3 reasoning: *"The docs not directly showing but known: int('abc')
   raises ValueError. Cite maybe builtins exceptions page?"*.

**Change.** `prompt_v3.txt` = v2 plus: the pre-retrieved context is "only a
first guess"; call `search_knowledge_base` before answering any Python/project
question; put the exact API name in the query and use the `source` filter;
**search at most twice**, then answer with lowered confidence or decline; arithmetic
needs no search; cite only documents that appeared in results. Config:
`recover_tool_answers=true`; temperature, top_k, max_iterations unchanged.

---

## v3 → v4: correct now, but paying twice for the same chunks

v4 is the first change made **after the golden set was frozen** (the references
were reviewed from v3's answers), so it is the first version the regression suite
guards honestly. It targets cost, not accuracy.

**What v3 did.** Completion 87.5%, no errors, 27,784 tokens (3,335 in / 138 out
per query). The traces show where the input tokens go:

- `v3/json-indent`: 2 turns, **5,460 tokens**. Step 1 searches with `"top_k": 5`
  and gets back `library/json.html` chunks. The loop had already pre-retrieved 4
  chunks (`top_k` 4) from the same page, so the model reads the same documentation
  twice before its final answer.
- `v3/read-lines` (5,403) and `v3/project-serving` (4,516) follow the same pattern:
  one search at `top_k` 5 on top of the pre-retrieved context.
- The turn budget is not the constraint: 3 of 4 turns is the most any query used
  (`max_iterations_used = 3`).

**Change.** `prompt_v4.txt` = v3 plus: pass `"top_k": 3`, one targeted search is
usually enough, and keep the answer short (direct answer, at most one small
example). Config: pre-retrieval `top_k` 4 → 3, `max_iterations` 4 → 3.

**Result.** Completion unchanged at 87.5%, tool correctness 100%, **17,468 tokens
(−37%)**, 2,094 in / 89 out per query, and every query now finishes in 2 turns.
Not every query got cheaper: `arith-units` rose from 1,895 to 3,440 tokens, so
the saving comes from the documentation questions (`int-error` 4,871 → 1,490,
`project-serving` 4,516 → 1,401, `read-lines` 5,403 → 1,606). Whether the shorter
answers lost information is what the regression suite decides.
