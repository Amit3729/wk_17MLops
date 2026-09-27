"""Scoring and failure classification in the evaluation harness."""

from app.llm import Completion, Usage
from eval.dataset import Case
from eval.harness import looks_declined, score, summarise, validate_arguments


def completion(**overrides):
    base = dict(
        answer="json.dumps indents the output.",
        confidence=0.8,
        citations=["library/json.html"],
        tool_calls=[{"name": "search_knowledge_base", "arguments": {"query": "json indent"}, "result": {"matches": [1]}}],
        retrieved=[{"document": "library/json.html", "chunk_id": 0, "score": 0.7, "text": "…"}],
        iterations=2,
        usage=Usage(prompt_tokens=900, completion_tokens=100, model_calls=2),
    )
    return Completion(**(base | overrides))


ANSWERABLE = Case("json-indent", "q", "single_hop",
                  expect_documents=("library/json.html",), expect_tools=("search_knowledge_base",))
UNANSWERABLE = Case("nope", "q", "unanswerable", expect_decline=True)


# ------------------------------------------------------------ argument checks


def test_valid_arguments_pass():
    assert validate_arguments("search_knowledge_base", {"query": "abc", "top_k": 3}) == []


def test_missing_required_argument_is_caught():
    assert "missing required argument 'query'" in validate_arguments("search_knowledge_base", {"top_k": 3})[0]


def test_wrong_type_is_caught():
    problems = validate_arguments("search_knowledge_base", {"query": "a", "top_k": "three"})
    assert "should be integer" in problems[0]


def test_out_of_range_integer_is_caught():
    problems = validate_arguments("search_knowledge_base", {"query": "a", "top_k": 99})
    assert "above maximum 10" in problems[0]


def test_unknown_argument_is_caught():
    problems = validate_arguments("calculator", {"expression": "1+1", "precision": 2})
    assert "unexpected argument 'precision'" in problems[0]


def test_tool_outside_the_allow_list_is_caught():
    assert validate_arguments("read_file", {"path": "/etc/passwd"}) == ["read_file: not an allow-listed tool"]


def test_unparsed_arguments_are_caught():
    assert validate_arguments("calculator", "{oops") == ["calculator: arguments were not a JSON object"]


# ----------------------------------------------------------------- taxonomy


def test_an_exception_is_a_hard_failure():
    result = score(ANSWERABLE, None, "RuntimeError: upstream died", 10, "single", "none")
    assert result.failure == "hard" and not result.answered


def test_an_empty_answer_is_a_hard_failure():
    result = score(ANSWERABLE, completion(answer="   "), "", 10, "single", "none")
    assert result.failure == "hard"


def test_a_good_run_completes_with_no_failure():
    result = score(ANSWERABLE, completion(), "", 10, "single", "none")
    assert result.completed and result.tools_correct and result.grounded
    assert result.failure == "" and result.total_tokens == 1000


def test_wrong_citation_with_working_retrieval_is_a_soft_failure():
    result = score(ANSWERABLE, completion(citations=["library/os.html"],
                                          retrieved=[{"document": "library/os.html", "chunk_id": 0, "score": 0.4, "text": "…"}]),
                   "", 10, "single", "none")
    assert result.failure == "soft" and not result.grounded


def test_a_confident_answer_built_on_a_broken_tool_is_a_cascading_soft_failure():
    broken = completion(
        citations=[],
        retrieved=[],
        confidence=0.9,
        answer="You can use json.dumps(indent=2) to pretty-print.",
        tool_calls=[{"name": "search_knowledge_base", "arguments": {"query": "json"},
                     "result": {"error": "Tool 'search_knowledge_base' timed out after 20s"}}],
    )
    result = score(ANSWERABLE, broken, "", 10, "multi", "retrieval_timeout")
    assert result.failure == "cascading_soft"
    assert result.tool_errors and "timed out" in result.tool_errors[0]


def test_admitting_the_gap_after_a_broken_tool_is_not_a_failure_of_honesty():
    honest = completion(
        citations=[], retrieved=[], confidence=0.1,
        answer="I could not retrieve anything: the search tool timed out, so I cannot answer this.",
        tool_calls=[{"name": "search_knowledge_base", "arguments": {"query": "json"},
                     "result": {"error": "timed out"}}],
    )
    result = score(ANSWERABLE, honest, "", 10, "multi", "retrieval_timeout")
    # It still misses the task, but it is a plain soft miss, not a confident fabrication.
    assert result.failure == "soft" and result.declined


def test_declining_an_unanswerable_question_counts_as_completed():
    declined = completion(answer="That is not covered by the indexed documentation.", confidence=0.1,
                          citations=[], retrieved=[], tool_calls=[])
    result = score(UNANSWERABLE, declined, "", 10, "multi", "none")
    assert result.completed and result.failure == ""


def test_answering_an_unanswerable_question_is_recorded_as_a_failure():
    invented = completion(answer="It costs $0.20 per million tokens.", confidence=0.85,
                          citations=[], retrieved=[], tool_calls=[])
    result = score(UNANSWERABLE, invented, "", 10, "multi", "none")
    assert result.failure and "instead of declining" in result.reason


def test_forbidden_tool_use_is_reported():
    arithmetic = Case("arith", "q", "arithmetic", expect_tools=("calculator",),
                      forbid_tools=("search_knowledge_base",), must_include=("55",))
    result = score(arithmetic, completion(answer="55"), "", 10, "single", "none")
    assert not result.tools_correct and "forbidden" in result.reason


# ------------------------------------------------------------------ helpers


def test_decline_detection_ignores_an_ordinary_answer():
    assert not looks_declined("Use json.dumps with indent=2.", 0.8)
    assert looks_declined("The documentation does not appear to cover that.", 0.8)
    assert looks_declined("Some answer", 0.1), "very low confidence is itself a decline"


def test_summary_aggregates_a_mixed_run():
    results = [
        score(ANSWERABLE, completion(), "", 10, "single", "none"),
        score(ANSWERABLE, None, "boom", 10, "single", "none"),
    ]
    summary = summarise(results)
    assert summary["runs"] == 2 and summary["completion_rate"] == 0.5 and summary["hard"] == 1


# --------------------------------------------- pre-retrieval is not grounding


def test_pre_retrieved_documents_are_reported_but_do_not_ground_the_answer():
    """Finding the page is the retriever's job; citing it is the agent's."""
    prefetched = [{"document": "library/json.html", "chunk_id": 0, "score": 0.6, "text": "…"}]
    blind = completion(citations=["library/os.html"], retrieved=[])

    result = score(ANSWERABLE, blind, "", 10, "single", "none", prefetched)

    assert "library/json.html" in result.documents, "still reported as a source"
    assert not result.grounded, "but the agent never cited or fetched it"
    assert result.failure == "soft", "usable evidence existed, so this is not cascading"


def test_an_agent_that_cites_what_it_was_given_is_grounded():
    prefetched = [{"document": "library/json.html", "chunk_id": 0, "score": 0.6, "text": "…"}]
    result = score(ANSWERABLE, completion(retrieved=[]), "", 10, "single", "none", prefetched)
    assert result.grounded and result.completed


# ----------------------------------------- whose fault was the tool failure?


def _with_tool_error(name, message, kind):
    """A good run plus one failing tool call, with arguments valid for that tool."""
    arguments = {"query": "json indent"} if name == "search_knowledge_base" else {"expression": "next Tuesday"}
    return completion(
        citations=["library/json.html"],
        retrieved=[{"document": "library/json.html", "chunk_id": 0, "score": 0.7, "text": "…"}],
        tool_calls=[
            {"name": "search_knowledge_base", "arguments": {"query": "json indent"}, "result": {"matches": [1]}},
            {"name": name, "arguments": arguments, "result": {"error": message, "kind": kind}},
        ],
    )


def test_a_tool_rejecting_the_arguments_counts_against_the_agent():
    bad = _with_tool_error("calculator", "Only basic arithmetic is supported", "arguments")
    result = score(ANSWERABLE, bad, "", 10, "multi", "none")
    assert not result.tools_correct
    assert "rejected the arguments" in result.reason


def test_an_unrelated_broken_tool_does_not_make_retrieval_look_broken():
    """A calculator failure says nothing about the quality of the evidence."""
    bad = _with_tool_error("calculator", "Only basic arithmetic is supported", "arguments")
    result = score(ANSWERABLE, bad, "", 10, "multi", "none")
    assert not result.retrieval_broke
    assert result.failure == "soft", "not cascading: the retrieved evidence was fine"


def test_infrastructure_failure_is_not_charged_to_the_agent():
    down = _with_tool_error("search_knowledge_base", "timed out after 20s", "infrastructure")
    result = score(ANSWERABLE, down, "", 10, "multi", "retrieval_timeout")
    assert result.tools_correct, "the agent picked the right tool; the tool was down"
    assert result.retrieval_broke
