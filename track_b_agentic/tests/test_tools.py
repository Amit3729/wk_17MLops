import json

import pytest

from app.retrieval import Retriever
from app.tools import TOOL_SCHEMAS, ToolError, ToolRegistry, calculator, heuristic_tool_calls


def test_schemas_are_valid_openai_function_definitions():
    names = set()
    for schema in TOOL_SCHEMAS:
        assert schema["type"] == "function"
        function = schema["function"]
        assert function["description"]
        assert function["parameters"]["type"] == "object"
        names.add(function["name"])
    assert names == {"search_knowledge_base", "calculator", "current_time"}


@pytest.mark.parametrize(
    ("expression", "expected"),
    [("12 * 4", 48), ("-3 + 2 ** 5", 29), ("7 // 2", 3), ("(1 + 2) * 3", 9)],
)
def test_calculator_evaluates_arithmetic(expression, expected):
    assert calculator(expression) == expected


@pytest.mark.parametrize("expression", ["__import__('os').system('ls')", "open('/etc/passwd')", "1 +", "x + 1"])
def test_calculator_rejects_non_arithmetic(expression):
    with pytest.raises(ToolError):
        calculator(expression)


async def test_registry_rejects_tools_outside_the_allow_list(settings):
    registry = ToolRegistry(Retriever(settings))
    payload, matches = await registry.execute("read_file", '{"path": "/etc/passwd"}')
    assert "Unknown tool" in json.loads(payload)["error"]
    assert matches is None


async def test_registry_reports_bad_arguments_back_to_the_model(settings):
    registry = ToolRegistry(Retriever(settings))
    assert "error" in json.loads((await registry.execute("calculator", "not json"))[0])
    assert "error" in json.loads((await registry.execute("calculator", '{"wrong_key": 1}'))[0])
    assert "error" in json.loads((await registry.execute("calculator", '{"expression": "1/0"}'))[0])


async def test_search_tool_returns_matches_for_citation(settings):
    registry = ToolRegistry(Retriever(settings))
    payload, matches = await registry.execute("search_knowledge_base", '{"query": "vLLM", "top_k": 2}')
    assert matches and matches == json.loads(payload)["matches"]
    assert {"document", "chunk_id", "score", "text"} <= set(matches[0])


async def test_search_tool_clamps_top_k(settings):
    registry = ToolRegistry(Retriever(settings))
    _, matches = await registry.execute("search_knowledge_base", '{"query": "vLLM", "top_k": 999}')
    assert len(matches) <= 10


async def test_current_time_takes_no_arguments(settings):
    registry = ToolRegistry(Retriever(settings))
    payload, _ = await registry.execute("current_time", "{}")
    assert "utc" in json.loads(payload)


def test_heuristic_only_fires_on_arithmetic(settings):
    registry = ToolRegistry(Retriever(settings))
    calls, context = heuristic_tool_calls("calculate 12 * 4", registry)
    assert calls[0]["result"] == 48 and "48" in context
    assert heuristic_tool_calls("what is RAG?", registry) == ([], "")
