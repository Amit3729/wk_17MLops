import pytest
from fastapi.testclient import TestClient

from app.main import app, cache


@pytest.fixture()
def client():
    cache._entries.clear()
    with TestClient(app) as test_client:
        yield test_client


def test_health_reports_index_and_embedder(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["index"]["backend"] == "chromadb"
    assert body["index"]["chunks"] > 0
    assert body["index"]["embedder"] == "hash:sha256"


def test_ingest_returns_corpus_size(client):
    body = client.post("/ingest", params={"force": True}).json()
    assert body["documents"] == 2 and body["chunks"] >= 2


def test_chat_returns_the_structured_contract(client):
    body = client.post("/chat", json={"question": "What is vLLM?"}).json()
    assert set(body) == {
        "answer", "confidence", "citations", "sources",
        "tool_calls", "provider", "model", "cached", "latency_ms",
        # Added in W16 so a caller can see which pattern ran and what it cost.
        "agent_mode", "iterations", "usage", "plan",
    }
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["cached"] is False


def test_chat_runs_the_calculator_tool(client):
    body = client.post("/chat", json={"question": "calculate 12 * 4"}).json()
    assert body["tool_calls"][0]["name"] == "calculator"
    assert body["tool_calls"][0]["result"] == 48
    assert "48" in body["answer"]


def test_identical_questions_are_served_from_cache(client):
    first = client.post("/chat", json={"question": "What is RAG?"}).json()
    second = client.post("/chat", json={"question": "What is RAG?"}).json()
    assert first["cached"] is False and second["cached"] is True
    assert second["answer"] == first["answer"]


def test_cache_key_includes_sampling_parameters(client):
    client.post("/chat", json={"question": "What is RAG?", "temperature": 0.2})
    other = client.post("/chat", json={"question": "What is RAG?", "temperature": 0.9}).json()
    assert other["cached"] is False


@pytest.mark.parametrize(
    "payload",
    [{"question": ""}, {"question": "hi", "temperature": 5}, {"question": "hi", "top_p": 0}, {}],
)
def test_invalid_requests_are_rejected(client, payload):
    assert client.post("/chat", json=payload).status_code == 422


def test_rate_limiter_returns_429_with_retry_after(client, monkeypatch):
    from app import main

    monkeypatch.setattr(main.settings, "rate_limit_requests", 2)
    main.requests_by_ip.clear()

    assert client.post("/chat", json={"question": "one"}).status_code == 200
    assert client.post("/chat", json={"question": "two"}).status_code == 200

    blocked = client.post("/chat", json={"question": "three"})
    assert blocked.status_code == 429
    assert int(blocked.headers["Retry-After"]) >= 1

    # /health stays reachable so orchestrators can still probe a throttled instance.
    assert client.get("/health").status_code == 200


def test_llm_failure_degrades_to_503(client, monkeypatch):
    from app import main

    async def boom(*args, **kwargs):
        raise RuntimeError("all providers down")

    monkeypatch.setattr(main.llm, "complete", boom)
    response = client.post("/chat", json={"question": "anything at all"})
    assert response.status_code == 503
    assert "temporarily unavailable" in response.json()["detail"]


def test_retrieval_failure_does_not_fail_the_request(client, monkeypatch):
    from app import main

    def boom(*args, **kwargs):
        raise RuntimeError("vector store offline")

    monkeypatch.setattr(main.retriever, "search", boom)
    response = client.post("/chat", json={"question": "calculate 2 + 2"})
    assert response.status_code == 200
    assert response.json()["sources"] == []
