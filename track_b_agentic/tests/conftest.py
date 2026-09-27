"""Test-wide isolation.

The API module builds its retriever/LLM singletons at import time, so the
environment has to be pinned before `app.main` is ever imported: no weight
downloads, no network provider, and a scratch corpus outside the repo.
"""

import os
import tempfile
from pathlib import Path

import pytest

CORPUS = Path(tempfile.mkdtemp(prefix="assistant-tests-"))
(CORPUS / "docs").mkdir()
(CORPUS / "docs" / "vllm.md").write_text(
    "vLLM is a high-throughput serving engine for large language models.\n\n"
    "It uses paged attention and continuous batching to raise throughput.\n",
    encoding="utf-8",
)
(CORPUS / "docs" / "rag.md").write_text(
    "Retrieval-Augmented Generation retrieves document chunks and passes them to a generator.\n",
    encoding="utf-8",
)

os.environ.update(
    PROVIDER="local",
    EMBEDDING_PROVIDER="hash",
    FALLBACK_PROVIDER="none",
    OPENAI_API_KEY="",
    DATA_DIR=str(CORPUS / "docs"),
    CHROMA_DIR=str(CORPUS / "chroma"),
    RATE_LIMIT_REQUESTS="1000",
)


@pytest.fixture()
def settings(tmp_path):
    """A Settings object pointed at a per-test corpus and Chroma directory."""
    from app.config import Settings

    corpus = tmp_path / "docs"
    corpus.mkdir()
    for name in ("vllm.md", "rag.md"):
        (corpus / name).write_text((CORPUS / "docs" / name).read_text(), encoding="utf-8")

    return Settings(
        provider="local",
        embedding_provider="hash",
        fallback_provider="none",
        data_dir=str(corpus),
        chroma_dir=str(tmp_path / "chroma"),
        collection_name="test-knowledge",
    )
