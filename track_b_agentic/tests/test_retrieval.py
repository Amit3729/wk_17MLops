from pathlib import Path

from app.retrieval import Retriever, chunk_text


def test_chunking_overlaps_and_drops_covered_tail():
    chunks = chunk_text("a b c d e f g h i j", size=4, overlap=1)
    assert chunks == ["a b c d", "d e f g", "g h i j"]
    # The overlap word is shared across the boundary.
    assert chunks[0].split()[-1] == chunks[1].split()[0]


def test_chunking_respects_paragraph_boundaries():
    chunks = chunk_text("first para here\n\nsecond para text", size=3, overlap=1)
    assert chunks[0].startswith("first")


def test_chunking_rejects_overlap_at_least_size():
    try:
        chunk_text("a b c", size=2, overlap=2)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_ingest_and_search_round_trip(settings):
    retriever = Retriever(settings)
    documents, chunks = retriever.ingest()
    assert documents == 2 and chunks >= 2

    stats = retriever.stats()
    assert stats["backend"] == "chromadb"
    assert stats["chunks"] == chunks

    matches = retriever.search("vLLM serving throughput", limit=2)
    assert matches, "expected at least one match"
    assert matches[0]["document"] == "vllm.md"
    assert -1.0 <= matches[0]["score"] <= 1.0
    # Results come back ordered by similarity.
    assert matches == sorted(matches, key=lambda m: m["score"], reverse=True)


def test_ingest_is_skipped_when_corpus_is_unchanged(settings):
    retriever = Retriever(settings)
    first = retriever.ingest()
    assert retriever.ingest() == first


def test_ingest_picks_up_a_new_document(settings):
    from pathlib import Path

    retriever = Retriever(settings)
    _, before = retriever.ingest()
    Path(settings.data_dir, "extra.md").write_text("ONNX Runtime executes the embedding graph.", encoding="utf-8")
    documents, after = retriever.ingest()
    assert documents == 3 and after > before


# --------------------------------------------------- injected retrieval faults


def test_malformed_retrieval_returns_unusable_matches_not_an_empty_list(settings):
    """The point of the fault is broken evidence, not absent evidence.

    An empty list reads to a model as "nothing was found", which is a reasonable
    thing to answer from. Corrupted rows are not.
    """
    retriever = Retriever(settings.model_copy(update={"fault_injection": "malformed_retrieval"}))
    matches = retriever.search("vLLM serving throughput")
    assert matches, "the fault returns rows"
    assert all(match["score"] == 0.0 for match in matches)
    assert not any(match["text"].strip().isascii() and match["text"].strip() for match in matches)


def test_retrieval_timeout_raises_rather_than_returning_nothing(settings):
    import pytest

    retriever = Retriever(settings.model_copy(update={"fault_injection": "retrieval_timeout"}))
    with pytest.raises(TimeoutError):
        retriever.search("vLLM serving throughput")


def test_an_injected_fault_does_not_affect_an_unfaulted_retriever(settings):
    Retriever(settings.model_copy(update={"fault_injection": "malformed_retrieval"})).search("vLLM")
    assert Retriever(settings).search("vLLM serving throughput")[0]["document"] == "vllm.md"


def test_html_pages_are_indexed_by_section(settings, tmp_path):
    """A Sphinx page becomes several chunks, each carrying its anchor."""
    page = """<html><head><title>json &#8212; Python 3.14.7 documentation</title></head>
    <body><div role="main">
      <section id="module-json"><h1>json</h1><p>JSON encoder and decoder for Python.</p>
        <section id="basic-usage"><h2>Basic Usage</h2><p>Use dump to write a stream.</p></section>
      </section>
    </div></body></html>"""
    (Path(settings.data_dir) / "json.html").write_text(page, encoding="utf-8")

    retriever = Retriever(settings)
    documents, _ = retriever.ingest(force=True)
    assert documents == 3

    matches = retriever.search("JSON encoder decoder", limit=5)
    html = [match for match in matches if match["document"] == "json.html"]
    assert html, "the HTML page is retrievable"
    assert {match["anchor"] for match in html} <= {"module-json", "basic-usage"}
    assert html[0]["title"] == "json"


def test_the_source_filter_restricts_a_search_to_one_corpus(settings):
    nested = Path(settings.data_dir) / "manual"
    nested.mkdir()
    (nested / "guide.md").write_text("Paged attention is a vLLM memory technique.", encoding="utf-8")

    retriever = Retriever(settings)
    retriever.ingest(force=True)

    assert set(retriever.sources()) == {"root", "manual"}
    assert all(m["source"] == "manual" for m in retriever.search("paged attention", limit=3, source="manual"))
    assert all(m["source"] == "root" for m in retriever.search("paged attention", limit=3, source="root"))


def test_parser_changes_invalidate_a_stale_index(settings):
    from app import documents as documents_module

    retriever = Retriever(settings)
    documents, chunks = retriever.ingest()
    corpus = retriever._corpus()
    before = retriever._fingerprint(corpus)

    original = documents_module.EXTRACTOR_VERSION
    try:
        documents_module.EXTRACTOR_VERSION = original + "-next"
        # A different extractor must not keep serving chunks the old one produced.
        assert retriever._fingerprint(corpus) != before
        assert retriever.ingest() == (documents, chunks), "the rebuild still yields the same corpus here"
    finally:
        documents_module.EXTRACTOR_VERSION = original


def test_the_corpus_list_comes_from_the_manifest_not_a_full_scan(settings):
    import json as json_module

    retriever = Retriever(settings)
    retriever.ingest(force=True)
    manifest = json_module.loads((Path(settings.chroma_dir) / "manifest.json").read_text())
    assert manifest["sources"] == ["root"]

    # A fresh object answers without touching the collection at all.
    fresh = Retriever(settings)
    fresh._collection = lambda: (_ for _ in ()).throw(AssertionError("scanned the collection"))
    assert fresh.sources() == ["root"]
