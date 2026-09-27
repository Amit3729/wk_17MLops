"""Extraction of indexable prose from the Sphinx HTML corpus."""

from pathlib import Path

from app.documents import DEFAULT_EXCLUDES, is_indexable, load, parse_html

SPHINX_PAGE = """<!DOCTYPE html>
<html><head><title>json &#8212; JSON encoder and decoder &#8212; Python 3.14.7 documentation</title></head>
<body>
  <div class="related">skip to content</div>
  <div class="sphinxsidebar"><h3>Navigation</h3><p>sidebar noise</p></div>
  <div class="body" role="main">
    <section id="module-json">
      <h1>json &#8212; JSON encoder and decoder<a class="headerlink" href="#module-json">&para;</a></h1>
      <p>JSON is a lightweight data interchange format.</p>
      <section id="basic-usage">
        <h2>Basic Usage<a class="headerlink" href="#basic-usage">&para;</a></h2>
        <dl class="py function">
          <dt class="sig sig-object py" id="json.dump"><span class="sig-prename">json.</span><span
             class="sig-name">dump</span><span class="sig-paren">(</span><em class="sig-param">obj</em>,
             <em class="sig-param">fp</em><span class="sig-paren">)</span></dt>
          <dd><p>Serialize <em>obj</em> as a JSON formatted stream.</p></dd>
        </dl>
        <div class="highlight-python"><div class="highlight"><pre>&gt;&gt;&gt; json.dumps([1, 2])
'[1, 2]'</pre></div></div>
      </section>
    </section>
  </div>
  <script>var noise = 1;</script>
</body></html>
"""


def test_title_drops_the_sphinx_suffix():
    assert parse_html(SPHINX_PAGE).title == "json — JSON encoder and decoder"


def test_sections_are_split_with_anchors_and_breadcrumbs():
    sections = parse_html(SPHINX_PAGE).sections
    assert [section.anchor for section in sections] == ["module-json", "basic-usage"]
    assert sections[1].heading == "json — JSON encoder and decoder > Basic Usage"


def test_parent_section_does_not_repeat_its_children():
    parent, child = parse_html(SPHINX_PAGE).sections
    assert "lightweight data interchange" in parent.text
    # The child's body belongs to the child chunk only, or the page is indexed twice.
    assert "Serialize" not in parent.text
    assert "Serialize" in child.text


def test_signatures_survive_as_one_searchable_token():
    child = parse_html(SPHINX_PAGE).sections[1]
    assert "json.dump(obj, fp)" in child.text
    # Space-joined rendering would have produced this instead.
    assert "json . dump" not in child.text


def test_code_blocks_keep_their_lines():
    child = parse_html(SPHINX_PAGE).sections[1]
    assert ">>> json.dumps([1, 2])\n'[1, 2]'" in child.text


def test_navigation_and_scripts_are_stripped():
    text = " ".join(section.text for section in parse_html(SPHINX_PAGE).sections)
    for noise in ("sidebar noise", "skip to content", "var noise", "¶"):
        assert noise not in text


def test_a_page_without_sections_still_yields_one():
    page = "<html><head><title>Flat</title></head><body><div role='main'><p>Only a paragraph.</p></div></body></html>"
    sections = parse_html(page).sections
    assert len(sections) == 1 and "Only a paragraph." in sections[0].text


def test_markdown_is_loaded_as_a_single_section(tmp_path):
    path = tmp_path / "notes.md"
    path.write_text("# Notes\n\nSome prose.", encoding="utf-8")
    document = load(path)
    assert document.title == "notes"
    assert len(document.sections) == 1 and "Some prose." in document.sections[0].text


def test_excludes_match_below_a_nested_corpus_directory(tmp_path):
    """The real corpus sits in its own directory, so patterns must not be root-anchored."""
    root = tmp_path
    build = root / "python-3.14-docs-html"
    (build / "whatsnew").mkdir(parents=True)
    (build / "_static").mkdir()
    (build / "library").mkdir()
    (build / "whatsnew" / "changelog.html").write_text(SPHINX_PAGE, encoding="utf-8")
    (build / "whatsnew" / "3.14.html").write_text(SPHINX_PAGE, encoding="utf-8")
    (build / "_static" / "notes.txt").write_text("asset", encoding="utf-8")
    (build / "library" / "json.html").write_text(SPHINX_PAGE, encoding="utf-8")

    kept = sorted(
        path.relative_to(root).as_posix() for path in root.rglob("*") if is_indexable(path, root, DEFAULT_EXCLUDES)
    )
    assert kept == [
        "python-3.14-docs-html/library/json.html",
        "python-3.14-docs-html/whatsnew/3.14.html",
    ], "the changelog and the asset directory are excluded; the narrative page is kept"


def test_build_artefacts_are_not_indexable(tmp_path):
    root = tmp_path
    for name in ("genindex-A.html", "search.html", "license.html"):
        (root / name).write_text("<html></html>", encoding="utf-8")
    (root / "_static").mkdir()
    (root / "_static" / "notes.txt").write_text("asset", encoding="utf-8")
    (root / "library").mkdir()
    (root / "library" / "json.html").write_text(SPHINX_PAGE, encoding="utf-8")
    (root / "readme.md").write_text("hello", encoding="utf-8")
    (root / "objects.inv").write_bytes(b"binary")

    kept = sorted(
        path.relative_to(root).as_posix() for path in root.rglob("*") if is_indexable(path, root, DEFAULT_EXCLUDES)
    )
    assert kept == ["library/json.html", "readme.md"]
