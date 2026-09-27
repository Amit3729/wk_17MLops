"""Download the CPython 3.14 HTML documentation into data/ (the RAG corpus).

    uv run python -m mlops.fetch_docs            # skips if data/python-3.14-docs-html exists
    uv run python -m mlops.fetch_docs --force

The corpus (66 MB) and the Chroma index built from it (~210 MB) are gitignored, so a fresh
clone runs this once and then `make ingest`. docs.python.org republishes the archive with each
3.14.x patch release, so a later download can differ slightly from the one the committed
results were produced with; the index manifest records the fingerprint of what was indexed.
"""

from __future__ import annotations

import argparse
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

URL = "https://docs.python.org/3.14/archives/python-3.14-docs-html.tar.bz2"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
TARGET = DATA_DIR / "python-3.14-docs-html"


def fetch(force: bool = False, data_dir: Path = DATA_DIR) -> Path:
    target = data_dir / TARGET.name
    if target.exists() and not force:
        print(f"{target} already present; use --force to re-download")
        return target
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "docs.tar.bz2"
        print(f"downloading {URL}")
        urllib.request.urlretrieve(URL, archive)
        with tarfile.open(archive, "r:bz2") as tar:
            tar.extractall(tmp, filter="data")
        if target.exists():
            shutil.rmtree(target)
        shutil.move(str(Path(tmp) / TARGET.name), target)
    print(f"extracted to {target} ({sum(1 for _ in target.rglob('*.html'))} html files)")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    fetch(args.force)


if __name__ == "__main__":
    main()
