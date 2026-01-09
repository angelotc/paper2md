"""Caching: hash-based incremental processing for papers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from lib.models import Paper


DEFAULT_CACHE_DIR = ".paper2md"
DEFAULT_CACHE_FILE = f"{DEFAULT_CACHE_DIR}/cache.json"


def compute_pdf_hash(pdf_path: Path) -> str:
    """Compute SHA-256 hash of PDF file content."""
    hasher = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


class PaperCache:
    """
    Cache for paper summaries keyed by PDF content hash.
    
    Schema:
    {
        "version": 1,
        "papers": {
            "filename.pdf": {
                "hash": "sha256...",
                "title": "Paper Title",
                "summary_md": "### TL;DR..."
            }
        }
    }
    """

    def __init__(self, cache_path: Path | str = DEFAULT_CACHE_FILE):
        self.cache_path = Path(cache_path)
        self._data: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        """Load cache from disk or initialize empty."""
        if self.cache_path.exists():
            try:
                data = json.loads(self.cache_path.read_text(encoding="utf-8"))
                if isinstance(data, dict) and data.get("version") == 1:
                    return data
            except (json.JSONDecodeError, OSError):
                pass
        return {"version": 1, "papers": {}}

    def save(self) -> None:
        """Persist cache to disk."""
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(self._data, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

    def get_cached(self, pdf_path: Path) -> Paper | None:
        """
        Return cached Paper if PDF hasn't changed, else None.
        
        Checks if:
        1. Entry exists for this filename
        2. Stored hash matches current file hash
        3. Summary exists
        """
        entry = self._data["papers"].get(pdf_path.name)
        if not entry:
            return None

        current_hash = compute_pdf_hash(pdf_path)
        if entry.get("hash") != current_hash:
            return None

        summary = entry.get("summary_md")
        if not summary:
            return None

        return Paper(
            pdf_path=pdf_path,
            title=entry.get("title", pdf_path.stem),
            text="",  # Don't cache full text - re-extract if needed
            summary_md=summary
        )

    def store(self, paper: Paper, pdf_hash: str | None = None) -> None:
        """
        Store paper summary in cache.
        
        Args:
            paper: Paper with summary_md populated
            pdf_hash: Pre-computed hash (to avoid re-hashing)
        """
        if not paper.summary_md:
            return

        if pdf_hash is None:
            pdf_hash = compute_pdf_hash(paper.pdf_path)

        self._data["papers"][paper.pdf_path.name] = {
            "hash": pdf_hash,
            "title": paper.title,
            "summary_md": paper.summary_md
        }

    def is_changed(self, pdf_path: Path) -> tuple[bool, str]:
        """
        Check if PDF has changed since last cache.
        
        Returns: (is_changed, current_hash)
        """
        current_hash = compute_pdf_hash(pdf_path)
        entry = self._data["papers"].get(pdf_path.name)

        if not entry:
            return True, current_hash

        return entry.get("hash") != current_hash, current_hash

    def clear(self) -> None:
        """Clear all cached entries."""
        self._data["papers"] = {}

    def stats(self) -> dict[str, int]:
        """Return cache statistics."""
        return {
            "total_entries": len(self._data["papers"]),
            "with_summaries": sum(
                1 for e in self._data["papers"].values()
                if e.get("summary_md")
            )
        }
