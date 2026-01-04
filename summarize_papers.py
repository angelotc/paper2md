#!/usr/bin/env python3
"""
Summarize PDFs in /papers into a single markdown file for repo context.

Behavior:
- Extracts text + title from each PDF via sophisticated metadata + text extraction.
- Generates structured summaries using OpenAI-compatible LLM (required).

Usage (PowerShell):
  python summarize_papers.py [--papers-dir papers] [--out output/PAPERS_SUMMARY.md]

Required env vars:
  OPENAI_API_KEY          -> API key for LLM provider

Optional env vars:
  OPENAI_MODEL            -> default: gpt-5-mini-2025-08-07
  OPENAI_BASE_URL         -> API base URL (e.g. OpenRouter, Gemini)
  PAPER2MD_CHUNK_MAX_CHARS -> max characters per chunk for map step (default: 12000)
  PAPER2MD_MAX_CHUNKS      -> max number of chunks per paper (default: 8)
"""

from __future__ import annotations

import argparse
import os
import re
import traceback
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from lib.models import Paper
from lib.pdf_extract import extract_paper_from_pdf
from lib.summarization import summarize_paper
from lib.content_analysis import extract_structured_content

# Load environment variables from root .env if it exists
load_dotenv(Path(__file__).parent / ".env")


def _truthy_env(name: str) -> bool:
    v = os.environ.get(name, "").strip().lower()
    return v not in {"", "0", "false", "no", "off"}


def _format_exc(e: Exception) -> str:
    msg = str(e).strip()
    if msg:
        return f"{type(e).__name__}: {msg}"
    return type(e).__name__


def _report_error(stage: str, pdf: Path, e: Exception) -> None:
    tqdm.write(f"[ERROR] {stage} failed for {pdf.name}: {_format_exc(e)}")
    if _truthy_env("PAPER2MD_DEBUG_TRACE"):
        tqdm.write(traceback.format_exc())


def load_papers(papers_dir: Path, max_pages: int | None = None) -> list[Paper]:
    """Extract title + text from all PDFs in directory."""
    pdfs = sorted(papers_dir.glob("*.pdf"))
    papers: list[Paper] = []
    failures = 0

    for pdf in tqdm(pdfs, desc="Extracting PDFs"):
        try:
            paper = extract_paper_from_pdf(pdf, max_pages=max_pages)
        except Exception as e:
            failures += 1
            _report_error("extract", pdf, e)
            continue

        if len(paper.text) < 500:
            tqdm.write(
                f"[WARN] Very little text extracted for {pdf.name} "
                f"(chars={len(paper.text)}). It may be scanned or protected."
            )
        papers.append(paper)

    if failures:
        tqdm.write(f"[WARN] Extraction failures: {failures}/{len(pdfs)} PDFs")

    return papers


def generate_summaries(papers: list[Paper]) -> list[Paper]:
    """Generate summaries for all papers (LLM or heuristic)."""
    summarized: list[Paper] = []
    failures = 0

    for paper in tqdm(papers, desc="Summarizing"):
        try:
            summarized.append(summarize_paper(paper))
        except Exception as e:
            failures += 1
            _report_error("summarize", paper.pdf_path, e)
            summarized.append(paper)

    if failures:
        tqdm.write(f"[WARN] Summarization failures: {failures}/{len(papers)} PDFs")

    return summarized


def build_markdown(papers: list[Paper]) -> str:
    """Build final markdown document from papers."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines: list[str] = []
    lines.append("# Papers summary (auto-generated)")
    lines.append("")
    lines.append(f"_Generated: {now}_")
    lines.append("")
    lines.append("This file summarizes the PDFs in `papers/` for use as context in this codebase.")
    lines.append("")
    lines.append("## DealSeek: current personalization context")
    lines.append(
        "_Source: `docs/PERSONALIZATION_PLAN.md` (summary of what the codebase currently implements)._"
    )
    lines.append("")
    lines.append("- **High-level flow**: `/deals?userId=...` runs **regular trending feed** and **personalized candidate fetch** in parallel, then blends/interleaves results (target: **1 personalized : 2 regular**).")
    lines.append("- **User preference representation**: time-weighted centroids per **(user, category)** stored in Milvus (`user_preferences`).")
    lines.append("- **Candidate retrieval**: Milvus vector search using top categories + deal embeddings from Milvus (`deals`), followed by SQL fetch of details with **quality gating** pushed down (e.g. `deal_score >= 25`).")
    lines.append("- **Diversity**: multi-centroid blending (top 3 categories) uses softmax allocation + a floor to avoid single-category domination.")
    lines.append("- **Latency**: parallelized via `errgroup`, plus **RAM preference cache** (`TTLCache`, 1h TTL) to avoid repeated Milvus calls.")
    lines.append("- **Fallbacks**: no `userId`, no prefs, Milvus failure, or all personalized candidates gated → standard ranking/trending only.")
    lines.append("- **Key implementation files**: `backend/go/internal/handlers/deals.go`, `backend/go/internal/service/personalization.go`, `backend/go/internal/milvus/preferences.go`, `backend/go/internal/service/deals_service.go`.")
    lines.append("")
    lines.append("## How to use this")
    lines.append("- Use each paper's **Practical takeaways for DealSeek** section for prompting/feature ideation.")
    lines.append("- If you set `OPENAI_API_KEY`, re-run the generator to get deeper method + results summaries.")
    lines.append("")
    lines.append("## Index")
    for p in papers:
        anchor = re.sub(r"[^a-z0-9]+", "-", p.title.lower()).strip("-")
        lines.append(f"- [{p.title}](#{anchor})")
    lines.append("")
    lines.append("## Summaries")
    lines.append("")
    for p in papers:
        lines.append(f"## {p.title}")
        lines.append("")
        lines.append(f"- **Source PDF**: `{p.pdf_path.as_posix()}`")

        content = extract_structured_content(p.text)
        if content.doi:
            lines.append(f"- **DOI**: `https://doi.org/{content.doi}`")
        lines.append("")

        if p.summary_md:
            lines.append(p.summary_md.strip())
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    """Main entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--papers-dir", default="papers", help="Directory containing PDFs (default: papers)")
    ap.add_argument(
        "--out",
        default="output/PAPERS_SUMMARY.md",
        help="Output markdown path (default: output/PAPERS_SUMMARY.md)",
    )
    ap.add_argument("--max-pages", type=int, default=0, help="Limit pages per PDF (0 = all pages)")
    args = ap.parse_args()

    papers_dir = Path(args.papers_dir)
    out_path = Path(args.out)
    max_pages = None if args.max_pages == 0 else args.max_pages

    if not papers_dir.exists():
        raise SystemExit(f"papers dir not found: {papers_dir}")

    # Simple pipeline: load → summarize → write
    papers = load_papers(papers_dir, max_pages=max_pages)
    papers = generate_summaries(papers)

    md = build_markdown(papers)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
