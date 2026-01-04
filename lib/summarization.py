"""Summarization: LLM-based (OpenAI required)."""

from __future__ import annotations

import dataclasses
import os
from typing import Final

from tqdm import tqdm

from lib.models import Paper
from lib.content_analysis import chunk_text_for_llm


DEFAULT_MAX_CHARS_PER_CHUNK: Final[int] = 12000
DEFAULT_MAX_CHUNKS: Final[int] = 8


def _get_int_env(name: str, default: int) -> int:
    """Read a positive integer from environment variables; fall back to default."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    raw = raw.strip()
    if not raw:
        return default
    try:
        val = int(raw)
    except ValueError:
        raise RuntimeError(f"{name} must be an integer (got {raw!r})")
    if val <= 0:
        raise RuntimeError(f"{name} must be > 0 (got {val})")
    return val


def _summarize_chunk(client, model: str, paper: Paper, chunk: str, idx: int, total: int) -> str:
    """Summarize a single chunk of paper text using OpenAI."""
    prompt = (
        "You are summarizing an ML/RecSys research paper for an engineering codebase context.\n"
        "Write a concise, high-signal summary of THIS CHUNK.\n"
        "Focus on: problem, key methods, key claims/results, assumptions/limitations.\n"
        "Return 6-10 bullet points.\n\n"
        f"Paper title: {paper.title}\n"
        f"Chunk {idx}/{total}:\n\n{chunk}"
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.choices[0].message.content.strip()


def _reduce_chunk_summaries(client, model: str, paper: Paper, chunk_summaries: list[str]) -> str:
    """Combine chunk summaries into final paper summary using OpenAI."""
    reduce_prompt = (
        "Combine the chunk summaries into a final paper summary for engineers.\n"
        "Output markdown with EXACTLY these sections:\n"
        "### TL;DR (3 bullets)\n"
        "### Problem\n"
        "### Approach\n"
        "### Results (include numbers)\n"
        "### Practical takeaways for DealSeek (deals/product recommendations)\n"
        "### Limitations / open questions\n"
        "Rules:\n"
        "- Prefer concrete metrics/numbers if stated.\n"
        "- If something isn't supported, write 'Unclear from text'.\n\n"
        f"Paper title: {paper.title}\n\n"
        "Chunk summaries:\n\n" + "\n\n".join(chunk_summaries)
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": reduce_prompt}]
    )
    return resp.choices[0].message.content.strip()


def summarize_paper(paper: Paper, max_chunks: int = 8) -> Paper:
    """
    Generate summary using OpenAI LLM (required).

    Strategy: Map-reduce over chunks
    1. Split text into chunks
    2. Summarize each chunk
    3. Reduce chunk summaries into final summary

    Requires: OPENAI_API_KEY environment variable

    Returns: New Paper object with summary_md populated.
    Raises: RuntimeError if OPENAI_API_KEY not set
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY environment variable is required. "
            "Set it in .env or export OPENAI_API_KEY=sk-..."
        )

    try:
        from openai import OpenAI  # type: ignore
    except ImportError:
        raise RuntimeError(
            "openai package not installed. Run: pip install openai"
        )

    base_url = os.environ.get("OPENAI_BASE_URL")
    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
    model = os.environ.get("OPENAI_MODEL", "gpt-5-mini-2025-08-07")

    max_chars = _get_int_env("PAPER2MD_CHUNK_MAX_CHARS", DEFAULT_MAX_CHARS_PER_CHUNK)
    env_max_chunks = _get_int_env("PAPER2MD_MAX_CHUNKS", DEFAULT_MAX_CHUNKS)
    # Preserve function argument as an override when explicitly passed by callers.
    effective_max_chunks = max_chunks if max_chunks != DEFAULT_MAX_CHUNKS else env_max_chunks

    chunks = chunk_text_for_llm(paper.text, max_chars=max_chars)[:effective_max_chunks]
    chunk_summaries: list[str] = []

    # Use tqdm for chunk summarization if there are multiple
    chunk_iter = enumerate(chunks, start=1)
    if len(chunks) > 1:
        chunk_iter = tqdm(
            chunk_iter,
            total=len(chunks),
            desc=f"  Summarizing {paper.pdf_path.name}",
            leave=False
        )

    for idx, ch in chunk_iter:
        summary = _summarize_chunk(client, model, paper, ch, idx, len(chunks))
        chunk_summaries.append(summary)

    final_summary = _reduce_chunk_summaries(client, model, paper, chunk_summaries)
    return dataclasses.replace(paper, summary_md=final_summary)
