"""Summarization: LLM-based (OpenAI or local LLM)."""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
from typing import Final, Any

from tqdm import tqdm

from lib.models import Paper
from lib.content_analysis import chunk_text_for_llm


DEFAULT_MAX_CHARS_PER_CHUNK: Final[int] = 12000
DEFAULT_MAX_CHUNKS: Final[int] = 8

# Global flag to enable local LLM mode
_USE_LOCAL_LLM: bool = False
_LOCAL_CLIENT: Any = None

DEFAULT_CHUNK_PROMPT: Final[str] = (
    "You are summarizing an ML/RecSys research paper for an engineering codebase context.\n"
    "Write a concise, high-signal summary of THIS CHUNK.\n"
    "Focus on: problem, key methods, key claims/results, assumptions/limitations.\n"
    "Return 6-10 bullet points.\n\n"
    "Paper title: {title}\n"
    "Chunk {idx}/{total}:\n\n{chunk}"
)

DEFAULT_REDUCE_PROMPT: Final[str] = (
    "Combine the chunk summaries into a final paper summary for engineers.\n"
    "Output markdown with EXACTLY these sections:\n"
    "### TL;DR (3 bullets)\n"
    "### Problem\n"
    "### Approach\n"
    "### Results (include numbers)\n"
    "### Practical Takeaways\n"
    "### Limitations / Open Questions\n"
    "Rules:\n"
    "- Prefer concrete metrics/numbers if stated.\n"
    "- If something isn't supported, write 'Unclear from text'.\n\n"
    "Paper title: {title}\n\n"
    "Chunk summaries:\n\n{summaries}"
)


_CONFIG_CACHE: dict[str, Any] | None = None


def enable_local_llm(model_id: str | None = None) -> None:
    """
    Enable local LLM mode using LiquidAI LFM2.5-1.2B-Instruct.
    
    This loads the model once and reuses it for all summarizations.
    Call this before running summarize_paper() to use local inference.
    
    Args:
        model_id: Optional HuggingFace model ID (default: LiquidAI/LFM2.5-1.2B-Instruct)
    """
    global _USE_LOCAL_LLM, _LOCAL_CLIENT
    
    from lib.local_llm import LocalLLMClient
    
    _LOCAL_CLIENT = LocalLLMClient(model_id=model_id)
    _USE_LOCAL_LLM = True


def disable_local_llm() -> None:
    """Disable local LLM mode, revert to OpenAI API."""
    global _USE_LOCAL_LLM, _LOCAL_CLIENT
    _USE_LOCAL_LLM = False
    _LOCAL_CLIENT = None


def is_using_local_llm() -> bool:
    """Check if local LLM mode is enabled."""
    return _USE_LOCAL_LLM


def _load_config() -> dict[str, Any]:
    """Load configuration from prompts.json if it exists (cached)."""
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE

    path = Path("prompts.json")
    if not path.exists():
        _CONFIG_CACHE = {}
        return _CONFIG_CACHE
    try:
        _CONFIG_CACHE = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        _CONFIG_CACHE = {}
    
    return _CONFIG_CACHE


def _build_chunk_prompt(paper: Paper, chunk: str, idx: int, total: int) -> str:
    """Build the prompt for summarizing a single chunk."""
    config = _load_config()
    prompt_template = config.get("chunk_prompt", DEFAULT_CHUNK_PROMPT)
    return (
        prompt_template
        .replace("{title}", paper.title)
        .replace("{idx}", str(idx))
        .replace("{total}", str(total))
        .replace("{chunk}", chunk)
    )


def _build_reduce_prompt(paper: Paper, chunk_summaries: list[str]) -> str:
    """Build the prompt for combining chunk summaries."""
    config = _load_config()
    reduce_prompt_template = config.get("reduce_prompt", DEFAULT_REDUCE_PROMPT)
    return (
        reduce_prompt_template
        .replace("{title}", paper.title)
        .replace("{summaries}", "\n\n".join(chunk_summaries))
    )


def _summarize_chunk_openai(client, model: str, paper: Paper, chunk: str, idx: int, total: int) -> str:
    """Summarize a single chunk of paper text using OpenAI."""
    prompt = _build_chunk_prompt(paper, chunk, idx, total)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.choices[0].message.content.strip()


def _reduce_chunk_summaries_openai(client, model: str, paper: Paper, chunk_summaries: list[str]) -> str:
    """Combine chunk summaries into final paper summary using OpenAI."""
    reduce_prompt = _build_reduce_prompt(paper, chunk_summaries)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": reduce_prompt}]
    )
    return resp.choices[0].message.content.strip()


def _summarize_chunk_local(paper: Paper, chunk: str, idx: int, total: int) -> str:
    """Summarize a single chunk of paper text using local LLM."""
    prompt = _build_chunk_prompt(paper, chunk, idx, total)
    return _LOCAL_CLIENT.generate(prompt, max_new_tokens=512)


def _reduce_chunk_summaries_local(paper: Paper, chunk_summaries: list[str]) -> str:
    """Combine chunk summaries into final paper summary using local LLM."""
    reduce_prompt = _build_reduce_prompt(paper, chunk_summaries)
    return _LOCAL_CLIENT.generate(reduce_prompt, max_new_tokens=1024)


def summarize_paper(paper: Paper, max_chunks: int = 8) -> Paper:
    """
    Generate summary using LLM (OpenAI API or local model).

    Strategy: Map-reduce over chunks
    1. Split text into chunks
    2. Summarize each chunk
    3. Reduce chunk summaries into final summary

    Modes:
    - Local: Call enable_local_llm() first to use LiquidAI LFM2.5-1.2B-Instruct
    - OpenAI: Requires OPENAI_API_KEY environment variable (default)

    Returns: New Paper object with summary_md populated.
    Raises: RuntimeError if required dependencies/keys not available
    """
    config = _load_config()
    
    # Priority: Config > Default
    max_chars = config.get("chunk_max_chars", DEFAULT_MAX_CHARS_PER_CHUNK)
    config_max_chunks = config.get("max_chunks", DEFAULT_MAX_CHUNKS)
    
    # Preserve function argument as an override when explicitly passed by callers.
    effective_max_chunks = max_chunks if max_chunks != DEFAULT_MAX_CHUNKS else config_max_chunks
    
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
    
    if _USE_LOCAL_LLM:
        # Local LLM mode (LFM2.5-1.2B-Instruct)
        if _LOCAL_CLIENT is None:
            raise RuntimeError(
                "Local LLM not initialized. Call enable_local_llm() first."
            )
        
        for idx, ch in chunk_iter:
            summary = _summarize_chunk_local(paper, ch, idx, len(chunks))
            chunk_summaries.append(summary)
        
        final_summary = _reduce_chunk_summaries_local(paper, chunk_summaries)
    else:
        # OpenAI API mode
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY environment variable is required. "
                "Set it in .env or export OPENAI_API_KEY=sk-... "
                "Or use --local flag for local LLM inference."
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
        
        for idx, ch in chunk_iter:
            summary = _summarize_chunk_openai(client, model, paper, ch, idx, len(chunks))
            chunk_summaries.append(summary)
        
        final_summary = _reduce_chunk_summaries_openai(client, model, paper, chunk_summaries)
    
    return dataclasses.replace(paper, summary_md=final_summary)
