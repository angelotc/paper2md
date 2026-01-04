"""PDF extraction: text + title metadata."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Optional

from lib.models import Paper
from lib.text_clean import clean_pdf_text, _WHITESPACE_RE, _BOM_PREFIX_RE


# Manual overrides for PDFs whose titles aren't discoverable
TITLE_OVERRIDES: dict[str, str] = {
    "2306.04833v2.pdf": "Unified Embedding Based Personalized Retrieval in Etsy Search",
}


def _decode_pdf_literal_string(raw: bytes) -> str:
    """
    Decode a PDF "literal string" body (without surrounding parentheses).
    Handles common escapes: \\n \\r \\t \\b \\f \\( \\) \\\\ and octal \\ddd.
    """
    out = bytearray()
    i = 0
    while i < len(raw):
        b = raw[i]
        if b != 0x5C:  # backslash
            out.append(b)
            i += 1
            continue

        # Escape
        i += 1
        if i >= len(raw):
            break
        c = raw[i]

        if c in b"nrtbf":
            out.extend(
                {
                    ord("n"): b"\n",
                    ord("r"): b"\r",
                    ord("t"): b"\t",
                    ord("b"): b"\b",
                    ord("f"): b"\f",
                }[c]
            )
            i += 1
            continue

        if c in b"()\\":
            out.append(c)
            i += 1
            continue

        # Octal escape: up to 3 digits
        if 0x30 <= c <= 0x37:  # '0'..'7'
            digits = [c]
            i += 1
            for _ in range(2):
                if i < len(raw) and 0x30 <= raw[i] <= 0x37:
                    digits.append(raw[i])
                    i += 1
                else:
                    break
            out.append(int(bytes(digits), 8) & 0xFF)
            continue

        # Line continuation (backslash followed by EOL)
        if c in (0x0A, 0x0D):
            if c == 0x0D and i + 1 < len(raw) and raw[i + 1] == 0x0A:
                i += 2
            else:
                i += 1
            continue

        # Unknown escape: keep char as-is
        out.append(c)
        i += 1

    try:
        return out.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return out.decode("latin-1", errors="replace")


def _extract_literal_string_after(marker: bytes, data: bytes) -> Optional[str]:
    """
    Find marker bytes, then parse the following PDF literal string "(...)"
    with balanced parentheses. Returns the decoded string or None.
    """
    start = data.find(marker)
    if start < 0:
        return None

    i = start + len(marker)
    # Allow whitespace between marker and "("
    while i < len(data) and data[i] in b" \t\r\n":
        i += 1
    if i >= len(data) or data[i] != 0x28:  # '('
        return None

    i += 1
    depth = 1
    body = bytearray()
    while i < len(data) and depth > 0:
        b = data[i]
        if b == 0x5C:  # backslash escape
            body.append(b)
            i += 1
            if i < len(data):
                body.append(data[i])
                i += 1
            continue
        if b == 0x28:  # '('
            depth += 1
            body.append(b)
            i += 1
            continue
        if b == 0x29:  # ')'
            depth -= 1
            if depth == 0:
                i += 1
                break
            body.append(b)
            i += 1
            continue
        body.append(b)
        i += 1

    if depth != 0:
        return None
    return _decode_pdf_literal_string(bytes(body))


def _is_suspicious_title(title: str) -> bool:
    """Check if extracted title looks like extraction garbage."""
    t = unicodedata.normalize("NFKC", title)
    if _BOM_PREFIX_RE.match(t):
        return True

    # Heuristic: lots of single-letter tokens indicates bad extraction
    tokens = _WHITESPACE_RE.sub(" ", t).strip().split(" ")
    if not tokens:
        return True

    single_letter = sum(1 for tok in tokens if len(tok) == 1 and tok.isalpha())
    if single_letter / max(len(tokens), 1) > 0.35:
        return True

    return False


def _extract_title_from_metadata(pdf_bytes: bytes) -> Optional[str]:
    """Extract title from PDF Info dict or XMP metadata."""
    # Fast path: Info dict title
    title = _extract_literal_string_after(b"/Title", pdf_bytes)
    if title:
        title = title.strip()
        if title and title.lower() not in {"untitled", "title"}:
            return title

    # XMP dc:title fallback
    try:
        text = pdf_bytes.decode("utf-8", errors="ignore")
    except Exception:
        text = ""

    xmp_match = re.search(
        r"<dc:title>\s*<rdf:Alt>\s*<rdf:li[^>]*>(.*?)</rdf:li>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if xmp_match:
        candidate = re.sub(r"<[^>]+>", "", xmp_match.group(1)).strip()
        candidate = _WHITESPACE_RE.sub(" ", candidate)
        if candidate:
            return candidate

    return None


def _extract_title_from_first_page(pdf_path: Path) -> Optional[str]:
    """Extract title from first-page text using pdfminer.six."""
    try:
        from pdfminer.high_level import extract_text as extract_miner  # type: ignore
        text = extract_miner(str(pdf_path), maxpages=1) or ""
    except Exception:
        return None

    lines = [_WHITESPACE_RE.sub(" ", ln).strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]

    # Filter obvious non-title lines
    def looks_like_non_title(ln: str) -> bool:
        low = ln.lower()
        if "arxiv" in low or "doi" in low or "http" in low:
            return True
        if low in {"abstract", "introduction", "contents"}:
            return True
        if "@" in ln:  # emails
            return True
        if len(ln) < 18:
            return True
        return False

    candidates = [ln for ln in lines[:40] if not looks_like_non_title(ln)]
    if not candidates:
        return None

    # Join first 1-2 lines if they look like a wrapped title
    first = candidates[0]
    if len(candidates) >= 2 and len(first) < 80 and len(candidates[1]) < 80:
        joined = f"{first} {candidates[1]}".strip()
        # Avoid accidentally joining with author line
        if joined.count(",") <= 1:
            return joined

    return first


def extract_text(pdf_path: Path, max_pages: int | None = None) -> str:
    """Extract text using pdfminer.six (best for two-column layouts)."""
    try:
        from pdfminer.high_level import extract_text as extract_miner  # type: ignore
        # maxpages=0 means "all pages" in pdfminer
        txt = extract_miner(str(pdf_path), maxpages=max_pages or 0) or ""
        return txt.strip()
    except Exception:
        return ""


def extract_paper_from_pdf(pdf_path: Path, max_pages: int | None = None) -> Paper:
    """
    Extract title + full text from a PDF.

    Title extraction priority:
    1. Manual override (TITLE_OVERRIDES)
    2. PDF Info dict /Title metadata
    3. XMP dc:title metadata
    4. First-page text heuristics (pdfminer.six)
    5. Fallback to filename

    Text extraction:
    - pdfminer.six

    Returns: Paper with title + cleaned text
    """
    # 1. Try manual override first
    title = TITLE_OVERRIDES.get(pdf_path.name)

    # 2. Try metadata extraction
    if not title:
        pdf_bytes = pdf_path.read_bytes()
        title = _extract_title_from_metadata(pdf_bytes)
        if title and _is_suspicious_title(title):
            title = None

    # 3. Try first-page heuristics
    if not title:
        title = _extract_title_from_first_page(pdf_path)

    # 4. Fallback to filename
    if not title:
        title = pdf_path.stem.replace("_", " ").strip()

    # Extract text
    txt = extract_text(pdf_path, max_pages=max_pages)

    # Clean the text
    if txt:
        txt = clean_pdf_text(txt)

    return Paper(pdf_path=pdf_path, title=title, text=txt)
