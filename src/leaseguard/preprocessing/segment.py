"""Split normalized text into headings, sections, clauses, tables, and exhibits."""

import re

from leaseguard.preprocessing.models import BlockKind, PageRecord, TextBlock
from leaseguard.preprocessing.normalize import normalize_whitespace

HEADING_PATTERN = re.compile(
    r"^(?:"
    r"\#{1,3}\s+.+"
    r"|article\s+[ivxlcdm0-9]+(?:\s*[.:-].*)?"
    r"|section\s+[0-9]+(?:\.[0-9]+)*(?:\s*[.:-].*)?"
    r"|(?:exhibit|schedule|appendix)\s+[a-z0-9]+(?:\s*[.:-].*)?"
    r"|[0-9]+\.(?:[0-9]+\.?)*\s+[A-Z].+"
    r")$",
    re.IGNORECASE,
)
EXHIBIT_PATTERN = re.compile(r"^(?:exhibit|schedule|appendix)\b", re.IGNORECASE)
TABLE_PATTERN = re.compile(r"\|")


def segment_pages(pages: list[PageRecord]) -> tuple[str, list[TextBlock]]:
    """Build normalized text and offset-aligned blocks from parsed pages."""
    blocks: list[TextBlock] = []
    parts: list[str] = []
    cursor = 0
    sequence = 1
    for page in pages:
        page_text = normalize_whitespace(page.text)
        if not page_text:
            continue
        if parts:
            cursor = _append_separator(parts, cursor)
        for raw_line in page_text.split("\n"):
            line = raw_line.strip()
            if not line:
                continue
            if parts and not parts[-1].endswith("\n"):
                cursor = _append_separator(parts, cursor, separator="\n")
            kind = _classify(line)
            heading = line.lstrip("# ").strip() if kind in {"heading", "exhibit"} else None
            start = cursor
            parts.append(line)
            cursor += len(line)
            blocks.append(
                TextBlock(
                    block_id=f"block-{sequence:04d}",
                    kind=kind,
                    text=line,
                    page_number=page.page_number,
                    heading=heading,
                    start_char=start,
                    end_char=cursor,
                )
            )
            sequence += 1
    normalized_text = "".join(parts)
    return normalized_text, blocks


def _append_separator(parts: list[str], cursor: int, separator: str = "\n\n") -> int:
    parts.append(separator)
    return cursor + len(separator)


def _classify(line: str) -> BlockKind:
    if EXHIBIT_PATTERN.match(line.lstrip("# ").strip()):
        return "exhibit"
    if TABLE_PATTERN.search(line) and line.count("|") >= 1:
        return "table"
    if HEADING_PATTERN.match(line):
        return "heading"
    return "paragraph"
