"""Prompt candidate-list parsing used by the standard GRIP evaluator."""

from __future__ import annotations

import re


def parse_listed_relations(question: str) -> list[str]:
    """Parse the semicolon-separated candidate list in a GRIP prompt."""
    match = re.search(
        r"candidate answers:\s*(.+)$",
        str(question),
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return []
    return [item.strip().rstrip(".") for item in match.group(1).split(";") if item.strip()]
