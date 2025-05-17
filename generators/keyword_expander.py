"""
Topic → Query-String generator
──────────────────────────────
Turn a short user topic into a handful of keyword-rich search strings that work well in academic APIs

Can be replaced with a LLM-based approach...?

"""

from __future__ import annotations
from typing import List, Set, Iterable
import itertools
import re

def topic_to_queries(topic: str, *, max_variants: int = 12) -> List[str]:
    """
    Expand a short topic into ≤ `max_variants` distinct query strings.
    """
    # TODO: Implement this
    return [topic]
