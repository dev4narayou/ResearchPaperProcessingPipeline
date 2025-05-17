"""
backend.searchers.meta_searcher
───────────────────────────────
Fan-out to each metadata provider concurrently, then normalise all
results into `Hit` objects and return a de-duplicated list.
"""

from external.crossref import CrossRefAPI
import asyncio
from typing import List, Iterable
import traceback
from dedupe.hit_merger import merge_hits
from core.Hit import Hit
import argparse
from pathlib import Path
import json

class MetaSearcher:

    def __init__(self):
        self.sources = [
            CrossRefAPI(),
            # TODO add more sources
        ]

    async def _gather(self, query: str) -> List[Hit]:
        """Run every provider for **one** query string in parallel."""
        async def _safe_call(provider):
            try:
                return await provider.search(query)
            except Exception as exc:
                # log and swallow so others still succeed
                print(f"[MetaSearcher] {provider.__class__.__name__} failed: {exc}")
                traceback.print_exc()
                return []

        grouped = await asyncio.gather(*(_safe_call(src) for src in self.sources))
        # flatten
        return [hit for group in grouped for hit in group]

    async def search(self, queries: Iterable[str]) -> List[Hit]:
        """
        Accepts *multiple* query strings, runs them concurrently using asyncio.gather(),
        merges all hits, and returns a de-duplicated, relevance-ranked list.
        """
        # concurrently run all queries
        results = await asyncio.gather(*(self._gather(q) for q in queries))
        # flatten
        raw: list[Hit] = [hit for group in results for hit in group]
        # de-duplicate & rank (merge_hits = our rapidfuzz logic)
        return merge_hits(raw)

