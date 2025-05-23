"""
ResolverManager - central lookup that tries a
stack of concrete resolvers in priority order.
"""

from __future__ import annotations
from typing import Iterable, List, Optional
import asyncio

from core.Hit import Hit
from resolvers.base import Resolved
from resolvers.doi_resolver import DoiResolver
from resolvers.playwright_resolver import PlaywrightResolver
from resolvers.pdf_resolver import PdfResolver

# TODO: implement more resolvers
# from resolvers.mdpi import MDPIResolver          # deterministic /pdf rule
# from resolvers.playwright_fallback import PlaywrightResolver  # last-ditch, manually extract pdf

# specify the order here
_RESOLVER_CLASSES = [
    PdfResolver, # checking if the provided url is to a direct pdf link
    DoiResolver,
    # MDPIResolver,
    PlaywrightResolver,
]

class ResolverManager:
    """
    Try each registered resolver in order until one returns a PDF.
    """

    def __init__(self) -> None:
        self._chain = [cls() for cls in _RESOLVER_CLASSES]

    async def resolve(self, hit: Hit) -> Optional[Resolved]:
        for resolver in self._chain:
            try:
                result = await resolver.resolve(hit)
                if result:
                    return result
            except Exception as exc:
                print(f"[resolver:{resolver.name}] {exc}")
        return None

    async def resolve_many(self, hits: Iterable[Hit]) -> List[Resolved]:
        tasks = [self.resolve(h) for h in hits]
        if not tasks:
            print("ResolverManager: No hits to resolve")
            return []
        results = await asyncio.gather(*tasks)
        resolved = [r for r in results if r]
        print(f"ResolverManager resolved {len(resolved)} hits, {(len(resolved) / len(tasks)) * 100:.1f}% success rate")
        return resolved
