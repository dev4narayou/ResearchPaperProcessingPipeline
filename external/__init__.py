"""
backend.external
────────────────
Package that groups thin async wrappers around external
research-metadata APIs.  Each sub-module implements exactly one
callable named `search_<provider>` that returns `list[Hit]`.

Re-export them here so other code can simply do:

    from backend.external.semantic_scholar import search_s2
"""

from importlib import import_module

_provider_modules = (
    "crossref",
)

for _mod in _provider_modules:
    import_module(f"{__name__}.{_mod}")

# Re-export the expected callables for convenient direct import
from .crossref import search_crossref          # noqa: E402

__all__ = [
    "search_crossref",
]
