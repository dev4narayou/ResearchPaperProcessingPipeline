from typing import List, Union
from core.Hit import Hit

def merge_hits(hits: List[Union[Hit, dict]]) -> List[Hit]:
    """Merge duplicate hits based on DOI."""
    seen_dois = set()
    unique_hits = []

    for hit in hits:
        if isinstance(hit, dict):
            hit = Hit(
                title=hit.get("title", ""),
                doi=hit.get("doi", ""),
                url=hit.get("url", ""),
                year=hit.get("year", 0),
                source=hit.get("source", "unknown"),
                score=hit.get("score", 0.0)
            )

        if hit.doi and hit.doi not in seen_dois:
            seen_dois.add(hit.doi)
            unique_hits.append(hit)

    return unique_hits
