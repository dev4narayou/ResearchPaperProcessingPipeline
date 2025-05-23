from core.Hit import Hit
from utils.http import get
from utils.semantic_filter import rank_sentences, rank_sentence
from typing import List

class CrossRefAPI:
    BASE = "https://api.crossref.org/works"

    async def search(self, query: str, limit: int = 1000) -> list[Hit]:
        r = await get(self.BASE, query=query, rows=limit, select="DOI,title,URL,issued")
        items = r.json()["message"]["items"]
        hits = []
        for it in items:
            hits.append(Hit(
                title=it.get("title", [""])[0],
                doi=it.get("DOI"),
                url=it.get("URL"),
                year=it.get("issued", {}).get("date-parts", [[None]])[0][0],
                source="crossref",
                score=0.8,
            ))
        return self.filter_hits(hits, query)

    def filter_hits(self, hits: list[Hit], original_query: str) -> list[Hit]:
        """
        largely for testing, filter out low quality hits, return modified list
        """
        # this is not very robust, all words are handled equivalently, when in reality they shouldn't be
        def _simple_keyword_in_title_filter(hit: Hit) -> bool:
            words = original_query.lower().split()
            for word in words:
                if word in hit.title.lower():
                    return True
            return False

        def _published_year_filter(hit: Hit, year_range: tuple[int, int]) -> bool:
            if hit.year is None:
                return False
            return year_range[0] <= hit.year <= year_range[1]

        def _semantic_filter(original_query: str, hits: List[Hit]) -> List[Hit]:
            """
            Uses semantic filtering to rank the hits.
            Modifies the hits in place.
            Returns the modified list of hits.
            """
            new_hits = []
            for hit in hits:
                hit.score = str(rank_sentence(original_query, hit.title))
                if float(hit.score) > -0.008331001:
                    new_hits.append(hit)
            return new_hits

        initial_hits = list(filter(lambda h: _simple_keyword_in_title_filter(h) and _published_year_filter(h, (2015, 2025)), hits))
        ranked_hits = _semantic_filter(original_query, initial_hits)
        return ranked_hits


async def search_crossref(query: str, max_results: int = 30) -> list[Hit]:
    """Search Crossref API and return results as Hit objects."""
    api = CrossRefAPI()
    return await api.search(query, limit=max_results)

if __name__ == "__main__":
    import asyncio

    x = asyncio.run(search_crossref("machine learning"))
    print(x)
