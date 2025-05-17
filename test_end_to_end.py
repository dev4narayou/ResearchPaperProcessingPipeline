from generators.keyword_expander import topic_to_queries
from searchers.meta_searcher import MetaSearcher
from typing import List
from core.Hit import Hit
import asyncio
import json
from dataclasses import asdict

topic = "creatine -impact on brain function"
queries: List[str] = topic_to_queries(topic)
hits: List[Hit] = asyncio.run(MetaSearcher().search(queries))

# for testing
def dump_hits_to_json(hits: List[Hit]) -> List[dict]:
    # Convert hits to JSON-serializable format
    hits_json = [asdict(hit) for hit in hits]

    # Export to JSON file
    with open('search_results.json', 'w', encoding='utf-8') as f:
        json.dump(hits_json, f, indent=2, ensure_ascii=False)

    print(f"Results exported to search_results.json")

dump_hits_to_json(hits)
