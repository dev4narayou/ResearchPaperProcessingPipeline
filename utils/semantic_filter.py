"""
Semantic relevance filter
─────────────────────────


Call:
    from backend.relevance.semantic_filter import semantic_filter

    filtered = semantic_filter(topic, hits, threshold=0.35)
"""


import functools
from sentence_transformers import CrossEncoder, SentenceTransformer


@functools.lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    try:
        model = CrossEncoder("models/cross-encoder-ms-marco-MiniLM-L6-v2")
        print("Successfully loaded local model")
    except:
        print("Could not load local model, downloading from HuggingFace...")
        model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")
        model.save_pretrained("models/cross-encoder-ms-marco-MiniLM-L6-v2", from_pt=True)
        print("Model downloaded and saved locally")
    return model

def rank_sentences(query: str, sentences: list[str]) -> list[str]:
    model = _get_model()
    scores = model.predict([(query, sentence) for sentence in sentences])
    return [{"sentence": sentence, "score": str(score)} for score, sentence in sorted(zip(scores, sentences), key=lambda x: x[0], reverse=True)]

def rank_sentence(query: str, sentence: str) -> float:
    model = _get_model()
    scores = model.predict([(query, sentence)])
    return scores[0]

if __name__ == "__main__":

    # # example usage, transforming a list of hits into a list of ranked sentences
    # query = "llm architecture"
    # import json
    # # make sure the json file is relevant
    # with open("search_results.json", "r", encoding="utf-8") as f:
    #     hits = json.load(f)
    #     titles = [hit["title"] for hit in hits]
    # ranked_titles = rank_sentences(query, titles)
    # json_object = json.dumps(ranked_titles, indent=4)
    # with open("ranked_titles.json", "w", encoding="utf-8") as f:
    #     f.write(json_object)

    # single sentence example
    query = "llm architecture"
    sentence = "A Novel LLM Architecture for Intelligent System Configuration"
    print(rank_sentence(query, sentence)) # should be pretty high (<10 but >0 or ideally >5)


