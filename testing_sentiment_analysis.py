from sentence_transformers import CrossEncoder

# # 1. Load a pretrained CrossEncoder model
# model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")

# # The texts for which to predict similarity scores
# query = "How many people live in Berlin?"
# passages = [
#     "Berlin had a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.",
#     "Berlin has a yearly total of about 135 million day visitors, making it one of the most-visited cities in the European Union.",
#     "In 2013 around 600,000 Berliners were registered in one of the more than 2,300 sport and fitness clubs.",
# ]

# # 2a. Either predict scores pairs of texts
# scores = model.predict([(query, passage) for passage in passages])
# print(scores)
# # => [8.607139 5.506266 6.352977]

# # 2b. Or rank a list of passages for a query
# ranks = model.rank(query, passages, return_documents=True)

# print("Query:", query)
# for rank in ranks:
#     print(f"- #{rank['corpus_id']} ({rank['score']:.2f}): {rank['text']}")
# """
# Query: How many people live in Berlin?
# - #0 (8.61): Berlin had a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.
# - #2 (6.35): In 2013 around 600,000 Berliners were registered in one of the more than 2,300 sport and fitness clubs.
# - #1 (5.51): Berlin has a yearly total of about 135 million day visitors, making it one of the most-visited cities in the European Union.
# """

from sentence_transformers import CrossEncoder

try:
    model = CrossEncoder("models/cross-encoder-ms-marco-MiniLM-L6-v2")
    print("Successfully loaded local model")
except:
    print("Could not load local model, downloading from HuggingFace...")
    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")
    model.save_pretrained("models/cross-encoder-ms-marco-MiniLM-L6-v2", from_pt=True)
    print("Model downloaded and saved locally")

# scores = model.predict([
#     ("How many people live in Berlin?", "Berlin had a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers."),
#     ("How many people live in Berlin?", "Berlin is well known for its museums."),
# ])
# print(scores)
# [ 8.607138 -4.320078]

def rank_sentences(query: str, sentences: list[str]) -> list[str]:
    scores = model.predict([(query, sentence) for sentence in sentences])
    return [{"sentence": sentence, "score": str(score)} for score, sentence in sorted(zip(scores, sentences), key=lambda x: x[0], reverse=True)]


if __name__ == "__main__":
    query = "llm architecture"
    import json
    with open("search_results.json", "r", encoding="utf-8") as f:
        hits = json.load(f)
        titles = [hit["title"] for hit in hits]
    ranked_titles = rank_sentences(query, titles)
    json_object = json.dumps(ranked_titles, indent=4)
    with open("ranked_titles.json", "w", encoding="utf-8") as f:
        f.write(json_object)



# e.g. download and save locally
# model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")
# model.save_pretrained("models/cross-encoder-ms-marco-MiniLM-L6-v2", from_pt=True)

# try:
#     model = CrossEncoder("models/cross-encoder-ms-marco-MiniLM-L6-v2")
#     print("Successfully loaded local model")
# except:
#     print("Could not load local model, downloading from HuggingFace...")
#     model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")
#     model.save_pretrained("models/cross-encoder-ms-marco-MiniLM-L6-v2", from_pt=True)
#     print("Model downloaded and saved locally")



