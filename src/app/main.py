from app.embedding.model import EmbeddingModel
from app.qdrant.qdrant import QdrantSingleton


def main():
    embedding_model = EmbeddingModel()
    sentences = ["Hello, World!", "I am Bob!", "I am a Sailer!"]

    embeddings = embedding_model.encode(sentences)
    print("Dimension: ", len(embeddings[0]))  # 384 個向量
    print("embeddings: ", embeddings)
    embedding = embeddings[0].tolist()
    print("type of embedding: ", type(embedding), type(embedding[0]))

    # qdrant
    drunken_sailer = [
        {"id": "1", "lyric": "What will we do with a drunken sailor"},
        {"id": "2", "lyric": "Early in the morning"},
        {"id": "3", "lyric": "Way hay and up she rises"},
        {"id": "4", "lyric": "have his belly with a rusty razor"},
        {"id": "5", "lyric": "Put him in a long boat till his sober"},
    ]

    qdrant = QdrantSingleton()
    collection_name = "Lyrics"
    qdrant.recreate_collection(collection_name)

    embedding_array = [qdrant.get_embedding(text["lyric"]) for text in drunken_sailer]

    qdrant.upsert_vectors(embedding_array, collection_name, drunken_sailer)

    query_text = "Is drunken sailor a good song?"
    search_result = qdrant.search_for_qdrant(query_text, collection_name, limit_k=1)

    print(f"尋找: {query_text}", search_result)
