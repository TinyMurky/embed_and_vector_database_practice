from app.embedding.model import EmbeddingModel
from app.qdrant.qdrant import QdrantSingleton


def main():
    embedding_model = EmbeddingModel()
    sentences = ["Hello, World!", "I am Kyaru!", "I am a virtual YouTuber!"]

    embeddings = embedding_model.encode(sentences)
    print("Dimension: ", len(embeddings[0]))  # 384 個向量
    print("embeddings: ", embeddings)
    embedding = embeddings[0].tolist()
    print("type of embedding: ", type(embedding), type(embedding[0]))

    # qdrant
    american_idiots = [
        {"id": "1", "lyric": "Don't wanna be an American idiot"},
        {"id": "2", "lyric": "Don't want a nation under the new media"},
        {"id": "3", "lyric": "And can you hear the sound of hysteria?"},
        {"id": "4", "lyric": "The subliminal mindfuck America"},
    ]

    qdrant = QdrantSingleton()
    collection_name = "Lyrics"
    qdrant.recreate_collection(collection_name)

    embedding_array = [qdrant.get_embedding(text["lyric"]) for text in american_idiots]

    qdrant.upsert_vectors(embedding_array, collection_name, american_idiots)

    query_text = "stupid american"
    search_result = qdrant.search_for_qdrant(query_text, collection_name, limit_k=1)

    print(f"尋找: {query_text}", search_result)
