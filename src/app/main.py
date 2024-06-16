from app.embedding.model import EmbeddingModel


def main():
    embedding_model = EmbeddingModel()
    sentences = ["Hello, World!", "I am Kyaru!", "I am a virtual YouTuber!"]

    embeddings = embedding_model.encode(sentences)
    print("Dimension: ", len(embeddings[0]))  # 384 個向量
    print("embeddings: ", embeddings)

    decoded_sentences = embedding_model.decode(embeddings)
    print("decoded_sentences: ", decoded_sentences)
