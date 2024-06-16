"""
this module is responsible for the embedding model
"""

from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """
    this class is responsible for the embedding model
    """

    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model: SentenceTransformer = SentenceTransformer(model_name)

    def encode(self, text: list[str]):
        """
        change the texts into embeddings
        """
        return self.model.encode(text)
