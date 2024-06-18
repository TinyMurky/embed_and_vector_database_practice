from app.embedding.model import EmbeddingModel
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import PointStruct


class QdrantSingleton:
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(QdrantSingleton, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not QdrantSingleton._initialized:
            self.qdrant_client = QdrantClient(
                url="http://localhost",
                port=6333,
                # api_key="api_key",
            )
            self.embedding_model = EmbeddingModel()
            QdrantSingleton._initialized = True

    def recreate_collection(self, collection_name: str):
        """
        這個方法會重新建立collection，並設定collection的參數
        https://ithelp.ithome.com.tw/articles/10335513
        """
        # https://python-client.qdrant.tech/qdrant_client.http.models.models#qdrant_client.http.models.models.VectorParams
        # 一個用cosine算距離的向量，長度是384
        vectors_config = models.VectorParams(
            distance=models.Distance.COSINE,
            size=384,
        )

        # m代表每個節點近鄰數量。m值越大，查詢速度越快，但內存和構建時間也會增加。
        # ef_construct這是用於構建圖時的效率參數。較大的ef_construct值會導致更好的查詢品質，但會增加構建時間。代表在構建索引時搜索的節點數量
        hnsw_config = models.HnswConfigDiff(on_disk=True, m=16, ef_construct=100)

        # memmap_threshold是這表示當數據大小超過20000時，將使用內存映射來管理數據，這可以有效地處理大量數據並減少內存使用。
        optimizers_config = models.OptimizersConfigDiff(memmap_threshold=20000)

        self.qdrant_client.recreate_collection(
            collection_name=collection_name,
            vectors_config=vectors_config,
            hnsw_config=hnsw_config,
            optimizers_config=optimizers_config,
        )
        return self.qdrant_client

    def create_collection(self, collection_name: str):
        """
        create collection
        """
        vectors_config = models.VectorParams(
            distance=models.Distance.COSINE,
            size=384,
        )

        hnsw_config = models.HnswConfigDiff(on_disk=True, m=16, ef_construct=100)

        optimizers_config = models.OptimizersConfigDiff(memmap_threshold=20000)
        if not self.qdrant_client.collection_exists(collection_name):
            self.qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=vectors_config,
                hnsw_config=hnsw_config,
                optimizers_config=optimizers_config,
            )
        return self.qdrant_client

    def get_embedding(self, text: str) -> list[float]:
        """
        get embedding
        """
        embedding_list = self.embedding_model.encode([text])
        embedding = embedding_list[0]
        embedding_to_float_list = embedding.tolist()
        return embedding_to_float_list

    def upsert_vectors(
        self, vectors: list[list[float]], collection_name: str, data: list
    ):
        """
        upsert vectors
        payload is metadata, can be any data in dict
        """
        for i, vector in enumerate(vectors):
            self.qdrant_client.upsert(
                collection_name=collection_name,
                points=[
                    PointStruct(
                        id=i,
                        vector=vector,
                        payload=data[i],
                    )
                ],
            )
        print("upsert_vectors done")

    def search_for_qdrant(self, text: str, collection_name: str, limit_k: int):
        """
        search for qdrant
        """
        embedding_vector = self.get_embedding(text)
        search_result = self.qdrant_client.search(
            collection_name=collection_name,
            query_vector=embedding_vector,
            limit=limit_k,
            append_payload=True,
        )
        return search_result
