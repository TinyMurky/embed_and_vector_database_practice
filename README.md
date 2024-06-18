# 受歡迎的embedding model
- all-MiniLM-L6-v2
這個模組可以直接用 [sentence-transformers](https://sbert.net/) 套件使用，他把很多的transformers套件包裝起來

> ex:

```python
from sentence_transformers import SentenceTransformer

# 1. Load a pretrained Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# The sentences to encode
sentences = [
    "The weather is lovely today.",
    "It's so sunny outside!",
    "He drove to the stadium.",
]

# 2. Calculate embeddings by calling model.encode()
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# 3. Calculate the embedding similarities
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.6660, 0.1046],
#         [0.6660, 1.0000, 0.1411],
#         [0.1046, 0.1411, 1.0000]])
```

# docker 啟用
```
docker-compose up -d
or
docker compose up d
```


# qdrant
瀏覽器輸入 `http://localhost:6333/dashboard` 即可看到網頁(`http://localhost:6333`可以看到啟動了嗎？)

config.yaml下載：[點我](https://github.com/qdrant/qdrant/blob/master/config/config.yaml)
找到 下面這個部份可以設定密碼
```yaml
service:
  api_key: your_secret_api_key_here
```