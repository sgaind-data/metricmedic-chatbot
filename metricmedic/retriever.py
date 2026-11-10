"""Core semantic retriever: Sentence Transformers + FAISS.

Same public API as the old rag.py (knowledge_base, documents, embeddings,
index, retrieve), plus an on-disk embedding cache.
"""

import hashlib
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from metricmedic.config import CACHE_DIR, DATA_PATH, EMBEDDING_MODEL


class MetricMedicRetriever:
    """Semantic retriever using Sentence Transformers and FAISS."""

    def __init__(
        self,
        data_path=DATA_PATH,
        model_name=EMBEDDING_MODEL,
        cache_dir=CACHE_DIR,
    ):
        self.data_path = Path(data_path)
        self.model_name = model_name
        self.cache_dir = Path(cache_dir) if cache_dir else None

        self.model = SentenceTransformer(model_name)
        self.knowledge_base = self.load_knowledge_base()
        self.documents = [self._build_document(item) for item in self.knowledge_base]
        self.embeddings = self._load_or_create_embeddings()
        self.index = self._build_faiss_index(self.embeddings)

    def load_knowledge_base(self):
        """Load troubleshooting scenarios from JSON."""
        with open(self.data_path, encoding="utf-8") as file:
            return json.load(file)

    @staticmethod
    def _build_document(item):
        """Combine issue, keywords, and response into one retrievable document."""
        return f"Issue: {item['issue']}. Keywords: {item['keywords']}. Guidance: {item['response']}"

    def _cache_path(self):
        """Cache key changes whenever the model or any KB text changes."""
        digest = hashlib.sha256(
            (self.model_name + "\n" + "\n".join(self.documents)).encode("utf-8")
        ).hexdigest()[:16]
        return self.cache_dir / f"kb_embeddings_{digest}.npy"

    def _load_or_create_embeddings(self):
        if self.cache_dir is None:
            return self._create_embeddings(self.documents)

        path = self._cache_path()
        if path.exists():
            return np.load(path).astype("float32")

        embeddings = self._create_embeddings(self.documents)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(path, embeddings)
        return embeddings

    def _create_embeddings(self, documents):
        """Create normalized semantic embeddings for the knowledge base."""
        embeddings = self.model.encode(
            documents,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.astype("float32")

    @staticmethod
    def _build_faiss_index(embeddings):
        """Create a FAISS index using inner product (cosine on normalized vectors)."""
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        return index

    def retrieve(self, query, top_k=3):
        """Return the top matching troubleshooting scenarios."""
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, index in zip(scores[0], indices[0], strict=True):
            if index == -1:
                continue
            item = self.knowledge_base[index]
            results.append(
                {
                    "score": float(score),
                    "issue": item["issue"],
                    "keywords": item["keywords"],
                    "response": item["response"],
                    "kb_index": int(index),
                }
            )
        return results
