import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


DATA_PATH = Path(__file__).parent / "data" / "troubleshooting_guide.json"


class MetricMedicRetriever:
    """Semantic retriever using Sentence Transformers and FAISS."""

    def __init__(
        self,
        data_path=DATA_PATH,
        model_name="all-MiniLM-L6-v2",
    ):
        self.data_path = data_path
        self.model = SentenceTransformer(model_name)

        self.knowledge_base = self.load_knowledge_base()

        self.documents = [
            self._build_document(item)
            for item in self.knowledge_base
        ]

        self.embeddings = self._create_embeddings(self.documents)

        self.index = self._build_faiss_index(self.embeddings)

    def load_knowledge_base(self):
        """Load troubleshooting scenarios from JSON."""
        with open(self.data_path, "r", encoding="utf-8") as file:
            return json.load(file)

    @staticmethod
    def _build_document(item):
        """Combine issue, keywords, and response into one retrievable document."""
        return (
            f"Issue: {item['issue']}. "
            f"Keywords: {item['keywords']}. "
            f"Guidance: {item['response']}"
        )

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
        """Create a FAISS index using inner product similarity."""
        dimension = embeddings.shape[1]

        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)

        return index

    def retrieve(self, query, top_k=3):
        """Return the top matching troubleshooting scenarios."""
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        scores, indices = self.index.search(
            query_embedding,
            top_k,
        )

        results = []

        for score, index in zip(scores[0], indices[0]):
            if index == -1:
                continue

            item = self.knowledge_base[index]

            results.append(
                {
                    "score": float(score),
                    "issue": item["issue"],
                    "keywords": item["keywords"],
                    "response": item["response"],
                }
            )

        return results


if __name__ == "__main__":
    retriever = MetricMedicRetriever()

    test_query = (
        "My dashboard customer count doubled after joining two tables"
    )

    results = retriever.retrieve(test_query)

    print("\nQuery:")
    print(test_query)

    print("\nTop Matches:")

    for result in results:
        print("-" * 60)
        print(f"Issue: {result['issue']}")
        print(f"Score: {result['score']:.4f}")
        print(f"Guidance: {result['response']}")