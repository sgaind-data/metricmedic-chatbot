"""Hybrid retrieval: BM25 keyword search fused with dense embeddings (Reciprocal Rank Fusion).

Why: the retrieval eval showed misses where users describe a symptom in plain words and
the KB uses different wording. Dense embeddings capture meaning; BM25 rewards exact
shared terms (e.g. "rounding", "union", "text"). RRF combines the two rankings without
needing their scores to be on the same scale.

All parameters are standard library defaults, fixed before evaluation (no tuning on the
golden set): BM25 k1=1.2, b=0.75 (Lucene/Elasticsearch defaults), RRF k=60 (Cormack et
al., 2009), scikit-learn's English stop-word list.

Each result keeps its dense cosine `score`, so the 0.25 out-of-scope threshold and the
CI gate keep their meaning. The fused rank is reported as `rrf_score`.
"""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from metricmedic.retriever import MetricMedicRetriever

BM25_K1 = 1.2
BM25_B = 0.75
RRF_K = 60


class BM25Index:
    """Okapi BM25 over a fixed list of documents."""

    def __init__(self, documents: list[str], k1: float = BM25_K1, b: float = BM25_B):
        self.k1, self.b = k1, b
        self.vectorizer = CountVectorizer(lowercase=True, stop_words="english")
        self.term_freqs = self.vectorizer.fit_transform(documents).tocsc().astype(float)
        n_docs = self.term_freqs.shape[0]
        self.doc_len = np.asarray(self.term_freqs.sum(axis=1)).ravel()
        self.avg_len = self.doc_len.mean()
        doc_freq = np.asarray((self.term_freqs > 0).sum(axis=0)).ravel()
        self.idf = np.log(1 + (n_docs - doc_freq + 0.5) / (doc_freq + 0.5))

    def scores(self, query: str) -> np.ndarray:
        """BM25 score of every document for the query (0 when no terms are shared)."""
        query_terms = self.vectorizer.transform([query]).indices
        norm = self.k1 * (1 - self.b + self.b * self.doc_len / self.avg_len)
        total = np.zeros(self.term_freqs.shape[0])
        for term in set(query_terms):
            freq = self.term_freqs[:, term].toarray().ravel()
            total += self.idf[term] * freq * (self.k1 + 1) / (freq + norm)
        return total


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = RRF_K) -> dict[int, float]:
    """Sum of 1 / (k + rank) over every ranking a document appears in (rank starts at 1)."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            fused[doc] = fused.get(doc, 0.0) + 1 / (k + rank)
    return fused


class HybridRetriever(MetricMedicRetriever):
    """Same API as MetricMedicRetriever; ranking = RRF(dense, BM25)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bm25 = BM25Index(self.documents)

    def retrieve(self, query, top_k=3):
        dense = super().retrieve(query, top_k=len(self.documents))
        dense_ranking = [r["kb_index"] for r in dense]

        keyword_scores = self.bm25.scores(query)
        keyword_ranking = [
            int(i) for i in np.argsort(-keyword_scores, kind="stable") if keyword_scores[i] > 0
        ]

        fused = reciprocal_rank_fusion([dense_ranking, keyword_ranking])
        by_index = {r["kb_index"]: r for r in dense}
        order = sorted(fused, key=lambda i: (-fused[i], dense_ranking.index(i)))[:top_k]
        return [
            {
                **by_index[i],
                "rrf_score": round(fused[i], 6),
                "bm25_score": round(float(keyword_scores[i]), 4),
            }
            for i in order
        ]
