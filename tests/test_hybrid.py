"""Unit tests for BM25 and Reciprocal Rank Fusion (no model download, no API calls)."""

import numpy as np

from metricmedic.hybrid import BM25Index, reciprocal_rank_fusion

DOCS = [
    "Rounding differences cause totals to differ by a few cents",
    "Duplicate records after UNION of two regional tables",
    "Data type mismatch when joining a text key to a numeric key",
]


def test_bm25_ranks_the_document_sharing_rare_terms_first():
    scores = BM25Index(DOCS).scores("totals differ by a few cents")
    assert int(np.argmax(scores)) == 0


def test_bm25_is_zero_when_no_terms_are_shared():
    scores = BM25Index(DOCS).scores("weather forecast tomorrow")
    assert np.all(scores == 0)


def test_bm25_ignores_case():
    index = BM25Index(DOCS)
    assert np.allclose(index.scores("UNION"), index.scores("union"))


def test_rrf_rewards_documents_ranked_well_by_both():
    fused = reciprocal_rank_fusion([[0, 1, 2], [1, 0, 2]], k=60)
    assert fused[0] == fused[1] > fused[2]


def test_rrf_includes_documents_from_either_ranking():
    fused = reciprocal_rank_fusion([[0], [5]], k=60)
    assert set(fused) == {0, 5}
    assert fused[0] == 1 / 61
