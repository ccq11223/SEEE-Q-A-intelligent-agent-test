"""重排：把「话题相邻但没答案」的资料压下去。

这组测试的由来：一道问「二端口网络 Z 参数怎么测量」的负样本，向量检索最高分
0.715（判为有依据），但召回的全是电容串并联的内容。模型于是用自己的知识
答了一遍，还挂上那些真实存在却内容无关的页码——引用校验只查页码存不存在，
查不出内容不对。这是最危险的一类错误。
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pipeline.retrieval import rerank_hits, retrieve
from pipeline.store import Hit
from tests.support import FakeEmbedder, FakeModelConfig, FakeStore, make_settings


class FakeReranker:
    def __init__(self, scores: list[float]) -> None:
        self.config = FakeModelConfig(name="fake-rerank")
        self.scores = scores
        self.calls: list[str] = []

    def rerank(self, *, query, documents, top_n=None):
        self.calls.append(query)
        return [(index, score) for index, score in enumerate(self.scores)]

    def close(self) -> None:
        pass


def hit(page_no: int, vector_score: float, content: str = "电容并联的等效电容") -> Hit:
    return Hit(
        chunk_key=f"slide-05#{page_no:04d}",
        page_no=page_no,
        page_end=None,
        chapter="第2章",
        section_title="电容的连接",
        content=content,
        trust_level=1,
        score=vector_score,
    )


class TestRerankHits(unittest.TestCase):
    def test_reorders_by_rerank_score(self) -> None:
        hits = [hit(1, 0.71), hit(2, 0.70), hit(3, 0.69)]
        reranked = rerank_hits(
            make_settings(Path(tempfile.mkdtemp())),
            FakeReranker([0.05, 0.92, 0.10]),
            "二端口网络的 Z 参数怎么测量",
            hits,
        )
        self.assertEqual([item.page_no for item in reranked], [2, 3, 1])
        self.assertAlmostEqual(reranked[0].score, 0.92)

    def test_keeps_original_vector_score_for_reference(self) -> None:
        hits = [hit(1, 0.71)]
        reranked = rerank_hits(
            make_settings(Path(tempfile.mkdtemp())),
            FakeReranker([0.05]),
            "q",
            hits,
        )
        self.assertAlmostEqual(reranked[0].score, 0.05)
        self.assertAlmostEqual(reranked[0].vector_score, 0.71)

    def test_without_client_returns_original_order(self) -> None:
        hits = [hit(1, 0.71), hit(2, 0.70)]
        result = rerank_hits(make_settings(Path(tempfile.mkdtemp())), None, "q", hits)
        self.assertEqual(result, hits)


class TestRetrieveGate(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.settings = make_settings(Path(self.tmp.name))

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_low_rerank_score_means_no_evidence(self) -> None:
        """正是 neg-003 的情形：向量分很高，重排分很低。"""
        store = FakeStore([hit(25, 0.715), hit(26, 0.70)])
        result = retrieve(
            self.settings,
            store,
            FakeEmbedder(),
            "二端口网络的 Z 参数怎么测量",
            rerank_client=FakeReranker([0.02, 0.01]),
        )
        self.assertFalse(result.has_evidence)
        self.assertIn("重排", result.reason)

    def test_high_rerank_score_means_evidence(self) -> None:
        store = FakeStore([hit(15, 0.60)])
        result = retrieve(
            self.settings,
            store,
            FakeEmbedder(),
            "串联电阻的分压公式",
            rerank_client=FakeReranker([0.88]),
        )
        self.assertTrue(result.has_evidence)

    def test_falls_back_to_vector_threshold_without_reranker(self) -> None:
        store = FakeStore([hit(15, 0.60)])
        result = retrieve(self.settings, store, FakeEmbedder(), "q")
        self.assertTrue(result.has_evidence)
        self.assertIn("向量", result.reason)

    def test_vector_low_score_still_gates_without_reranker(self) -> None:
        store = FakeStore([hit(15, 0.10)])
        result = retrieve(self.settings, store, FakeEmbedder(), "q")
        self.assertFalse(result.has_evidence)


if __name__ == "__main__":
    unittest.main()
