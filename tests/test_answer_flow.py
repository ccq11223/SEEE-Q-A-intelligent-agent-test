from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pipeline.answer import MODE_HANDOFF, MODE_KB, answer_question
from pipeline.store import Hit
from tests.support import FakeEmbedder, FakeGenerator, FakeStore, make_settings


def hit(page_no: int, score: float = 0.92, content: str = "节点电流定律：流入等于流出。") -> Hit:
    return Hit(
        chunk_key="doc-x#0001",
        page_no=page_no,
        page_end=None,
        chapter="第 1 章",
        section_title="1.1 基尔霍夫定律",
        content=content,
        trust_level=1,
        score=score,
    )


class TestAnswerFlow(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.settings = make_settings(Path(self.tmp.name))
        self.embedder = FakeEmbedder()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def run_answer(self, store, generator, question="节点电流定律是什么？"):
        return answer_question(
            self.settings, store, self.embedder, generator, question
        )

    def test_answers_from_knowledge_base_with_valid_citation(self) -> None:
        store = FakeStore([hit(12)])
        generator = FakeGenerator(["节点电流定律说明流入等于流出（第 12 页）。"])
        answer = self.run_answer(store, generator)
        self.assertEqual(answer.mode, MODE_KB)
        self.assertEqual(answer.citations, [12])
        self.assertTrue(answer.citation_valid)
        self.assertEqual(len(store.qa_logs), 1)
        self.assertEqual(store.gaps, [])

    def test_low_score_triggers_handoff_and_records_gap(self) -> None:
        store = FakeStore([hit(12, score=0.05)])
        generator = FakeGenerator(["不该被调用（第 12 页）"])
        answer = self.run_answer(store, generator)
        self.assertEqual(answer.mode, MODE_HANDOFF)
        self.assertEqual(len(store.gaps), 1)
        self.assertEqual(generator.prompts, [])

    def test_empty_knowledge_base_triggers_handoff(self) -> None:
        store = FakeStore([])
        answer = self.run_answer(store, FakeGenerator())
        self.assertEqual(answer.mode, MODE_HANDOFF)
        self.assertEqual(len(store.gaps), 1)

    def test_invalid_citation_is_retried_then_handed_off(self) -> None:
        store = FakeStore([hit(12)])
        generator = FakeGenerator(
            ["结论见（第 99 页）。", "仍然见（第 98 页）。"]
        )
        answer = self.run_answer(store, generator)
        self.assertEqual(answer.mode, MODE_HANDOFF)
        self.assertFalse(answer.citation_valid)
        self.assertEqual(len(generator.prompts), 2)
        self.assertIn("第 99 页", generator.prompts[1])
        self.assertEqual(len(store.gaps), 1)

    def test_retry_succeeds_with_correct_citation(self) -> None:
        store = FakeStore([hit(12)])
        generator = FakeGenerator(
            ["结论见（第 99 页）。", "更正：结论见（第 12 页）。"]
        )
        answer = self.run_answer(store, generator)
        self.assertEqual(answer.mode, MODE_KB)
        self.assertEqual(answer.citations, [12])

    def test_answer_without_citation_is_rejected(self) -> None:
        store = FakeStore([hit(12)])
        generator = FakeGenerator(["这是一个没有任何页码引用的回答。"] * 2)
        answer = self.run_answer(store, generator)
        self.assertEqual(answer.mode, MODE_HANDOFF)

    def test_complex_question_is_handed_off_without_calling_models(self) -> None:
        store = FakeStore([hit(12)])
        generator = FakeGenerator()
        answer = self.run_answer(store, generator, question="请给一道跨章节综合题并推导")
        self.assertEqual(answer.mode, MODE_HANDOFF)
        self.assertIn("复杂题", answer.note)
        self.assertEqual(generator.prompts, [])
        self.assertEqual(self.embedder.calls, [])

    def test_page_range_citation_is_accepted(self) -> None:
        store = FakeStore([Hit("k", 12, 14, "第1章", "1.1", "跨页内容", 1, 0.9)])
        generator = FakeGenerator(["跨页结论（第 13 页）。"])
        answer = self.run_answer(store, generator)
        self.assertEqual(answer.mode, MODE_KB)
        self.assertEqual(answer.citations, [13])

    def test_web_branch_is_off_by_default(self) -> None:
        self.assertFalse(self.settings.retrieval.get("allow_web"))
        store = FakeStore([hit(12, score=0.01)])
        answer = self.run_answer(store, FakeGenerator())
        self.assertEqual(answer.mode, MODE_HANDOFF)


if __name__ == "__main__":
    unittest.main()
