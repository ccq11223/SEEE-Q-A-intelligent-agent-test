from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pipeline.answer import (
    MODE_GENERAL,
    MODE_HANDOFF,
    MODE_KB,
    answer_question,
    build_retrieval_query,
)
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

    def test_followup_question_uses_previous_turn_for_retrieval(self) -> None:
        history = [{"role": "user", "content": "怎么列写节点支路方程？"}]
        store = FakeStore([hit(12)])
        generator = FakeGenerator(["说明（第 12 页）。"])
        answer_question(
            self.settings,
            store,
            self.embedder,
            generator,
            "不是，是那种通用的结点方程",
            history=history,
        )
        self.assertIn("怎么列写节点支路方程", self.embedder.calls[-1][0])
        self.assertIn("不是，是那种通用的结点方程", self.embedder.calls[-1][0])
        # 生成时也带上上下文，模型才知道「那种」指什么
        self.assertIn("之前的对话", generator.prompts[0])

    def test_standalone_question_is_not_polluted_by_history(self) -> None:
        history = [{"role": "user", "content": "怎么理解特勒根定理？"}]
        self.assertEqual(
            build_retrieval_query("戴维南定理的适用条件是什么？", history),
            "戴维南定理的适用条件是什么？",
        )

    def test_followup_query_keeps_recent_two_user_turns(self) -> None:
        history = [
            {"role": "user", "content": "第一问"},
            {"role": "assistant", "content": "第一答"},
            {"role": "user", "content": "第二问"},
        ]
        query = build_retrieval_query("那它呢？", history)
        self.assertIn("第一问", query)
        self.assertIn("第二问", query)
        self.assertNotIn("第一答", query)

    def test_low_score_falls_back_to_general_and_records_gap(self) -> None:
        store = FakeStore([hit(12, score=0.05)])
        generator = FakeGenerator(["节点电流定律说明流入等于流出。"])
        answer = self.run_answer(store, generator)
        self.assertEqual(answer.mode, MODE_GENERAL)
        self.assertEqual(len(store.gaps), 1)
        self.assertEqual(len(generator.prompts), 1)
        self.assertIn("课件未覆盖", answer.text.splitlines()[0])

    def test_empty_knowledge_base_still_answers_from_general_knowledge(self) -> None:
        store = FakeStore([])
        answer = self.run_answer(store, FakeGenerator(["节点电流定律是 KCL 的另一种叫法。"]))
        self.assertEqual(answer.mode, MODE_GENERAL)
        self.assertEqual(len(store.gaps), 1)

    def test_invalid_citation_is_retried_then_answered_externally(self) -> None:
        store = FakeStore([hit(12)])
        generator = FakeGenerator(
            ["结论见（第 99 页）。", "仍然见（第 98 页）。", "改用通用知识作答。"]
        )
        answer = self.run_answer(store, generator)
        self.assertEqual(answer.mode, MODE_GENERAL)
        self.assertTrue(answer.citation_valid)
        self.assertEqual(len(generator.prompts), 3)
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

    def test_answer_without_citation_falls_back_to_general(self) -> None:
        store = FakeStore([hit(12)])
        generator = FakeGenerator(["这是一个没有任何页码引用的回答。"] * 2)
        answer = self.run_answer(store, generator)
        self.assertEqual(answer.mode, MODE_GENERAL)
        self.assertEqual(len(generator.prompts), 3)

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

    def test_web_is_off_by_default_but_general_still_answers(self) -> None:
        self.assertFalse(self.settings.retrieval.get("allow_web"))
        store = FakeStore([hit(12, score=0.01)])
        answer = self.run_answer(store, FakeGenerator(["通用知识回答。"]))
        self.assertEqual(answer.mode, MODE_GENERAL)

    def test_no_fallback_configured_hands_off(self) -> None:
        settings = make_settings(
            Path(self.tmp.name), retrieval={"allow_general": False}
        )
        store = FakeStore([hit(12, score=0.01)])
        answer = answer_question(
            settings, store, self.embedder, FakeGenerator(), "节点电流定律是什么？"
        )
        self.assertEqual(answer.mode, MODE_HANDOFF)
        self.assertEqual(len(store.gaps), 1)

    def test_answer_without_citation_is_retried_with_clear_instruction(self) -> None:
        # 实测：模型写出一段很好的回答但一个页码都没标，被引用校验拦下。
        # 老代码在这种情况拼出的是「引用了参考资料中不存在的页码（）」这种空话。
        store = FakeStore([hit(12)])
        generator = FakeGenerator(
            ["这个定理适用于线性电路和非线性电路。", "这个定理适用于线性电路（第 12 页）。"]
        )
        answer = self.run_answer(store, generator)
        self.assertEqual(answer.mode, MODE_KB)
        self.assertIn("第 12 页", generator.prompts[1])
        self.assertIn("一个页码都没有标注", generator.prompts[1])
        self.assertNotIn("不存在的页码（）", generator.prompts[1])

    def test_answer_that_admits_a_gap_is_recorded(self) -> None:
        # 实测：学生问「回转器怎么用在有源滤波器设计里」，模型老实说
        # 「现有资料没有覆盖回转器」，然后列了资料里确实有的滤波器内容并标了页码。
        # 这个回答诚实且有用，但走的是 kb 分支，缺口以前不会被记下来。
        store = FakeStore([hit(12)])
        generator = FakeGenerator(
            [
                "现有资料没有覆盖回转器，不过资料里有这些有源滤波器内容（第 12 页）。"
            ]
        )
        answer = self.run_answer(store, generator)
        self.assertEqual(answer.mode, MODE_KB)
        self.assertEqual(store.gaps, ["节点电流定律是什么？"])
        self.assertIn("知识缺口", answer.note)

    def test_normal_kb_answer_records_no_gap(self) -> None:
        store = FakeStore([hit(12)])
        generator = FakeGenerator(["节点电流定律说明流入等于流出（第 12 页）。"])
        self.run_answer(store, generator)
        self.assertEqual(store.gaps, [])

    def test_mentioning_a_missing_distractor_is_not_a_gap(self) -> None:
        # 实测：一道选择题答对了 B，但答案里说「资料里没有提到 C 那种情况」，
        # 宽泛的匹配把它误记成了知识缺口。
        store = FakeStore([hit(12)])
        generator = FakeGenerator(
            [
                "答案：B\n\n"
                "直流下电感相当于短路、电容相当于开路（第 12 页）。"
                "资料里没有提到 C 选项所说的那种频率关系。"
            ]
        )
        answer = self.run_answer(store, generator)
        self.assertEqual(answer.mode, MODE_KB)
        self.assertEqual(store.gaps, [])

    def test_gap_stated_late_in_the_answer_is_ignored(self) -> None:
        store = FakeStore([hit(12)])
        generator = FakeGenerator(
            [
                "根据定义（第 12 页），结论如下。" + "补充说明。" * 100
                + "另外，现有资料没有覆盖这个扩展问题。"
            ]
        )
        self.run_answer(store, generator)
        self.assertEqual(store.gaps, [])

    def test_gap_stated_at_the_start_is_still_recorded(self) -> None:
        store = FakeStore([hit(12)])
        generator = FakeGenerator(
            ["**结论**：现有资料没有覆盖回转器，不过资料里有这些滤波器内容（第 12 页）。"]
        )
        self.run_answer(store, generator)
        self.assertEqual(len(store.gaps), 1)

    def test_explicit_marker_counts_even_late_in_the_answer(self) -> None:
        store = FakeStore([hit(12)])
        generator = FakeGenerator(
            [
                "先说明一下背景（第 12 页）。" + "补充。" * 80
                + "\n\n【资料未覆盖】这道题涉及的内容课件里没有。"
            ]
        )
        self.run_answer(store, generator)
        self.assertEqual(len(store.gaps), 1)

    def test_bare_no_coverage_phrase_at_start_counts(self) -> None:
        # 实测 neg-002 的措辞是「这些内容没有覆盖」，不含「资料」二字
        store = FakeStore([hit(12)])
        generator = FakeGenerator(
            ["结论：非线性电路的小信号分析法，这些内容没有覆盖（第 12 页）。"]
        )
        self.run_answer(store, generator)
        self.assertEqual(len(store.gaps), 1)


if __name__ == "__main__":
    unittest.main()
