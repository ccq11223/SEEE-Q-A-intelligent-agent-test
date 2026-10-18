from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pipeline.assessment import (
    Grade,
    grade_answer,
    load_bank,
    pick_questions,
    summarize,
)
from tests.support import FakeGenerator, make_settings

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TestAssessmentBank(unittest.TestCase):
    def test_real_bank_is_valid(self) -> None:
        bank = load_bank(PROJECT_ROOT)
        self.assertGreaterEqual(len(bank), 5)
        self.assertEqual(len({item.id for item in bank}), len(bank))
        types = [item.type for item in bank]
        self.assertGreaterEqual(types.count("concept"), 2)
        self.assertGreaterEqual(types.count("calculation"), 2)
        self.assertGreaterEqual(types.count("comprehensive"), 1)
        for item in bank:
            self.assertTrue(item.question)
            self.assertTrue(item.answer)
            self.assertTrue(item.points)
            self.assertTrue(item.chapter)
            self.assertTrue(item.review)

    def test_pick_is_deterministic_and_follows_the_2_2_1_plan(self) -> None:
        bank = load_bank(PROJECT_ROOT)
        first = pick_questions(bank, "xh-abc", 5)
        second = pick_questions(bank, "xh-abc", 5)
        self.assertEqual([item.id for item in first], [item.id for item in second])
        self.assertEqual(
            [item.type for item in first],
            ["concept", "concept", "calculation", "calculation", "comprehensive"],
        )
        self.assertEqual(len({item.chapter for item in first}), 5)

    def test_different_seed_can_pick_different_set(self) -> None:
        bank = load_bank(PROJECT_ROOT)
        sets = {
            tuple(item.id for item in pick_questions(bank, seed, 5))
            for seed in ("a", "b", "c", "d", "e", "f")
        }
        self.assertGreater(len(sets), 1)


class TestSummary(unittest.TestCase):
    def setUp(self) -> None:
        self.bank = pick_questions(load_bank(PROJECT_ROOT), "xh-test", 5)

    def test_strong(self) -> None:
        result = summarize([(item, Grade(1, "")) for item in self.bank])
        self.assertEqual(result["level"], "strong")
        self.assertEqual(result["score"], 5)
        self.assertEqual(result["correct_ids"], [1, 2, 3, 4, 5])
        self.assertEqual(result["wrong_ids"], [])
        self.assertIn("仅供辅导同学参考", result["report_text"])

    def test_medium(self) -> None:
        result = summarize([(item, Grade(0.5, "", "符号不清")) for item in self.bank])
        self.assertEqual(result["level"], "medium")

    def test_weak(self) -> None:
        result = summarize([(item, Grade(0, "", "概念混淆")) for item in self.bank])
        self.assertEqual(result["level"], "weak")
        self.assertIn("概念混淆", result["weak_point_list"])
        self.assertEqual(result["diagnosis"], "基本概念模糊")
        self.assertEqual(len(result["wrong_ids"]), 5)
        self.assertTrue(result["weak_chapters"])
        self.assertTrue(result["review_points"])
        self.assertIn("从定义和物理意义讲起", result["tutor_note"])

    def test_concept_ok_but_calculation_weak(self) -> None:
        results = []
        for item in self.bank:
            score = 1 if item.type == "concept" else 0
            results.append((item, Grade(score, "")))
        summary = summarize(results)
        self.assertEqual(summary["diagnosis"], "概念基本掌握，但计算能力不足")
        self.assertIn("计算容易出错", summary["tutor_note"])


class TestGrading(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.settings = make_settings(Path(self.tmp.name))
        self.item = load_bank(PROJECT_ROOT)[0]

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_parses_json_grade(self) -> None:
        client = FakeGenerator(
            ['```json\n{"score": 0.5, "feedback": "方向说反了", "weak_point": "电流方向"}\n```']
        )
        grade = grade_answer(
            self.settings, client, self.item, "我的回答", course_name="电路理论"
        )
        self.assertEqual(grade.score, 0.5)
        self.assertEqual(grade.feedback, "方向说反了")
        self.assertEqual(grade.weak_point, "电流方向")

    def test_bad_json_falls_back_without_blocking(self) -> None:
        client = FakeGenerator(["我判断这道题是对的"])
        grade = grade_answer(
            self.settings, client, self.item, "我的回答", course_name="电路理论"
        )
        self.assertEqual(grade.score, 0.5)
        self.assertTrue(grade.feedback)
