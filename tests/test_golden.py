"""黄金测试集：格式校验与指标计算。

这组测试的意义：题集是「判断调教有没有变好」的唯一客观依据，
所以它的格式约束和指标算法必须可靠——写错的题集会给出看着正常、
实际没在测东西的结论。
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pipeline.golden import (
    CaseResult,
    GoldenCase,
    GoldenError,
    GoldenReport,
    describe_hits,
    load_cases,
    pages_from_hits,
    point_covered,
    render_report,
    summarize_cases,
)
from pipeline.store import Hit


def write_cases(rows: list[dict]) -> Path:
    tmp = Path(tempfile.mkdtemp()) / "qa.jsonl"
    tmp.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    return tmp


POSITIVE = {
    "id": "p1",
    "question": "替代定理的适用条件？",
    "negative": False,
    "expect_pages": [25, 26],
}
NEGATIVE = {"id": "n1", "question": "回转器怎么用？", "negative": True}


class TestLoadCases(unittest.TestCase):
    def test_loads_valid_file(self) -> None:
        cases = load_cases(write_cases([POSITIVE, NEGATIVE]))
        self.assertEqual([case.id for case in cases], ["p1", "n1"])
        self.assertEqual(cases[0].expect_pages, [25, 26])
        self.assertTrue(cases[1].negative)
        self.assertEqual(cases[1].expect_pages, [])

    def test_ignores_blank_lines_and_comments(self) -> None:
        path = write_cases([POSITIVE])
        path.write_text(
            "# 这是注释\n\n" + path.read_text(encoding="utf-8"), encoding="utf-8"
        )
        self.assertEqual(len(load_cases(path)), 1)

    def test_missing_file_mentions_readme(self) -> None:
        with self.assertRaises(GoldenError) as ctx:
            load_cases(Path(tempfile.mkdtemp()) / "nope.jsonl")
        self.assertIn("README", str(ctx.exception))

    def test_invalid_json_is_reported_with_line_number(self) -> None:
        path = write_cases([POSITIVE])
        path.write_text(path.read_text(encoding="utf-8") + "{坏的\n", encoding="utf-8")
        with self.assertRaises(GoldenError) as ctx:
            load_cases(path)
        self.assertIn("第 2 行", str(ctx.exception))

    def test_duplicate_id_is_rejected(self) -> None:
        with self.assertRaises(GoldenError) as ctx:
            load_cases(write_cases([POSITIVE, POSITIVE]))
        self.assertIn("重复", str(ctx.exception))

    def test_empty_question_is_rejected(self) -> None:
        with self.assertRaises(GoldenError) as ctx:
            load_cases(write_cases([{**POSITIVE, "question": "  "}]))
        self.assertIn("question", str(ctx.exception))

    def test_negative_with_expect_pages_is_rejected(self) -> None:
        """负样本期望命中页码，说明作者没想清楚它在测什么。"""
        with self.assertRaises(GoldenError) as ctx:
            load_cases(write_cases([{**NEGATIVE, "expect_pages": [3]}]))
        self.assertIn("负样本", str(ctx.exception))

    def test_positive_without_expect_pages_is_rejected(self) -> None:
        with self.assertRaises(GoldenError) as ctx:
            load_cases(write_cases([{**POSITIVE, "expect_pages": []}]))
        self.assertIn("expect_pages", str(ctx.exception))

    def test_draft_positive_may_omit_expect_pages(self) -> None:
        """还没想好页码的题先标 draft，不参与计分，也不该卡住录入。"""
        cases = load_cases(
            write_cases([{**POSITIVE, "expect_pages": [], "status": "draft"}])
        )
        self.assertFalse(cases[0].scorable)
        self.assertEqual(len([c for c in cases if c.scorable]), 0)

    def test_bad_status_is_rejected(self) -> None:
        with self.assertRaises(GoldenError) as ctx:
            load_cases(write_cases([{**POSITIVE, "status": "随时"}]))
        self.assertIn("status", str(ctx.exception))

    def test_non_integer_pages_are_rejected(self) -> None:
        with self.assertRaises(GoldenError):
            load_cases(write_cases([{**POSITIVE, "expect_pages": ["二十五"]}]))

    def test_summary_counts_drafts_separately(self) -> None:
        cases = load_cases(
            write_cases([POSITIVE, NEGATIVE, {**POSITIVE, "id": "p2", "status": "draft"}])
        )
        summary = summarize_cases(cases)
        self.assertIn("待计分 2 题", summary)
        self.assertIn("正样本 1", summary)
        self.assertIn("负样本 1", summary)
        self.assertIn("草稿 1 题", summary)


class TestPagesFromHits(unittest.TestCase):
    def test_expands_page_range(self) -> None:
        hits = [Hit("k", 25, 26, "", "", "", 1, 0.9)]
        self.assertEqual(pages_from_hits(hits), [25, 26])

    def test_dedupes_and_sorts(self) -> None:
        hits = [
            Hit("a", 5, None, "", "", "", 1, 0.9),
            Hit("b", 3, 4, "", "", "", 1, 0.8),
            Hit("c", 4, None, "", "", "", 1, 0.7),
        ]
        self.assertEqual(pages_from_hits(hits), [3, 4, 5])

    def test_skips_hits_without_page(self) -> None:
        hits = [Hit("a", None, None, "", "", "", 1, 0.9)]
        self.assertEqual(pages_from_hits(hits), [])

    def test_describe_hits_labels_range(self) -> None:
        hits = [
            Hit("a", 25, 26, "", "", "", 1, 0.687),
            Hit("b", 5, None, "", "", "", 1, 0.5),
        ]
        self.assertEqual(
            describe_hits(hits), [("第25-26页", 0.687), ("第5页", 0.5)]
        )


class TestCaseVerdicts(unittest.TestCase):
    def positive(self, **kwargs) -> CaseResult:
        case = GoldenCase(id="p1", question="q", expect_pages=[25, 26])
        return CaseResult(case=case, **kwargs)

    def negative(self, **kwargs) -> CaseResult:
        case = GoldenCase(id="n1", question="q", negative=True)
        return CaseResult(case=case, **kwargs)

    def test_retrieval_hit_is_loose_front_hit_is_strict(self) -> None:
        # 召回 20 条里有 25 页，但前 5 条（真正交给模型的）里没有
        result = self.positive(hit_pages=[25, 40, 41], front_pages=[40, 41])
        self.assertTrue(result.retrieval_hit)
        self.assertFalse(result.front_hit)
        self.assertEqual(result.case.expect_pages, [25, 26])

    def test_both_hit_when_page_is_in_front(self) -> None:
        result = self.positive(hit_pages=[25, 40], front_pages=[25])
        self.assertTrue(result.retrieval_hit)
        self.assertTrue(result.front_hit)

    def test_negative_never_counts_as_hit(self) -> None:
        result = self.negative(hit_pages=[1, 2], front_pages=[1])
        self.assertFalse(result.retrieval_hit)
        self.assertFalse(result.front_hit)

    def test_handoff_is_not_fabrication(self) -> None:
        self.assertFalse(self.negative(mode="handoff").fabricated)

    def test_admitting_the_gap_is_not_fabrication(self) -> None:
        result = self.negative(
            mode="kb", answer="现有资料没有覆盖回转器，不过资料里有这些滤波器内容（第31页）。"
        )
        self.assertFalse(result.fabricated)

    def test_answering_as_if_it_were_in_the_material_is_fabrication(self) -> None:
        result = self.negative(mode="kb", answer="回转器的回转电阻通常取 1kΩ 左右（第31页）。")
        self.assertTrue(result.fabricated)

    def test_retrieval_only_mode_does_not_judge_fabrication(self) -> None:
        # 没调用生成模型，mode 为空，不能拿它判编造
        self.assertFalse(self.negative(mode="").fabricated)


class TestPointCoverage(unittest.TestCase):
    """要点覆盖率是唯一能发现「页码引用正确、但结论说反了」的自动指标。"""

    def test_matches_rewritten_wording(self) -> None:
        # 题集里写「必须保证唯一解」，模型答成「要保证唯一解」
        self.assertTrue(point_covered("必须保证唯一解", "替代前后都要保证唯一解（第 25 页）。"))

    def test_detects_a_missing_point(self) -> None:
        self.assertFalse(
            point_covered("不能出现电压源回路", "替代定理适用于线性电路和非线性电路")
        )

    def test_ignores_punctuation_and_latex_marks(self) -> None:
        self.assertTrue(
            point_covered("原电路与替代后电路都必须有唯一解", "原电路、替代后的电路，都必须有唯一解。")
        )

    def test_short_points_use_substring(self) -> None:
        self.assertTrue(point_covered("唯一解", "要保证唯一解"))
        self.assertFalse(point_covered("唯一解", "要保证解的个数不限"))

    def test_empty_point_is_ignored(self) -> None:
        self.assertTrue(point_covered("   ", "任意回答"))

    def test_case_result_counts_hit_and_missed(self) -> None:
        case = GoldenCase(
            id="p1",
            question="q",
            expect_pages=[1],
            expect_points=["必须有唯一解", "不能出现电压源回路"],
        )
        result = CaseResult(
            case=case,
            mode="kb",
            citation_valid=True,
            points_hit=["必须有唯一解"],
            points_missed=["不能出现电压源回路"],
        )
        self.assertAlmostEqual(result.point_rate, 0.5)

    def test_point_rate_is_one_when_no_points_written(self) -> None:
        case = GoldenCase(id="p1", question="q", expect_pages=[1])
        self.assertEqual(CaseResult(case=case, mode="kb").point_rate, 1.0)

    def test_report_lists_low_coverage_cases(self) -> None:
        case = GoldenCase(
            id="p1",
            question="替代定理的适用条件",
            expect_pages=[25],
            expect_points=["不能出现电压源回路"],
        )
        report = GoldenReport(
            results=[
                CaseResult(
                    case=case,
                    mode="kb",
                    citation_valid=True,
                    hit_pages=[25],
                    front_pages=[25],
                    citations=[25],
                    points_missed=["不能出现电压源回路"],
                )
            ]
        )
        text = render_report(
            report,
            thresholds={"hit_rate": 0.8, "fabrication_rate": 0.05},
            generated_at="now",
        )
        self.assertIn("要点覆盖偏低", text)
        self.assertIn("不能出现电压源回路", text)

    def test_report_average_only_counts_answered_cases(self) -> None:
        # 转人工的题不该把要点覆盖率的均值拉低
        case_ok = GoldenCase(id="p1", question="q", expect_pages=[1], expect_points=["甲要点"])
        case_handoff = GoldenCase(
            id="p2", question="q", expect_pages=[2], expect_points=["乙要点"]
        )
        report = GoldenReport(
            results=[
                CaseResult(case=case_ok, mode="kb", points_hit=["甲要点"]),
                CaseResult(case=case_handoff, mode="handoff", points_missed=["乙要点"]),
            ]
        )
        self.assertEqual(report.point_rate, 1.0)


class TestReportMetrics(unittest.TestCase):
    def build(self, results: list[CaseResult]) -> GoldenReport:
        return GoldenReport(results=results)

    def test_rates(self) -> None:
        report = self.build(
            [
                CaseResult(
                    case=GoldenCase(id="p1", question="q", expect_pages=[1]),
                    hit_pages=[1],
                    front_pages=[1],
                    mode="kb",
                    citation_valid=True,
                ),
                CaseResult(
                    case=GoldenCase(id="p2", question="q", expect_pages=[9]),
                    hit_pages=[2],
                    front_pages=[2],
                    mode="handoff",
                    citation_valid=False,
                ),
                CaseResult(
                    case=GoldenCase(id="n1", question="q", negative=True),
                    mode="kb",
                    answer="回转电阻通常取 1kΩ。",
                ),
                CaseResult(
                    case=GoldenCase(id="n2", question="q", negative=True),
                    mode="handoff",
                ),
            ]
        )
        self.assertAlmostEqual(report.hit_rate, 0.5)
        self.assertAlmostEqual(report.front_hit_rate, 0.5)
        self.assertAlmostEqual(report.answer_rate, 0.5)
        self.assertAlmostEqual(report.handoff_rate, 0.5)
        self.assertAlmostEqual(report.fabrication_rate, 0.5)

    def test_empty_sets_do_not_divide_by_zero(self) -> None:
        report = self.build([])
        self.assertEqual(report.hit_rate, 1.0)
        self.assertEqual(report.fabrication_rate, 0.0)

    def test_report_highlights_fabrication_first(self) -> None:
        report = self.build(
            [
                CaseResult(
                    case=GoldenCase(id="n1", question="回转器怎么用？", negative=True),
                    mode="kb",
                    answer="回转电阻取 1kΩ。",
                )
            ]
        )
        text = render_report(
            report, thresholds={"hit_rate": 0.8, "fabrication_rate": 0.05}, generated_at="now"
        )
        self.assertIn("编造嫌疑", text)
        self.assertIn("n1", text)
        self.assertIn("不达标", text)

    def test_clean_report_says_达标(self) -> None:
        report = self.build(
            [
                CaseResult(
                    case=GoldenCase(id="p1", question="q", expect_pages=[1]),
                    hit_pages=[1],
                    front_pages=[1],
                    mode="kb",
                    citation_valid=True,
                )
            ]
        )
        text = render_report(
            report, thresholds={"hit_rate": 0.8, "fabrication_rate": 0.05}, generated_at="now"
        )
        self.assertIn("达标", text)
        # 只判小节标题：报告末尾的「怎么用」里也会提到这几个字
        self.assertNotIn("## ⚠️ 编造嫌疑", text)


if __name__ == "__main__":
    unittest.main()
