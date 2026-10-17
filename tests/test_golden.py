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
    PLAN_TARGETS,
    CaseResult,
    GoldenCase,
    GoldenError,
    GoldenReport,
    build_worksheet,
    compose_question,
    describe_hits,
    extract_choice,
    foreign_docs,
    load_cases,
    pages_from_hits,
    plan_status,
    point_covered,
    render_report,
    run_golden,
    summarize_cases,
)
from pipeline.store import Hit
from tests.support import FakeEmbedder, FakeGenerator, FakeStore


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

    def test_filters_by_document(self) -> None:
        """不同资料各有自己的第 25 页，必须按资料过滤。

        实测 bug：期望 slide-08 第 9、13 页，实际命中的却是别的资料第 52、53 页，
        而只比页码数字的判分把它当成了命中。
        """
        hits = [
            Hit("slide-08#0003", 9, 10, "", "", "", 1, 0.9),
            Hit("slide-10#0007", 52, 53, "", "", "", 1, 0.8),
        ]
        self.assertEqual(pages_from_hits(hits, doc_key="slide-08"), [9, 10])
        self.assertEqual(pages_from_hits(hits, doc_key="slide-10"), [52, 53])

    def test_foreign_docs_reports_offtarget_retrieval(self) -> None:
        hits = [
            Hit("slide-08#0003", 9, None, "", "", "", 1, 0.9),
            Hit("slide-10#0007", 52, None, "", "", "", 1, 0.8),
            Hit("slide-10#0008", 53, None, "", "", "", 1, 0.7),
        ]
        self.assertEqual(foreign_docs(hits, "slide-08"), ["slide-10×2"])
        self.assertEqual(foreign_docs(hits, "slide-10"), ["slide-08×1"])

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


CHOICE = {
    "id": "c1",
    "question": "下列关于替代定理适用条件的说法，哪一项是错误的？",
    "negative": False,
    "expect_pages": [25, 26],
    "choices": ["必须有唯一解", "不能有电压源回路", "可以出现电流源节点", "应无耦合关系"],
    "expected_choice": "C",
}


class TestWorksheetAndDraftRuns(unittest.TestCase):
    """填写闭环：工作单看原文 → 试跑草稿 → 满意再改 ready。"""

    def setUp(self) -> None:
        from tests.support import make_settings

        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.settings = make_settings(self.root)
        pages = self.settings.paths.doc_pages("doc-x")
        pages.mkdir(parents=True, exist_ok=True)
        for page_no in (10, 11):
            (pages / f"doc-x_page_{page_no:04d}.md").write_text(
                f"# 第{page_no}页内容\n\n叠加定理的定义。", encoding="utf-8"
            )
            (pages / f"doc-x_page_{page_no:04d}.jpg").write_bytes(b"\xff\xd8\xff\xd9")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def draft(self, **kwargs) -> GoldenCase:
        base = dict(
            id="draft-p09",
            question="【待写·问答】叠加定理能不能叠加功率？",
            expect_pages=[10, 11],
            chapter="第4章 电路定理",
            source_doc="doc-x",
            status="draft",
        )
        base.update(kwargs)
        return GoldenCase(**base)

    def test_worksheet_includes_source_pages_and_image_paths(self) -> None:
        text = build_worksheet([self.draft()], self.settings)
        self.assertIn("draft-p09", text)
        self.assertIn("# 第10页内容", text)
        self.assertIn("doc-x_page_0010.jpg", text)
        self.assertIn("预填的页码只是起点", text)

    def test_worksheet_says_negative_has_no_pages(self) -> None:
        case = GoldenCase(id="draft-n01", question="回转器怎么用？", negative=True, status="draft")
        text = build_worksheet([case], self.settings)
        self.assertIn("负样本", text)
        self.assertIn("不该命中任何页", text)

    def test_worksheet_flags_missing_page(self) -> None:
        text = build_worksheet([self.draft(expect_pages=[99])], self.settings)
        self.assertIn("找不到这一页", text)

    def test_draft_can_be_run_when_explicitly_requested(self) -> None:
        """草稿必须能试跑——否则「改之前先验」这个动作做不了。"""
        from tests.support import FakeEmbedder

        case = self.draft()
        report = run_golden(
            [case],
            self.settings,
            store=FakeStore([]),
            embed_client=FakeEmbedder(),
            generate_client=FakeGenerator(),
            retrieval_only=True,
            include_ids={"draft-p09"},
        )
        self.assertEqual(len(report.results), 1)
        self.assertEqual(report.results[0].case.id, "draft-p09")

    def test_draft_is_skipped_when_not_requested(self) -> None:
        from tests.support import FakeEmbedder

        report = run_golden(
            [self.draft()],
            self.settings,
            store=FakeStore([]),
            embed_client=FakeEmbedder(),
            generate_client=FakeGenerator(),
            retrieval_only=True,
        )
        self.assertEqual(report.results, [])


class TestPlanStatus(unittest.TestCase):
    """骨架的意义：让「还差多少题」一直是可见的，而不是靠记忆。"""

    def test_counts_ready_and_drafts_separately(self) -> None:
        ready_pos = GoldenCase(id="p1", question="q", expect_pages=[1])
        ready_neg = GoldenCase(id="n1", question="q", negative=True)
        ready_choice = GoldenCase(
            id="c1",
            question="q",
            expect_pages=[1],
            choices=["甲", "乙"],
            expected_choice="A",
        )
        draft = GoldenCase(id="d1", question="q", status="draft")
        status = plan_status([ready_pos, ready_neg, ready_choice, draft])
        self.assertEqual(status.ready_positive, 2)
        self.assertEqual(status.ready_negative, 1)
        self.assertEqual(status.ready_choice, 1)
        self.assertEqual(status.drafts, 1)
        self.assertEqual(status.ready_total, 3)

    def test_reports_remaining_quota(self) -> None:
        status = plan_status([GoldenCase(id="p1", question="q", expect_pages=[1])])
        self.assertFalse(status.complete)
        self.assertEqual(
            status.missing["负样本"], PLAN_TARGETS["negative"]
        )
        self.assertEqual(
            status.missing["正样本"], PLAN_TARGETS["positive"] - 1
        )

    def test_complete_when_all_targets_met(self) -> None:
        cases = []
        for index in range(PLAN_TARGETS["positive"]):
            is_choice = index < PLAN_TARGETS["choice"]
            cases.append(
                GoldenCase(
                    id=f"p{index}",
                    question="q",
                    expect_pages=[1],
                    choices=["甲", "乙"] if is_choice else [],
                    expected_choice="A" if is_choice else "",
                )
            )
        for index in range(PLAN_TARGETS["negative"]):
            cases.append(GoldenCase(id=f"n{index}", question="q", negative=True))
        status = plan_status(cases)
        self.assertTrue(status.complete)
        self.assertTrue(
            any("已达标" in line for line in status.lines())
        )


class TestChoiceQuestions(unittest.TestCase):
    """选择题：唯一正确答案，能机械判分。

    这是整套指标里唯一能可靠发现「结论说错」的形式——问答题只能靠
    要点覆盖率那种模糊信号，选择题可以直接判对错。
    """

    def test_extracts_answer_prefix(self) -> None:
        self.assertEqual(extract_choice("答案：B\n理由如下（第25页）"), "B")
        self.assertEqual(extract_choice("答案: c"), "C")
        self.assertEqual(extract_choice("**答案：** D"), "D")

    def test_extracts_inline_phrasings(self) -> None:
        self.assertEqual(extract_choice("所以应选B，因为……"), "B")
        self.assertEqual(extract_choice("正确选项是A。"), "A")
        self.assertEqual(extract_choice("正确的是C"), "C")

    def test_extracts_line_leading_option(self) -> None:
        self.assertEqual(extract_choice("A. 第一条理由"), "A")

    def test_returns_empty_when_no_option(self) -> None:
        self.assertEqual(extract_choice("替代定理适用于线性电路（第25页）。"), "")

    def test_does_not_match_letters_inside_words(self) -> None:
        # ABC 公司、KCL 这类不该被当成选项
        self.assertEqual(extract_choice("ABC 公司的教材里 KCL 是这样写的"), "")

    def test_compose_question_appends_options(self) -> None:
        case = load_cases(write_cases([CHOICE]))[0]
        text = compose_question(case)
        self.assertIn("A. 必须有唯一解", text)
        self.assertIn("D. 应无耦合关系", text)
        self.assertTrue(text.startswith(CHOICE["question"]))

    def test_compose_question_leaves_plain_question_untouched(self) -> None:
        case = GoldenCase(id="p1", question="替代定理的适用条件？", expect_pages=[1])
        self.assertEqual(compose_question(case), "替代定理的适用条件？")

    def test_choice_letters_follow_order(self) -> None:
        case = load_cases(write_cases([CHOICE]))[0]
        self.assertEqual(case.choice_letters, ["A", "B", "C", "D"])
        self.assertTrue(case.is_choice)

    def test_too_few_choices_is_rejected(self) -> None:
        bad = {**CHOICE, "choices": ["只有一个"], "expected_choice": "A"}
        with self.assertRaises(GoldenError) as ctx:
            load_cases(write_cases([bad]))
        self.assertIn("至少", str(ctx.exception))

    def test_out_of_range_answer_is_rejected(self) -> None:
        bad = {**CHOICE, "expected_choice": "F"}
        with self.assertRaises(GoldenError) as ctx:
            load_cases(write_cases([bad]))
        self.assertIn("expected_choice", str(ctx.exception))

    def test_expected_choice_without_choices_is_rejected(self) -> None:
        bad = {**CHOICE, "choices": []}
        with self.assertRaises(GoldenError) as ctx:
            load_cases(write_cases([bad]))
        self.assertIn("choices", str(ctx.exception))

    def test_choice_accuracy_metric(self) -> None:
        case = load_cases(write_cases([CHOICE]))[0]
        right = CaseResult(case=case, mode="kb", chosen="C", choice_correct=True)
        wrong = CaseResult(case=case, mode="kb", chosen="A", choice_correct=False)
        missing = CaseResult(case=case, mode="kb", chosen="", choice_correct=None)
        self.assertAlmostEqual(GoldenReport(results=[right]).choice_accuracy, 1.0)
        self.assertAlmostEqual(GoldenReport(results=[wrong]).choice_accuracy, 0.0)
        self.assertAlmostEqual(
            GoldenReport(results=[right, wrong]).choice_accuracy, 0.5
        )
        # 没答出选项也算错，不能因为「判不了」就当对
        self.assertAlmostEqual(GoldenReport(results=[missing]).choice_accuracy, 0.0)

    def test_report_lists_wrong_choices(self) -> None:
        case = load_cases(write_cases([CHOICE]))[0]
        report = GoldenReport(
            results=[
                CaseResult(
                    case=case,
                    mode="kb",
                    chosen="A",
                    choice_correct=False,
                    citations=[25],
                    answer="答案：A\n……",
                )
            ]
        )
        text = render_report(
            report,
            thresholds={"hit_rate": 0.8, "fabrication_rate": 0.05},
            generated_at="now",
        )
        self.assertIn("选择题答错", text)
        self.assertIn("选了 A，正确是 C", text)

    def test_report_marks_unreadable_choice(self) -> None:
        case = load_cases(write_cases([CHOICE]))[0]
        report = GoldenReport(
            results=[CaseResult(case=case, mode="kb", chosen="", answer="……")]
        )
        text = render_report(
            report,
            thresholds={"hit_rate": 0.8, "fabrication_rate": 0.05},
            generated_at="now",
        )
        self.assertIn("没能从回答里识别出选项", text)

    def test_negative_choice_is_not_scored_for_correctness(self) -> None:
        case = GoldenCase(
            id="n1", question="q", negative=True, choices=["甲", "乙"], expected_choice="A"
        )
        self.assertEqual(CaseResult(case=case, mode="kb").choice_rate, 1.0)


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
