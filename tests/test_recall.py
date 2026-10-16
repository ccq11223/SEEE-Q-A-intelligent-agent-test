"""补漏校对：把「模型漏掉的文字」补回来。

存在的理由：实测第一遍转写会稳定地漏掉图上的短标签、口诀、例题答案这类
孤立文字（例如某页例题只抄了设方程过程、答案 U2=4.86V 整个丢掉）。
temperature=0 时重跑不会变，所以需要一次「拿原图和已有结果再问一遍」的专门校对。
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pipeline.ai_parser import (
    RECALL_EMPTY,
    RECALL_MARKER,
    _clean_recall,
    filter_new_lines,
    judge_recall,
    merge_recall,
    recall_page,
)
from pipeline.convert import PageAsset
from pipeline.llm import LLMError
from pipeline.manifest import ManifestEntry
from pipeline.settings import Settings
from tests.support import FakeModelConfig, copy_config_files


class FakeVision:
    def __init__(self, reply: str = RECALL_EMPTY) -> None:
        self.config = FakeModelConfig(name="fake-vision")
        self.reply = reply
        self.calls = 0
        self.last_user = ""

    def chat_with_images(self, *, system, user, image_paths, max_tokens=None) -> str:
        self.calls += 1
        self.last_user = user
        return self.reply

    def chat(self, *, system, user, max_tokens=None) -> str:  # pragma: no cover
        raise AssertionError("补漏必须带图片")


class FailingVision(FakeVision):
    def chat_with_images(self, *, system, user, image_paths, max_tokens=None) -> str:
        self.calls += 1
        raise LLMError("vision 调用被拒绝（HTTP 403）：Access denied")


class TestCleanRecall(unittest.TestCase):
    def test_empty_reply_means_nothing_missing(self) -> None:
        self.assertEqual(_clean_recall(RECALL_EMPTY), "")

    def test_plain_chinese_nothing_missing(self) -> None:
        self.assertEqual(_clean_recall("这一页没有遗漏。"), "")

    def test_keeps_real_additions(self) -> None:
        text = "- 模相乘，角相加\n- $U_2 = 4.86\\ \\text{V}$"
        self.assertEqual(_clean_recall(text), text)

    def test_drops_declaration_lines_but_keeps_content(self) -> None:
        text = "没有遗漏以下内容：\n- 削弱型耦合"
        self.assertEqual(_clean_recall(text), "- 削弱型耦合")

    def test_blank_is_empty(self) -> None:
        self.assertEqual(_clean_recall("   \n  "), "")


class TestFilterNewLines(unittest.TestCase):
    """去重：补漏不能把已经存在的内容再写一遍。

    这条规则的由来：实测有一页补漏返回 10 行，其中 8 行是原结果里已有的公式，
    只有 2 行（两句口诀）是真正新增的。如果不去重，知识库里同一段会存两份。
    """

    def test_drops_line_already_present(self) -> None:
        existing = "## (2) 乘除运算——采用极坐标形式"
        added = "## (2) 乘除运算——采用极坐标形式"
        self.assertEqual(filter_new_lines(existing, added), "")

    def test_drops_line_that_differs_only_in_punctuation_and_spacing(self) -> None:
        existing = "$A_1 = |A_1|e^{\\mathrm{j}\\theta_1}$，$A_2 = |A_2|e^{\\mathrm{j}\\theta_2}$"
        added = "$A_1=|A_1|e^{\\mathrm{j}\\theta_1}$,$A_2=|A_2|e^{\\mathrm{j}\\theta_2}$"
        self.assertEqual(filter_new_lines(existing, added), "")

    def test_keeps_genuinely_new_line(self) -> None:
        existing = "# 复数运算\n\n加法公式"
        added = "- 乘法：模相乘，角相加"
        self.assertEqual(filter_new_lines(existing, added), "- 乘法：模相乘，角相加")

    def test_keeps_new_line_when_only_a_short_fragment_matches(self) -> None:
        # 「电压」这类两字片段重合不算重复，否则会误删新内容
        existing = "电压的方向与电流的关联参考方向"
        added = "- 电压表量程为 50V"
        self.assertEqual(filter_new_lines(existing, added), "- 电压表量程为 50V")

    def test_dedups_short_line_that_already_exists(self) -> None:
        # 短行也要去重：实测「举例说明：」「$u_C'$」这类短行被重复加过
        self.assertEqual(filter_new_lines("模相除", "- 模相除"), "")
        self.assertEqual(filter_new_lines("## 举例说明：", "- 举例说明："), "")

    def test_keeps_short_line_that_is_new(self) -> None:
        self.assertEqual(filter_new_lines("已有正文", "- 削弱型耦合"), "- 削弱型耦合")

    def test_blank_lines_are_dropped(self) -> None:
        self.assertEqual(filter_new_lines("正文", "  \n\n  "), "")

    def test_drops_empty_math_fences(self) -> None:
        added = "$$\n$$\n- 真正的公式 $x=1$"
        self.assertEqual(filter_new_lines("正文", added), "- 真正的公式 $x=1$")

    def test_drops_lone_page_number(self) -> None:
        self.assertEqual(filter_new_lines("正文", "- 13", page_no=13), "")
        self.assertEqual(filter_new_lines("正文", "- 7"), "")

    def test_keeps_meaningful_numbers(self) -> None:
        self.assertEqual(filter_new_lines("正文", "- 4.86"), "- 4.86")
        self.assertEqual(filter_new_lines("正文", "- U_2 = 4.86V"), "- U_2 = 4.86V")

    def test_drops_short_symbol_only_lines(self) -> None:
        self.assertEqual(filter_new_lines("正文", "=\nX\nY"), "")


class TestCleanRecallDropsMetaTalk(unittest.TestCase):
    """补漏模型偶尔会把思考过程当结果写出来（实测发生过）。"""

    def test_drops_meta_reasoning_lines(self) -> None:
        text = (
            "等等，以上其实已在已有结果中。\n"
            "重新严格比对图片与已有转写：\n"
            "- 削弱型耦合"
        )
        self.assertEqual(_clean_recall(text), "- 削弱型耦合")

    def test_keeps_only_real_content(self) -> None:
        self.assertEqual(_clean_recall("- 三要素法\n- 彼德生法确定"), "- 三要素法\n- 彼德生法确定")


class TestJudgeRecall(unittest.TestCase):
    """补漏结果异常膨胀时必须停下来交给人看，而不是自动写进页文件。"""

    def test_normal_addition_is_not_suspicious(self) -> None:
        suspicious, reason = judge_recall("一" * 1000, "- 三要素法\n- 彼德生法确定")
        self.assertFalse(suspicious)
        self.assertEqual(reason, "")

    def test_empty_addition_is_not_suspicious(self) -> None:
        self.assertEqual(judge_recall("正文", "   "), (False, ""))

    def test_too_many_lines_is_suspicious(self) -> None:
        added = "\n".join(f"- 第{i}行内容" for i in range(50))
        suspicious, reason = judge_recall("正文", added)
        self.assertTrue(suspicious)
        self.assertIn("行", reason)

    def test_oversized_addition_is_suspicious(self) -> None:
        # 实测案例：已有约 1500 字，补漏返回 3000 字，里面大半是自言自语
        suspicious, reason = judge_recall("一" * 1500, "二" * 3000)
        self.assertTrue(suspicious)
        self.assertIn("超过", reason)


class TestMergeRecall(unittest.TestCase):
    def test_appends_with_marker(self) -> None:
        merged = merge_recall("# 标题\n\n正文", "- 漏掉的一行")
        self.assertIn("# 标题", merged)
        self.assertIn(RECALL_MARKER, merged)
        self.assertTrue(merged.rstrip().endswith("- 漏掉的一行"))
        self.assertLess(merged.index("# 标题"), merged.index(RECALL_MARKER))

    def test_empty_addition_leaves_file_untouched(self) -> None:
        original = "# 标题\n\n正文\n"
        self.assertEqual(merge_recall(original, "  "), original)

    def test_does_not_duplicate_marker_on_repeat(self) -> None:
        once = merge_recall("正文", "- A")
        twice = merge_recall(once, "- B")
        self.assertEqual(twice.count(RECALL_MARKER), 1)
        self.assertIn("- A", twice)
        self.assertIn("- B", twice)


class TestRecallPage(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        copy_config_files(self.root, embedding_dim=5)
        self.settings = Settings.load(self.root)
        image = self.settings.paths.doc_pages("doc-x") / "doc-x_page_0007.jpg"
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"\xff\xd8\xff\xd9")
        self.asset = PageAsset(page_no=7, image_path=image, has_embedded_image=True)
        self.entry = ManifestEntry(
            doc_key="doc-x",
            course_id="circuit-theory-1",
            doc_type="slide",
            trust_level=1,
            filename="doc.pdf",
            title="测试课件",
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_returns_additions(self) -> None:
        client = FakeVision("- 模相除，角相减")
        result = recall_page(self.entry, self.settings, self.asset, "已有正文", client)
        self.assertIsNone(result.error)
        self.assertEqual(result.added, "- 模相除，角相减")
        self.assertEqual(result.page_no, 7)
        self.assertIn("已有正文", client.last_user)
        self.assertIn("第 7 页", client.last_user)

    def test_no_missing_reports_empty(self) -> None:
        result = recall_page(
            self.entry, self.settings, self.asset, "已有正文", FakeVision()
        )
        self.assertEqual(result.added, "")
        self.assertIsNone(result.error)

    def test_api_failure_is_reported_not_raised(self) -> None:
        result = recall_page(
            self.entry, self.settings, self.asset, "已有正文", FailingVision()
        )
        self.assertIsNotNone(result.error)
        self.assertIn("403", result.error or "")
        self.assertEqual(result.added, "")

    def test_page_without_image_is_rejected(self) -> None:
        asset = PageAsset(page_no=7, image_path=None)
        result = recall_page(self.entry, self.settings, asset, "正文", FakeVision())
        self.assertIsNotNone(result.error)

    def test_prompt_truncates_very_long_existing_text(self) -> None:
        client = FakeVision()
        recall_page(
            self.entry, self.settings, self.asset, "字" * 20000, client
        )
        self.assertLess(len(client.last_user), 12000)


if __name__ == "__main__":
    unittest.main()
