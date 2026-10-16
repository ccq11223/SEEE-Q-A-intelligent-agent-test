from __future__ import annotations

import unittest

from pipeline.textutils import (
    estimate_tokens,
    extract_citation_pages,
    has_formula,
    is_probably_empty,
    normalize_markdown,
    page_marker,
    split_sentences,
    strip_page_markers,
    validate_citations,
)


class TestNormalize(unittest.TestCase):
    def test_strips_markdown_fence(self) -> None:
        raw = "```markdown\n# 标题\n\n正文\n```"
        self.assertEqual(normalize_markdown(raw), "# 标题\n\n正文")

    def test_converts_escaped_latex_delimiters(self) -> None:
        raw = "电压 \\(u(t)\\) 满足 \\[u = iR\\]"
        result = normalize_markdown(raw)
        self.assertIn("$u(t)$", result)
        self.assertIn("$$u = iR$$", result)

    def test_removes_zero_width_and_extra_blank_lines(self) -> None:
        raw = "第一行\u200b\n\n\n\n第二行"
        self.assertEqual(normalize_markdown(raw), "第一行\n\n第二行")

    def test_strip_page_markers(self) -> None:
        text = f"{page_marker(1)}\n内容\n{page_marker(2)}\n更多内容"
        self.assertEqual(strip_page_markers(text), "内容\n更多内容")


class TestEmptyDetection(unittest.TestCase):
    def test_blank_page_is_empty(self) -> None:
        self.assertTrue(is_probably_empty(f"{page_marker(3)}\n\n##   \n"))

    def test_unclear_page_is_empty(self) -> None:
        self.assertTrue(is_probably_empty("[此处原图不清晰]"))

    def test_real_content_is_not_empty(self) -> None:
        self.assertFalse(is_probably_empty("基尔霍夫电流定律：任一节点电流代数和为零。"))


class TestFormula(unittest.TestCase):
    def test_detects_inline_math(self) -> None:
        self.assertTrue(has_formula("电流为 $i = u/R$"))

    def test_detects_latex_command(self) -> None:
        self.assertTrue(has_formula("相量记作 \\dot{U}"))

    def test_plain_text_is_false(self) -> None:
        self.assertFalse(has_formula("这是一段纯文字，没有公式。"))


class TestTokensAndSentences(unittest.TestCase):
    def test_estimate_tokens_counts_cjk_and_words(self) -> None:
        self.assertEqual(estimate_tokens("电阻 R1 是 10 欧"), 6)

    def test_split_sentences_keeps_terminators(self) -> None:
        self.assertEqual(
            split_sentences("第一句。第二句；第三句！"), ["第一句。", "第二句；", "第三句！"]
        )

    def test_split_sentences_handles_no_terminator(self) -> None:
        self.assertEqual(split_sentences("没有标点"), ["没有标点"])


class TestCitation(unittest.TestCase):
    def test_extracts_chinese_page_numbers(self) -> None:
        answer = "根据定义（第 3 页），电流方向与参考方向相反（第 12 页）。"
        self.assertEqual(extract_citation_pages(answer), [3, 12])

    def test_ignores_step_numbers(self) -> None:
        self.assertEqual(extract_citation_pages("第一步：写出方程；第二步：求解。"), [])

    def test_validate_passes_when_all_pages_exist(self) -> None:
        ok, invalid = validate_citations("见（第 2 页）和（第 5 页）", {2, 5, 9})
        self.assertTrue(ok)
        self.assertEqual(invalid, [])

    def test_validate_reports_out_of_range_pages(self) -> None:
        ok, invalid = validate_citations("见（第 2 页）和（第 99 页）", {2, 5})
        self.assertFalse(ok)
        self.assertEqual(invalid, [99])

    def test_validate_deduplicates(self) -> None:
        ok, invalid = validate_citations("（第 7 页）（第 7 页）（第 8 页）", {7})
        self.assertFalse(ok)
        self.assertEqual(invalid, [8])


if __name__ == "__main__":
    unittest.main()
