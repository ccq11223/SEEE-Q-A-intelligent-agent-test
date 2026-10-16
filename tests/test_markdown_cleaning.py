"""转写结果的清洗规则。

规则来自一次真实的 5 页试水：模型把课件里的项目符号 ◆□ 一起转写进了标题，
而标题会作为章节名进入切块元数据、再出现在回答的引用里，属于纯噪声。
"""

from __future__ import annotations

import unittest

from pipeline.textutils import (
    count_figure_placeholders,
    count_figure_text_marks,
    normalize_markdown,
)


class TestBulletGlyphStripping(unittest.TestCase):
    def test_strips_glyph_from_heading(self) -> None:
        self.assertEqual(
            normalize_markdown("## □ 实际电路器件种类繁多、工作原理各异"),
            "## 实际电路器件种类繁多、工作原理各异",
        )

    def test_strips_glyph_from_list_items(self) -> None:
        self.assertEqual(
            normalize_markdown("- ◆ 如何获得电路模型\n1. ○ 第一条"),
            "- 如何获得电路模型\n1. 第一条",
        )

    def test_strips_multiple_leading_glyphs(self) -> None:
        self.assertEqual(normalize_markdown("### ■★ 标题"), "### 标题")

    def test_keeps_glyph_inside_body_text(self) -> None:
        # 正文中间出现的符号可能是有意义的，不能动
        text = "关键问题：◆如何获得电路模型"
        self.assertEqual(normalize_markdown(text), text)

    def test_keeps_non_glyph_content_untouched(self) -> None:
        text = "## 1.1 电路\n\n正文里的 ■ 符号\n\n- 正常列表项"
        self.assertEqual(normalize_markdown(text), text)

    def test_does_not_touch_headings_without_glyphs(self) -> None:
        self.assertEqual(normalize_markdown("# 第一章\n## 1.1 电路"), "# 第一章\n## 1.1 电路")


class TestFigureTextCounting(unittest.TestCase):
    def test_counts_placeholders(self) -> None:
        text = (
            "[图 第4页-1：电路分析流程图]\n"
            "[图上文字] 实际电路；电路模型；电路分析；分析结果\n"
            "[图 第4页-2：实际电路与电路模型对比图]\n"
            "[图上文字] 电压源 U_S；内阻 R_i；电阻 R_f；开关"
        )
        self.assertEqual(count_figure_placeholders(text), 2)
        self.assertEqual(count_figure_text_marks(text), 2)

    def test_zero_when_no_figure_text(self) -> None:
        text = "[图 第1页-1：示意图]\n\n正文"
        self.assertEqual(count_figure_placeholders(text), 1)
        self.assertEqual(count_figure_text_marks(text), 0)

    def test_tolerates_extra_spaces(self) -> None:
        self.assertEqual(count_figure_text_marks("[ 图上文字 ] a；b"), 1)


if __name__ == "__main__":
    unittest.main()
