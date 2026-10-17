"""提示词模板的占位符自检。

踩过一次坑：正文里写了 LaTeX 的 ``\\begin{bmatrix}``，``str.format`` 把
``{bmatrix}`` 当成了占位符，直到真实提问才炸。这个测试把每个会被 format
的提示词的占位符集合钉死，多一个少一个都会立刻失败。
"""

from __future__ import annotations

import unittest
from pathlib import Path
from string import Formatter

from pipeline.settings import Settings

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPECTED = {
    ("qa_answer", "system"): {"course_name", "learner_block"},
    ("qa_answer", "user"): {"context", "question"},
    ("qa_answer_external", "system"): {"course_name", "mode_note", "learner_block"},
    ("qa_answer_external", "user"): {"context_block", "question"},
    ("assessment_grade", "system"): {"course_name"},
    ("assessment_grade", "user"): {"question", "answer", "points", "student_answer"},
}


class TestPromptTemplates(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.settings = Settings.load(PROJECT_ROOT)

    def test_placeholders_are_exactly_as_expected(self) -> None:
        for (section, key), expected in EXPECTED.items():
            text = self.settings.prompt(section, key)
            names = {
                name
                for _, name, _, _ in Formatter().parse(text)
                if name
            }
            self.assertEqual(
                names,
                expected,
                f"{section} > {key} 的占位符不对：{sorted(names)}",
            )

    def test_templates_can_be_rendered(self) -> None:
        values = {
            "course_name": "电路理论（上）",
            "learner_block": "",
            "mode_note": "【测试】",
            "context": "材料",
            "context_block": "",
            "question": "问题",
            "answer": "答案",
            "points": "要点",
            "student_answer": "学生回答",
        }
        for section, key in EXPECTED:
            text = self.settings.prompt(section, key).format(**values)
            if "course_name" in EXPECTED[(section, key)]:
                self.assertIn("电路理论", text)
