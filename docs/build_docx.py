# -*- coding: utf-8 -*-
"""Render docs/_plan_source.md into a formatted Word document."""
import os
import re
import sys

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "课程辅导智能体实施方案.md")
OUT = os.path.join(HERE, "课程辅导智能体实施方案_v1.0.docx")

BODY_FONT = "微软雅黑"
MONO_FONT = "Consolas"
DARK = RGBColor(0x1A, 0x1A, 0x1A)


def style_font(style, name, size, bold=None, color=None):
    style.font.name = name
    style.font.size = Pt(size)
    if bold is not None:
        style.font.bold = bold
    if color is not None:
        style.font.color.rgb = color
    rpr = style.element.get_or_add_rPr()
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = rpr.makeelement(qn("w:rFonts"), {})
        rpr.append(rf)
    for attr in ("w:eastAsia", "w:ascii", "w:hAnsi"):
        rf.set(qn(attr), name)


def run_font(run, name, size, bold=None, color=None):
    run.font.name = name
    run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if color is not None:
        run.font.color.rgb = color
    rpr = run._element.get_or_add_rPr()
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = rpr.makeelement(qn("w:rFonts"), {})
        rpr.append(rf)
    for attr in ("w:eastAsia", "w:ascii", "w:hAnsi"):
        rf.set(qn(attr), name)


def add_inline(par, text, name=BODY_FONT, size=10.5, bold=False, color=DARK):
    for idx, part in enumerate(re.split(r"\*\*", text)):
        if part == "":
            continue
        is_bold = bold or idx % 2 == 1
        for sub_idx, sub in enumerate(part.split("`")):
            if sub == "":
                continue
            run = par.add_run(sub)
            run_font(
                run,
                MONO_FONT if sub_idx % 2 == 1 else name,
                size - 0.5 if sub_idx % 2 == 1 else size,
                bold=is_bold,
                color=color,
            )


def is_table_row(line):
    stripped = line.strip()
    return stripped.startswith("|") and stripped.endswith("|") and len(stripped) > 1


def is_separator_row(line):
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    return all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c != "")


def split_row(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def build():
    with open(SRC, encoding="utf-8") as handle:
        lines = handle.read().split("\n")

    doc = Document()
    style_font(doc.styles["Normal"], BODY_FONT, 10.5, color=DARK)
    doc.styles["Normal"].paragraph_format.space_after = Pt(4)
    doc.styles["Normal"].paragraph_format.line_spacing = 1.25
    style_font(doc.styles["Title"], BODY_FONT, 20, bold=True, color=DARK)
    style_font(doc.styles["Heading 1"], BODY_FONT, 16, bold=True, color=DARK)
    style_font(doc.styles["Heading 2"], BODY_FONT, 13, bold=True, color=DARK)
    style_font(doc.styles["Heading 3"], BODY_FONT, 11.5, bold=True, color=DARK)

    section = doc.sections[0]
    section.page_height = Cm(29.7)
    section.page_width = Cm(21.0)
    section.top_margin = Cm(2.2)
    section.bottom_margin = Cm(2.2)
    section.left_margin = Cm(2.0)
    section.right_margin = Cm(2.0)

    i = 0
    total = len(lines)
    while i < total:
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("```"):
            i += 1
            while i < total and not lines[i].strip().startswith("```"):
                code = lines[i].rstrip()
                par = doc.add_paragraph()
                par.paragraph_format.space_after = Pt(0)
                par.paragraph_format.line_spacing = 1.0
                par.paragraph_format.left_indent = Cm(0.3)
                run = par.add_run(code if code else " ")
                run_font(run, MONO_FONT, 7.5, color=RGBColor(0x28, 0x28, 0x28))
                i += 1
            i += 1
            continue

        if is_table_row(line):
            block = []
            while i < total and is_table_row(lines[i]):
                block.append(lines[i])
                i += 1
            rows = [split_row(r) for r in block if not is_separator_row(r)]
            if not rows:
                continue
            width = max(len(r) for r in rows)
            table = doc.add_table(rows=len(rows), cols=width)
            table.style = "Table Grid"
            table.autofit = True
            size = 9.0 if width <= 4 else 8.0
            for r_idx, row in enumerate(rows):
                for c_idx in range(width):
                    cell = table.cell(r_idx, c_idx)
                    cell.text = ""
                    par = cell.paragraphs[0]
                    par.paragraph_format.space_after = Pt(1)
                    par.paragraph_format.line_spacing = 1.0
                    text = row[c_idx] if c_idx < len(row) else ""
                    add_inline(
                        par,
                        text,
                        size=size,
                        bold=(r_idx == 0),
                    )
            doc.add_paragraph()
            continue

        if stripped.startswith("### "):
            doc.add_heading(stripped[4:], level=3)
        elif stripped.startswith("## "):
            doc.add_heading(stripped[3:], level=2)
        elif stripped.startswith("# "):
            par = doc.add_heading(stripped[2:], level=1)
            par.alignment = WD_ALIGN_PARAGRAPH.CENTER
        elif stripped.startswith("- "):
            par = doc.add_paragraph(style="List Bullet")
            par.paragraph_format.space_after = Pt(2)
            add_inline(par, stripped[2:])
        elif re.match(r"^\d+\.\s", stripped):
            par = doc.add_paragraph(style="List Number")
            par.paragraph_format.space_after = Pt(2)
            add_inline(par, re.sub(r"^\d+\.\s", "", stripped))
        elif stripped == "---":
            par = doc.add_paragraph()
            par.paragraph_format.space_after = Pt(0)
            run = par.add_run("")
            run_font(run, BODY_FONT, 4)
        elif stripped == "":
            pass
        else:
            par = doc.add_paragraph()
            add_inline(par, stripped)
        i += 1

    doc.save(OUT)
    print("written:", OUT)


if __name__ == "__main__":
    build()
